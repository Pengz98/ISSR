import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import argparse

class Transform_Net(nn.Module):
    def __init__(self, args):
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)


class DGCNN_partseg(nn.Module):
    def __init__(self, args, seg_num_all=50):
        super(DGCNN_partseg, self).__init__()
        self.args = args
        self.seg_num_all = seg_num_all
        self.k = args.k
        # self.transform_net = Transform_Net(args)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
        #                            self.bn7,
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1216, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                    self.bn10,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)

    def forward(self, inputs):
        x = inputs.point.reshape(1,-1,3).transpose(2,1)
        label = inputs.class_label.reshape(-1)
        label_one_hot = np.zeros((label.shape[0], 16))
        for idx in range(label.shape[0]):
            label_one_hot[idx, label[idx]] = 1
        l = torch.from_numpy(label_one_hot.astype(np.float32))
        x, l = x.to('cuda' if torch.cuda.is_available() else 'cpu'), l.to('cuda' if torch.cuda.is_available() else 'cpu')

        batch_size = x.size(0)
        num_points = x.size(2)

        # x0 = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        # t = self.transform_net(x0)  # (batch_size, 3, 3)
        # x = x.transpose(2, 1)  # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        # x = torch.bmm(x, t)  # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        # x = x.transpose(2, 1)  # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        # l = l.view(batch_size, -1, 1)  # (batch_size, num_categoties, 1)
        # l = self.conv7(l)  # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        # x = torch.cat((x, l), dim=1)  # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)  # (batch_size, 1088, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1088+64*3, num_points)

        x = self.conv8(x)  # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)  # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv11(x)  # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)

        return x.transpose(2,1).reshape(-1,self.seg_num_all)


DGCNN_partset_parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
DGCNN_partset_parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                    help='Name of the experiment')
DGCNN_partset_parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                    choices=['dgcnn'],
                    help='Model to use, [dgcnn]')
DGCNN_partset_parser.add_argument('--dataset', type=str, default='shapenetpart', metavar='N',
                    choices=['shapenetpart'])
DGCNN_partset_parser.add_argument('--class_choice', type=str, default=None, metavar='N',
                    choices=['airplane', 'bag', 'cap', 'car', 'chair',
                             'earphone', 'guitar', 'knife', 'lamp', 'laptop',
                             'motor', 'mug', 'pistol', 'rocket', 'skateboard', 'table'])
DGCNN_partset_parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                    help='Size of batch)')
DGCNN_partset_parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                    help='Size of batch)')
DGCNN_partset_parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of episode to train ')
DGCNN_partset_parser.add_argument('--use_sgd', type=bool, default=True,
                    help='Use SGD')
DGCNN_partset_parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001, 0.1 if using sgd)')
DGCNN_partset_parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
DGCNN_partset_parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                    choices=['cos', 'step'],
                    help='Scheduler to use, [cos, step]')
DGCNN_partset_parser.add_argument('--no_cuda', type=bool, default=False,
                    help='enables CUDA training')
DGCNN_partset_parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
DGCNN_partset_parser.add_argument('--eval', type=bool,  default=True,
                    help='evaluate the model')
DGCNN_partset_parser.add_argument('--num_points', type=int, default=2048,
                    help='num of points to use')
DGCNN_partset_parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout rate')
DGCNN_partset_parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                    help='Dimension of embeddings')
DGCNN_partset_parser.add_argument('--k', type=int, default=40, metavar='N',
                    help='Num of nearest neighbors to use')
DGCNN_partset_parser.add_argument('--model_path', type=str, default='/home/vcg8004/Tsuki/segmentation/PCT-dgcnn.pytorch-master/pretrained/model.partseg.t7', metavar='N',
                    help='Pretrained model path')
DGCNN_partset_parser.add_argument('--visu', type=str, default='',
                    help='visualize the model')
DGCNN_partset_parser.add_argument('--visu_format', type=str, default='ply',
                    help='file format of visualization')


class DGCNN_semseg_s3dis(nn.Module):
    def __init__(self, args):
        super(DGCNN_semseg_s3dis, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(18, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, 13, kernel_size=1, bias=False)

    def forward(self, x):
        x = x.input.reshape(1, -1, 9).transpose(2, 1)
        x = x.to('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k, dim9=True)  # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)  # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)  # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)  # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        # x = self.dp1(x)

        feat = x
        x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 13, num_points)

        return x.transpose(2,1).reshape(-1,13)

DGCNN_semseg_parser = argparse.ArgumentParser(description='Point Cloud Semantic Segmentation')
DGCNN_semseg_parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                    help='Name of the experiment')
DGCNN_semseg_parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                    choices=['dgcnn'],
                    help='Model to use, [dgcnn]')
DGCNN_semseg_parser.add_argument('--dataset', type=str, default='S3DIS', metavar='N',
                    choices=['S3DIS'])
DGCNN_semseg_parser.add_argument('--test_area', type=str, default='6', metavar='N',
                    choices=['1', '2', '3', '4', '5', '6', 'all'])
DGCNN_semseg_parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size',
                    help='Size of batch)')
DGCNN_semseg_parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size',
                    help='Size of batch)')
DGCNN_semseg_parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of episode to train ')
DGCNN_semseg_parser.add_argument('--use_sgd', type=bool, default=True,
                    help='Use SGD')
DGCNN_semseg_parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001, 0.1 if using sgd)')
DGCNN_semseg_parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
DGCNN_semseg_parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                    choices=['cos', 'step'],
                    help='Scheduler to use, [cos, step]')
DGCNN_semseg_parser.add_argument('--no_cuda', type=bool, default=False,
                    help='enables CUDA training')
DGCNN_semseg_parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 1)')
DGCNN_semseg_parser.add_argument('--eval', type=bool, default=False,
                    help='evaluate the model')
DGCNN_semseg_parser.add_argument('--num_points', type=int, default=4096,
                    help='num of points to use')
DGCNN_semseg_parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout rate')
DGCNN_semseg_parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                    help='Dimension of embeddings')
DGCNN_semseg_parser.add_argument('--k', type=int, default=20, metavar='N',
                    help='Num of nearest neighbors to use')
DGCNN_semseg_parser.add_argument('--model_root', type=str, default='pretrained/semseg_s3dis/', metavar='N',
                    help='Pretrained model root')
DGCNN_semseg_parser.add_argument('--visu', type=str, default='',
                    help='visualize the model')
DGCNN_semseg_parser.add_argument('--visu_format', type=str, default='ply',
                    help='file format of visualization')