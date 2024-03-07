import open3d as o3d
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
import sklearn.metrics.pairwise
from open3d.ml.torch.modules import losses, metrics
from open3d._ml3d.torch.modules import filter_valid_label
from open3d.ml.torch.datasets import InferenceDummySplit

import utils.models

SemSegLoss = losses.SemSegLoss
SemSegMetric = metrics.SemSegMetric
from open3d.ml.torch.dataloaders import get_sampler, TorchDataloader
from open3d.ml.torch.datasets import InferenceDummySplit

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

import math
import numpy as np
import os
import sys
import copy
import time
import argparse
from tqdm import tqdm
from datetime import datetime
import logging
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree


timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

log_dir = os.path.join('my_logs/' + timestamp)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file_path = os.path.join(log_dir, 'log.txt')
logging.basicConfig(filename=log_file_path, filemode='w', level=logging.INFO)
log = logging.getLogger(__name__)
log.addHandler(logging.FileHandler(log_file_path))

os.system(f'cp InterPCSeg.py {log_dir}')

from utils.vis_utils import vis, vis_and_interact
from utils.adapt_loss import softmax_entropy, cross_entropy
from utils.sample_utils import voxel_sampling, random_sampling

from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity


class InterPCSeg_pipeline():
    def __init__(self, model, ckpt_path, name='default', max_num_click=10, color_map=None):
        self.name = name
        self.color_map = color_map
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.model.to(self.device)
        self.model.device = self.device

        self.method = 'ours'
        self.reg_strength = 0.005


        if self.name=='scannet_sparseconvunet':
            self.model_copy = copy.deepcopy(model)
            self.model_copy.to(self.device)
            self.model_copy.device = self.device


        self.warm_lr = 5e-3
        self.lr = 1e-3

        self.momentum = 0 #default:0
        self.betas = (0,0) #default:0,0

        self.optim_weight_decay = 1e-2

        # self.optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.0, weight_decay=1e-2)
        if self.name == 's3dis_pointtransformer':
            self.optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.optim_weight_decay)
        else:
            self.optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.lr, betas=self.betas, weight_decay=self.optim_weight_decay)

        try:
            self.model_params = torch.load(ckpt_path, map_location=self.device)['model_state_dict']
        except:
            self.model_params = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(self.model_params)

        self.max_all_round = 500
        self.interact_mode = 'simulated'

        self.alpha = 1.
        self.beta = 100.
        self.entropy_threshold_increase = 0.1
        self.entropy_threshold_decrease = 0.01
        self.DBSCAN_eps=0.1
        self.DBSCAN_min_cluster = 10

        self.warm_round = 5
        self.adapt_round = 3
        self.max_adapt_round=5

        self.unsup_loss_record = np.zeros((3,))
        self.max_optimize_round = 10
        self.max_num_click = max_num_click
        self.num_clicks_per_round = 1

        # self.Loss = SemSegLoss(self, model, model, self.device)
        if 's3dis' in self.name:
            self.num_classes = 13
        else:
            self.num_classes = 20

        self.min_error_size = 100
        self.expected_IoU = [0.85, 0.9, 0.95]
        # self.expected_IoU = [0.7, 0.8, 0.8]

        self.use_warm_up = False
        self.use_confidence_filtering = True

    def input_preprocess_ml3d_s3dis(self, data):
        from open3d.ml.torch.dataloaders import get_sampler, TorchDataloader, ConcatBatcher
        from open3d.ml.torch.datasets import InferenceDummySplit
        infer_dataset = InferenceDummySplit(data)
        infer_sampler = infer_dataset.sampler
        infer_split = TorchDataloader(dataset=infer_dataset,
                                      preprocess=self.model.preprocess,
                                      transform=self.model.transform,
                                      sampler=infer_sampler,
                                      use_cache=False)
        infer_loader = DataLoader(infer_split,
                                  batch_size=1,
                                  sampler=get_sampler(infer_sampler),
                                  collate_fn=ConcatBatcher(self.device, self.model.cfg.name).collate_fn)
        self.model.trans_point_sampler = infer_sampler.get_point_sampler()
        infer_data = next(iter(infer_loader))
        return infer_data['data']

    def metric_cal(self, pred):
        metric = SemSegMetric()
        metric.reset()
        if self.name == 'scannet_sparseconvunet':
            # pred, label = filter_valid_label(pred, self.label, self.num_classes, [-1], self.device)
            metric.update(pred[self.valid_idx, :], self.label[self.valid_idx])
        else:
            metric.update(pred, self.label)
        return metric

    def test_time_warm_up(self, inputs, num_warm_rounds=5):
        self.model.eval()
        with torch.no_grad():
            eval_results = self.model(inputs)
        if eval_results.shape[0] == 1:
            eval_results = eval_results[0]
        target = eval_results.max(-1)[1].detach()

        self.initial_pred = target.cpu()

        eval_metric = self.metric_cal(eval_results.cpu().data)
        # self.vis_on_point(target.cpu().data.reshape(-1,1),
        #                   hint=f'(eval_mode)acc: {eval_metric.acc()[-1]:.2f}, iou:{eval_metric.iou()[-1]:.2f}')
        self.model.train()

        # self.model_config_comparison()
        # self.model_config()
        # self.model_config_comparison_1()
        # optim = torch.optim.SGD(self.model.parameters(), lr=5e-3, momentum=0.9, weight_decay=1e-2)
        if self.name == 's3dis_pointtransformer':
            optim = torch.optim.SGD(self.model.parameters(), lr=self.warm_lr, momentum=0.9, weight_decay=self.optim_weight_decay)
        else:
            optim = torch.optim.Adam(self.model.parameters(),
                                     lr=self.warm_lr, betas=(0.9, 0.99), weight_decay=self.optim_weight_decay)

        break_flag = False
        max_warmup_rounds = num_warm_rounds
        start_time = time.time()
        for i in range(max_warmup_rounds):
            optim.zero_grad()
            train_results = self.model(inputs)
            if train_results.shape[0] == 1:
                train_results = train_results[0]
            train_pred = train_results.max(1)[1]

            train_metric = self.metric_cal(train_results.cpu().data)
            # self.vis_on_point(train_pred.cpu().data.reshape(-1, 1),
            #                   hint=f'(eval_mode)acc: {train_metric.acc()[-1]:.2f}, iou:{train_metric.iou()[-1]:.2f}')

            loss_s = softmax_entropy(train_results).mean()
            loss_t = cross_entropy(train_results[self.valid_idx], target[self.valid_idx])
            loss = loss_t + loss_s
            # loss = loss_t
            loss.backward()
            # break_flag = self.check_stop_optimize(loss.item(), i)
            # if break_flag:
            #     print(f'model warmed after {i} rounds')
            #     break
            optim.step()
        end_time = time.time()
        print(f'time cost for warm-up: {end_time-start_time}')
        if not break_flag:
            print(
                f'model warmed after {max_warmup_rounds} rounds, acc({eval_metric.acc()[-1]}->{train_metric.acc()[-1]}),iou({eval_metric.iou()[-1]}->{train_metric.iou()[-1]})')
        warmed_params = copy.deepcopy(self.model.state_dict())
        self.warmed_params = warmed_params
        self.model.load_state_dict(warmed_params)
        self.model.requires_grad_(True)
        return eval_metric.acc()[-1], eval_metric.iou()[-1]

    def vis_on_point(self, feat, hint='default', max_value=None, use_color_map=False, heat_map=False):
        if type(feat) is not torch.Tensor:
            feat = torch.from_numpy(feat)
        if self.color_map is not None and use_color_map:
            color_np = np.concatenate(list(self.color_map.values())).reshape(-1, 3)
            color = torch.from_numpy(color_np[feat.reshape(-1)])
            vis(torch.cat([self.point, color.reshape(-1, 3)], -1), hint=hint, max_value=max_value, heat_map=heat_map)
        else:
            vis(torch.cat([self.point, feat], -1), hint=hint, max_value=max_value, heat_map=heat_map)

    def vis_with_interact(self, pred):
        hint = f'{len(self.picked_idx_buffer)}clicks; acc:{self.metric.acc()[-1]:.2f}, iou:{self.metric.iou()[-1]:.2f}'
        if self.color_map is not None:
            color_np = np.concatenate(list(self.color_map.values())).reshape(-1, 3)
            color = torch.from_numpy(color_np[pred])
            picked_idx = vis_and_interact(torch.cat([self.point, color.reshape(-1, 3)], -1), hint=hint)
        else:
            picked_idx = vis_and_interact(torch.cat([self.point, pred.reshape(-1, 1)], -1), hint=hint)
        return picked_idx

    def vis_clicks_on_pred(self, pred):
        hint = f'{len(self.picked_idx_buffer)}clicks; acc:{self.metric.acc()[-1]:.2f}, iou:{self.metric.iou()[-1]:.2f}'
        if self.color_map is not None:
            color_map = np.concatenate(list(self.color_map.values())).reshape(-1, 3)
            colors = color_map[pred]
        else:
            colors = plt.get_cmap("tab20")(pred / pred.max())[:, :3]
        colors[self.picked_idx_buffer, :] = [0, 0, 0]
        colors[np.where(pred==-1)] = [0,0,0]

        pc_np = self.point.cpu().data.numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_np[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(colors)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=hint)
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()

    def one_hot_encoding(self, pred):
        one_hots = torch.zeros(pred.shape[0], self.num_classes)
        one_hots[np.arange(pred.shape[0]), pred] = 1
        return one_hots

    def simulate_user_clicks(self, error_map, num_clicks=None, random=True, num_point_threashold=100):
        point_np = self.point.cpu().data.numpy()

        if self.name=='scannet_sparseconvunet':
            centroid = np.mean(point_np, axis=0)
            point_np = point_np - centroid
            m = np.max(np.sqrt(np.sum(point_np ** 2, axis=1)))
            point_np = point_np / m

        error_map_np = error_map.cpu().data.numpy()
        # self.vis_on_point(error_map_np.reshape(-1,1),'error_map')
        error_idx = np.where(error_map_np == 1)[0]

        db = DBSCAN(eps=self.DBSCAN_eps, min_samples=self.DBSCAN_min_cluster).fit(point_np[error_idx])
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        # tsne_visualization_color(point_np[error_idx], labels)

        cluster_num_point = []
        cluster_labels = []
        for l in np.unique(labels):
            if l != -1:
                cluster = point_np[error_idx[labels == l]]
                cluster_num_point.append(cluster.shape[0])
                cluster_labels.append(l)
        cluster_num_point = np.array(cluster_num_point)
        cluster_labels = np.array(cluster_labels)
        candidate_labels = cluster_labels[np.argsort(cluster_num_point)[-num_clicks:]]

        cluster_centroids = []
        # density_heat_map = -1. * np.ones((point_np.shape[0],)).astype('float')
        for l in np.unique(candidate_labels):
            if l != -1:
                cluster = point_np[error_idx[labels == l]]
                if cluster.shape[0]>1000:
                    cluster, samp_idx = random_sampling(cluster, num_sample=1000)
                else:
                    samp_idx = np.arange(cluster.shape[0])
                kd = KernelDensity()
                kd.fit(cluster)
                score = kd.score_samples(cluster)
                cluster_centroid = score.argsort()[-10:]
                cluster_centroid = samp_idx[cluster_centroid]
                cluster_centroid = error_idx[np.where(labels == l)[0][cluster_centroid]]
                cluster_centroids.append(cluster_centroid)

                # density_heat_map[error_idx[labels == l]] = (score - score.min()) / (score.max() - score.min())
        # self.vis_on_point(density_heat_map.reshape(-1,1), max_value=1., heat_map=True)

        colors = -1 * np.ones((point_np.shape[0],))
        colors[error_idx] = labels
        # colors[cluster_centroids] = labels.max() + 1
        # self.vis_on_point(colors.reshape(-1,1))


        if num_clicks is None:
            num_big_cluster = np.sum(cluster_num_point > num_point_threashold)
            if num_big_cluster < 1:
                num_clicks = 1
            elif num_big_cluster > 3:
                num_clicks = 3
            else:
                num_clicks = num_big_cluster
            if self.metric.iou()[-1] > 0.85:
                num_clicks = 1

        user_clicks = []
        for c in range(len(cluster_centroids)):
            centroids = cluster_centroids[c]
            if random:
                user_click = np.random.choice(centroids, 1).item()
            else:
                user_click = centroids[-1]
            user_clicks.append(user_click)
        return user_clicks, cluster_num_point

    def filter_confident_data(self, inputs):
        # model_copy = self.model
        if self.name!='scannet_sparseconvunet':
            model_copy = copy.deepcopy(self.model)
        else:
            state_dict = copy.deepcopy(self.model.state_dict())
            self.model_copy.load_state_dict(state_dict)
            self.model_copy.train()
            model_copy = self.model_copy

        if self.name == 's3dis_pointtransformer':
            optimizer = optim.SGD(model_copy.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.optim_weight_decay)
        else:
            optimizer = torch.optim.Adam(model_copy.parameters(),
                                     lr=self.lr, betas=self.betas, weight_decay=self.optim_weight_decay)

        optimizer.zero_grad()
        results = model_copy(inputs)
        if results.shape[0] == 1:
            results = results[0]
        entropy = softmax_entropy(results)
        loss_entropy_weight = torch.ones((entropy.shape[0],))
        loss_entropy = torch.mul(loss_entropy_weight.to(self.device), entropy).mean()
        loss_pick = cross_entropy(results[self.picked_idx_buffer, :],
                                  self.label[self.picked_idx_buffer].to(self.device),
                                  reduction='sum')
        loss = loss_pick + self.beta * loss_entropy
        loss.backward()
        optimizer.step()
        results_new = model_copy(inputs)
        if results_new.shape[0] == 1:
            results_new = results_new[0]
        entropy_new = softmax_entropy(results_new)
        entropy_change = entropy_new.cpu().data.numpy() - entropy.cpu().data.numpy()
        entropy_increase_idx = np.where(entropy_change > self.entropy_threshold_increase)[0]
        loss_entropy_weight = torch.ones((entropy.shape[0],))
        loss_entropy_weight[entropy_increase_idx] = 0.
        del optimizer
        if self.name!='scannet_sparseconvunet':
            del model_copy
        return loss_entropy_weight

    def run_interact(self, inputs):
        if self.name == 's3dis_pointtransformer':
            self.point = inputs.point
            self.label = inputs.label
            self.color = inputs.feat
            self.valid_idx = np.arange(self.label.shape[0]).astype('int')
        elif self.name == 'scannet_sparseconvunet':
            self.point = inputs.point[0]
            self.label = inputs.label[0]
            self.color = inputs.feat[0]
            self.valid_idx = np.where(self.label!=-1)[0]

        self.invalid_idx = np.arange(self.label.shape[0])
        invalid_mask = np.ones((self.label.shape[0],))
        invalid_mask[self.valid_idx] = 0
        self.invalid_idx = self.invalid_idx[invalid_mask==1]

        if hasattr(inputs, 'to'):
            inputs.to(self.device)

        self.model.load_state_dict(self.model_params)
        eval_acc, eval_iou = self.test_time_warm_up(inputs, num_warm_rounds=self.warm_round)

        # for n, m in self.model.named_modules():
        #     if 'encoders' in n:
        #         print(n)
        #         m.requires_grad_(True)
        #     else:
        #         m.requires_grad_(False)

        if not self.use_warm_up:
            self.model.load_state_dict(self.model_params)  # for checking the effectiveness of warmup

        noc = -1 * np.ones((3,))  # IoU 85, 90, 95
        self.picked_idx_buffer = []
        iou_record = []
        acc_record = []
        max_error_region_size = []
        mean_error_region_size = []
        interact_flag = True
        interact_round = 0


        # self.vis_build(f'eval acc:{eval_acc:.2f}, iou: {eval_iou:.2f}')
        # self.vis_on_point(self.color.reshape(-1,3))
        run_time_eval = False
        sum_time = 0.
        sum_cnt = 0
        start_time = time.time()
        for round in range(self.max_all_round):
            self.optimizer.zero_grad()
            '''do inference'''
            self.results = self.model(inputs)
            if self.results.shape[0] == 1:
                self.results = self.results[0]

            self.pred = self.results.cpu().data.max(1)[1]
            self.pred[self.invalid_idx] = -1

            if not run_time_eval:
                self.metric = self.metric_cal(self.results)

                if round == 0:
                    self.max_entropy_value = softmax_entropy(self.results[self.valid_idx]).cpu().data.numpy().max()

            # if (self.metric.iou()[-1] >= 0.90) or (len(self.picked_idx_buffer)>50):
            #     torch.cuda.empty_cache()
            #     break



            if interact_flag:
                end_time = time.time()
                print(f'time cost for test-time training: {end_time-start_time}')
                if round!=0:
                    sum_time += end_time-start_time
                    sum_cnt += 1
                    print(f'avg time cost for test-time training: {sum_time/sum_cnt}')

                if len(self.picked_idx_buffer) >= self.max_num_click:
                    torch.cuda.empty_cache()
                    break

                # just for visu exp
                # if len(self.picked_idx_buffer)!=0:
                #     pick_label = self.label[self.picked_idx_buffer[0]]
                #     color = np.ones_like(self.color)
                #     color[:,1:] = 0
                #     color[np.where(self.pred!=pick_label)] = [0.5,0.5,0.5]
                #     point1_idx, point2_idx = self.picked_idx_buffer[1], self.picked_idx_buffer[2]
                #     dist_th = square_distance(self.point[point1_idx].reshape(-1,3), self.point[point2_idx].reshape(-1,3))
                #     dist = square_distance(self.point[point1_idx].reshape(-1,3), self.point)
                #     to_balck_idx = np.where(dist.reshape(-1)<dist_th.reshape(-1))[0]
                #     color[to_balck_idx] = [0.5,0.5,0.5]
                #
                #     point3_idx, point4_idx = self.picked_idx_buffer[3], self.picked_idx_buffer[4]
                #     dist_th = square_distance(self.point[point3_idx].reshape(-1,3), self.point[point4_idx].reshape(-1,3))
                #     dist = square_distance(self.point[point3_idx].reshape(-1,3), self.point)
                #     to_balck_idx = np.where(dist.reshape(-1)<dist_th.reshape(-1))[0]
                #     color[to_balck_idx] = [0.5,0.5,0.5]
                #
                #     self.vis_on_point(color)
                #     print('ok')


                if self.interact_mode == 'real':
                    if eval_iou < 10:
                        '''info visualization before interact(necessary for experiment but not for practical use)'''
                        self.vis_on_point(self.label.reshape(-1, 1), hint='reference', use_color_map=True)
                        entropy = softmax_entropy(self.results).cpu().data.numpy()
                        entropy[self.invalid_idx] = -1
                        # self.vis_on_point(entropy.reshape(-1, 1), hint='entropy map', max_value=self.max_entropy_value)
                        error_map = 1.0 * (self.pred != self.label)
                        error_map[self.invalid_idx] = -1
                        self.vis_on_point(error_map.reshape(-1, 1), hint='error map')
                        picked_idx = self.vis_with_interact(self.pred.cpu().data)
                        if np.intersect1d(picked_idx,self.valid_idx).shape[0]==len(picked_idx):
                            self.picked_idx_buffer = self.picked_idx_buffer + picked_idx
                    else:
                        # picked_idx = [0, 0]
                        picked_idx = [round]

                    if not run_time_eval:
                        iou_record.append(self.metric.iou()[-1])
                        acc_record.append(self.metric.acc()[-1])
                        max_error_region_size.append(0)
                        mean_error_region_size.append(0)

                    if (np.bincount(picked_idx) > 1).sum():
                        # if len(self.picked_idx_buffer) > 10:
                        iou_record = iou_record + [iou_record[-1]] * (1 + self.max_num_click - len(iou_record))
                        acc_record = acc_record + [acc_record[-1]] * (1 + self.max_num_click - len(acc_record))
                        max_error_region_size = max_error_region_size + [max_error_region_size[-1]] * (
                                1 + self.max_num_click - len(max_error_region_size))
                        mean_error_region_size = mean_error_region_size + [mean_error_region_size[-1]] * (
                                1 + self.max_num_click - len(mean_error_region_size))
                        torch.cuda.empty_cache()
                        break
                    interact_round = round
                    interact_flag = False

                else:
                    iou_record.append(self.metric.iou()[-1])
                    acc_record.append(self.metric.acc()[-1])
                    error_map = 1.0 * (self.pred != self.label)
                    error_map[self.invalid_idx] = -1
                    if iou_record[-1] >= self.expected_IoU[0] and noc[0]==-1:
                        noc[0] = len(self.picked_idx_buffer)
                    if iou_record[-1] >= self.expected_IoU[1] and noc[1]==-1:
                        noc[1] = len(self.picked_idx_buffer)
                    if iou_record[-1] >= self.expected_IoU[2] and noc[2] == -1:
                        noc[2] = len(self.picked_idx_buffer)

                    # num_clicks = 5 if round==0 else 3
                    # if self.metric.iou()[-1] > 0.9:
                    #     num_clicks = 1
                    num_clicks = self.num_clicks_per_round
                    if (error_map==1).sum()!=0:
                        picked_idx, error_region_sizes = self.simulate_user_clicks(error_map, num_clicks=num_clicks, num_point_threashold=self.min_error_size)
                    else:
                        picked_idx = []
                        error_region_sizes = np.array([])
                    if error_region_sizes.shape[0] != 0:
                        max_error_size, mean_error_size = error_region_sizes.max(), error_region_sizes.mean()
                        max_error_region_size.append(max_error_size)
                        mean_error_region_size.append(mean_error_size)
                        print(f'click on the max error region with {max_error_size} points')
                        if iou_record[-1] > self.expected_IoU[2]:
                        # if max_error_size <= 150 and iou_record[-1] > 0.9:
                            # self.vis_on_point(error_map.reshape(-1,1), f'error_map(max error size:{max_error_size})')
                            iou_record = iou_record + [iou_record[-1]] * (1 + self.max_num_click - len(iou_record))
                            acc_record = acc_record + [acc_record[-1]] * (1 + self.max_num_click - len(acc_record))
                            max_error_region_size = max_error_region_size + [max_error_region_size[-1]] * (
                                        1 + self.max_num_click - len(max_error_region_size))
                            mean_error_region_size = mean_error_region_size + [mean_error_region_size[-1]] * (
                                        1 + self.max_num_click - len(mean_error_region_size))
                            torch.cuda.empty_cache()
                            break
                    else:
                        iou_record = iou_record + [iou_record[-1]] * (1 + self.max_num_click - len(iou_record))
                        acc_record = acc_record + [acc_record[-1]] * (1 + self.max_num_click - len(acc_record))
                        if len(max_error_region_size) == 0:
                            max_error_region_size = [10] * (1 + self.max_num_click - len(iou_record))
                            mean_error_region_size = [10] * (1 + self.max_num_click - len(iou_record))
                            torch.cuda.empty_cache()
                            break
                        else:
                            max_error_region_size = max_error_region_size + [max_error_region_size[-1]] * (
                                    1 + self.max_num_click - len(max_error_region_size))
                            mean_error_region_size = mean_error_region_size + [mean_error_region_size[-1]] * (
                                    1 + self.max_num_click - len(mean_error_region_size))
                            torch.cuda.empty_cache()
                            break

                    # if len(iou_record) >= 1 + self.max_num_click:
                    #     torch.cuda.empty_cache()
                    #     break

                    self.picked_idx_buffer = self.picked_idx_buffer + picked_idx

                    # self.vis_clicks_on_pred(self.pred.cpu().data)
                    # self.vis_on_point(error_map.reshape(-1,1))
                interact_flag = False
                interact_round = round
                start_time = time.time()

                # self.picked_idx_buffer = list(predefined_clicks)
            # self.vis_clicks_on_pred(self.pred.cpu().data)

            # self.vis_update(self.pred.cpu().data, picked_idx)

            # density_heat_map = self.check_boundary_click(sp_graph)
            # self.vis_on_point(density_heat_map.reshape(-1, 1), max_value=1., heat_map=True)

            '''test-time loss calculation'''
            '''loss_sparse by user pick'''
            loss_pick_weight = (self.pred[self.picked_idx_buffer] != self.label[self.picked_idx_buffer]) * 1.
            # loss_pick = cross_entropy(self.results[self.picked_idx_buffer, :],
            #                           self.label[self.picked_idx_buffer].to(self.device),
            #                           reduction='sum')
            loss_pick = cross_entropy(self.results[self.picked_idx_buffer, :],
                                      self.label[self.picked_idx_buffer].to(self.device),
                                      reduction='mean')
            # loss_pick = torch.mul(loss_pick, loss_pick_weight.to(self.device)).sum()
            # loss_pick = loss_pick.sum()

            '''loss_self-supervised by softmax entropy'''
            entropy = softmax_entropy(self.results)

            if round - interact_round == 0:
                loss_entropy_weight = self.filter_confident_data(inputs)
                # loss_entropy_weight = torch.ones((entropy.shape[0],))
                # self.vis_clicks_on_pred(self.pred)
            else:
                entropy_change = entropy.cpu().data.numpy() - entropy_np
                entropy_decrease_idx = np.where(entropy_change < -self.entropy_threshold_decrease)[0]
                entropy_increase_idx = np.where(entropy_change > self.entropy_threshold_increase)[0]
                loss_entropy_weight[entropy_decrease_idx] = 1.
                loss_entropy_weight[entropy_increase_idx] = 0.

            if not self.use_confidence_filtering:
                loss_entropy_weight = torch.ones((entropy.shape[0],))

            loss_entropy_weight[self.invalid_idx]=-1
            entropy_np = entropy.cpu().data.numpy()

            entropy_confidence_vis = torch.ones((self.point.shape[0],))
            entropy_confidence_vis[np.where(loss_entropy_weight == 0)[0]] = 0
            # self.vis_on_point(entropy_np.reshape(-1,1))
            # if round-interact_round==1:
            #     self.vis_on_point(entropy_confidence_vis.reshape(-1,1))
            loss_entropy = torch.mul(loss_entropy_weight.to(self.device), entropy)[self.valid_idx].mean()

            loss_pick = self.alpha * loss_pick
            loss_entropy = self.beta * loss_entropy

            # interact_flag = (loss_sup==0)
            interact_flag = (loss_pick_weight.sum() == 0) and (round - interact_round >= self.adapt_round) or (
                        self.alpha == 0) or (round - interact_round >= self.max_adapt_round)

            loss = loss_pick + loss_entropy

            if not run_time_eval:
                print(f'losses: {loss_pick}, {loss_entropy};noc:{len(self.picked_idx_buffer) - 1}; iou:{self.metric.iou()[-1]}; acc:{self.metric.acc()[-1]}')
            loss.backward()
            self.optimizer.step()

            self.last_pred = self.pred
        # noc = len(self.picked_idx_buffer)
        log.info(f'{noc[-1]} clicks, acc:{eval_acc} -> {self.metric.acc()[-1]}, iou:{eval_iou} -> {self.metric.iou()[-1]}')
        # return eval_acc, eval_iou, self.metric.acc()[-1], self.metric.iou()[-1], noc
        return np.array(iou_record)[:self.max_num_click], np.array(acc_record)[:self.max_num_click], np.array(
            max_error_region_size)[:self.max_num_click], np.array(mean_error_region_size)[:self.max_num_click], noc, eval_iou, eval_acc

    def crop_data(self, data):
        from utils.indoor3d_util import room2blocks
        points, labels, feats = room2blocks(data)

        data_1 = self.input_preprocess_crop_s3dis(points[0], labels[0], feats[0])
        data_2 = self.input_preprocess_crop_s3dis(points[1], labels[1], feats[1])
        data_1.row_splits = torch.LongTensor([0, data_1.point.shape[0]])
        data_2.row_splits = torch.LongTensor([0, data_2.point.shape[0]])
        return data_1, data_2


def infer_and_count(InterPCSeg, data, iou_record_list, acc_record_list,
                    iou_list, acc_list, eval_iou_list, eval_acc_list, noc_list):
    if InterPCSeg.method == 'ours':
        iou_record, acc_record, max_error_size_record, mean_error_size_record, noc, eval_iou, eval_acc = InterPCSeg.run_interact(data)
    else:
        iou_record, acc_record, max_error_size_record, mean_error_size_record, noc, eval_iou, eval_acc = InterPCSeg.run_interact_traditional(data)

    iou_record_list.append(iou_record.reshape(1, -1))
    acc_record_list.append(acc_record.reshape(1, -1))
    iou = iou_record[-1]
    acc = acc_record[-1]
    iou_list.append(iou)
    acc_list.append(acc)
    eval_iou_list.append(eval_iou)
    eval_acc_list.append(eval_acc)
    noc_list.append(noc.reshape(1,-1))
    return iou_record_list, acc_record_list, \
           iou_list, acc_list, eval_iou_list, eval_acc_list, noc_list

def s3dis_pointtransformer():
    cfg_file = "Open3D-ML/ml3d/configs/pointtransformer_s3dis.yml"
    cfg = _ml3d.utils.Config.load_from_file(cfg_file)
    dataset_path = "/media/vcg8004/WD_BLACK/dataset/Stanford3dDataset_v1.2_Aligned_Version"
    cfg.dataset['dataset_path'] = dataset_path
    '''load s3dis in open3d framework'''
    dataset = ml3d.datasets.S3DIS(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
    test_split = dataset.get_split("test")
    '''load pointtransformer trained on s3dis'''

    model = ml3d.models.PointTransformer(**cfg.model)
    InterPCSeg = InterPCSeg_pipeline(model=model, ckpt_path='Open3D-ML/logs/pointtransformer_s3dis_202109241350utc.pth',
                         name='s3dis_pointtransformer', max_num_click=30)

    InterPCSeg.interact_mode = 'real'

    InterPCSeg.method = 'ours'  # default: 'ours'

    InterPCSeg.use_warm_up = True
    InterPCSeg.use_confidence_filtering = True

    InterPCSeg.entropy_threshold_increase = 0.03
    InterPCSeg.entropy_threshold_decrease = 0.03
    eval_iou_list = []
    eval_acc_list = []
    iou_list = []
    acc_list = []
    noc_list = []
    drop_samples = []
    iou_record_list = []
    acc_record_list = []

    InterPCSeg.max_num_click = 30
    InterPCSeg.num_clicks_per_round = 1

    user_name = 'ours'

    try:
        iou_record_tmp = np.load(f'iou_record_PT_{user_name}_NoC30_IoU95_tmp.npy')
        noc_record_tmp = np.load(f'noc_record_PT_{user_name}_NoC30_IoU95_tmp.npy')
        acc_record_tmp = np.load(f'acc_record_PT_{user_name}_NoC30_IoU95_tmp.npy')
        eval_iou_record_tmp = np.load(f'eval_iou_PT_{user_name}_NoC30_IoU95_tmp.npy')
        start_data_id = iou_record_tmp.shape[0]
    except:
        record_step = InterPCSeg.max_num_click // InterPCSeg.num_clicks_per_round
        iou_record_tmp, noc_record_tmp, acc_record_tmp, eval_iou_record_tmp \
            = np.array([]).reshape(0,record_step), np.array([]).reshape(0,3), np.array([]).reshape(0,record_step), np.array([])
        start_data_id = 0

    selected_ids = [0,2,28,30,46,49,52,53,56,70]
    # start_data_id = selected_ids[start_data_id]
    real_data_id = -1

    for data_id in range(len(test_split)):
        print(f'data{data_id}')
        data = test_split.get_data(data_id)
        data = InterPCSeg.input_preprocess_ml3d_s3dis(data)
        log.info(f'data {data_id}, {data.point.shape[0]} points')

        data_list = []
        if data.point.shape[0] > 150000:
            data_1, data_2 = InterPCSeg.crop_data(data)
            if max(data_1.point.shape[0], data_2.point.shape[0]) < 150000:
                data_list = [data_1, data_2]
            elif min(data_1.point.shape[0], data_2.point.shape[0]) > 150000:
                drop_samples.append(data_id)
                continue
            elif data_1.point.shape[0] > 150000:
                data_11, data_12 = InterPCSeg.crop_data(data_1)
                if max(data_11.point.shape[0], data_12.point.shape[0]) > 150000:
                    drop_samples.append(data_id)
                    continue
                data_list = [data_11, data_12, data_2]
            else:
                data_21, data_22 = InterPCSeg.crop_data(data_2)
                if max(data_21.point.shape[0], data_22.point.shape[0]) > 150000:
                    drop_samples.append(data_id)
                    continue
                data_list = [data_1, data_21, data_22]
        else:
            data_list.append(data)


        for data in data_list:
            real_data_id += 1
            # if real_data_id < start_data_id: continue
            # if real_data_id not in selected_ids:
            #     continue
            iou_record_list, acc_record_list,\
            iou_list, acc_list, eval_iou_list, eval_acc_list, noc_list = \
                infer_and_count(InterPCSeg, data, iou_record_list, acc_record_list, iou_list, acc_list, eval_iou_list, eval_acc_list, noc_list)

            iou_record_tmp = np.concatenate([iou_record_tmp, iou_record_list[-1]])
            noc_record_tmp = np.concatenate([noc_record_tmp, noc_list[-1]])
            acc_record_tmp = np.concatenate([acc_record_tmp, acc_record_list[-1]])
            eval_iou_record_tmp = np.concatenate([eval_iou_record_tmp, [eval_iou_list[-1]]])

            # np.save(f'iou_record_PT_{user_name}_NoC30_IoU95_tmp.npy', iou_record_tmp)
            # np.save(f'noc_record_PT_{user_name}_NoC30_IoU95_tmp.npy', noc_record_tmp)
            # np.save(f'acc_record_PT_{user_name}_NoC30_IoU95_tmp.npy', acc_record_tmp)
            # np.save(f'eval_iou_PT_{user_name}_NoC30_IoU95_tmp.npy', eval_iou_record_tmp)


    log.info(f'iou:{np.mean(eval_iou_list):.2f} -> {np.mean(iou_list):.2f}')
    log.info(f'acc:{np.mean(eval_acc_list):.2f} -> {np.mean(acc_list):.2f}')

    if len(drop_samples) > 0:
        log.info(f'data {drop_samples} are dropped due to its scale')

    # np.save(f'iou_record_PT_{user_name}_NoC30_IoU95.npy', iou_record_tmp)
    # np.save(f'noc_record_PT_{user_name}_NoC30_IoU95.npy', noc_record_tmp)
    # np.save(f'acc_record_PT_{user_name}_NoC30_IoU95.npy', acc_record_tmp)
    # np.save(f'eval_iou_PT_{user_name}_NoC30_IoU95.npy', eval_iou_record_tmp)

def scannet_sparseconvunet():
    cfg_file = "Open3D-ML/ml3d/configs/sparseconvunet_scannet.yml"
    cfg = _ml3d.utils.Config.load_from_file(cfg_file)
    '''load s3dis in open3d framework'''
    dataset = ml3d.datasets.Scannet(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
    test_split = dataset.get_split("test")
    '''load pointtransformer trained on s3dis'''

    model = ml3d.models.SparseConvUnet(**cfg.model)
    InterPCSeg = InterPCSeg_pipeline(model=model, ckpt_path=cfg.model.ckpt_path,
                         name='scannet_sparseconvunet', max_num_click=30)

    InterPCSeg.interact_mode = 'real'
    InterPCSeg.alpha = 1.
    InterPCSeg.beta = 100.
    InterPCSeg.warm_round = 10
    InterPCSeg.adapt_round = 5
    InterPCSeg.max_adapt_round = 10

    InterPCSeg.method = 'ours'

    InterPCSeg.use_warm_up = True
    InterPCSeg.use_confidence_filtering = True

    InterPCSeg.optim_weight_decay = 5e-1

    InterPCSeg.expected_IoU = [0.8,0.85,0.9]

    InterPCSeg.entropy_threshold_increase = 0.1
    InterPCSeg.entropy_threshold_decrease = 0.01
    InterPCSeg.DBSCAN_eps = 0.01
    eval_iou_list = []
    eval_acc_list = []
    iou_list = []
    acc_list = []
    noc_list = []
    drop_samples = []
    iou_record_list = []
    acc_record_list = []
    max_error_size_record_list = []
    mean_error_size_record_list = []
    selected_id = 0

    InterPCSeg.max_num_click = 30
    InterPCSeg.num_clicks_per_round = 1

    user_name = 'ours'

    try:
        iou_record_tmp = np.load(f'iou_record_SCU_{user_name}_NoC30_IoU90_tmp.npy')
        noc_record_tmp = np.load(f'noc_record_SCU_{user_name}_NoC30_IoU90_tmp.npy')
        acc_record_tmp = np.load(f'acc_record_SCU_{user_name}_NoC30_IoU90_tmp.npy')
        start_data_id = iou_record_tmp.shape[0]
    except:
        record_step = InterPCSeg.max_num_click // InterPCSeg.num_clicks_per_round
        iou_record_tmp, noc_record_tmp, acc_record_tmp, eval_iou_record_tmp \
            = np.array([]).reshape(0,record_step), np.array([]).reshape(0,3), np.array([]).reshape(0,record_step), np.array([])
        start_data_id = 0

    # selected_ids = [40,43,56,90,121,134,159,176,182,186,221,232,249,275,290]
    # start_data_id = selected_ids[start_data_id]

    for data_id in range(len(test_split)):
        if data_id < start_data_id: continue
        # if data_id not in selected_ids:
        #     continue
        print(f'data{data_id}')
        data = test_split.get_data(data_id)
        data = InterPCSeg.input_preprocess_ml3d_s3dis(data)

        log.info(f'data {data_id}, {data.point[0].shape[0]} points')

        iou_record_list, acc_record_list, \
        iou_list, acc_list, eval_iou_list, eval_acc_list, noc_list = \
            infer_and_count(InterPCSeg, data, iou_record_list, acc_record_list,
                            iou_list, acc_list, eval_iou_list, eval_acc_list, noc_list)

        iou_record_tmp = np.concatenate([iou_record_tmp, iou_record_list[-1]])
        noc_record_tmp = np.concatenate([noc_record_tmp, noc_list[-1]])
        acc_record_tmp = np.concatenate([acc_record_tmp, acc_record_list[-1]])
        # np.save(f'iou_record_SCU_{user_name}_NoC30_IoU90_tmp.npy', iou_record_tmp)
        # np.save(f'noc_record_SCU_{user_name}_NoC30_IoU90_tmp.npy', noc_record_tmp)
        # np.save(f'acc_record_SCU_{user_name}_NoC30_IoU90_tmp.npy', acc_record_tmp)

    log.info(f'iou:{np.mean(eval_iou_list):.2f} -> {np.mean(iou_list):.2f}')
    log.info(f'acc:{np.mean(eval_acc_list):.2f} -> {np.mean(acc_list):.2f}')

    if len(drop_samples) > 0:
        log.info(f'data {drop_samples} are dropped due to its scale')
    iou_record_all = np.concatenate(iou_record_list, 0)
    noc_record_all = np.concatenate(noc_list, 0)
    acc_record_all = np.concatenate(acc_record_list, 0)

    # np.save(f'iou_record_SCU_{user_name}_NoC30_IoU90.npy', iou_record_all)
    # np.save(f'noc_record_SCU_{user_name}_NoC30_IoU90.npy', noc_record_all)
    # np.save(f'acc_record_SCU_{user_name}_NoC30_IoU90.npy', acc_record_all)


if __name__ == '__main__':

    s3dis_pointtransformer()
    scannet_sparseconvunet()
