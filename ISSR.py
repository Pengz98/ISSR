import open3d as o3d
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
from open3d.ml.torch.modules import losses, metrics

SemSegLoss = losses.SemSegLoss
SemSegMetric = metrics.SemSegMetric

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import os
import copy
import time
import argparse
from datetime import datetime
import logging
import matplotlib.pyplot as plt

torch.random.manual_seed(0)
np.random.seed(0)

timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

log_dir = os.path.join('my_logs/' + timestamp)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file_path = os.path.join(log_dir, 'log.txt')
logging.basicConfig(filename=log_file_path, filemode='w', level=logging.INFO)
log = logging.getLogger(__name__)
log.addHandler(logging.FileHandler(log_file_path))

os.system(f'cp ISSR.py {log_dir}')

from utils.vis_utils import vis, vis_and_interact
from utils.adapt_loss import softmax_entropy, cross_entropy
from utils.sample_utils import random_sampling
from utils.vis_utils import tsne_visualization_color

from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity


class FTIS_pipeline():
    def __init__(self, model, ckpt_path, name='default', max_num_click=10, color_map=None):
        self.name = name
        self.color_map = color_map
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.model.to(self.device)
        self.model.device = self.device

        self.tsne_vis = False
        self.entropy_vis = False
        self.input_vis = False

        if self.name=='scannet_sparseconvunet':
            self.model_copy = copy.deepcopy(model)
            self.model_copy.to(self.device)
            self.model_copy.device = self.device

        self.warm_lr = 5e-3
        self.lr = 1e-3

        self.momentum = 0
        self.betas = (0,0)
        self.optim_weight_decay = 1e-2

        try:
            self.model_params = torch.load(ckpt_path, map_location=self.device)['model_state_dict']
        except:
            self.model_params = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(self.model_params)

        self.max_all_round = 1000
        self.interact_mode = 'simulated'

        self.alpha = 1. # loss_correction_weight
        self.beta = 100. # loss_stabilization_weight
        self.entropy_threshold_increase = 0.1
        self.entropy_threshold_decrease = 0.01
        self.DBSCAN_eps=0.1
        self.DBSCAN_min_cluster = 10

        self.warm_round = 5
        self.adapt_round = 2
        self.max_adapt_round = 10

        self.max_optimize_round = 10
        self.max_num_click = max_num_click
        self.num_clicks_per_round = 1

        self.min_error_size = 100
        self.expected_IoU = [0.85, 0.9, 0.95]

        self.use_warm_up = True
        self.use_confidence_filtering = True


    def input_preprocess_crop_s3dis(self, point, label, feat):
        class D():
            def __init__(self):
                self.point = torch.from_numpy(point)
                self.label = torch.from_numpy(label)
                self.feat = torch.from_numpy(feat)
        inputs = D()
        return inputs


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

    def input_preprocess_ml3d_semantickitti(self, data):
        from open3d.ml.torch.dataloaders import get_sampler, TorchDataloader, DefaultBatcher, ConcatBatcher
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
                                  collate_fn=DefaultBatcher().collate_fn)
        self.model.trans_point_sampler = infer_sampler.get_point_sampler()
        infer_data = next(iter(infer_loader))
        return infer_data['data']


    def crop_data(self, data):
        from utils.indoor3d_util import room2blocks
        points, labels, feats = room2blocks(data)

        data_1 = self.input_preprocess_crop_s3dis(points[0], labels[0], feats[0])
        data_2 = self.input_preprocess_crop_s3dis(points[1], labels[1], feats[1])
        data_1.row_splits = torch.LongTensor([0, data_1.point.shape[0]])
        data_2.row_splits = torch.LongTensor([0, data_2.point.shape[0]])

        return data_1, data_2


    def metric_cal(self, pred):
        metric = SemSegMetric()
        metric.reset()
        if 's3dis' in self.name:
            metric.update(pred, self.label)
        else:
            metric.update(pred[self.valid_idx, :], self.label[self.valid_idx])
        return metric

    def evaluate_model(self, inputs):
        self.model.eval()
        with torch.no_grad():
            if self.name == 'scannet_stratified_transformer':
                model_input, _ = self.input_process_st(inputs)
                eval_results = self.model(model_input)
            else:
                eval_results = self.model(inputs)
        if eval_results.shape[0] == 1:
            eval_results = eval_results[0]
        target = eval_results.max(-1)[1].detach()
        self.initial_pred = target.cpu()
        eval_metric = self.metric_cal(eval_results.cpu().data)
        self.model.train()
        return eval_metric.acc()[-1], eval_metric.iou()[-1]


    def test_time_warm_up(self, inputs, num_warm_rounds=5):
        self.model.eval()
        with torch.no_grad():
            if self.name == 'scannet_stratified_transformer':
                model_input, _ = self.input_process_st(inputs)
                eval_results = self.model(model_input)
            else:
                eval_results = self.model(inputs)
        if eval_results.shape[0] == 1:
            eval_results = eval_results[0]
        target = eval_results.max(-1)[1].detach()

        self.initial_pred = target.cpu()

        eval_metric = self.metric_cal(eval_results.cpu().data)
        self.model.train()

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
            train_results, self.embedding = self.model(inputs, return_feat=True)

            if train_results.shape[0] == 1:
                train_results = train_results[0]
            self.pred = train_results.max(1)[1]

            if self.tsne_vis:
                if i==0:
                    self.tsne_vis_feat(window_name='before warm-up')
                elif i == max_warmup_rounds-1:
                    self.tsne_vis_feat(window_name='after warm-up')

            train_metric = self.metric_cal(train_results.cpu().data)

            loss = cross_entropy(train_results[self.valid_idx], target[self.valid_idx])
            loss.backward()

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

    def vis_on_point(self, feat, hint='default', max_value=None, use_color_map=False, heat_map=False, invalid_idx=None, visible_idx=None):
        if type(feat) is not torch.Tensor:
            feat = torch.from_numpy(feat)
        if self.color_map is not None and use_color_map:
            color_np = np.concatenate(list(self.color_map.values())).reshape(-1, 3)
            color = torch.from_numpy(color_np[feat.reshape(-1)])
            if visible_idx is not None:
                color = color[visible_idx]
                point = self.point[visible_idx]
                vis(torch.cat([point, color.reshape(-1, 3)], -1), hint=hint, max_value=max_value,
                    heat_map=heat_map)
            else:
                if invalid_idx is not None:
                    color[invalid_idx,:] = 0
                vis(torch.cat([self.point, color.reshape(-1, 3)], -1), hint=hint, max_value=max_value, heat_map=heat_map)
        else:
            if visible_idx is not None:
                feat = feat[visible_idx]
                point = self.point[visible_idx]
                vis(torch.cat([point, feat], -1), hint=hint, max_value=max_value,
                    heat_map=heat_map)
            else:
                if invalid_idx is not None:
                    feat[invalid_idx,:] = -1
                if feat.shape[0]>0:
                    vis(torch.cat([self.point, feat], -1), hint=hint, max_value=max_value, heat_map=heat_map)
                else:
                    vis(self.point, hint=hint, max_value=max_value, heat_map=heat_map)

    def vis_with_interact(self, pred, use_color_map=True, invalid_idx=None, visible_idx=None):
        hint = f'{len(self.picked_idx_buffer)}clicks; acc:{self.metric.acc()[-1]:.2f}, iou:{self.metric.iou()[-1]:.2f}'
        if self.color_map is not None and use_color_map:
            color_np = np.concatenate(list(self.color_map.values())).reshape(-1, 3)
            color = torch.from_numpy(color_np[pred])
            if visible_idx is not None:
                color = color[visible_idx]
                point = self.point[visible_idx]
                picked_idx = vis_and_interact(torch.cat([point, color.reshape(-1, 3)], -1), hint=hint)
                picked_idx = list(visible_idx[picked_idx])
            else:
                if invalid_idx is not None:
                    color[invalid_idx,:] = 0
                picked_idx = vis_and_interact(torch.cat([self.point, color.reshape(-1, 3)], -1), hint=hint)
        else:
            if visible_idx is not None:
                pred = pred[visible_idx]
                point = self.point[visible_idx]
                picked_idx = vis_and_interact(torch.cat([point, pred.reshape(-1, 3)], -1), hint=hint)
                picked_idx = visible_idx[picked_idx]
            else:
                if invalid_idx is not None:
                    pred[invalid_idx] = -1
                picked_idx = vis_and_interact(torch.cat([self.point, pred.reshape(-1, 1)], -1), hint=hint)
        return picked_idx

    def simulate_user_clicks(self, error_map, num_clicks=None, random=True, num_point_threashold=100):
        point_np = self.point.cpu().data.numpy()

        if 'scannet' in self.name or 'semantickitti' in self.name:
            centroid = np.mean(point_np, axis=0)
            point_np = point_np - centroid
            m = np.max(np.sqrt(np.sum(point_np ** 2, axis=1)))
            point_np = point_np / m

        error_map_np = error_map.cpu().data.numpy()
        error_idx = np.where(error_map_np == 1)[0]

        db = DBSCAN(eps=self.DBSCAN_eps, min_samples=self.DBSCAN_min_cluster).fit(point_np[error_idx])
        labels = db.labels_

        global_labels = -1 * np.ones((error_map.shape[0],))
        global_labels[error_idx] = labels

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
        for l in np.unique(candidate_labels):
            if l != -1:
                cluster = point_np[error_idx[labels == l]]
                if cluster.shape[0]>1000:
                    samp_idx = np.random.choice(cluster.shape[0],(1000,))
                    cluster = cluster[samp_idx]
                else:
                    samp_idx = np.arange(cluster.shape[0])

                if self.interact_mode=='simulated':
                    kd = KernelDensity()
                    kd.fit(cluster)
                    score = kd.score_samples(cluster)
                else:
                    cluster_center = np.mean(cluster, axis=0)
                    score = -((cluster-cluster_center)**2).sum(1)

                cluster_centroid = score.argsort()[-10:]
                cluster_centroid = samp_idx[cluster_centroid]
                cluster_centroid = error_idx[np.where(labels == l)[0][cluster_centroid]]
                cluster_centroids.append(cluster_centroid)

        colors = -1 * np.ones((point_np.shape[0],))
        colors[error_idx] = labels

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
        if self.name == 'scannet_stratified_transformer':
            model_input, _ = self.input_process_st(inputs)
            results = model_copy(model_input)
        else:
            results = model_copy(inputs)
        if results.shape[0] == 1:
            results = results[0]
        entropy = softmax_entropy(results)
        loss_entropy_weight = torch.ones((entropy.shape[0],))
        loss_entropy = torch.mul(loss_entropy_weight.to(self.device), entropy).mean()
        loss_pick = cross_entropy(results[self.picked_idx_buffer, :],
                                  self.label[self.picked_idx_buffer].to(self.device),
                                  reduction='sum')
        loss = self.alpha * loss_pick + self.beta * loss_entropy
        loss.backward()
        optimizer.step()
        if self.name == 'scannet_stratified_transformer':
            model_input, _ = self.input_process_st(inputs)
            results_new = model_copy(model_input)
        else:
            results_new = model_copy(inputs)
        if results_new.shape[0] == 1:
            results_new = results_new[0]
        entropy_new = softmax_entropy(results_new)
        entropy_change = entropy_new.cpu().data.numpy() - entropy.cpu().data.numpy()
        entropy_increase_idx = np.where(entropy_change > self.entropy_threshold_increase)[0]
        loss_entropy_weight = torch.ones((entropy.shape[0],))
        loss_entropy_weight[entropy_increase_idx] = 0.
        # loss_entropy_weight[entropy_increase_idx] = -1.
        del optimizer
        if self.name!='scannet_sparseconvunet':
            del model_copy
        return loss_entropy_weight

    def tsne_vis_feat(self, window_name='default', vis_pred=True, num_point=5000):
        feat_bank = self.embedding.cpu().data
        feat_anchor = torch.empty((0, feat_bank.shape[1]))
        np.random.seed(0)
        random_samp_idx = np.random.choice(feat_bank[self.valid_idx].shape[0], (num_point,), replace=False)

        if vis_pred:
            colors = self.pred[self.valid_idx][random_samp_idx].cpu().data
        else:
            colors = self.label[self.valid_idx][random_samp_idx].cpu().data

        tsne_visualization_color(feat_bank[self.valid_idx][random_samp_idx].numpy(), feat_anchor.numpy(), colors.numpy(), window=window_name)

    def tsne_vis_feat_click(self, window_name='default', vis_pred=True, num_point=5000):
        feat_bank = self.embedding.cpu().data
        feat_anchor = feat_bank[self.picked_idx_buffer]
        np.random.seed(0)
        random_samp_idx = np.random.choice(feat_bank[self.valid_idx].shape[0], (num_point,), replace=False)
        if vis_pred:
            colors = torch.cat([self.pred[self.valid_idx][random_samp_idx], self.label[self.picked_idx_buffer]]).cpu().data
        else:
            colors = torch.cat([self.label[self.valid_idx][random_samp_idx], self.label[self.picked_idx_buffer]]).cpu().data
        tsne_visualization_color(feat_bank[self.valid_idx][random_samp_idx].numpy(), feat_anchor.numpy(), colors.numpy(), window=window_name)

    def run_interact(self, inputs):
        '''input data processing'''
        if self.name == 's3dis_pointtransformer':
            self.point = inputs.point
            self.label = inputs.label
            self.color = inputs.feat
            self.valid_idx = np.arange(self.label.shape[0]).astype('int')
            self.invalid_idx = None
        elif self.name == 'scannet_sparseconvunet':
            self.point = inputs.point[0]
            self.label = inputs.label[0]
            self.color = inputs.feat[0]
            self.valid_idx = np.where(self.label!=-1)[0]
            self.invalid_idx = np.arange(self.label.shape[0])
            invalid_mask = np.ones((self.label.shape[0],))
            invalid_mask[self.valid_idx] = 0
            self.invalid_idx = self.invalid_idx[invalid_mask == 1]
        elif self.name == 'semantickitti_randlanet':
            self.point = inputs['point'].reshape(-1, 3)
            self.label = inputs['labels'].reshape(-1)
            self.color = torch.ones((self.point.shape[0], 3)) * 0.3
            self.valid_idx = np.where(self.label!=0)[0]
            self.invalid_idx = np.where(self.label==0)[0]
            self.label = self.label - 1

        if hasattr(inputs, 'to'):
            inputs.to(self.device)
        elif self.name!='semantickitti_randlanet':
            for attr in dir(inputs):
                if not attr.startswith('__'):
                    setattr(inputs, attr, getattr(inputs, attr).to(self.device))

        self.model.load_state_dict(self.model_params) # load pre-trained weights

        eval_acc, eval_iou = self.test_time_warm_up(inputs, num_warm_rounds=self.warm_round) # network warm-up

        noc = -1 * np.ones((3,))
        self.picked_idx_buffer = []
        iou_record = []
        acc_record = []
        max_error_region_size = []
        mean_error_region_size = []
        interact_flag = True
        interact_round = 0

        '''optimizer setting for test-time training'''
        if self.name == 's3dis_pointtransformer':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.optim_weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr, betas=self.betas, weight_decay=self.optim_weight_decay)

        if self.input_vis:
            invisible_ceil = False
            if self.name == 's3dis_pointtransformer' and invisible_ceil:
                visible_idx = np.where(self.label!=0)[0]
            else:
                visible_idx = None
            self.vis_on_point(self.color.reshape(-1, 3), visible_idx=visible_idx)
        else:
            visible_idx = None

        for round in range(self.max_all_round):
            self.optimizer.zero_grad()
            self.results, self.embedding = self.model(inputs, return_feat=True) # do inference
            if self.results.shape[0] == 1:
                self.results = self.results[0]

            self.pred = self.results.cpu().data.max(1)[1]
            if self.invalid_idx is not None:
                self.pred[self.invalid_idx] = -1
            self.metric = self.metric_cal(self.results) # record the prediction

            if round == 0:
                # save the max entropy value for uncertainty visualization
                self.max_entropy_value = softmax_entropy(self.results[self.valid_idx]).cpu().data.numpy().max()

            if interact_flag:
                if len(self.picked_idx_buffer) >= self.max_num_click:
                    # stop the interactive loop if the click budget is used out
                    torch.cuda.empty_cache()
                    break

                '''get interactions by real or simulated users'''
                if self.interact_mode == 'real':
                    if eval_iou < 10:
                        '''gt visualization before interact(necessary for experiment but not for practical use)'''
                        self.vis_on_point(self.label.reshape(-1, 1), hint='reference', use_color_map=True,
                                              invalid_idx=self.invalid_idx, visible_idx=visible_idx)
                        entropy = softmax_entropy(self.results).cpu().data.numpy()
                        error_map = 1.0 * (self.pred != self.label)
                        if self.invalid_idx is not None:
                            entropy[self.invalid_idx] = -1
                            error_map[self.invalid_idx] = -1
                        self.vis_on_point(error_map.reshape(-1, 1), hint='error map', visible_idx=visible_idx)
                        picked_idx = self.vis_with_interact(self.pred.cpu().data, use_color_map=True, invalid_idx=self.invalid_idx, visible_idx=visible_idx)

                        if len(picked_idx) == 0:
                            picked_idx, error_region_sizes = self.simulate_user_clicks(error_map, num_clicks=1, num_point_threashold=self.min_error_size)

                        if np.intersect1d(picked_idx,self.valid_idx).shape[0]==len(picked_idx):
                            self.picked_idx_buffer = self.picked_idx_buffer + picked_idx
                    else:
                        picked_idx = [round]

                    iou_record.append(self.metric.iou()[-1])
                    acc_record.append(self.metric.acc()[-1])
                    max_error_region_size.append(0)
                    mean_error_region_size.append(0)

                    if (np.bincount(picked_idx) > 1).sum():
                        iou_record = iou_record + [iou_record[-1]] * (1 + self.max_num_click - len(iou_record))
                        acc_record = acc_record + [acc_record[-1]] * (1 + self.max_num_click - len(acc_record))
                        max_error_region_size = max_error_region_size + [max_error_region_size[-1]] * (
                                1 + self.max_num_click - len(max_error_region_size))
                        mean_error_region_size = mean_error_region_size + [mean_error_region_size[-1]] * (
                                1 + self.max_num_click - len(mean_error_region_size))
                        torch.cuda.empty_cache()
                        break

                    if iou_record[-1] > self.expected_IoU[2]:
                        iou_record = iou_record + [iou_record[-1]] * (1 + self.max_num_click - len(iou_record))
                        acc_record = acc_record + [acc_record[-1]] * (1 + self.max_num_click - len(acc_record))
                        max_error_region_size = max_error_region_size + [max_error_region_size[-1]] * (
                                1 + self.max_num_click - len(max_error_region_size))
                        mean_error_region_size = mean_error_region_size + [mean_error_region_size[-1]] * (
                                1 + self.max_num_click - len(mean_error_region_size))
                        torch.cuda.empty_cache()
                        break
                else:
                    iou_record.append(self.metric.iou()[-1])
                    acc_record.append(self.metric.acc()[-1])
                    error_map = 1.0 * (self.pred != self.label)
                    if self.invalid_idx is not None:
                        error_map[self.invalid_idx] = -1
                    if iou_record[-1] >= self.expected_IoU[0] and noc[0]==-1:
                        noc[0] = len(self.picked_idx_buffer)
                    if iou_record[-1] >= self.expected_IoU[1] and noc[1]==-1:
                        noc[1] = len(self.picked_idx_buffer)
                    if iou_record[-1] >= self.expected_IoU[2] and noc[2] == -1:
                        noc[2] = len(self.picked_idx_buffer)

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
                            max_error_region_size = [10] * (1 + self.max_num_click - len(max_error_region_size))
                            mean_error_region_size = [10] * (1 + self.max_num_click - len(max_error_region_size))
                            torch.cuda.empty_cache()
                            break
                        else:
                            max_error_region_size = max_error_region_size + [max_error_region_size[-1]] * (
                                    1 + self.max_num_click - len(max_error_region_size))
                            mean_error_region_size = mean_error_region_size + [mean_error_region_size[-1]] * (
                                    1 + self.max_num_click - len(mean_error_region_size))
                            torch.cuda.empty_cache()
                            break

                    if round==(self.max_all_round-1):
                        iou_record = iou_record + [iou_record[-1]] * (1 + self.max_num_click - len(iou_record))
                        acc_record = acc_record + [acc_record[-1]] * (1 + self.max_num_click - len(acc_record))
                        max_error_region_size = max_error_region_size + [max_error_region_size[-1]] * (
                                1 + self.max_num_click - len(max_error_region_size))
                        mean_error_region_size = mean_error_region_size + [mean_error_region_size[-1]] * (
                                1 + self.max_num_click - len(mean_error_region_size))
                        torch.cuda.empty_cache()
                        break

                    self.picked_idx_buffer = self.picked_idx_buffer + picked_idx

                if len(iou_record) >= 1 + self.max_num_click:
                    torch.cuda.empty_cache()
                    break

                interact_round = round

            '''test-time loss calculation'''
            '''correction energy'''
            loss_pick_weight = (self.pred[self.picked_idx_buffer] != self.label[self.picked_idx_buffer]) * 1.
            loss_correction = cross_entropy(self.results[self.picked_idx_buffer, :],
                                      self.label[self.picked_idx_buffer].to(self.device),
                                      reduction='sum')

            '''loss_self-supervised by softmax entropy'''
            entropy = softmax_entropy(self.results)
            if round - interact_round == 0:
                loss_entropy_weight = self.filter_confident_data(inputs)
            else:
                entropy_change = entropy.cpu().data.numpy() - entropy_np
                entropy_decrease_idx = np.where(entropy_change < -self.entropy_threshold_decrease)[0]
                entropy_increase_idx = np.where(entropy_change > self.entropy_threshold_increase)[0]
                loss_entropy_weight[entropy_decrease_idx] = 1.
                loss_entropy_weight[entropy_increase_idx] = 0.

            if self.invalid_idx is not None:
                loss_entropy_weight[self.invalid_idx]=-1
            entropy_np = entropy.cpu().data.numpy()

            if self.tsne_vis:
                self.tsne_vis_feat_click(window_name=f'{round} overall round, {round-interact_round}rounds since last interaction')

            if self.entropy_vis:
                entropy_confidence_vis = torch.ones((self.point.shape[0],))
                entropy_confidence_vis[np.where(loss_entropy_weight == 0)[0]] = 0
                self.vis_on_point(self.pred.reshape(-1,1), use_color_map=True, visible_idx=visible_idx, hint=f'{round} overall round, {round-interact_round}rounds since last interaction, iou: {self.metric.iou()[-1]}, acc:{self.metric.acc()[-1]}',
                                  invalid_idx=self.invalid_idx)
                self.vis_on_point(entropy_np.reshape(-1,1), hint=f'{round} overall round, {round-interact_round}rounds since last interaction, mean Ent: {entropy_np.mean()}',
                                  heat_map=True, invalid_idx=self.invalid_idx, max_value=self.max_entropy_value, visible_idx=visible_idx)

            loss_stabilization = torch.mul(loss_entropy_weight.to(self.device), entropy)[self.valid_idx].mean()

            loss_correction = self.alpha * loss_correction
            loss_stabilization = self.beta * loss_stabilization

            interact_flag = (loss_pick_weight.sum() == 0) and (round - interact_round >= self.adapt_round) or (
                        self.alpha == 0) or (round - interact_round >= self.max_adapt_round)

            loss = loss_correction + loss_stabilization

            print(f'losses: {loss.item()}; noc:{len(self.picked_idx_buffer) - 1}; iou:{self.metric.iou()[-1]}; acc:{self.metric.acc()[-1]}')
            loss.backward()
            self.optimizer.step()

        log.info(f'{noc[-1]} clicks, acc:{eval_acc} -> {self.metric.acc()[-1]}, iou:{eval_iou} -> {self.metric.iou()[-1]}')
        return np.array(iou_record)[:self.max_num_click], np.array(acc_record)[:self.max_num_click], np.array(
            max_error_region_size)[:self.max_num_click], np.array(mean_error_region_size)[:self.max_num_click], noc, eval_iou, eval_acc



def infer_and_count(FTIS, data, iou_record_list, acc_record_list, max_error_size_record_list,
                    mean_error_size_record_list,
                    iou_list, acc_list, eval_iou_list, eval_acc_list, noc_list):
    iou_record, acc_record, max_error_size_record, mean_error_size_record, noc, eval_iou, eval_acc = FTIS.run_interact(data)

    iou_record_list.append(iou_record.reshape(1, -1))
    acc_record_list.append(acc_record.reshape(1, -1))
    max_error_size_record_list.append(max_error_size_record.reshape(1, -1))
    mean_error_size_record_list.append(mean_error_size_record.reshape(1, -1))
    iou = iou_record[-1]
    acc = acc_record[-1]
    iou_list.append(iou)
    acc_list.append(acc)
    eval_iou_list.append(eval_iou)
    eval_acc_list.append(eval_acc)
    noc_list.append(noc.reshape(1,-1))
    return iou_record_list, acc_record_list, max_error_size_record_list, mean_error_size_record_list, \
           iou_list, acc_list, eval_iou_list, eval_acc_list, noc_list


def s3dis_pointtransformer(args):
    cfg_file = "Open3D-ML/ml3d/configs/pointtransformer_s3dis.yml"
    cfg = _ml3d.utils.Config.load_from_file(cfg_file)
    dataset_path = "/media/vcg8004/WD_BLACK/dataset/Stanford3dDataset_v1.2_Aligned_Version"
    cfg.dataset['dataset_path'] = dataset_path
    '''load s3dis in open3d framework'''
    dataset = ml3d.datasets.S3DIS(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
    test_split = dataset.get_split("test")
    '''load pointtransformer trained on s3dis'''

    colormap = plt.get_cmap("tab20")(np.arange(20))[:, :3]
    colormap_dict = dict()
    for i in range(colormap.shape[0]):
        colormap_dict[i] = colormap[i]

    model = ml3d.models.PointTransformer(**cfg.model)
    FTIS = FTIS_pipeline(model=model, ckpt_path='Open3D-ML/logs/pointtransformer_s3dis_202109241350utc.pth',
                         name='s3dis_pointtransformer', max_num_click=30, color_map=colormap_dict)

    FTIS.interact_mode = args.interact_mode
    FTIS.entropy_vis = args.entropy_vis
    FTIS.tsne_vis = args.tsne_vis
    FTIS.input_vis = args.input_vis

    FTIS.alpha = args.alpha
    FTIS.beta = args.beta  # 100.
    FTIS.gamma = 0.
    FTIS.lr = args.lr # 1e-3

    FTIS.warm_round = 5
    FTIS.adapt_round = args.adapt_round  #2
    FTIS.max_adapt_round = 10

    FTIS.momentum = 0.0
    FTIS.optim_weight_decay = 1e-3 #1e-2


    FTIS.use_warm_up = args.warm_up
    FTIS.use_confidence_filtering = args.filter

    FTIS.entropy_threshold_increase = args.entropy_threshold_increase # 0.03
    FTIS.entropy_threshold_decrease = args.entropy_threshold_decrease # 0.03
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

    FTIS.max_num_click = 30
    FTIS.num_clicks_per_round = 1

    save_dir = args.save_dir
    if save_dir is None:
        save_dir = timestamp
        os.makedirs(save_dir, exist_ok=True)
        os.system(f'cp ISSR.py {save_dir}/')


    file = open(f'{save_dir}/args.txt', 'w')
    file.write(str(args))
    file.close()

    user_name = args.user_name
    record_step = FTIS.max_num_click // FTIS.num_clicks_per_round

    try:
        iou_record_tmp = np.load(f'{save_dir}/iou_record_PT_{user_name}_NoC{record_step}_IoU95_tmp.npy')
        noc_record_tmp = np.load(f'{save_dir}/noc_record_PT_{user_name}_NoC{record_step}_IoU95_tmp.npy')
        acc_record_tmp = np.load(f'{save_dir}/acc_record_PT_{user_name}_NoC{record_step}_IoU95_tmp.npy')
        eval_iou_record_tmp = np.load(f'{save_dir}/eval_iou_PT_{user_name}_NoC{record_step}_IoU95_tmp.npy')
        mes_record_tmp = np.load(f'{save_dir}/mes_record_PT_{user_name}_NoC{record_step}_IoU95_tmp.npy')
        start_data_id = iou_record_tmp.shape[0]
    except:
        iou_record_tmp, noc_record_tmp, acc_record_tmp, eval_iou_record_tmp, mes_record_tmp \
            = np.array([]).reshape(0,record_step), np.array([]).reshape(0,3), np.array([]).reshape(0,record_step), np.array([]), np.array([]).reshape(0,record_step)
        start_data_id = 0

    selected_ids = []
    real_data_id = -1

    for data_id in range(len(test_split)):

        print(f'data{data_id}')
        data = test_split.get_data(data_id)
        data = FTIS.input_preprocess_ml3d_s3dis(data)
        log.info(f'data {data_id}, {data.point.shape[0]} points')

        data_list = []
        if data.point.shape[0] > 150000:
            data_1, data_2 = FTIS.crop_data(data)
            if max(data_1.point.shape[0], data_2.point.shape[0]) < 150000:
                data_list = [data_1, data_2]
            elif min(data_1.point.shape[0], data_2.point.shape[0]) > 150000:
                drop_samples.append(data_id)
                continue
            elif data_1.point.shape[0] > 150000:
                data_11, data_12 = FTIS.crop_data(data_1)
                if max(data_11.point.shape[0], data_12.point.shape[0]) > 150000:
                    drop_samples.append(data_id)
                    continue
                data_list = [data_11, data_12, data_2]
            else:
                data_21, data_22 = FTIS.crop_data(data_2)
                if max(data_21.point.shape[0], data_22.point.shape[0]) > 150000:
                    drop_samples.append(data_id)
                    continue
                data_list = [data_1, data_21, data_22]
        else:
            data_list.append(data)


        for data in data_list:
            real_data_id += 1
            if real_data_id < start_data_id: continue
            if len(selected_ids)>0:
                if real_data_id not in selected_ids:
                    continue
            iou_record_list, acc_record_list, max_error_size_record_list, mean_error_size_record_list, \
            iou_list, acc_list, eval_iou_list, eval_acc_list, noc_list = \
                infer_and_count(FTIS, data, iou_record_list, acc_record_list, max_error_size_record_list,
                                mean_error_size_record_list, iou_list, acc_list, eval_iou_list, eval_acc_list, noc_list)


            iou_record_tmp = np.concatenate([iou_record_tmp, iou_record_list[-1]])
            noc_record_tmp = np.concatenate([noc_record_tmp, noc_list[-1]])
            acc_record_tmp = np.concatenate([acc_record_tmp, acc_record_list[-1]])
            eval_iou_record_tmp = np.concatenate([eval_iou_record_tmp, [eval_iou_list[-1]]])
            mes_record_tmp = np.concatenate([mes_record_tmp, max_error_size_record_list[-1]])

            np.save(f'{save_dir}/iou_record_PT_{user_name}_NoC{record_step}_IoU95_tmp.npy', iou_record_tmp)
            np.save(f'{save_dir}/noc_record_PT_{user_name}_NoC{record_step}_IoU95_tmp.npy', noc_record_tmp)
            np.save(f'{save_dir}/acc_record_PT_{user_name}_NoC{record_step}_IoU95_tmp.npy', acc_record_tmp)
            np.save(f'{save_dir}/eval_iou_PT_{user_name}_NoC{record_step}_IoU95_tmp.npy', eval_iou_record_tmp)
            np.save(f'{save_dir}/mes_record_PT_{user_name}_NoC{record_step}_IoU95_tmp.npy', mes_record_tmp)


    log.info(f'iou:{np.mean(eval_iou_list):.2f} -> {np.mean(iou_list):.2f}')
    log.info(f'acc:{np.mean(eval_acc_list):.2f} -> {np.mean(acc_list):.2f}')

    if len(drop_samples) > 0:
        log.info(f'data {drop_samples} are dropped due to its scale')

    np.save(f'{save_dir}/iou_record_PT_{user_name}_NoC{record_step}_IoU95.npy', iou_record_tmp)
    np.save(f'{save_dir}/noc_record_PT_{user_name}_NoC{record_step}_IoU95.npy', noc_record_tmp)
    np.save(f'{save_dir}/acc_record_PT_{user_name}_NoC{record_step}_IoU95.npy', acc_record_tmp)
    np.save(f'{save_dir}/eval_iou_PT_{user_name}_NoC{record_step}_IoU95.npy', eval_iou_record_tmp)
    np.save(f'{save_dir}/mes_record_PT_{user_name}_NoC{record_step}_IoU95.npy', mes_record_tmp)


def semantickitti_randlanet(args):
    cfg_file = "Open3D-ML/ml3d/configs/randlanet_semantickitti.yml"
    cfg = _ml3d.utils.Config.load_from_file(cfg_file)
    dataset_path = "/media/vcg8004/WD_BLACK/dataset/SemanticKITTI/data_odometry_velodyne/"
    cfg.dataset['dataset_path'] = dataset_path
    '''load s3dis in open3d framework'''
    dataset = ml3d.datasets.SemanticKITTI(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
    test_split = dataset.get_split("val")
    '''load pointtransformer trained on s3dis'''

    colormap = plt.get_cmap("tab20")(np.arange(20))[:, :3]
    colormap_dict = dict()
    for i in range(colormap.shape[0]):
        colormap_dict[i] = colormap[i]

    model = ml3d.models.RandLANet(**cfg.model)
    FTIS = FTIS_pipeline(model=model, ckpt_path='Open3D-ML/logs/randlanet_semantickitti_202201071330utc.pth',
                         name='semantickitti_randlanet', max_num_click=30, color_map=colormap_dict)

    FTIS.expected_IoU = [0.6,0.7,0.8]

    FTIS.interact_mode = args.interact_mode
    FTIS.entropy_vis = args.entropy_vis
    FTIS.tsne_vis = args.tsne_vis
    FTIS.input_vis = args.input_vis

    FTIS.alpha = 1 # args.alpha
    FTIS.beta = args.beta  # 100.
    FTIS.gamma = 0.
    FTIS.lr = args.lr  #1e-3

    FTIS.warm_lr = 1e-2

    FTIS.warm_round = 5
    FTIS.adapt_round = args.adapt_round #4
    FTIS.max_adapt_round = 10

    FTIS.momentum = 0.0
    FTIS.optim_weight_decay = 1e-2


    FTIS.use_warm_up = args.warm_up
    FTIS.use_confidence_filtering = args.filter
    FTIS.DBSCAN_eps = 0.01

    FTIS.entropy_threshold_increase = args.entropy_threshold_increase
    FTIS.entropy_threshold_decrease = args.entropy_threshold_decrease

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

    FTIS.max_num_click = 30
    FTIS.num_clicks_per_round = 1

    save_dir = args.save_dir
    if save_dir is None:
        save_dir = timestamp
        os.makedirs(save_dir, exist_ok=True)
        os.system(f'cp ISSR.py {save_dir}/')

    file = open(f'{save_dir}/args.txt', 'w')
    file.write(str(args))
    file.close()

    user_name = args.user_name
    record_step = FTIS.max_num_click // FTIS.num_clicks_per_round

    try:
        iou_record_tmp = np.load(f'{save_dir}/iou_record_RandLANet_ab_th_NoC{record_step}_IoU80_tmp.npy')
        noc_record_tmp = np.load(f'{save_dir}/noc_record_RandLANet_ab_th_NoC{record_step}_IoU80_tmp.npy')
        acc_record_tmp = np.load(f'{save_dir}/acc_record_RandLANet_ab_th_NoC{record_step}_IoU80_tmp.npy')
        eval_iou_record_tmp = np.load(f'{save_dir}/eval_iou_RandLANet_ab_th_NoC{record_step}_IoU80_tmp.npy')
        start_data_id = iou_record_tmp.shape[0]
    except:
        iou_record_tmp, noc_record_tmp, acc_record_tmp, eval_iou_record_tmp, mes_record_tmp \
            = np.array([]).reshape(0,record_step), np.array([]).reshape(0,3), np.array([]).reshape(0,record_step), np.array([]), np.array([]).reshape(0,record_step)
        start_data_id = 0

    for data_id in range(len(test_split)):
        if data_id>500: break  # for rapid validation
        if data_id < start_data_id: continue

        print(f'data {data_id}')
        data = test_split.get_data(data_id)
        data = FTIS.input_preprocess_ml3d_semantickitti(data)
        point = data['coords'][0].squeeze(0)
        data['point'] = point
        log.info(f'data {data_id}, {point.shape[0]} points')

        iou_record_list, acc_record_list, max_error_size_record_list, mean_error_size_record_list, \
            iou_list, acc_list, eval_iou_list, eval_acc_list, noc_list = \
            infer_and_count(FTIS, data, iou_record_list, acc_record_list, max_error_size_record_list,
                            mean_error_size_record_list, iou_list, acc_list, eval_iou_list, eval_acc_list, noc_list)
        iou_record_tmp = np.concatenate([iou_record_tmp, iou_record_list[-1]])
        noc_record_tmp = np.concatenate([noc_record_tmp, noc_list[-1]])
        acc_record_tmp = np.concatenate([acc_record_tmp, acc_record_list[-1]])
        eval_iou_record_tmp = np.concatenate([eval_iou_record_tmp, [eval_iou_list[-1]]])
        np.save(f'{save_dir}/iou_record_RandLANet_{user_name}_NoC{record_step}_IoU80_tmp.npy', iou_record_tmp)
        np.save(f'{save_dir}/noc_record_RandLANet_{user_name}_NoC{record_step}_IoU80_tmp.npy', noc_record_tmp)
        np.save(f'{save_dir}/acc_record_RandLANet_{user_name}_NoC{record_step}_IoU80_tmp.npy', acc_record_tmp)
        np.save(f'{save_dir}/eval_iou_RandLANet_{user_name}_NoC{record_step}_IoU80_tmp.npy', eval_iou_record_tmp)

    log.info(f'{np.mean(noc_list):.2f} clicks, iou:{np.mean(eval_iou_list):.2f} -> {np.mean(iou_list):.2f}')
    if len(drop_samples) > 0:
        log.info(f'data {drop_samples} are dropped due to its scale')

    np.save(f'{save_dir}/iou_record_RandLANet_{user_name}_NoC{record_step}_IoU80.npy', iou_record_tmp)
    np.save(f'{save_dir}/noc_record_RandLANet_{user_name}_NoC{record_step}_IoU80.npy', noc_record_tmp)
    np.save(f'{save_dir}/acc_record_RandLANet_{user_name}_NoC{record_step}_IoU80.npy', acc_record_tmp)
    np.save(f'{save_dir}/eval_iou_RandLANet_{user_name}_NoC{record_step}_IoU80.npy', eval_iou_record_tmp)


def scannet_sparseconvunet():
    cfg_file = "Open3D-ML/ml3d/configs/sparseconvunet_scannet.yml"
    cfg = _ml3d.utils.Config.load_from_file(cfg_file)
    '''load s3dis in open3d framework'''
    dataset = ml3d.datasets.Scannet(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
    test_split = dataset.get_split("val")
    '''load pointtransformer trained on s3dis'''

    colormap = plt.get_cmap("tab20")(np.arange(20))[:, :3]
    colormap_dict = dict()
    for i in range(colormap.shape[0]):
        colormap_dict[i] = colormap[i]

    model = ml3d.models.SparseConvUnet(**cfg.model)
    FTIS = FTIS_pipeline(model=model, ckpt_path=cfg.model.ckpt_path,
                         name='scannet_sparseconvunet', max_num_click=30, color_map=colormap_dict)

    FTIS.interact_mode = args.interact_mode
    FTIS.alpha = args.alpha
    FTIS.beta = args.beta
    FTIS.gamma = 1.
    FTIS.warm_round = 5
    FTIS.adapt_round = args.adapt_round
    FTIS.max_adapt_round = 10

    FTIS.entropy_vis = args.entropy_vis
    FTIS.tsne_vis = args.tsne_vis
    FTIS.input_vis = args.input_vis

    FTIS.use_warm_up = args.warm_up
    FTIS.use_confidence_filtering = args.filter

    FTIS.optim_weight_decay = args.optim_weight_decay #5e-1
    # FTIS.warm_lr = 5e-2
    FTIS.lr = args.lr

    FTIS.expected_IoU = [0.8,0.85,0.9]

    FTIS.entropy_threshold_increase = args.entropy_threshold_increase #0.1
    FTIS.entropy_threshold_decrease = args.entropy_threshold_decrease #0.01
    FTIS.DBSCAN_eps = 0.01
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

    FTIS.max_num_click = 30
    FTIS.num_clicks_per_round = 1
    record_step = FTIS.max_num_click // FTIS.num_clicks_per_round

    user_name = args.user_name
    save_dir = args.save_dir
    if save_dir is None:
        save_dir = timestamp
        os.makedirs(save_dir, exist_ok=True)
        os.system(f'cp ISSR.py {save_dir}/')

    file = open(f'{save_dir}/args.txt', 'w')
    file.write(str(args))
    file.close()

    try:
        iou_record_tmp = np.load(f'{save_dir}/iou_record_SCU_{user_name}_NoC30_IoU90_tmp.npy')
        noc_record_tmp = np.load(f'{save_dir}/noc_record_SCU_{user_name}_NoC30_IoU90_tmp.npy')
        acc_record_tmp = np.load(f'{save_dir}/acc_record_SCU_{user_name}_NoC30_IoU90_tmp.npy')
        mes_record_tmp = np.load(f'{save_dir}/mes_record_SCU_{user_name}_NoC30_IoU90_tmp.npy')
        eval_iou_record_tmp = np.load(f'{save_dir}/eval_iou_SCU_{user_name}_NoC{record_step}_IoU90_tmp.npy')
        start_data_id = iou_record_tmp.shape[0]
    except:

        iou_record_tmp, noc_record_tmp, acc_record_tmp, eval_iou_record_tmp \
            = np.array([]).reshape(0,record_step), np.array([]).reshape(0,3), np.array([]).reshape(0,record_step), np.array([])
        mes_record_tmp = np.array([]).reshape(0,record_step)
        start_data_id = 0

    selected_ids = []

    for data_id in range(len(test_split)):
        if data_id < start_data_id: continue
        if len(selected_ids) > 0:
            if data_id not in selected_ids:
                continue
        print(f'data{data_id}')
        data = test_split.get_data(data_id)
        data = FTIS.input_preprocess_ml3d_s3dis(data)

        log.info(f'data {data_id}, {data.point[0].shape[0]} points')

        iou_record_list, acc_record_list, max_error_size_record_list, mean_error_size_record_list, \
        iou_list, acc_list, eval_iou_list, eval_acc_list, noc_list = \
            infer_and_count(FTIS, data, iou_record_list, acc_record_list, max_error_size_record_list,
                            mean_error_size_record_list, iou_list, acc_list, eval_iou_list, eval_acc_list, noc_list)

        iou_record_tmp = np.concatenate([iou_record_tmp, iou_record_list[-1]])
        noc_record_tmp = np.concatenate([noc_record_tmp, noc_list[-1]])
        acc_record_tmp = np.concatenate([acc_record_tmp, acc_record_list[-1]])
        eval_iou_record_tmp = np.concatenate([eval_iou_record_tmp, [eval_iou_list[-1]]])
        mes_record_tmp = np.concatenate([mes_record_tmp, max_error_size_record_list[-1]])

        np.save(f'{save_dir}/iou_record_SCU_{user_name}_NoC30_IoU90_tmp.npy', iou_record_tmp)
        np.save(f'{save_dir}/noc_record_SCU_{user_name}_NoC30_IoU90_tmp.npy', noc_record_tmp)
        np.save(f'{save_dir}/acc_record_SCU_{user_name}_NoC30_IoU90_tmp.npy', acc_record_tmp)
        np.save(f'{save_dir}/eval_iou_SCU_{user_name}_NoC{record_step}_IoU90_tmp.npy', eval_iou_record_tmp)
        np.save(f'{save_dir}/mes_record_SCU_{user_name}_NoC{record_step}_IoU90_tmp.npy', mes_record_tmp)

    log.info(f'iou:{np.mean(eval_iou_list):.2f} -> {np.mean(iou_list):.2f}')
    log.info(f'acc:{np.mean(eval_acc_list):.2f} -> {np.mean(acc_list):.2f}')

    if len(drop_samples) > 0:
        log.info(f'data {drop_samples} are dropped due to its scale')

    np.save(f'{save_dir}/iou_record_SCU_{user_name}_NoC{record_step}_IoU90.npy', iou_record_tmp)
    np.save(f'{save_dir}/noc_record_SCU_{user_name}_NoC{record_step}_IoU90.npy', noc_record_tmp)
    np.save(f'{save_dir}/acc_record_SCU_{user_name}_NoC{record_step}_IoU90.npy', acc_record_tmp)
    np.save(f'{save_dir}/eval_iou_SCU_{user_name}_NoC{record_step}_IoU90.npy', eval_iou_record_tmp)
    np.save(f'{save_dir}/mes_record_SCU_{user_name}_NoC{record_step}_IoU90.npy', mes_record_tmp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', default=1., type=float)
    parser.add_argument('--beta', default=1000., type=float)
    parser.add_argument('--optim_weight_decay', default=1e-2, type=float)
    parser.add_argument('--filter', action='store_true')
    parser.add_argument('--warm_up', action='store_true')
    parser.add_argument('--user_name', default='test', type=str)
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--exp', default='semantickitti_randlanet', type=str)
    parser.add_argument('--interact_mode', default='simulated',type=str)
    parser.add_argument('--entropy_vis', action='store_true')
    parser.add_argument('--tsne_vis', action='store_true')
    parser.add_argument('--input_vis', action='store_true')

    parser.add_argument('--entropy_threshold_increase', default=0.1, type=float)
    parser.add_argument('--entropy_threshold_decrease', default=0.1, type=float)
    parser.add_argument('--adapt_round', default=5, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    args = parser.parse_args()

    if args.exp == 's3dis_pointtransformer':
        s3dis_pointtransformer(args)
    elif args.exp == 'scannet_sparseconvunet':
        scannet_sparseconvunet()
    elif args.exp == 'semantickitti_randlanet':
        semantickitti_randlanet(args)
    else:
        Exception('no supported yet')