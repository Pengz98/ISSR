from utils.knn_utils import query_ball_point, pc_normalize, square_distance
import torch
import numpy as np

def get_effect_radius(picked_idx, xyz, pred):
    picked_xyz = xyz[picked_idx, :].reshape(len(picked_idx),-1)
    picked_pred = pred[picked_idx]
    other_idx = torch.where(pred!=picked_pred)[0]
    other_xyz = xyz[other_idx, :]
    dist = square_distance(picked_xyz, other_xyz)
    min_dist = dist.squeeze().min()
    radius = 3. * min_dist**0.5
    return radius



def extract_effect_point_idx(picked_idx, xyz, sp_graph, radius=0.1):
    # xyz = pc_normalize(xyz)
    new_xyz = xyz[picked_idx,:]
    ball_idx = query_ball_point(nsample=5000, xyz=xyz, new_xyz=new_xyz, radius=radius)
    ball_idx = np.unique(ball_idx.squeeze(0))
    sp_label = torch.from_numpy(sp_graph['in_component'])
    components = sp_graph['components']
    ball_sp_label = sp_label[ball_idx]
    ball_sp_label_cnt = np.unique(ball_sp_label)
    valid_point_idx = []
    for s in range(ball_sp_label_cnt.shape[0]):
        valid_point_idx.append(components[ball_sp_label_cnt[s]])
    valid_point_idx = np.concatenate(valid_point_idx)
    return valid_point_idx, ball_idx