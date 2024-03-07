import numpy as np
import torch
import open3d as o3d


def index_points(points, idx):
    '''
    :param points: [B,N,C]
    :param idx: [B,S]
    :return: indexed points: [B,S,C]
    '''
    device = points.device
    B = points.shape[0]

    # if idx.dim() != 2:
    #     idx = idx.squeeze()

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)    # view_shape=[B,1...1], [B,1] typically
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1     # repeat_shape=[1,S] typically
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def random_sampling(points, num_sample, seed=None):
    numpy = type(points)==np.ndarray
    ndim = points.ndim
    if ndim == 2:
        points = points.reshape(1,-1,points.shape[-1])
    if numpy:
        points = torch.from_numpy(points)
    batch_size, num_point = points.shape[0], points.shape[1]
    sample_idx_list = []
    for b in range(batch_size):
        batch_sample_idx = np.random.choice(num_point, (1, num_sample), replace=False)
        sample_idx_list.append(batch_sample_idx)
    sample_idx = np.concatenate(sample_idx_list, 0)
    sampled_points = index_points(points, sample_idx)
    if ndim == 2:
        sampled_points = sampled_points[0]
        sample_idx = sample_idx[0]
    if numpy:
        sampled_points = sampled_points.numpy()
    return sampled_points, sample_idx

def voxel_sampling(points, voxel_size=0.1):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd_samp, trace_np, trace_list = pcd.voxel_down_sample_and_trace(voxel_size, pcd.get_min_bound(), pcd.get_max_bound(), False)
    points_samp = np.asarray(pcd_samp.points)
    return points_samp, trace_np

def proj_after_vs(data, trace_np):
    mask = trace_np != -1
    nn_data = data[trace_np]
    proj_data = (nn_data * mask.reshape(mask.shape[0],-1,1)).sum(1) / mask.sum(-1).reshape(-1,1)
    return proj_data


