import open3d as o3d
import torch
import numpy as np
import open3d.ml.torch as ml3d

class KDTree_search():
    def __init__(self, point):
        self.pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point))
        self.kd_tree = o3d.geometry.KDTreeFlann(self.pcd)

    def search_knn_points(self, query_idx=None, k=10):
        if query_idx is None:
            [k, idx, _] = self.kd_tree.search_knn_vector_3d(self.pcd.points, k)
        else:
            [k, idx, _] = self.kd_tree.search_knn_vector_3d(self.pcd.points[query_idx], k)
        return idx


def query_knn_points(points, queries, k):
    if type(points) is np.ndarray:
        points = torch.from_numpy(points)
    if type(queries) is np.ndarray:
        queries = torch.from_numpy(queries)
    nsearch = ml3d.layers.KNNSearch(return_distances=False, ignore_query_point=False)
    ans = nsearch(points, queries, k)
    neighbors_idx = ans.neighbors_index.reshape(queries.shape[0],k)
    return neighbors_idx


def pc_normalize(pc):
    pc = pc.cpu().data.numpy()
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return torch.from_numpy(pc)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    N, _ = src.shape
    M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(1,0))
    dist += torch.sum(src ** 2, -1).view(N, 1)
    dist += torch.sum(dst ** 2, -1).view(1, M)
    return dist

def query_ball_point(nsample, xyz, new_xyz, radius=3.):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    N, C = xyz.shape
    S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).view(1, N).repeat([S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :nsample]
    if nsample is None:
        nsample = group_idx.shape[-1]
    group_first = group_idx[:, 0].view(S, 1).repeat([1,nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx