import torch
import numpy as np
import open3d as o3d
from utils.knn_utils import query_knn_points

def simulated_interact(point, error_map, num_pick=1):
    knn_idx = query_knn_points(point, point, k=5).long()
    knn_error = error_map[knn_idx].sum(-1)
    candidate_ids = knn_error.argsort()[-20:]
    picked_idx = np.random.choice(candidate_ids.cpu().data.numpy(), num_pick)

    # components = sp_graph['components']
    # sp_ce_error_mean = []
    # sp_pick_idx = []
    # for i in range(components.shape[0]):
    #     in_sp_idx = components[i]
    #     if len(in_sp_idx) < 10:
    #         sp_ce_error_mean.append(0)
    #         sp_pick_idx.append(-1)
    #     in_sp_ce_error = ce_error_map[in_sp_idx]
    #     ce_error_mean = in_sp_ce_error.mean()
    #     sp_ce_error_mean.append(ce_error_mean)
    #     in_sp_pick = np.random.choice(in_sp_ce_error.argsort()[-10:], 1)[0]
    #     pick = in_sp_idx[in_sp_pick]
    #     sp_pick_idx.append(pick)
    # sp_ce_error_mean = np.array(sp_ce_error_mean)
    # sp_ce_error_max_idx = np.array(sp_pick_idx)
    # picked_sp_idx = sp_ce_error_mean.argsort()[-num_pick:]
    # picked_idx = sp_ce_error_max_idx[picked_sp_idx]
    return list(picked_idx)


