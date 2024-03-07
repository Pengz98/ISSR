import copy

import numpy as np
import torch
from utils.adapt_loss import softmax_entropy
import open3d as o3d


def in_component_to_components(in_component):
    components = []
    elem = np.unique(in_component)
    for i in range(elem.shape[0]):
        in_sp_idx = np.where(in_component==elem[i])[0]
        components.append(list(in_sp_idx))
    components = np.array(components, dtype='object')
    return components


def update_sp_graph(sp_graph, sp_graph_local, update_idx):
    updated_graph = copy.deepcopy(sp_graph)
    in_component = sp_graph['in_component'] + 1
    in_component_local = sp_graph_local['in_component'] + 2

    solid_idx = np.ones((in_component.shape[0],))
    solid_idx[update_idx] = 0
    solid_idx = np.where(solid_idx==1)[0]

    in_component[update_idx] = 0
    max_in_component = in_component.max()+1
    in_component_local += max_in_component
    in_component_local[solid_idx] = 0

    updated_in_component = in_component + in_component_local
    elements = np.unique(updated_in_component)
    updated_components = []
    norm_in_component = np.zeros((in_component.shape[0],))
    for i in range(elements.shape[0]):
        in_sp_idx = np.where(updated_in_component==elements[i])[0]
        norm_in_component[in_sp_idx] = i
        updated_components.append(list(in_sp_idx))
    updated_components = np.array(updated_components, dtype='object')
    updated_graph['components'] = updated_components
    updated_graph['in_components'] = norm_in_component
    return updated_graph



def get_active_component_ids(change_map, sp_graph, min_active=10):
    components = sp_graph['components']
    active_ids = []
    for i in range(components.shape[0]):
        in_sp_idx = components[i]
        changed_cnt = change_map[in_sp_idx].sum()
        if changed_cnt>=min_active:
            active_ids.append(i)
    return active_ids


def estimate_normals(point):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    normals = np.asarray(pcd.normals)


    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(pcd)
    # vis.run()
    # vis.destroy_window()
    return normals

def get_sp_idx_point_idx(picked_idx, sp_graph):
    sp_label = torch.from_numpy(sp_graph['in_component'])
    picked_sp_idx = sp_label[picked_idx]

    components = sp_graph['components']
    picked_point_idx = components[picked_sp_idx]
    return picked_sp_idx.item(), np.array(picked_point_idx)



# def extract_effect_point_idx(picked_idx, sp_graph, knn_sp=3):
#     sp_label = torch.from_numpy(sp_graph['in_component'])
#     cur_picked_sp_idx = sp_label[picked_idx]
#     sp_source = sp_graph['source'].squeeze()
#     sp_target = sp_graph['target'].squeeze()
#     sp_neighbor_idx = []
#     for k in range(cur_picked_sp_idx.shape[0]):
#         cur_sp_neighbor_idx = sp_target[np.where(sp_source == cur_picked_sp_idx[k].item())[0]]
#         sp_neighbor_idx.append(cur_sp_neighbor_idx)
#     sp_neighbor_idx = np.concatenate(sp_neighbor_idx)
#
#     components = sp_graph['components']
#     sp_neighbor_point_idx = []
#     for n in range(sp_neighbor_idx.shape[0]):
#         sp_neighbor_point_idx.append(components[sp_neighbor_idx[n]])
#     sp_neighbor_point_idx = np.concatenate(sp_neighbor_point_idx[:knn_sp])
#     sp_picked_point_idx = []
#     for p in range(cur_picked_sp_idx.shape[0]):
#         sp_picked_point_idx.append(components[cur_picked_sp_idx[p]])
#     sp_picked_point_idx = np.concatenate(sp_picked_point_idx)
#     return sp_picked_point_idx, sp_neighbor_point_idx


def get_sp_sup(result, sp_graph, picked_idx, gt):
    pred = result.argmax(-1).cpu().data
    sp_confidence_eval(sp_graph, pred, picked_idx, gt)
    components = sp_graph['components']
    sp_vote = sp_graph['sp_vote']
    sp_confidence = sp_graph['sp_confidence']
    sp_picked = sp_graph['sp_picked']
    sp_sup = torch.zeros_like(pred)
    sp_sup_confidence = torch.zeros_like(pred).float()
    for i in range(components.shape[0]):
        in_sp_idx = components[i]
        in_sp_entropy = softmax_entropy(result[in_sp_idx]).mean()
        cur_sp_vote = sp_vote[i]
        cur_sp_confidence = sp_confidence[i]
        sp_sup[in_sp_idx] = cur_sp_vote
        if sp_picked[i] is None:
            sp_sup_confidence[in_sp_idx] = cur_sp_confidence
        elif np.unique(sp_picked[i][1]).shape[0] > 1:
            sp_sup_confidence[in_sp_idx] = 0.
        else:
            if np.unique(sp_picked[i][1]) == cur_sp_vote:
                sp_sup_confidence[in_sp_idx] = 1.
            else:
                sp_sup_confidence[in_sp_idx] = 0.
    return sp_sup, sp_sup_confidence


def get_sp_vec(result, sp_label):
    hard_pred = result.argmax(-1).cpu().data
    confidence = result.softmax(-1)
    confidence = confidence.cpu().data[np.arange(result.shape[0]), hard_pred]
    one_hot_pred = torch.zeros((result.shape[0], result.shape[1]))
    one_hot_pred[np.arange(result.shape[0]),hard_pred] = 1
    sp_vec = torch.zeros((result.shape[0], result.shape[1]))
    for l in np.unique(sp_label):
        idx = np.where(sp_label==l)[0]
        cur_sp_vec = confidence[idx].reshape(-1,1) * one_hot_pred[idx, :]
        cur_sp_vec = cur_sp_vec.sum(0) / idx.shape[0]
        sp_vec[idx, :] = cur_sp_vec
    return sp_vec



def sp_confidence_eval(sp_graph, pred, picked_idx, gt, min_sp=20):
    components = sp_graph['components']
    sp_vote = np.zeros((components.shape[0],))
    sp_confidence = np.zeros((components.shape[0],))
    sp_confident_idx = np.zeros((components.shape[0],), dtype=object)
    sp_pickd = np.empty((components.shape[0],), dtype=object)
    for i in range(components.shape[0]):
        in_sp_idx = np.array(components[i])
        in_sp_picked_idx = np.intersect1d(in_sp_idx, picked_idx)
        sp_preds = pred[in_sp_idx]
        if sp_preds.shape[0] < min_sp:
            sp_confidence[i] = 0.0

        sp_vote[i] = torch.bincount(sp_preds).argmax()
        sp_pickd[i] = None
        sp_confident_idx_local = torch.where(sp_preds == sp_vote[i])[0]
        sp_confident_idx[i] = in_sp_idx[sp_confident_idx_local]
        sp_confidence[i] = sp_confident_idx_local.shape[0] / sp_preds.shape[0]

        if in_sp_picked_idx.shape[0] == 0:
            sp_pickd[i] = None
        else:
            # overwrite the sp_vote results with user provided label
            sp_picked_preds = gt[in_sp_picked_idx]
            sp_pickd[i] = (in_sp_picked_idx, sp_picked_preds)
            # sp_vote[i] = torch.bincount(sp_picked_preds).argmax()
            # sp_confident_idx_local = torch.where(sp_preds==sp_vote[i])[0]
            # sp_confident_idx[i] = in_sp_idx[sp_confident_idx_local]
            # sp_confidence[i] = 1.0
    sp_graph['sp_vote'] = sp_vote
    sp_graph['sp_confidence'] = sp_confidence
    sp_graph['sp_confident_idx'] = sp_confident_idx
    sp_graph['picked_idx'] = picked_idx
    sp_graph['sp_picked'] = sp_pickd


