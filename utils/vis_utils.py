import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import seaborn as sns

global Flag
Flag = True

def vis(pc_np, hint='default', max_label=None, max_value=None, heat_map=False, Flag=Flag):
    if Flag==False:
        return 0
    if type(pc_np) is not np.ndarray:
        pc_np = pc_np.cpu().data.numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_np[:, :3])
    if pc_np.shape[-1] == 4:
        labels = pc_np[:, -1]
        max_label = max_label if max_label is not None else labels.max()
        min_label = labels.min()
        if np.unique(labels).shape[0] == 2:
            color_1, color_2 = np.unique(labels)
            colors = np.zeros((pc_np.shape[0],3))
            colors[np.where(labels == color_1)[0]] = [0, 0, 1]
            colors[np.where(labels == color_2)[0]] = [1, 0, 0]

        elif np.unique(labels).shape[0] == 3 and labels.min()==-1:
            _, color_1, color_2 = np.unique(labels)
            colors = np.zeros((pc_np.shape[0],3))
            colors[np.where(labels == color_1)[0]] = [0, 0, 1]
            colors[np.where(labels == color_2)[0]] = [1, 0, 0]
            colors[np.where(labels == -1)[0]] = [0,0,0]

        elif heat_map:
            if max_value is not None:
                max_label = max_value
            colors = plt.get_cmap("gist_heat")(labels / max_label)[:,:3]
            if labels.min()==-1:
                colors[np.where(labels==-1)[0]] = [0,0,0]
        else:
            if min_label >= 0:
                colors = plt.get_cmap("tab20")((labels-min_label) / (max_label-min_label))[:, :3]
            if min_label == -1:
                idx = np.where(labels == -1)[0]
                colors = plt.get_cmap("tab20")((labels-min_label) / (max_label-min_label))[:, :3]
                colors[idx] = [0, 0, 0]

        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    elif pc_np.shape[-1] == 6:
        colors = pc_np[:,3:]
        if colors.max() > 1:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=hint)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

def vis_and_interact(pc_np, hint=None, Flag=True, max_value=None, max_label=None):
    if Flag==False:
        return []
    if type(pc_np) is not np.ndarray:
        pc_np = pc_np.cpu().data.numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_np[:,:3])
    if pc_np.shape[-1] == 4:
        labels = pc_np[:, -1]
        max_label = max_label if max_label is not None else labels.max()
        min_label = labels.min()
        if np.unique(labels).shape[0] == 2:
            color_1, color_2 = np.unique(labels)
            colors = np.zeros((pc_np.shape[0], 3))
            colors[np.where(labels == color_1)[0]] = [0, 0, 1]
            colors[np.where(labels == color_2)[0]] = [1, 0, 0]
        elif (labels % 1).sum() == 0:
            if min_label >= 0:
                colors = plt.get_cmap("tab20")((labels-min_label) / (max_label-min_label))[:, :3]
            if min_label == -1:
                idx = np.where(labels == -1)[0]
                colors = plt.get_cmap("tab20")((labels-min_label) / (max_label-min_label))[:, :3]
                colors[idx] = [0, 0, 0]
        else:
            if max_value is not None:
                max_label = max_value
            colors = plt.get_cmap("gist_heat")(labels / max_label)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    elif pc_np.shape[-1] == 6:
        colors = pc_np[:, 3:]
        if colors.max() > 1:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=hint)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    picked_idx = vis.get_picked_points()
    return picked_idx


def tsne_visualization_color(feature_bank, feature_anchor, colors, window='default'):
    palette = np.array(sns.color_palette("hls", colors.max()+1))
    # feature_bank: (file_size, 128)
    feat_dim = feature_bank.shape[-1]
    num_anchor = feature_anchor.shape[0]
    feature_bank = np.concatenate([feature_bank, feature_anchor],0)
    # Random state.
    RS = 20150101
    num_null = np.isnan(feature_bank).sum()
    num_inf = np.isinf(feature_bank).sum()
    if num_inf + num_inf > 0:
        print(f'num_null:{num_null},num_inf:{num_inf}')
        feature_bank = feature_bank.replace([np.inf, -np.inf], np.nan)
        feature_bank = feature_bank.fillna(value=np.zeros(feat_dim))
    features_proj = TSNE(random_state=RS).fit_transform(feature_bank)
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc_anchor = ax.scatter(features_proj[-num_anchor:, 0], features_proj[-num_anchor:, 1], s=15, c=palette[colors[-num_anchor:].astype(np.int32)])
    # plt.title(window)
    # plt.show()

    sc_feat = ax.scatter(features_proj[:-num_anchor, 0], features_proj[:-num_anchor, 1], s=5, c=palette[colors[:-num_anchor].astype(np.int32)])
    # img_path = '/TSNE' + '/tsne-generated-color-' + str(epoch) + '.png'
    # plt.savefig(img_path, dpi=120)
    plt.title(window)
    plt.show()