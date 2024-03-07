import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors


class check_stop_optimize():
    def __init__(self):
        self.loss_unsup = np.zeros((3,))
        self.max_round = 5
    def loss_unsup_update(self, loss_unsup):
        self.loss_unsup[0] = self.loss_unsup[1]
        self.loss_unsup[1] = self.loss_unsup[2]
        self.loss_unsup[2] = loss_unsup
    def loss_unsup_trend(self, beta_1=1.5, beta_2=0.2):
        l1 = self.loss_unsup[0]-self.loss_unsup[1]
        l2 = self.loss_unsup[1]-self.loss_unsup[2]
        if l1/l2 < beta_1 and l2/self.loss_unsup[1] < beta_2:
            return True
        else:
            return False
    def by_loss_treand(self, loss_sup, loss_unsup, continued_round):
        if continued_round > self.max_round:
            return True
        self.loss_unsup_update(loss_unsup)
        if continued_round >= 2:
            sup_flag = True if loss_sup == 0 else False
            unsup_flag = self.loss_unsup_trend()
            stop_flag = sup_flag * unsup_flag
            return stop_flag
        else:
            return False


def test_time_warm_up(to_warm_model, inputs):
    from utils.vis_utils import vis
    model = copy.deepcopy(to_warm_model)
    model.eval()
    eval_results = model(inputs['data'])
    if eval_results.shape[0]==1:
        eval_results = eval_results.squeeze(0)
    target = eval_results.max(-1)[1].detach()
    # vis(torch.cat([inputs['data'].point, target.cpu().reshape(-1, 1)], -1), hint='eval pred')
    model.train()
    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-2)
    break_flag = False
    for i in range(10):
        optim.zero_grad()
        train_results = model(inputs['data'])
        if train_results.shape[0] == 1:
            train_results = train_results.squeeze(0)
        train_pred = train_results.max(1)[1]
        acc = (train_pred == target).sum() / target.shape[0]
        # vis(torch.cat([inputs['data'].point, train_pred.cpu().reshape(-1, 1)], -1), hint=f'train pred(acc={acc})')
        if acc > 0.99:
            print(f'model warmed after {i} rounds')
            break_flag = True
            break
        loss = cross_entropy(train_results, target)
        loss.backward()
        optim.step()
    if not break_flag:
        print(f'model warmed after 20 rounds')
    warmed_params = copy.deepcopy(model.state_dict())
    del model
    return warmed_params

def eval_fishers(results, model):
    ewc_optimizer = torch.optim.SGD(model.parameters(), 1e-3)
    fishers = {}
    criterion = nn.CrossEntropyLoss().to(results.device)
    targets = results.max(1)[1]
    loss = criterion(results, targets)
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            fisher = param.grad.data.clone().detach() ** 2
            fishers.update({name: [fisher, param.data.clone().detach()]})
    del ewc_optimizer
    return fishers


def cal_consist_loss(results, sp_graph, sp_features):
    components = sp_graph['components']
    sp_vote = sp_graph['sp_vote']
    sp_source = sp_graph['source'].squeeze(1)
    sp_target = sp_graph['target'].squeeze(1)
    se_delta_norm = sp_graph['se_delta_norm'].squeeze(1)

    similarity_matrix = np.zeros((components.shape[0], components.shape[0]))
    similarity_matrix[sp_source, sp_target] = se_delta_norm

    sp_result_pooling = []
    indicator_matrix = np.zeros((components.shape[0], components.shape[0]))
    for i in range(components.shape[0]):
        sp_result_pooling.append(torch.mean(results[components[i]], dim=0).reshape(1,-1))
        indicator_matrix[i] = (sp_vote[i] == sp_vote) * 1
    sp_result_pooling = torch.cat(sp_result_pooling,0)

    I_matrix = torch.from_numpy(indicator_matrix).to(results.device).float()
    W_matrix = torch.from_numpy(similarity_matrix).to(results.device).float()
    D_matrix = torch.diag(torch.sum(W_matrix, dim=1)).float()
    loss_smooth = sp_result_pooling.transpose(1,0) @ I_matrix @ (D_matrix - W_matrix) @ sp_result_pooling
    loss_smooth = loss_smooth.sum()

    loss_fidelity = cross_entropy(sp_result_pooling, torch.from_numpy(sp_vote).to(results.device))

    loss = loss_smooth / components.shape[0]**2 + loss_fidelity
    return loss

def cal_inter_sp_loss(results, sp_graph, sp_features, confidence_threshold=0.6):
    components = sp_graph['components']
    sp_confidence = sp_graph['sp_confidence']
    sp_vote = sp_graph['sp_vote']
    # sp_centroids = sp_graph['sp_centroids']
    sp_source = sp_graph['source'].squeeze(1)
    sp_target = sp_graph['target'].squeeze(1)
    se_delta_norm = sp_graph['se_delta_norm'].squeeze(1)
    loss_inter_sp = 0.0
    cnt = 0
    for i in range(sp_confidence.shape[0]):
        if sp_confidence[i] < confidence_threshold:
            loss_inter_sp += 0.0
            continue
        spedge_idx = np.where(sp_source == i)

        source_sp_idx = i
        source_point_idx = components[source_sp_idx]
        source_point_result = results[source_point_idx]
        source_result_pooling = torch.mean(source_point_result, dim=0)

        target_sp_idx = sp_target[spedge_idx]
        target_vote = sp_vote[target_sp_idx]
        target_vote = torch.from_numpy(target_vote)
        target_confidence = sp_confidence[target_sp_idx]

        spedge_dist = se_delta_norm[spedge_idx]
        weight = torch.from_numpy(target_confidence / spedge_dist)
        weight = weight / weight.sum()


        loss = 0.0
        for j in range(target_confidence.shape[0]):
            loss += weight[j] * cross_entropy(source_result_pooling.reshape(1,-1), target_vote[j].to(results.device))
        loss_inter_sp += loss / target_confidence.shape[0]
        cnt += 1
    loss_inter_sp = loss_inter_sp / cnt
    return loss_inter_sp

def cal_intra_sp_cross_point_loss(graph_nn, idx, confident_idx, results):
    confused_idx = np.setdiff1d(idx, confident_idx, assume_unique=True)
    if confused_idx.shape[0] == 0:
        return 0.
    source_idx = []
    for confused_id in confused_idx:
        source_idx.append(10 * confused_id + np.arange(10))
    source_idx = np.concatenate(source_idx)
    source = graph_nn['source'].astype('int')[source_idx]
    target = graph_nn['target'].astype('int')[source_idx]

    target_idx = []
    for tar in target:
        target_idx.append(tar in confident_idx)
    target_idx = np.nonzero(target_idx)
    source = source[target_idx]
    target = target[target_idx]
    edge_weight = torch.from_numpy(graph_nn['edge_weight']).to(results.device)[source_idx][target_idx]

    target_result = results[target,:].softmax(-1)
    loss_cross_point = torch.nn.functional.kl_div(torch.log_softmax(results[source,:],-1), target_result, reduction='none').sum(-1)
    loss_cross_point = torch.mul(loss_cross_point, edge_weight.softmax(0)).sum()
    return loss_cross_point


def cal_intra_sp_loss(results, sp_graph, sp_features, confidence_threshold=0.6):
    sp_picked = sp_graph['sp_picked']
    picked_idx = sp_graph['picked_idx']
    sp_vote = sp_graph['sp_vote']
    sp_confidence = sp_graph['sp_confidence']
    graph_nn = sp_features['graph_nn']
    xyz = sp_features['xyz']
    all_loss = 0.0
    cnt_sp_points = 0
    all_confident_idx = np.empty(shape=(1,))
    user_affect_idx = np.empty((1,))
    for i in range(sp_confidence.shape[0]):
        in_sp_idx = np.array(sp_graph['components'][i])
        if sp_confidence[i] < confidence_threshold or in_sp_idx.shape[0] < 10:
            continue
        sp_confident_idx = sp_graph['sp_confident_idx'][i]

        # loss_unary_point = softmax_entropy(results[sp_confident_idx]).mean()

        if sp_picked[i] is not None:
            sp_picked_idx, sp_picked_label = sp_picked[i]
            user_affect_idx = np.concatenate([user_affect_idx, in_sp_idx])
            loss_unary_point = 0.0
            nn = NearestNeighbors(n_neighbors=in_sp_idx.shape[0], algorithm='kd_tree').fit(xyz[in_sp_idx])
            distance, idx = nn.kneighbors(xyz[sp_picked_idx])
            del nn
            loss_weight_cache = np.empty((in_sp_idx.shape[0], 0))
            for k in range(sp_picked_idx.shape[0]):
                distance_norm = distance[k, idx[k].argsort()]
                loss_weight = np.exp(-1 * distance_norm) / np.exp(-1 * distance_norm).sum()
                loss_weight_cache = np.concatenate([loss_weight_cache, loss_weight.reshape(-1,1)], -1)
            indicator = np.argmax(loss_weight_cache, -1)
            pick_label = sp_picked_label[indicator].to(results.device)
            loss_weight = loss_weight_cache[np.arange(indicator.shape[0]), indicator]
            loss_weight = torch.from_numpy(loss_weight).to(results.device)
            loss_picked_point = in_sp_idx.shape[0] * cross_entropy(results[in_sp_idx], pick_label, weights=loss_weight, reduction='sum')
        else:
            loss_unary_point = cross_entropy(results[in_sp_idx],
                                             sp_vote[i] * torch.ones((in_sp_idx.shape[0],)).to(results.device),
                                             reduction='sum')
            loss_picked_point = 0.0

        # loss_cross_point = cal_intra_sp_cross_point_loss(graph_nn, in_sp_idx, sp_confident_idx, results)
        loss_cross_point = 0.0
        cnt_sp_points += in_sp_idx.shape[0]
        loss = loss_unary_point + 0. * loss_cross_point + 0. * loss_picked_point
        all_loss += loss
        all_confident_idx = np.concatenate([all_confident_idx, in_sp_idx])
    all_loss = all_loss / cnt_sp_points
    return all_loss, all_confident_idx, user_affect_idx


# def cal_inter_sp_loss(results, sp_graph, sp_features, confidence_threshold=0.5):

def cal_loss_under_pseudo(result, sp_vec, uncertain_threashold=0.1):
    uncertainty = softmax_entropy(result)
    uncertain_idx = torch.where(uncertainty>uncertain_threashold)
    pseudo_label = result.max(1)[1].detach()
    weights = sp_vec[np.arange(sp_vec.shape[0]),pseudo_label.cpu().data].to(result.device)
    # weights[uncertain_idx] = 0.
    loss_under_pseudo = cross_entropy(result, pseudo_label, weights=weights)
    return loss_under_pseudo


def cal_loss_over_pseudo(result, sp_vec, uncertain_threashold=0.8):
    uncertainty = sp_vec.max(-1)[0]
    pseudo_label = sp_vec.max(1)[1]
    weights = result.softmax(-1)[np.arange(result.shape[0]), pseudo_label].to(result.device)
    loss_over_pseudo = cross_entropy(result, pseudo_label.to(result.device), weights=weights)
    return loss_over_pseudo


def softmax_entropy(x):
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(-1) * x.log_softmax(-1)).sum(-1)

def class_entropy(x, c):
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(-1) * x.log_softmax(-1))[:, c]

def cross_entropy(pred, gold, smoothing=False, ignore_index=255, weights=None, reduction='mean'):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = CrossEntropyLoss(aggregate=reduction)(pred, gold, weights)  # my instance-weighted loss

    return loss


class CrossEntropyLoss(nn.Module):
    """
    Cross entropy with instance-wise weights. Leave `aggregate` to None to obtain a loss
    vector of shape (batch_size,).loss_weight
    """
    def __init__(self, aggregate='mean'):
        super(CrossEntropyLoss, self).__init__()
        assert aggregate in ['sum', 'mean', None]
        self.aggregate = aggregate

    def log_sum_exp(self, x):
        # See implementation detail in
        # http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        # b is a shift factor. see link.
        # x.size() = [N, C]:
        b, _ = torch.max(x, 1)
        y = b + torch.log(torch.exp(x - b.unsqueeze(1).expand_as(x)).sum(1))
        # y.size() = [N, 1]. Squeeze to [N] and return
        return y

    def class_select(self, logits, target):
        # in numpy, this would be logits[:, target].
        batch_size, num_classes = logits.size()
        if target.is_cuda:
            device = target.data.get_device()
            one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                                   .long()
                                                   .repeat(batch_size, 1)
                                                   .cuda(device)
                                                   .eq(target.data.repeat(num_classes, 1).t()))
        else:
            one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                                   .long()
                                                   .repeat(batch_size, 1)
                                                   .eq(target.data.repeat(num_classes, 1).t()))
        return logits.masked_select(one_hot_mask)

    def cross_entropy_with_weights(self, logits, target, weights=None):
        assert logits.dim() == 2
        assert not target.requires_grad
        target = target.squeeze(1) if target.dim() == 2 else target
        assert target.dim() == 1
        loss = self.log_sum_exp(logits) - self.class_select(logits, target)
        if weights is not None:
            # loss.size() = [N]. Assert weights has the same shape
            assert list(loss.size()) == list(weights.size())
            # Weight the loss
            loss = loss * weights
        return loss

    def forward(self, input, target, weights=None):
        if self.aggregate == 'sum':
            return self.cross_entropy_with_weights(input, target, weights).sum()
        elif self.aggregate == 'mean':
            return self.cross_entropy_with_weights(input, target, weights).mean()
        elif self.aggregate is None:
            return self.cross_entropy_with_weights(input, target, weights)
