import torch
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F


def sym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, rmse


def hard_sample(p):
    anchor, pos, neg = [], [], []
    length = p.shape[3]  # 19
    for i in range(length):
        minn, maxn = float('inf'), -float('inf')
        kmin, kmax = i, i
        for j in range(length):
            if i == j:
                continue
            dist = torch.sqrt(torch.sum(torch.square(p[:, :, :, i] - p[:, :, :, j])))
            if dist > maxn:
                maxn = dist
                kmax = j
            if dist < minn:
                minn = dist
                kmin = j
        anchor.append(p[:, :, :, i])
        pos.append(p[:, :, :, kmin])
        neg.append(p[:, :, :, kmax])

    anchor = torch.stack(anchor)
    pos = torch.stack(pos)
    neg = torch.stack(neg)

    return anchor, pos, neg


class NodeLoss_Con(torch.nn.Module):
    def __init__(self, args, num_nodes, dtw_graph, geo_graph):
        super(NodeLoss_Con, self).__init__()
        self.temperature = args.tempe  # 0.05
        self.device = args.device
        self.num_nodes = num_nodes
        self.dtw_graph = dtw_graph
        self.geo_graph = geo_graph

    def forward(self, rep_node, rep_node_aug):
        # rep_node, rep_node_aug: (bs, node_number, hidden)
        tempo_rep = rep_node.transpose(0, 1)
        tempo_rep_aug = rep_node_aug.transpose(0, 1)
        tempo_norm = tempo_rep.norm(dim=2).unsqueeze(dim=2)
        tempo_norm_aug = tempo_rep_aug.norm(dim=2).unsqueeze(dim=2)
        tempo_matrix = torch.matmul(tempo_rep, tempo_rep_aug.transpose(1, 2)) / torch.matmul(tempo_norm, tempo_norm_aug.transpose(1, 2))
        tempo_matrix = torch.exp(tempo_matrix / self.temperature)
        tempo_neg = torch.sum(tempo_matrix, dim=2)  # (node, bs)

        # spatial contrast
        spatial_norm = rep_node.norm(dim=2).unsqueeze(dim=2)
        spatial_norm_aug = rep_node_aug.norm(dim=2).unsqueeze(dim=2)
        spatial_matrix = torch.matmul(rep_node, rep_node_aug.transpose(1, 2)) / torch.matmul(spatial_norm,  spatial_norm_aug.transpose(1, 2))
        spatial_matrix = torch.exp(spatial_matrix / self.temperature)
        diag = torch.eye(self.num_nodes, dtype=torch.bool).to(self.device)
        pos_sum = torch.sum(spatial_matrix * diag, dim=2)  # (bs, node)
        # spatial negative filter
        if self.geo_graph is not None:
            adj_m = self.dtw_graph + self.geo_graph
        else:
            adj_m = self.dtw_graph
        adj_m_norm = sym_adj(adj_m)
        adj = (adj_m_norm == 0)
        adj = torch.tensor(adj).to(self.device)
        adj = adj + diag
        spatial_matrix = spatial_matrix * adj
        spatial_neg = torch.sum(spatial_matrix, dim=2)  # (bs, node)
        ratio = pos_sum / (spatial_neg + tempo_neg.transpose(0, 1) - pos_sum)
        node_u_loss = torch.mean(-torch.log(ratio))
        return node_u_loss


class TimeLoss_Con(torch.nn.Module):
    def __init__(self, args):
        super(TimeLoss_Con, self).__init__()
        self.temperature = args.tempe
        self.device = args.device

    def forward(self, anchor, zis, zjs):
        shape = zis.shape
        anchor = anchor.reshape(shape[0], -1)
        zis = zis.reshape(shape[0], -1)
        zjs = zjs.reshape(shape[0], -1)

        anchor1 = F.normalize(anchor, p=2, dim=-1)
        zis1 = F.normalize(zis, p=2, dim=-1)
        zjs1 = F.normalize(zjs, p=2, dim=-1)

        sim_pos = torch.matmul(anchor1, zis1.permute(1, 0))
        sim_neg = torch.matmul(anchor1, zjs1.permute(1, 0))

        positives = sim_pos / self.temperature
        negatives = sim_neg / self.temperature

        loss = -torch.log((torch.exp(positives)) / (torch.exp(negatives)))

        return loss.mean()
