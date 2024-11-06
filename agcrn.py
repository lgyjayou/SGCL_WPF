from collections import OrderedDict

from einops import rearrange

import util
import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from aug import (
    sim_global,
    aug_topology,
    aug_feature
)


class AVWGCN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.sa = SpatialAttention(node_num)
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.weights_pool)
        nn.init.xavier_uniform_(self.bias_pool)
    
    def forward(self, x, node_embeddings, adj_m):
        node_num = node_embeddings.shape[0]

        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        s2 = self.sa(supports, adj_m)
        supports = (s2 * adj_m) + ((1-s2) * supports)

        support_set = [torch.eye(node_num).to(supports.device), supports]
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)  # Tensor(2, 307, 307)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)   # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)       # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)   # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias   # b, N, dim_out
        return x_gconv


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(node_num, dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(node_num, dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings, adj_m):
        state = state.to(x.device)  # Tensor(64, 307, 64)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings, adj_m))  # z_r: Tensor(64, 307, 128)
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)  # 根据最后一个维度将z_r分两半, z和r均为Tensor(64, 307, 64)
        candidate = torch.cat((x, z*state), dim=-1)  # Tensor(64, 307, 65)
        hc = torch.tanh(self.update(candidate, node_embeddings, adj_m))  # Tensor(64, 307, 64)
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings, adj_m):
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings, adj_m)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)  # (num_layers, B, N, hidden_dim)


class SpatialAttention(nn.Module):
    def __init__(self, num_nodes, kernel_size=1):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2*num_nodes, num_nodes, kernel_size, bias=False)
        nn.init.xavier_normal_(self.conv1.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        try:
            z = torch.cat([x, y], dim=1)
            z = z.unsqueeze(-1).unsqueeze(-1)
            z = self.conv1(z)
        except Exception as e:
            print("error：", e)
        z = torch.squeeze(z)
        return self.sigmoid(z)


class AGCRN(nn.Module):
    def __init__(self, args, num_nodes, embed_dim, dtw_graph, geo_graph):
        super(AGCRN, self).__init__()
        self.im_t = args.im_t
        self.device = args.device
        self.num_nodes = num_nodes
        self.output_dim = args.out_dim
        self.hidden_dim = args.rnn_units
        self.horizon = args.horizon
        self.method = args.method

        self.dtw_graph = dtw_graph
        self.geo_graph = geo_graph

        self.sa = SpatialAttention(num_nodes=num_nodes)

        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.node_embeddings)

        self.encoder = AVWDCRNN(num_nodes, args.in_dim, args.rnn_units, args.cheb_k, embed_dim, args.num_layers)

        self.lstm = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
        self.fc = nn.Linear(32, 64)

        self.end_conv = nn.Conv2d(1, args.horizon * args.out_dim, kernel_size=(1, args.rnn_units), bias=True)

        self.project_linear1 = nn.Linear(args.rnn_units, args.rnn_units)
        self.project_bn = nn.BatchNorm1d(args.rnn_units)
        self.project_relu = nn.ReLU()
        self.project_linear2 = nn.Linear(args.rnn_units, args.rnn_units)

        self.project_linear = nn.Conv2d(2 * args.rnn_units, 16, kernel_size=(1, args.history - args.horizon + 1), bias=True)

        self.nodeLoss_con = util.NodeLoss_Con(args, num_nodes, dtw_graph, geo_graph)  # SCL-loss
        self.timeLoss_con = util.TimeLoss_Con(args)  # TCL-loss

    def forward(self, input_val):
        # input_val (B, L, N, D)
        num_turbine = input_val.shape[2]
        init_state = self.encoder.init_hidden(input_val.shape[0])

        dtw_adj = torch.tensor(util.sym_adj(self.dtw_graph), device=self.device)
        if self.geo_graph is not None:
            geo_adj = torch.tensor(util.sym_adj(self.geo_graph), device=self.device)
            s1 = self.sa(dtw_adj, geo_adj)
            adj_m = (s1 * geo_adj) + ((1 - s1) * dtw_adj)
        else:
            adj_m = dtw_adj
        output, _ = self.encoder(input_val, init_state, self.node_embeddings, adj_m)

        x = rearrange(output, 'B L N D -> (B N) L D')
        lstm_output, _ = self.lstm(x)
        fc_output = self.fc(lstm_output)
        pred = rearrange(fc_output, '(B N) L D -> B L N D', N=num_turbine)
        pred = self.end_conv(pred[:, -1:, :, :])

        if self.training and self.method == 'contrastive_learning':
            s_sim_mx = sim_global(input_val, sim_type='cos')
            input_aug = aug_feature(input_val, self.im_t, self.device)  # feature-level augmentation

            dtw_graph_aug = aug_topology(s_sim_mx, torch.tensor(self.dtw_graph), percent=0.2)   # topology-level augmentation
            dtw_graph_aug = dtw_graph_aug.numpy()
            dtw_adj_aug = torch.tensor(util.sym_adj(dtw_graph_aug), device=self.device)
            if self.geo_graph is not None:
                geo_graph_aug = aug_topology(s_sim_mx, torch.tensor(self.geo_graph), percent=0.2)
                geo_graph_aug = geo_graph_aug.numpy()
                geo_adj_aug = torch.tensor(util.sym_adj(geo_graph_aug), device=self.device)
                s1_aug = self.sa(dtw_adj_aug, geo_adj_aug)
                adj_m_aug = (s1_aug * geo_adj_aug) + ((1 - s1_aug) * dtw_adj_aug)
            else:
                adj_m_aug = dtw_adj_aug
            output_aug, _ = self.encoder(input_aug, init_state, self.node_embeddings, adj_m_aug)

            # SCL-project head
            rep = torch.squeeze(output[:, -1:, :, :])  # (B, N, D)
            rep_node = self.project_linear1(rep)
            rep_node = rep_node.transpose(1, 2)
            rep_node = self.project_bn(rep_node)
            rep_node = rep_node.transpose(1, 2)
            rep_node = self.project_relu(rep_node)
            rep_node = self.project_linear2(rep_node)

            rep_aug = torch.squeeze(output_aug[:, -1:, :, :])  # (B, N, D)
            rep_node_aug = self.project_linear1(rep_aug)
            rep_node_aug = rep_node_aug.transpose(1, 2)
            rep_node_aug = self.project_bn(rep_node_aug)
            rep_node_aug = rep_node_aug.transpose(1, 2)
            rep_node_aug = self.project_relu(rep_node_aug)
            rep_node_aug = self.project_linear2(rep_node_aug)

            node_u_loss = self.nodeLoss_con(rep_node, rep_node_aug)

            # TCL
            h_l = output.transpose(1, 3)  # (64, 64, 134, 36)
            h_g = output_aug.transpose(1, 3)  # (64, 64, 134, 36)
            p = torch.cat((h_l, h_g), 1)  # (64, 128, 134, 36)
            p = self.project_linear(p)  # (64, 16, 134, 12)
            anchor, pos, neg = util.hard_sample(p)  # (36, 64, 128, 134)
            tc_loss = self.timeLoss_con(anchor, pos, neg)

            return pred, node_u_loss, tc_loss
        else:
            return pred





