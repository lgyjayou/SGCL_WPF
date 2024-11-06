import copy
import numpy as np
import torch


def sim_global(flow_data, sim_type='cos'):
    """
    :param flow_data: tensor, original flow [n,l,v,c] or location embedding [n,v,c]
    :param type: str, type of similarity, attention or cosine. ['att', 'cos']
    :return sim: tensor, symmetric similarity, [v,v]
    """
    if len(flow_data.shape) == 4:
        n, l, v, c = flow_data.shape
        att_scaling = n * l * c
        cos_scaling = torch.norm(flow_data, p=2, dim=(0, 1, 3)) ** -1  # cal 2-norm of each node, dim N
        sim = torch.einsum('btnc, btmc->nm', flow_data, flow_data)
    elif len(flow_data.shape) == 3:
        n, v, c = flow_data.shape
        att_scaling = n * c
        cos_scaling = torch.norm(flow_data, p=2, dim=(0, 2)) ** -1  # cal 2-norm of each node, dim N
        sim = torch.einsum('bnc, bmc->nm', flow_data, flow_data)
    else:
        raise ValueError('sim_global only support shape length in [3, 4] but got {}.'.format(len(flow_data.shape)))

    if sim_type == 'cos':
        # cosine similarity
        scaling = torch.einsum('i, j->ij', cos_scaling, cos_scaling)
        sim = sim * scaling
    elif sim_type == 'att':
        # scaled dot product similarity
        scaling = float(att_scaling) ** -0.5
        sim = torch.softmax(sim * scaling, dim=-1)
    else:
        raise ValueError('sim_global only support sim_type in [att, cos].')

    return sim


def aug_topology(sim_mx, input_graph, percent=0.2):
    # edge dropping starts here
    diag_indices = np.diag_indices(input_graph.shape[0])
    input_graph[diag_indices] = 0

    drop_percent = percent / 2

    index_list = input_graph.nonzero()  # list of edges [row_idx, col_idx]

    edge_num = int(index_list.shape[0] / 2)  # treat one undirected edge as two edges
    edge_mask = (input_graph > 0).tril(diagonal=-1)
    add_drop_num = int(edge_num * drop_percent / 2)
    aug_graph = copy.deepcopy(input_graph)

    drop_prob = torch.softmax(sim_mx[edge_mask], dim=0)
    drop_prob = (1. - drop_prob).cpu().numpy()  # normalized similarity to get sampling probability
    drop_prob /= drop_prob.sum()
    drop_list = np.random.choice(edge_num, size=add_drop_num, p=drop_prob)
    drop_index = index_list[drop_list]

    zeros = torch.zeros_like(aug_graph[0, 0])
    aug_graph[drop_index[:, 0], drop_index[:, 1]] = zeros
    aug_graph[drop_index[:, 1], drop_index[:, 0]] = zeros

    # edge adding starts here
    node_num = input_graph.shape[0]
    x, y = np.meshgrid(range(node_num), range(node_num), indexing='ij')
    mask = y < x
    x, y = x[mask], y[mask]

    add_prob = sim_mx[torch.ones(sim_mx.size(), dtype=bool).tril(diagonal=-1)]
    add_prob = torch.softmax(add_prob, dim=0).cpu().numpy()
    add_list = np.random.choice(int((node_num * node_num - node_num) / 2),
                                size=add_drop_num, p=add_prob)

    ones = torch.ones_like(aug_graph[0, 0])
    aug_graph[x[add_list], y[add_list]] = ones
    aug_graph[y[add_list], x[add_list]] = ones

    aug_graph = aug_graph + np.eye(len(aug_graph))

    return aug_graph


def aug_feature(input, im_t, device):
    bs = input.shape[0]
    frame = input.shape[1]
    num_node = input.shape[2]
    rand = torch.rand(bs, frame, num_node).to(device)
    input[:, :, :, -1] = input[:, :, :, -1] * (rand >= im_t)
    return input
