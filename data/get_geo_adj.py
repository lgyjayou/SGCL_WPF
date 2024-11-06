import argparse
import os

import math
import numpy as np
import pandas as pd


def get_sdwpf_graph(args):
    geo = pd.read_csv(os.path.join(args.data_path, args.file_name), low_memory=False)
    coord_list = []
    for row in geo.values:
        coord_list.append((row[1], row[2]))
    num_nodes = len(coord_list)

    graph = np.zeros((num_nodes, num_nodes))  # (134, 134)
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            dist = math.sqrt((coord_list[i][0] - coord_list[j][0]) ** 2 + (coord_list[i][1] - coord_list[j][1]) ** 2)
            graph[i][j] = dist
            graph[j][i] = dist
    if args.binary:
        distances = graph.flatten()  # (134*134, )
        dist_std = distances.std()
        graph = np.exp(-np.square(graph / dist_std))
        graph[graph < args.weight_adj_epsilon] = 0
        graph[graph >= args.weight_adj_epsilon] = 1
        np.save(args.data_path + "/geo_graph_no_weight.npy", graph)
    else:
        np.save(args.data_path + "/geo_graph_weight.npy", graph)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./SDWPF', help='dataset path')
    parser.add_argument('--file_name', type=str, default='sdwpf_baidukddcup2022_turb_location.CSV', help='file name')

    parser.add_argument("--weight_adj_epsilon", type=float, default=0.8, help='epsilon for geo graph')
    parser.add_argument("--binary", type=bool, default=True, help='Whether to set the adjacency matrix as binary')
    args = parser.parse_args()

    get_sdwpf_graph(args)
    print("SDWPF geo graph generation success!")
