import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from fastdtw import fastdtw


def get_dtw_graph(args):
    csv_data = pd.read_csv(args.data_path, low_memory=False)
    data = csv_data.iloc[:, -1]
    day_len = 24 * 6  # Number of sampling times a day
    if args.nan2zero:
        data = data.replace(to_replace=np.nan, value=0, inplace=False)
        data = np.maximum(data, 0)
    data = data.to_numpy().reshape([args.n_turbine, -1])

    num_samples = data.shape[1]
    num_train = int(num_samples * 0.6)

    df = data[:, 0:num_train]  # (134, t)  Patv
    data_mean = np.mean(
        [df[:, day_len * i: day_len * (i + 1)]
         for i in range(df.shape[1] // day_len)], axis=0)  # (134, 144)

    dtw_distance = np.zeros((args.n_turbine, args.n_turbine))
    for i in tqdm(range(args.n_turbine)):
        for j in range(i, args.n_turbine):
            dtw_distance[i][j], _ = fastdtw(data_mean[i, :], data_mean[j, :], radius=6)
    for i in range(args.n_turbine):
        for j in range(i):
            dtw_distance[i][j] = dtw_distance[j][i]
    np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)), args.output_dir + "/dtw_graph.npy"), dtw_distance)


def get_dtw_graph_topk(args):
    dtw_distance = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), args.output_dir + "/dtw_graph.npy"))
    ind = np.argsort(dtw_distance)[:, 0:args.dtw_topk]  # (n, k)
    graph = np.zeros((args.n_turbine, args.n_turbine))
    for i in range(ind.shape[0]):
        for j in range(ind.shape[1]):
            graph[i][ind[i][j]] = 1
            graph[ind[i][j]][i] = 1
    np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         args.output_dir + "/dtw_graph_top{}.npy".format(args.dtw_topk)), graph)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./SDWPF/wtbdata_245days.csv', help='dataset path')
    parser.add_argument('--output_dir', type=str, default='./SDWPF', help='output dir')
    parser.add_argument('--n_turbine', type=int, default=134, help='number of turbine')
    parser.add_argument('--nan2zero', type=bool, default=True, help='Whether to set NAN to 0')
    parser.add_argument('--dtw_topk', type=int, default=5, help='the top-k of similarity as the edges of the graph')
    args = parser.parse_args()

    get_dtw_graph(args)
    get_dtw_graph_topk(args)
