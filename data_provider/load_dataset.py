import os
import numpy as np
import pandas as pd
import time
from tqdm import tqdm


def load_SDWPFDataset(data_path="./data/SDWPF/wtbdata_245days.csv", nan2zero=True):
    if os.path.exists("./data/SDWPF/wtbdata_245days.npz"):
        dict_data = np.load("./data/SDWPF/wtbdata_245days.npz")
        return dict_data['data'], dict_data['times']

    n_turbine = 134
    csv_data = pd.read_csv(data_path)
    data = csv_data.iloc[:, 3:]
    if nan2zero:
        data = data.replace(to_replace=np.nan, value=0, inplace=False)
        data = np.maximum(data, 0)
    data = data.to_numpy().reshape([n_turbine, -1, 10])
    data = data.transpose([1, 0, 2])

    # process time
    time_info = csv_data[['Day', 'Tmstamp']]

    def process_str_time(row):
        str_time = "{} {}".format(row['Day'], row['Tmstamp'])
        data_sj = time.strptime(str_time, "%j %H:%M")
        return (
            data_sj.tm_mon,
            data_sj.tm_yday,
            data_sj.tm_hour,
            data_sj.tm_min,
            data_sj.tm_sec
        )

    tqdm.pandas(desc='load data from SDWPF Dir')
    times = time_info.progress_apply(process_str_time, axis=1).to_numpy()
    times = np.stack(times).reshape([n_turbine, -1, 5]).transpose([1, 0, 2])
    np.savez("./data/SDWPF/wtbdata_245days", data=data, times=times)
    return data, times


def load_MYDataset(data_path="./data/MyData/wpData.csv", nan2zero=True):
    if os.path.exists("./data/MyData/wpData.npz"):
        dict_data = np.load("./data/MyData/wpData.npz")
        return dict_data['data'], dict_data['times']

    n_turbine = 33
    csv_data = pd.read_csv(data_path)

    data = csv_data.iloc[:, 2:]
    if nan2zero:
        data = data.replace(to_replace=np.nan, value=0, inplace=False)
        data = np.maximum(data, 0)

    data = data.to_numpy().reshape([n_turbine, -1, 7])
    data = data.transpose([1, 0, 2])

    # process time
    def process_str_time(str_time):
        data_sj = time.strptime(str_time, "%d/%m/%Y %H:%M:%S")
        return (
            data_sj.tm_mon,
            data_sj.tm_yday,
            data_sj.tm_hour,
            data_sj.tm_min,
            data_sj.tm_sec
        )

    tqdm.pandas(desc='load data from MyData Dir')
    times = csv_data['TmStamp'].progress_apply(process_str_time).to_numpy()
    times = np.stack(times).reshape([n_turbine, -1, 5]).transpose([1, 0, 2])
    np.savez("./data/MyData/wpData", data=data, times=times)
    return data, times


def load_wp_dataset(dataset, nan2zero=True):
    if dataset == 'SDWPF':
        data, times = load_SDWPFDataset(nan2zero=nan2zero)
    elif dataset == 'MyData':
        data, times = load_MYDataset(nan2zero=nan2zero)
    else:
        raise ValueError("dataset must selected in ['SDWPF', 'MyData']")
    return data, times


if __name__ == '__main__':
   a = np.array([[-1, 2, 3], [-1, -1, 0]])
   a = np.maximum(a, 0)
   tmp = 1
