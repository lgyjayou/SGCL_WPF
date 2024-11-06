import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from data_provider.load_dataset import load_wp_dataset
from data_provider.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler


def normalize_dataset(data, normalizer):
    shape, dim = data.shape, data.shape[-1]
    data = data.reshape([-1, dim])
    if normalizer == 'max01':
        minimum = data.min(axis=0)
        maximum = data.max(axis=0)
        scaler = MinMax01Scaler(minimum, maximum)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        minimum = data.min(axis=0)
        maximum = data.max(axis=0)
        scaler = MinMax11Scaler(minimum, maximum)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        scaler = StandardScaler(mean, std)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        print('Does not normalize the dataset')
    else:
        raise ValueError
    data = data.reshape(shape)
    data = scaler.transform(data)
    return data, scaler


def add_window(x_data, y_label, history, horizon, single=False):
    # x_data.shape = [size, n_turbine, dim]
    # y_label.shape = [size, n_turbine]
    length = len(x_data)
    end_idx = length - (horizon + history) + 1
    x, y = [], []
    idx = 0
    if single:
        while idx < end_idx:
            x.append(x_data[idx:idx + history])
            y.append(y_label[idx + history + horizon - 1:idx + history + horizon])
            idx += 1
    else:
        while idx < end_idx:
            x.append(x_data[idx:idx + history])
            y.append(y_label[idx + history:idx + history + horizon])
            idx += 1

    x = np.array(x)  # (length - (horizon + history), history, n_turbine, dim)
    y = np.array(y)  # (length - (horizon + history), horizon, n_turbine)
    return x, y


def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len * test_ratio):]
    val_data = data[-int(data_len * (test_ratio + val_ratio)):-int(data_len * test_ratio)]
    train_data = data[:-int(data_len * (test_ratio + val_ratio))]
    return train_data, val_data, test_data


def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y)
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader


def get_dataloader(dataset, history, horizon, batch_size,
                   val_ratio=0.2, test_ratio=0.2, normalizer='std', single=False):
    # load raw dataset
    data, times = load_wp_dataset(dataset)
    data_x, data_y = data, data[..., -1]

    # normalize data
    data_x, scaler = normalize_dataset(data_x, normalizer)

    # concat with times
    month_of_year = times[..., 0] - 1  # 0 ~ 11
    day_of_year = times[..., 1] - 1    # 0 ~ 365
    time_of_day = (times[..., 2] * 3600 + times[..., 3] * 60 + times[..., 4]) / 86400
    data_x = np.concatenate([
            time_of_day.reshape([*time_of_day.shape, 1]),
            day_of_year.reshape([*day_of_year.shape, 1]),
            month_of_year.reshape([*month_of_year.shape, 1]),
            data_x
        ],
        axis=-1
    )

    # add window
    x_data, y_label = add_window(data_x, data_y, history, horizon, single)

    # split dataset by ratio
    random_idx = np.random.permutation(x_data.shape[0])
    x_data = x_data[random_idx, ...]
    y_label = y_label[random_idx, ...]
    x_train, x_val, x_test = split_data_by_ratio(x_data, val_ratio, test_ratio)
    y_train, y_val, y_test = split_data_by_ratio(y_label, val_ratio, test_ratio)

    print('Train: ', x_train.shape, y_train.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)

    # get dataloader
    train_dataloader = data_loader(x_train, y_train, batch_size, shuffle=True, drop_last=True)
    val_dataloader = data_loader(x_val, y_val, batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, batch_size, shuffle=False, drop_last=True)

    return train_dataloader, val_dataloader, test_dataloader, scaler


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch dataloader')
    parser.add_argument('--dataset', default='SDWPF', type=str)
    parser.add_argument('--column_wise', default=False, type=bool)
    parser.add_argument('--val_ratio', default=0.2, type=float)
    parser.add_argument('--test_ratio', default=0.2, type=float)
    parser.add_argument('--lag', default=12, type=int)
    parser.add_argument('--horizon', default=12, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    args = parser.parse_args()
    train_dataloader, val_dataloader, test_dataloader, scaler = get_dataloader(args.dataset, args.lag, args.horizon, args.batch_size, normalizer='std')

    length = len(train_dataloader)
    tmp = 1
