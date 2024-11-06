import numpy as np
import torch
import abc


class AbstractScaler(metaclass=abc.ABCMeta):
    """An abstract class to standard the input"""
    @abc.abstractmethod
    def transform(self, x_data):
        raise NotImplementedError

    @abc.abstractmethod
    def inverse_transform(self, y_label):
        raise NotImplementedError


class NScaler(AbstractScaler):
    def transform(self, x_data):
        return x_data

    def inverse_transform(self, y_label):
        return y_label


class StandardScaler(AbstractScaler):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, x_data):
        return (x_data - self.mean) / self.std

    def inverse_transform(self, y_label):
        if type(y_label) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(y_label.device).type(y_label.dtype)
            self.mean = torch.from_numpy(self.mean).to(y_label.device).type(y_label.dtype)
        if len(self.mean.shape) == 1:
            return (y_label * self.std[-1]) + self.mean[-1]
        return (y_label * self.std) + self.mean


class MinMax01Scaler(AbstractScaler):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, x_data):
        return (x_data - self.min) / (self.max - self.min)

    def inverse_transform(self, y_label):
        if type(y_label) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(y_label.device).type(y_label.dtype)
            self.max = torch.from_numpy(self.max).to(y_label.device).type(y_label.dtype)
        if len(self.min.shape) == 1:
            return y_label * (self.max[-1] - self.min[-1]) + self.min[-1]
        return y_label * (self.max - self.min) + self.min


class MinMax11Scaler(AbstractScaler):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, x_data):
        return ((x_data - self.min) / (self.max - self.min)) * 2. - 1.

    def inverse_transform(self, y_label):
        if type(y_label) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(y_label.device).type(y_label.dtype)
            self.max = torch.from_numpy(self.max).to(y_label.device).type(y_label.dtype)
        if len(self.min.shape) == 1:
            return ((y_label + 1.) / 2.) * (self.max[-1] - self.min[-1]) + self.min[-1]
        return ((y_label + 1.) / 2.) * (self.max - self.min) + self.min
