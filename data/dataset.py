import torch
from torch.utils.data import Dataset
import numpy as np

from tool.util import load_input_and_output
from tool.configs import args

class TrainingDataset(Dataset):
    def __init__(self, input, output):
        self.input = torch.Tensor(input)
        self.output = torch.Tensor(output)

    def __getitem__(self, i):
        return self.input[i], self.output[i]

    def __len__(self):
        return self.input.shape[0]

    def data_preproccess(self, data):
        pass


class ForecastingDataset(Dataset):
    def __init__(self, input):
        self.input = torch.Tensor(input)

    def __getitem__(self, i):
        return self.input[i]

    def __len__(self):
        return self.input.shape[0]

    def data_preproccess(self, data):
        pass


def load_train_data():
    input, output = load_input_and_output(filename=args['train_npz_file'])
    return TrainingDataset(input, output)


def load_forecast_data():
    input, _ = load_input_and_output(filename=args['forecast_npz_file'])
    return ForecastingDataset(input)
