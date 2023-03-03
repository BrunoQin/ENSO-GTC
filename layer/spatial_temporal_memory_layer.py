import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Sequential, ModuleList, Conv2d, BatchNorm2d, ReLU, AvgPool3d, Dropout, BatchNorm1d, Conv1d, Linear


class STMLayer(nn.Module):
    def __init__(self, in_seqlen, in_channels, dropout):
        super(STMLayer, self).__init__()
        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.in_seqlen = in_seqlen
        self.in_channels = in_channels
        self.dropout = dropout

        self.tcnn = self.make_tcnn()
        self.scnn = self.make_scnn()

    def make_tcnn(self):
        return Sequential(
            Conv1d(self.in_seqlen, self.in_seqlen, 1, stride=1),
            BatchNorm1d(self.in_seqlen),
            ReLU(inplace=True),
            Conv1d(self.in_seqlen, self.in_seqlen, 3, stride=2),
            BatchNorm1d(self.in_seqlen),
            ReLU(inplace=True),
            Conv1d(self.in_seqlen, self.in_seqlen, 3, stride=2),
            BatchNorm1d(self.in_seqlen),
            ReLU(inplace=True),
            Conv1d(self.in_seqlen, self.in_seqlen, 3, stride=2),
            BatchNorm1d(self.in_seqlen),
            ReLU(inplace=True)
        )

    def make_scnn(self):
        return Sequential(
            Conv2d(self.in_channels, self.in_channels, 1, stride=1),
            BatchNorm2d(self.in_channels),
            ReLU(inplace=True),
            Conv2d(self.in_channels, self.in_channels, 3, stride=2),
            BatchNorm2d(self.in_channels),
            ReLU(inplace=True),
            Conv2d(self.in_channels, self.in_channels, 3, stride=2),
            BatchNorm2d(self.in_channels),
            ReLU(inplace=True),
            Conv2d(self.in_channels, self.in_channels, 3, stride=2),
            BatchNorm2d(self.in_channels),
            ReLU(inplace=True)
        )

    def forward(self, input):
        _, s, n, h, w = input.size()
        t = input.view(_, s, n * h * w)
        t_out = self.tcnn.to(input.device)(t)
        t_out = t_out.flatten(start_dim=1)

        s = input.view(_, s * n, h, w)
        s_out = self.scnn.to(input.device)(s)
        s_out = s_out.flatten(start_dim=1)

        out = torch.cat([t_out, s_out], dim=1)
        return out


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    x = torch.rand(4, 3, 10, 30, 30).to(device)
    model = STMNN(3, 30, 400, 0.2)
    y = model(x)
    print(y.shape) # torch.Size([1, 30, 114, 5, 5])
