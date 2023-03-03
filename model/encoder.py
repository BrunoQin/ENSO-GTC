import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Sequential, ModuleList, Conv3d, MaxPool3d, BatchNorm3d, ReLU


class _DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, bn_size, dropout):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', BatchNorm3d(in_channels))
        self.add_module('relu1', ReLU(inplace=True))
        self.add_module('conv2', Conv3d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        # self.add_module('conv1', Conv3d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, bias=False))
        # self.add_module('norm2', BatchNorm3d(bn_size * growth_rate))
        # self.add_module('relu2', ReLU(inplace=True))
        # self.add_module('conv2', Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate, dropout):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            self.add_module('denselayer%d' % (i+1), _DenseLayer(in_channels + growth_rate * i, growth_rate, bn_size, dropout))


class _DownTransition(nn.Sequential):
    def __init__(self, in_channels, out_channels, pool_size, dropout):
        super(_DownTransition, self).__init__()
        self.add_module('norm', BatchNorm3d(in_channels))
        self.add_module('relu', ReLU(inplace=True))
        self.add_module('conv', Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', MaxPool3d(kernel_size=(1, pool_size, pool_size), stride=(1, pool_size, pool_size)))


class Encoder(nn.Module):
    def __init__(self, growth_rate=12, bn_size=4, theta=0.5, dense_config=(2, 4), pool_size=(2, 3), dropout=0.2):
        super(Encoder, self).__init__()
        self.growth_rate = growth_rate
        self.bn_size = bn_size
        self.theta = theta
        self.dropout = dropout

        num_init_feature = 2 * self.growth_rate
        self.features = Sequential(Conv3d(1, num_init_feature, kernel_size=3, stride=1, padding=1, bias=False))

        num_feature = num_init_feature
        for i, num_layers in enumerate(dense_config):
            self.features.add_module('denseblock%d' % (i + 1), _DenseBlock(num_layers, num_feature, self.bn_size, self.growth_rate, self.dropout))
            num_feature = num_feature + growth_rate * num_layers
            if i < 2:
                self.features.add_module('transition%d' % (i + 1), _DownTransition(num_feature, int(num_feature * self.theta), pool_size[i], self.dropout))
                num_feature = int(num_feature * self.theta)

    def forward(self, input):
        batch_size, s, n, h, w = input.size()
        x = input.view(batch_size, 1, s * n, h, w)
        x = self.features(x)
        output = x.permute(0, 2, 1, 3, 4)
        return output

if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    x = torch.rand(1, 3, 10, 30, 30).to(device)
    model = Encoder().to(device)
    y = model(x)
    print(y.shape) # torch.Size([1, 30, 114, 5, 5])
