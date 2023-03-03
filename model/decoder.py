import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Sequential, ConvTranspose3d, ReLU, Upsample, BatchNorm3d, Sigmoid


class _ReduceLayer(nn.Sequential):
    def __init__(self, in_channels, reduce_rate, bn_size, dropout):
        super(_ReduceLayer, self).__init__()
        self.add_module('norm1', BatchNorm3d(in_channels))
        self.add_module('relu1', ReLU(inplace=True))
        self.add_module('conv2', ConvTranspose3d(in_channels, in_channels - reduce_rate, kernel_size=1, stride=1, bias=False))
        # self.add_module('conv1', ConvTranspose3d(in_channels, bn_size * reduce_rate, kernel_size=1, stride=1, bias=False))
        # self.add_module('norm2', BatchNorm3d(bn_size * reduce_rate))
        # self.add_module('relu2', ReLU(inplace=True))
        # self.add_module('conv2', ConvTranspose3d(bn_size * reduce_rate, in_channels - reduce_rate, kernel_size=1, stride=1, bias=False))


class _ReduceBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, bn_size, reduce_rate, dropout):
        super(_ReduceBlock, self).__init__()
        for i in range(num_layers):
            self.add_module('denselayer%d' % (i+1), _ReduceLayer(in_channels - reduce_rate * i, reduce_rate, bn_size, dropout))


class _UpTransition(nn.Sequential):
    def __init__(self, in_channels, out_channels, pool_size, dropout):
        super(_UpTransition, self).__init__()
        self.add_module('norm', BatchNorm3d(in_channels))
        self.add_module('relu', ReLU(inplace=True))
        self.add_module('conv', ConvTranspose3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', Upsample(scale_factor=(1, pool_size, pool_size), mode='trilinear', align_corners=True))


class Decoder(nn.Module):
    def __init__(self, init_feature, reduce_rate=3, bn_size=4, theta=0.5, reduce_config=(2, 4), pool_size=(3, 2), dropout=0.2):
        super(Decoder, self).__init__()
        self.reduce_rate = reduce_rate
        self.bn_size = bn_size
        self.theta = theta
        self.init_feature = init_feature
        self.dropout = dropout
        self.features = Sequential(ConvTranspose3d(self.init_feature, self.init_feature, kernel_size=3, stride=1, padding=1, bias=False))

        num_feature = self.init_feature
        for i, num_layers in enumerate(reduce_config):
            self.features.add_module('denseblock%d' % (i + 1), _ReduceBlock(num_layers, num_feature, self.bn_size, self.reduce_rate, self.dropout))
            num_feature = num_feature - reduce_rate * num_layers
            if i < 2:
                self.features.add_module('transition%d' % (i + 1), _UpTransition(num_feature, int(num_feature * self.theta), pool_size[i], self.dropout))
                num_feature = int(num_feature * self.theta)
        self.features.add_module('conv_transpose1', ConvTranspose3d(num_feature, 16, kernel_size=1, stride=1, padding=0, bias=False))
        self.features.add_module('conv_transpose2', ConvTranspose3d(16, 1, kernel_size=1, stride=1, padding=0, bias=False))
        self.features.add_module('sigmoid', Sigmoid())

    def forward(self, input):
        x = input.permute(0, 2, 1, 3, 4)
        x = self.features(x)
        output = x.permute(0, 2, 1, 3, 4)
        return output

if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    x = torch.rand(1, 30, 114, 5, 5).to(device)
    model = Decoder(x.size(2)).to(device)
    y = model(x)
    print(y.shape)
