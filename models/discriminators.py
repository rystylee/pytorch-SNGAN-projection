import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def downsample(x):
    return F.avg_pool2d(x, 2)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, hidden_ch=None, ksize=3, pad=1,
                 activation=F.relu, downsampling=False):
        super(ResBlock, self).__init__()

        self.activation = activation
        self.downsampling = downsampling
        self.learnable_sc = (in_ch != out_ch) or downsampling
        hidden_ch = in_ch if hidden_ch is None else hidden_ch

        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_ch, hidden_ch, kernel_size=ksize, stride=1, padding=pad))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(hidden_ch, out_ch, kernel_size=ksize, stride=1, padding=pad))
        if self.learnable_sc:
            self.c_sc = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0))

        self._initialize()

    def _initialize(self):
        nn.init.xavier_uniform_(self.conv1.weight.data, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, gain=math.sqrt(2))
        if self.learnable_sc:
            nn.init.xavier_uniform_(self.c_sc.weight.data, gain=1.)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.conv1(h)
        h = self.activation(h)
        h = self.conv2(h)
        if self.downsampling:
            h = downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsampling:
                x = downsample(x)
            return x
        else:
            return x


class OptimizeBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, pad=1, activation=F.relu):
        super(OptimizeBlock, self).__init__()

        self.activation = activation

        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=ksize, stride=1, padding=pad))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, kernel_size=ksize, stride=1, padding=pad))
        self.c_sc = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0))

        self._initialize()

    def _initialize(self):
        nn.init.xavier_uniform_(self.conv1.weight.data, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.c_sc.weight.data, gain=1.)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

    def residual(self, x):
        h = x
        h = self.conv1(h)
        h = self.activation(h)
        h = self.conv2(h)
        h = downsample(h)
        return h

    def shortcut(self, x):
        h = x
        h = downsample(h)
        h = self.c_sc(h)
        return h


class SNResNetProjectionDiscriminator(nn.Module):
    def __init__(self, ch=64, n_classes=0, activation=F.relu):
        super(SNResNetProjectionDiscriminator, self).__init__()

        self.activation = activation
        self.n_classes = n_classes

        self.block1 = OptimizeBlock(3, ch)
        self.block2 = ResBlock(ch * 1,  ch * 2,  activation=activation, downsampling=True)
        self.block3 = ResBlock(ch * 2,  ch * 4,  activation=activation, downsampling=True)
        self.block4 = ResBlock(ch * 4,  ch * 8,  activation=activation, downsampling=True)
        self.block5 = ResBlock(ch * 8,  ch * 8,  activation=activation, downsampling=True)
        self.block6 = ResBlock(ch * 8,  ch * 16, activation=activation, downsampling=True)
        self.block7 = ResBlock(ch * 16, ch * 16, activation=activation, downsampling=False)
        self.l8 = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        if n_classes > 0:
            self.l_y = nn.utils.spectral_norm(nn.Embedding(n_classes, ch * 16))

        self._initialize()

    def _initialize(self):
        nn.init.xavier_uniform_(self.l8.weight.data)
        if self.n_classes:
            nn.init.xavier_uniform_(self.l_y.weight.data)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.block7(h)
        h = self.activation(h)
        h = torch.sum(h, dim=(2, 3))
        out = self.l8(h)
        if y is not None:
            w_y = self.l_y(y)
            out += torch.sum(w_y * h, dim=1, keepdim=True)
        return out
