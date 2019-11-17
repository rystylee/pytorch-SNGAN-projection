import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import ConditionalBatchNorm2d, CategoricalConditionalBatchNorm


def upsample(x, scale_factor=2):
    h, w = x.size()[2:]
    return F.interpolate(x, size=(h * scale_factor, w * scale_factor), mode='bilinear', align_corners=True)


def upsample_conv(x, conv, scale_factor=2):
    return conv(upsample(x, scale_factor))


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, hidden_ch=None, ksize=3, pad=1,
                 activation=F.relu, upsampling=False, n_classes=0):
        super(ResBlock, self).__init__()

        self.activation = activation
        self.upsampling = upsampling
        self.learnable_sc = (in_ch != out_ch) or upsampling
        hidden_ch = out_ch if hidden_ch is None else hidden_ch

        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=ksize, padding=pad)
        self.conv2 = nn.Conv2d(hidden_ch, out_ch, kernel_size=ksize, padding=pad)
        if n_classes > 0:
            self.bn1 = CategoricalConditionalBatchNorm(in_ch, n_classes)
            self.bn2 = CategoricalConditionalBatchNorm(out_ch, n_classes)
        else:
            self.bn1 = nn.BatchNorm2d(in_ch)
            self.bn2 = nn.BatchNorm2d(out_ch)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        self._initialize()

    def _initialize(self):
        nn.init.xavier_uniform_(self.conv1.weight.data, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, gain=math.sqrt(2))
        if self.learnable_sc:
            nn.init.xavier_uniform_(self.c_sc.weight.data, gain=1.)

    def forward(self, x, y=None):
        return self.residual(x, y) + self.shortcut(x)

    def residual(self, x, y=None):
        h = x
        h = self.bn1(h, y) if y is not None else self.bn1(h)
        h = self.activation(h)
        h = upsample_conv(h, self.conv1) if self.upsampling else self.conv1(h)
        h = self.bn2(h, y) if y is not None else self.bn2(h)
        h = self.activation(h)
        return self.conv2(h)

    def shortcut(self, x):
        if self.learnable_sc:
            x = upsample_conv(x, self.c_sc, scale_factor=2) if self.upsampling else self.c_sc(x)
            return x
        else:
            return x


class ResNetGenerator(nn.Module):
    def __init__(self, ch=64, dim_z=128, bottom_width=4,
                 activation=F.relu, n_classes=0, distribution='normal'):
        super(ResNetGenerator, self).__init__()

        self.bottom_width = bottom_width
        self.activation = activation

        self.l1 = nn.Linear(dim_z, (bottom_width ** 2) * ch * 16)
        self.block2 = ResBlock(ch * 16, ch * 16, activation=activation, upsampling=True, n_classes=n_classes)
        self.block3 = ResBlock(ch * 16, ch * 8,  activation=activation, upsampling=True, n_classes=n_classes)
        self.block4 = ResBlock(ch * 8,  ch * 8,  activation=activation, upsampling=True, n_classes=n_classes)
        self.block5 = ResBlock(ch * 8,  ch * 4,  activation=activation, upsampling=True, n_classes=n_classes)
        self.block6 = ResBlock(ch * 4,  ch * 2,  activation=activation, upsampling=True, n_classes=n_classes)
        self.block7 = ResBlock(ch * 2,  ch,      activation=activation, upsampling=True, n_classes=n_classes)
        self.bn8 = nn.BatchNorm2d(ch)  # or ConditionalBatchNorm2d?
        self.conv8 = nn.Conv2d(ch, out_channels=3, kernel_size=3, stride=1, padding=1)

        self._initialize()

    def _initialize(self):
        nn.init.xavier_uniform_(self.l1.weight.data)
        nn.init.xavier_uniform_(self.conv8.weight.data)

    def forward(self, x, y):
        h = x
        h = self.l1(h)
        h = h.view(x.size(0), -1, self.bottom_width, self.bottom_width)
        h = self.block2(h, y)
        h = self.block3(h, y)
        h = self.block4(h, y)
        h = self.block5(h, y)
        h = self.block6(h, y)
        h = self.block7(h, y)
        h = self.bn8(h)
        h = self.activation(h)
        h = self.conv8(h)
        # h = F.tanh(h)
        h = torch.tanh(h)
        return h
