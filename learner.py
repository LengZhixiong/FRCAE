import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.distributions import Beta
from numpy import linalg as LA
import numpy as np

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze())
        max_out = self.fc(self.max_pool(x).squeeze())

        return torch.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)

class Conv_Standard(nn.Module):
    def __init__(self, args, x_dim, hid_dim, z_dim, final_layer_size):
        super(Conv_Standard, self).__init__()
        self.args = args
        if self.args.cae == 1:
            self.net = nn.Sequential(self.transpose_conv_block(x_dim, hid_dim), self.conv_block(hid_dim, hid_dim), self.conv_block(hid_dim, hid_dim),
                                     self.conv_block(hid_dim, hid_dim), self.conv_block(hid_dim, z_dim))
        else:
            self.net = nn.Sequential(self.conv_block(x_dim, hid_dim), self.conv_block(hid_dim, hid_dim),
                                     self.conv_block(hid_dim, hid_dim), self.conv_block(hid_dim, z_dim))            
        self.dist = Beta(torch.FloatTensor([2]), torch.FloatTensor([2]))
        self.hid_dim = hid_dim

        self.logits = nn.Linear(final_layer_size, self.args.num_classes)

        self.ca = ChannelAttention(hid_dim)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def transpose_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def functional_conv_transpose_block(self, x, weights, biases,
                              bn_weights, bn_biases, dropout=0, is_training=False):
        x = F.conv2d(x, weights, biases, padding=1)
        x = F.batch_norm(x, running_mean=None, running_var=None, weight=bn_weights, bias=bn_biases, training=True)
        return x

    def functional_conv_block(self, x, weights, biases,
                              bn_weights, bn_biases, dropout=0, is_training=False):

        x = F.conv2d(x, weights, biases, padding=1)
        x = F.batch_norm(x, running_mean=None, running_var=None, weight=bn_weights, bias=bn_biases, training=True)
        x = F.relu(x)
        x, indices = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        x = F.dropout(x, p=dropout, training=is_training)
        return x, indices

    def forward(self, x):
        x = self.net(x)
        print(x.shape)
        x = x.view(x.size(0), -1)

        return self.logits(x)

    def forward_anil(self, x, inin_weights, weights, dropout=0, is_training=False):
        if self.args.cae == 1:
            x = self.functional_conv_transpose_block(x, inin_weights[f'net.0.0.weight'], inin_weights[f'net.0.0.bias'],
                                               inin_weights.get(f'net.0.1.weight'), inin_weights.get(f'net.0.1.bias'), dropout, is_training)
            x = torch.sigmoid(x)
            x = x * self.ca(x)

            for block in range(1, 5, 1):
                x, indices = self.functional_conv_block(x, inin_weights[f'net.{block}.0.weight'], inin_weights[f'net.{block}.0.bias'],
                                               inin_weights.get(f'net.{block}.1.weight'), inin_weights.get(f'net.{block}.1.bias'), dropout, is_training)
        else:
            for block in range(0, 4, 1):
                x, indices = self.functional_conv_block(x, inin_weights[f'net.{block}.0.weight'], inin_weights[f'net.{block}.0.bias'],
                                               inin_weights.get(f'net.{block}.1.weight'), inin_weights.get(f'net.{block}.1.bias'), dropout, is_training)            

        x = x.view(x.size(0), -1)

        x = F.linear(x, weights['weight'], weights['bias'])

        return x        

    def forward_maml(self, x, weights, dropout=0, is_training=False):
        if self.args.cae == 1:
            x = self.functional_conv_transpose_block(x, weights[f'net.0.0.weight'], weights[f'net.0.0.bias'],
                                               weights.get(f'net.0.1.weight'), weights.get(f'net.0.1.bias'), dropout, is_training)
            x = torch.sigmoid(x)
            x = x * self.ca(x)

            for block in range(1, 5, 1):
                x, indices = self.functional_conv_block(x, weights[f'net.{block}.0.weight'], weights[f'net.{block}.0.bias'],
                                               weights.get(f'net.{block}.1.weight'), weights.get(f'net.{block}.1.bias'), dropout, is_training)
        else:
            for block in range(0, 4, 1):
                x, indices = self.functional_conv_block(x, weights[f'net.{block}.0.weight'], weights[f'net.{block}.0.bias'],
                                               weights.get(f'net.{block}.1.weight'), weights.get(f'net.{block}.1.bias'), dropout, is_training)

        x = x.view(x.size(0), -1)

        x = F.linear(x, weights['logits.weight'], weights['logits.bias'])

        return x