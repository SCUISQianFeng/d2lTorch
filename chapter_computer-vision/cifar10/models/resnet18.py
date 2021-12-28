#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuislishuai 
@license: Apache Licence 
@file: resnet18.py 
@time: 2021/12/28
@contact: scuislishuai@gmail.com
@site:  
@software: PyCharm 
"""

import os
import torch
from torch import nn as nn
from torch.nn import functional as F


class Config(object):
    def __init__(self, dataset=r'E:\DataSet\DataSet\ClassicalDatasets\cifar\cifar-10'):
        self.model_name = 'resnet18'
        self.train_path = os.path.join(dataset, 'train')
        self.test_path = os.path.join(dataset, 'test')
        self.result_path = 'result'
        self.log_path = os.path.join(self.result_path, 'log', self.model_name)
        self.save_path = os.path.join(self.result_path, 'saved_dict', self.model_name + '.ckpt')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.require_improvement = 3000  # 若超过3000次效果没有提升，则提前结束训练

        self.num_classes = 10
        self.in_channels = 3
        self.out_channels = 64
        self.kernel_size = 3
        self.padding = 1
        self.stride = 1
        self.demo = False
        self.batch_size = 32 if self.demo else 128
        self.lr = 2e-4
        self.lr_period = 4
        self.lr_decay = 0.9
        self.num_epochs = 20
        self.weight_decay = 5e-4  # 权重衰减
        self.valid_ratio = 0.1


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=3, padding=1,
                               stride=strides)
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_features=num_channels)
        self.bn2 = nn.BatchNorm2d(num_features=num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X  # self.conv3 is None， 卷积的结果直接和原始数据相加做条连接
        return F.relu(Y)


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    block = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            block.append(Residual(input_channels=in_channels, num_channels=out_channels, use_1x1conv=True, strides=2))
        else:
            block.append(Residual(input_channels=out_channels, num_channels=out_channels))
    return nn.Sequential(*block)


class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(config.in_channels, config.out_channels, kernel_size=config.kernel_size,
                                           stride=config.kernel_size, padding=config.padding),
                                 nn.BatchNorm2d(num_features=64),
                                 nn.ReLU())
        self.net.add_module('resnet_block1',
                            resnet_block(in_channels=64, out_channels=64, num_residuals=2, first_block=True))
        self.net.add_module('resnet_block2', resnet_block(in_channels=64, out_channels=128, num_residuals=2))
        self.net.add_module('resnet_block3', resnet_block(in_channels=128, out_channels=256, num_residuals=2))
        self.net.add_module('resnet_block4', resnet_block(in_channels=256, out_channels=512, num_residuals=2))
        self.net.add_module('global_avg_pool', nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.net.add_module('fc', nn.Sequential(nn.Flatten(), nn.Linear(512, config.num_classes)))

    def forward(self, X):
        return self.net(X)
