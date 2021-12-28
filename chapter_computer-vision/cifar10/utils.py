#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuislishuai 
@license: Apache Licence 
@file: utils.py 
@time: 2021/12/28
@contact: scuislishuai@gmail.com
@site:  
@software: PyCharm 
"""

import collections
import math
import os
import shutil
import sys
import torchvision
from torch import nn
import torch
import time
from datetime import timedelta
import numpy as np

sys.path.append('..')

# static parameter
data_path = r'E:\DataSet\DataSet\ClassicalDatasets\cifar\cifar-10'

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists.

    Defined in :numref:`sec_use_gpu`"""
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


def read_csv_label(data_path=r'E:\DataSet\DataSet\ClassicalDatasets\cifar\cifar-10'):
    with open(os.path.join(data_path, 'trainLabels.csv'), 'r') as f:
        # 跳过文件头行
        lines = f.readlines()[1:]
    tokens = [line.rstrip().split(',') for line in lines]
    return dict(((name, label) for name, label in tokens))


def copy_file(filename, target_dir):
    """将文件复制到目标目录"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)


def get_time_diff(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_diff = end_time - start_time
    return timedelta(seconds=int(round(time_diff)))


def split_train_valid(data_path, labels, valid_ration):
    """将验证集从原始的训练集中拆分出来"""
    # 训练数据集中样本最少的类别中的样本数
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # 验证集中每个类别的样本数
    n_valid_per_label = max(1, math.floor(n * valid_ration))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_path, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_path, 'train', train_file)
        copy_file(fname, os.path.join(data_path, 'train_valid_test', 'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copy_file(fname, os.path.join(data_path, 'train_valid_test', 'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copy_file(fname, os.path.join(data_path, 'train_valid_test', 'train', label))
    return n_valid_per_label


def split_test(data_path):
    for test_file in os.listdir(os.path.join(data_path, 'test')):
        copy_file(filename=os.path.join(data_path, 'test', test_file),
                  target_dir=os.path.join(data_path, 'train_valid_test', 'test', 'unknown'))


def build_train_val_test(data_path, valid_ratio):
    labels = read_csv_label()
    split_train_valid(data_path, labels, valid_ratio)
    split_test(data_path)


def build_train_val_compose():
    return torchvision.transforms.Compose([
        # 在高度和宽度上将图像放大到40像素的正方形
        torchvision.transforms.Resize(size=40),
        # 随机裁剪出一个高度和宽度均为40像素的正方形图像，
        # 生成一个面积为原始图像面积0.64到1倍的小正方形，
        # 然后将其缩放为高度和宽度均为32像素的正方形
        torchvision.transforms.RandomResizedCrop(size=32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        # 标准化图像的每个通道
        torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
    ])


def build_test_compose():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
    ])


def build_train_dataLoader(batch_size):
    train_compose = build_train_val_compose()
    train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_path, 'train_valid_test', 'train'), transform=train_compose)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
    return train_data_loader


def build_val_dataLoader(batch_size):
    valid_compose = build_test_compose()
    valid_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_path, 'train_valid_test', 'valid'),
                                                     transform=valid_compose)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size, shuffle=False, drop_last=True)
    return valid_data_loader


def build_test_dataLoader(batch_size):
    test_compose = build_test_compose()
    test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_path, 'train_valid_test', 'test'), transform=test_compose)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False, drop_last=False)
    return test_data_loader


def set_seed(seed=1):
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)
    torch.backends.cudnn.deterministic = True




if __name__ == '__main__':
    data_path = r'E:\DataSet\DataSet\ClassicalDatasets\cifar\cifar-10'
    train_data, label = read_csv_label(data_path=data_path)
    print('pass')
