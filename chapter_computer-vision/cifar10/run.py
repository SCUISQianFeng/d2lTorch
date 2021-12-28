#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuislishuai 
@license: Apache Licence 
@file: run.py 
@time: 2021/12/28
@contact: scuislishuai@gmail.com
@site:  
@software: PyCharm 
"""

import argparse
import os
from utils import get_time_diff, set_seed, build_train_dataLoader, build_val_dataLoader, build_test_dataLoader, \
    build_train_val_test
from train_val import train, init_network
import time

from importlib import import_module

parser = argparse.ArgumentParser(description='Cifar10 train')
# parser.add_argument('--model', type=str, required=True, help='choose a model: resnet18,....')
parser.add_argument('--model', type=str, default='resnet18', help='choose a model: resnet18,....')
args = parser.parse_args()

if __name__ == '__main__':
    model_name = args.model
    x = import_module(name='models.' + model_name)
    config = x.Config()
    data_path = r'E:\DataSet\DataSet\ClassicalDatasets\cifar\cifar-10'
    if os.path.exists(os.path.join(data_path, 'train_valid_test', 'train_valid')) == False:
        build_train_val_test(data_path, config.valid_ratio)
    # 保证每次结果一样
    set_seed()
    start_time = time.time()
    print("Loading data...")
    train_loader = build_train_dataLoader(config.batch_size)
    val_loader = build_val_dataLoader(config.batch_size)
    test_loader = build_test_dataLoader(config.batch_size)
    time_dif = get_time_diff(start_time)
    print("Time usage:", time_dif)

    model = x.Model(config).to(config.device)
    init_network(model)
    print(model.parameters)
    train(model, train_loader, val_loader, test_loader, config)
