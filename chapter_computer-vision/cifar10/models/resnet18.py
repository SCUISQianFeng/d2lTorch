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

class Config(object):
    def __init__(self, dataset=r'E:\DataSet\DataSet\ClassicalDatasets\cifar\cifar-10'):
        self.model_name = 'resnet18'
        self.train_path = os.path.join(dataset, 'train')
        self.test_path = os.path.join(dataset, 'test')
