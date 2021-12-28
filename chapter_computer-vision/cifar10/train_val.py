#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuislishuai 
@license: Apache Licence 
@file: train_val.py
@time: 2021/12/28
@contact: scuislishuai@gmail.com
@site:  
@software: PyCharm 
"""
import numpy as np
import torch
from torch import nn
from utils import try_all_gpus, get_time_diff
from d2l import torch as d2l
from tensorboardX import SummaryWriter
import time
from sklearn import metrics
from torch.nn import functional as F


def init_network(model: nn.Module, method='xavier', exclude='embedding'):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if len(w.shape) >= 2:
                    if method == 'xavier':
                        nn.init.xavier_normal_(w)
                    elif method == 'kaiming':
                        nn.init.kaiming_normal_(w)
                    else:
                        nn.init.normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, val=0)
            else:
                pass


def train(model: nn.Module, train_iter, valid_iter, test_iter, config):
    start_time = time.time()
    trainer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, config.lr_period, config.lr_decay)
    devices = try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    net = nn.DataParallel(model, device_ids=devices).to(devices[0])
    total_batch = 0
    best_loss = float('inf')
    last_improve = 0
    flag = True
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        net.train()
        pred_all = []
        true_all = []
        for i, (features, labels) in enumerate(train_iter, 0):
            train_loss, train_acc, y_pred, y_true = train_batch(net, features, labels, loss, trainer, devices)
            pred_all = np.append(pred_all, y_pred)
            true_all = np.append(true_all, y_true)
            val_acc, val_loss = evaluate(config, model, valid_iter, devices)
            if total_batch % 10 == 0:
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_diff(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2}, Val Acc: {4:>6.2%},  Time: {5} {6}'
                current_train_acc = metrics.accuracy_score(pred_all, true_all)
                print(msg.format(total_batch, train_loss, current_train_acc, val_loss, val_acc, time_dif, improve))
                writer.add_scalar('loss/train', train_loss, total_batch)
                writer.add_scalar("loss/dev", val_loss, total_batch)
                writer.add_scalar("acc/train", current_train_acc, total_batch)
                writer.add_scalar("acc/dev", val_acc, total_batch)
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
        scheduler.step()
    writer.close()
    # 测试集运行
    test(config, model, test_iter, devices)


def test(config, model: nn.Module, test_iter, devices):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, devices, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_diff(start_time)
    print("Time usage:", time_dif)


def train_batch(net, X, y, loss, trainer, devices):
    """Train for a minibatch with mutiple GPUs (defined in Chapter 13).

    Defined in :numref:`sec_image_augmentation`"""
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc = metrics.accuracy_score(y.data.cpu().numpy(), d2l.argmax(pred, axis=1).data.cpu().numpy())
    return train_loss_sum, train_acc, d2l.argmax(pred, axis=1).data.cpu().numpy(), y.data.cpu().numpy()


def evaluate(config, model, data_iter, devices, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            texts = texts.to(devices[0])
            labels = labels.to(devices[0])
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)
