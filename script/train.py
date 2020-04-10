#!/usr/bin/env python
# coding: utf-8

# # S2Loc Training
#
# Description: We propose to lear a descriptor of point clouds for global localization.
#
# Author: Lukas Bernreiter (lukas.bernreiter@ieee.org)
#

from data_source import DataSource
from visualize import Visualize
from sphere import Sphere
from model import Model
from loss import TripletLoss, ImprovedTripletLoss
from training_set import TrainingSet
from average_meter import AverageMeter
from data_splitter import DataSplitter

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import sys
import time
import math
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from tqdm.auto import tqdm
from scipy import spatial


# ## Load All Data
#
# Load the dataset, project each point cloud on a sphere and derive a function for it.


#ds = DataSource('/home/berlukas/data/spherical/training-set', 1.0)
ds = DataSource('/media/scratch/berlukas/spherical/training', 1.0)
ds.load(10000)

# ## Initialize the model and the training set

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
net = Model().cuda()
restore = False
optimizer = torch.optim.SGD(net.parameters(), lr=5e-3, momentum=0.9)
n_epochs = 50
batch_size = 12
num_workers = 12
descriptor_size = 64
bandwith = 100
net_input_size = 2*bandwith
n_features = 2
criterion = ImprovedTripletLoss(margin=2, alpha=0.5, margin2=0.2)
writer = SummaryWriter()
model_save = 'net_params_new_1.pkl'

summary(net, input_size=[(2, 200, 200), (2, 200, 200), (2, 200, 200)])

train_set = TrainingSet(ds, restore, bandwith)
print("Total size: ", len(train_set))
split = DataSplitter(train_set, restore, shuffle=True)

train_loader, val_loader, test_loader = split.get_split(batch_size=batch_size, num_workers=num_workers)
train_size = split.get_train_size()
val_size = split.get_val_size()
test_size = split.get_test_size()
print("Training size: ", train_size)
print("Validation size: ", val_size)
print("Testing size: ", test_size)

visualize = False
if visualize:
    first_anchor = Sphere(ds.anchors_training[0])
    len(first_anchor.point_cloud)

    viz = Visualize()
    viz.visualizeRawPointCloud(first_anchor, True)
    viz.visualizeSphere(first_anchor, True)

# ## Train model

def adjust_learning_rate_exp(optimizer, epoch_num, lr=5e-3):
    decay_rate = 0.96
    new_lr = lr * math.pow(decay_rate, epoch_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    return new_lr

val_accs = AverageMeter()
loss_ = 0

def accuracy(dista, distb):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    acc = ((pred < 0).sum()).float()/dista.size(0)
    return acc

def train(net, criterion, optimizer, writer, epoch, n_iter, loss_, t0):
    net.train()
    for batch_idx, (data1, data2, data3) in enumerate(train_loader):
        data1, data2, data3 = data1.cuda().float(), data2.cuda().float(), data3.cuda().float()
        embedded_a, embedded_p, embedded_n = net(data1, data2, data3)
        optimizer.zero_grad()

        dista, distb, loss_triplet, loss_total = criterion(embedded_a, embedded_p, embedded_n)
        loss_embedd = embedded_a.norm(2) + embedded_p.norm(2) + embedded_n.norm(2)
        loss = loss_triplet + 0.001 * loss_embedd
        #loss = loss_triplet

        loss.backward()
        optimizer.step()
        loss_ += loss_total.item()

        writer.add_scalar('Train/Loss_Triplet', loss_triplet, n_iter)
        writer.add_scalar('Train/Loss_Embedd', loss_embedd, n_iter)
        writer.add_scalar('Train/Loss', loss, n_iter)
        n_iter += 1

        if batch_idx % 100 == 99:
            t1 = time.time()
            print('[Epoch %d, Batch %4d] loss: %.5f time: %.5f lr: %.3e' %
                (epoch + 1, batch_idx + 1, loss_ / 100, (t1-t0) / 60, lr))
            t0 = t1
            loss_ = 0.0
    return n_iter

def validate(net, criterion, optimizer, writer, epoch, n_iter):
    net.eval()
    with torch.no_grad():
        for batch_idx, (data1, data2, data3) in enumerate(val_loader):
            data1, data2, data3 = data1.cuda().float(), data2.cuda().float(), data3.cuda().float()
            embedded_a, embedded_p, embedded_n = net(data1, data2, data3)
            optimizer.zero_grad()

            dista, distb, loss_triplet, loss_total = criterion(embedded_a, embedded_p, embedded_n)
            loss_embedd = embedded_a.norm(2) + embedded_p.norm(2) + embedded_n.norm(2)
            loss = loss_triplet + 0.001 * loss_embedd


            acc = accuracy(dista, distb)
            val_accs.update(acc, data1.size(0))
            writer.add_scalar('Validation/Loss_Triplet', loss_triplet, n_iter)
            writer.add_scalar('Validation/Loss_Embedd', loss_embedd, n_iter)
            writer.add_scalar('Validation/Loss', loss, n_iter)
            writer.add_scalar('Validation/Accuracy', val_accs.avg, n_iter)
            n_iter += 1
    return n_iter

def test(net, criterion, writer):
    with open('test_indices.txt','wb') as f:
        np.savetxt(f, np.array(split.test_indices), fmt='%d')

    n_iter = 0
    net.eval()
    test_accs = AverageMeter()
    test_pos_dist = AverageMeter()
    test_neg_dist = AverageMeter()
    anchor_embeddings = np.empty(1)
    positive_embeddings = np.empty(1)
    for batch_idx, (data1, data2, data3) in enumerate(test_loader):
            embedded_a, embedded_p, embedded_n = net(data1.cuda().float(), data2.cuda().float(), data3.cuda().float())
            dist_to_pos, dist_to_neg, loss, loss_total = criterion(embedded_a, embedded_p, embedded_n)
            writer.add_scalar('Test/Loss', loss, n_iter)

            acc = accuracy(dist_to_pos, dist_to_neg)
            test_accs.update(acc, data1.size(0))
            test_pos_dist.update(dist_to_pos.cpu().data.numpy().sum())
            test_neg_dist.update(dist_to_neg.cpu().data.numpy().sum())

            writer.add_scalar('Test/Accuracy', test_accs.avg, n_iter)
            writer.add_scalar('Test/Distance/Positive', test_pos_dist.avg, n_iter)
            writer.add_scalar('Test/Distance/Negative', test_neg_dist.avg, n_iter)

            anchor_embeddings = np.append(anchor_embeddings, embedded_a.cpu().data.numpy().reshape([1,-1]))
            positive_embeddings = np.append(positive_embeddings, embedded_p.cpu().data.numpy().reshape([1,-1]))
            n_iter = n_iter + 1

    #import pdb; pdb.set_trace()

    desc_anchors = anchor_embeddings[1:].reshape([test_size, descriptor_size])
    desc_positives = positive_embeddings[1:].reshape([test_size, descriptor_size])
    print(np.array(desc_positives).shape)

    sys.setrecursionlimit(50000)
    tree = spatial.KDTree(desc_positives)
    p_norm = 2
    max_pos_dist = 0.05
    max_anchor_dist = 1
    for n_nearest_neighbors in range(1,21):
        pos_count = 0
        anchor_count = 0
        idx_count = 0
        for idx in range(test_size):
            nn_dists, nn_indices = tree.query(desc_anchors[idx,:], p = p_norm, k = n_nearest_neighbors)
            nn_indices = [nn_indices] if n_nearest_neighbors == 1 else nn_indices

            for nn_i in nn_indices:
                if (nn_i >= test_size):
                    break;
                dist = spatial.distance.euclidean(desc_positives[nn_i,:], desc_positives[idx,:])
                if (dist <= max_pos_dist):
                    pos_count = pos_count + 1;
                    break
            for nn_i in nn_indices:
                if (nn_i >= test_size):
                    break;
                dist = spatial.distance.euclidean(desc_positives[nn_i,:], desc_anchors[idx,:])
                if (dist <= max_anchor_dist):
                    anchor_count = anchor_count + 1;
                    break
            for nn_i in nn_indices:
                if (nn_i == idx):
                    idx_count = idx_count + 1;
                    break
        pos_precision = (pos_count*1.0) / test_size
        anchor_precision = (anchor_count*1.0) / test_size
        idx_precision = (idx_count*1.0) / test_size
        writer.add_scalar('Test/Precision/Positive_Distance', pos_precision, n_nearest_neighbors)
        writer.add_scalar('Test/Precision/Anchor_Distance', anchor_precision, n_nearest_neighbors)
        writer.add_scalar('Test/Precision/Index_Count', idx_precision, n_nearest_neighbors)

if not restore:
    train_iter = 0
    val_iter = 0
    loss_ = 0.0
    print(f'Starting training using {n_epochs} epochs');
    for epoch in tqdm(range(n_epochs)):
        lr = adjust_learning_rate_exp(optimizer, epoch_num=epoch)
        t0 = time.time()

        train_iter = train(net, criterion, optimizer, writer, epoch, train_iter, loss_, t0)

        val_iter = validate(net, criterion, optimizer, writer, epoch, val_iter)

        writer.add_scalar('Train/lr', lr, epoch)

    print("Training finished!")
    torch.save(net.state_dict(), model_save)
else:
    net.load_state_dict(torch.load(model_save))

## Test

print("Starting testing...")
test(net, criterion, writer)
print("Testing finished!")
writer.close()
