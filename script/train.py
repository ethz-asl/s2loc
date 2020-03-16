#!/usr/bin/env python
# coding: utf-8

# # S2Loc Training
# 
# Description: We propose to lear a descriptor of point clouds for global localization. 
# 
# Author: Lukas Bernreiter (lukas.bernreiter@ieee.org)
# 

# In[1]:


from data_source import DataSource
from visualize import Visualize
from sphere import Sphere
from model import Model
from loss import TripletLoss
from training_set import TrainingSet
from average_meter import AverageMeter

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import time
import math
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ## Load All Data
# 
# Load the dataset, project each point cloud on a sphere and derive a function for it.


ds = DataSource('/home/berlukas/data/spherical/training-set')
ds.load(100)


# ## Initialize the model and the training set


torch.backends.cudnn.benchmark = True
net = Model().cuda()
restore = 0
optimizer = torch.optim.SGD(net.parameters(), lr=5e-3, momentum=0.9)
n_epochs = 20
batch_size = 16
num_workers = 1
criterion = TripletLoss(margin=2)

result_save = 'triplet_result.txt'
progress_save = 'triplet_progress.txt'
model_save = 'net_params_new_1.pkl'

fp = open(result_save,'w')
fpp = open(progress_save, 'w')
n_parameters = sum([p.data.nelement() for p in net.parameters()])
fp.write('Number of params: {}\n'.format(n_parameters))
fp.write('features: [2, 10, 16, 20, 60]\n')
fp.write('bandwidths: [512, 50, 25, 15, 5]\n')
fp.write('batch_size = 16\n')
fp.write('training epoch: 20\n')
fp.write('TripletLoss(margin=2.0\n')


# In[6]:


bandwith = 100
train_set = TrainingSet(ds, bandwith)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)


# ## Train model

# In[7]:


def adjust_learning_rate_exp(optimizer, epoch_num, lr=5e-3):
    decay_rate = 0.96
    new_lr = lr * math.pow(decay_rate, epoch_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    return new_lr


# In[9]:


if restore ==0:
    net.train()

    for epoch in range(n_epochs):
        lr = adjust_learning_rate_exp(optimizer, epoch_num=epoch)
        loss_ = 0.0
        t0 = time.time()
        for batch_idx, (data1, data2, data3) in enumerate(train_loader):
            data1, data2, data3 = data1.cuda().float(), data2.cuda().float(), data3.cuda().float()
            
            embedded_a, embedded_p, embedded_n = net(data1, data2, data3)
            optimizer.zero_grad()

            dista, distb, loss, loss_total = criterion(embedded_a, embedded_p, embedded_n)
            loss.backward()
            optimizer.step()
            loss_ += loss_total.item()
            if batch_idx % 100 == 99:
                t1 = time.time()
                fpp.write('%.5f\n' %(loss_ / 100))
                print('[Epoch %d, Batch %4d] loss: %.5f time: %.5f lr: %.3e' %
                    (epoch + 1, batch_idx + 1, loss_ / 100, (t1-t0) / 60, lr))
                t0 = t1
                loss_ = 0.0

    print('training finished!')
    torch.save(net.state_dict(), model_save)
    # validating
    net.eval()

else:
    net.load_state_dict(torch.load(model_read))
    net.eval()


# In[7]:


accs = AverageMeter()
test_set = TrainingSet(ds, bandwith, False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
# record the error of each triplet
list_pos = []
list_neg = []


# In[8]:


for batch_idx, (data1, data2, data3) in enumerate(test_loader):
        data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        # data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)
        embedded_a, embedded_p, embedded_n = net(data1, data2, data3)
        dista, distb, loss, loss_total = criterion(embedded_a, embedded_p, embedded_n)

        record(dista, distb)

        acc = accuracy(dista, distb)
        accs.update(acc, data1.size(0))

array_pos = np.array(list_pos)
array_pos = array_pos.reshape(-1,)
dataframe = pd.DataFrame(array_pos)
dataframe.to_csv("pos_error.csv", index=0)
array_neg = np.array(list_neg)
array_neg = array_neg.reshape(-1,)
dataframe = pd.DataFrame(array_neg)
dataframe.to_csv("neg_error.csv", index=0)


# In[9]:


fp.write('Validation set:  Accuracy: {:.5f}%\n'.format(100. * accs.avg))
print('Validation set:  Accuracy: {:.5f}%\n'.format(100. * accs.avg))
fp.close()
fpp.close()

