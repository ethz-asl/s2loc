import math
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import spatial

import open3d as o3d
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from average_meter import AverageMeter
from data_source import DataSource
from data_splitter import DataSplitter
from database_parser import DatabaseParser
from loss import ImprovedTripletLoss, TripletLoss
from mission_indices import MissionIndices
from model import Model
from sphere import Sphere
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm.auto import tqdm
from training_set import TrainingSet
from visualize import Visualize

# ## Load All Data
# Load the dataset, project each point cloud on a sphere and derive a function for it.

# ## Initialize the model and the training set
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
bandwidth = 100
n_features = 3
net = Model(n_features,bandwidth).cuda()
restore = False
learning_rate = 4.5e-3
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
n_epochs = 55
batch_size = 13
num_workers = 32
descriptor_size = 256
net_input_size = 2 * bandwidth
criterion = ImprovedTripletLoss(margin=2, alpha=0.5, margin2=0.2)
writer = SummaryWriter()
model_save = 'net_params.pkl'
feature_dim = bandwidth * 2
#summary(net, input_size=[(n_features, feature_dim, feature_dim), (n_features, feature_dim, feature_dim), (n_features, feature_dim, feature_dim)])

# ## Load the data
n_data = 20000
#n_data = 22
cache = n_data
dataset_path = "../../data/arche_low_res/"
db_parser = DatabaseParser(dataset_path)

training_missions, test_missions = MissionIndices.get_arche_low_res()
#training_missions, test_missions = MissionIndices.get_arche_high_res()
training_indices, test_indices = db_parser.extract_training_and_test_indices(
    training_missions, test_missions)
idx = np.array(training_indices['idx'].tolist())

ds = DataSource(dataset_path, cache)
train_set = TrainingSet(restore, bandwidth)
generate_features = True
if generate_features:
    ds.load(n_data, idx, filter_clusters=False)
    train_set.generateAll(ds)
    anchor_poses = ds.anchor_poses
    positive_poses = ds.positive_poses
    negative_poses = ds.negative_poses
    train_set.exportGeneratedFeatures('../../data/spherical/arche_low_res/')
else:
    anchor_poses,positive_poses,negative_poses = train_set.loadFeatures('../../data/spherical/arche_low_res/')

# tmp for removing the images
#train_set.anchor_features = train_set.anchor_features[:,0:2,:,:]
#train_set.positive_features = train_set.positive_features[:,0:2,:,:]
#train_set.negative_features = train_set.negative_features[:,0:2,:,:]


print("Total size: ", len(train_set))
split = DataSplitter(train_set, restore, test_train_split=1.0, shuffle=True)

train_loader, val_loader, test_loader = split.get_split(
    batch_size=batch_size, num_workers=num_workers)
train_size = split.get_train_size()
val_size = split.get_val_size()
test_size = split.get_test_size()
print("Training size: ", train_size)
print("Validation size: ", val_size)
if test_size == 0:
    print('Test size is 0. Configured for external tests')
else:
    print("Testing size: ", test_size)


visualize = False
if visualize:
    first_anchor = Sphere(ds.anchors_training[0])
    len(first_anchor.point_cloud)

    viz = Visualize()
    viz.visualizeRawPointCloud(first_anchor, True)
    viz.visualizeSphere(first_anchor, True)

# ## Train model
def adjust_learning_rate_exp(optimizer, epoch_num, lr):
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
    acc = ((pred < 0).sum()).float() / dista.size(0)
    return acc


def train(net, criterion, optimizer, writer, epoch, n_iter, loss_, t0):
    train_pos_dist = AverageMeter()
    train_neg_dist = AverageMeter()
    net.train()
    for batch_idx, (data1, data2, data3) in enumerate(train_loader):
        data1, data2, data3 = data1.cuda().float(), data2.cuda().float(), data3.cuda().float()
        embedded_a, embedded_p, embedded_n = net(data1, data2, data3)

        dista, distb, loss_triplet, loss_total = criterion(embedded_a, embedded_p, embedded_n)
        loss_embedd = embedded_a.norm(2) + embedded_p.norm(2) + embedded_n.norm(2)
        loss = loss_triplet + 0.001 * loss_embedd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_ += loss_total.item()

        train_pos_dist.update(dista.cpu().data.numpy().sum())
        train_neg_dist.update(distb.cpu().data.numpy().sum())
        writer.add_scalar('Train/Loss_Triplet', loss_triplet, n_iter)
        writer.add_scalar('Train/Loss_Embedd', loss_embedd, n_iter)
        writer.add_scalar('Train/Loss', loss, n_iter)
        writer.add_scalar('Train/Distance/Positive',
                          train_pos_dist.avg, n_iter)
        writer.add_scalar('Train/Distance/Negative',
                          train_neg_dist.avg, n_iter)
        n_iter += 1

        if batch_idx % 5 == 4:
            t1 = time.time()
            print('[Epoch %d, Batch %4d] loss: %.8f time: %.5f lr: %.3e' %
                  (epoch + 1, batch_idx + 1, loss_ / 5, (t1 - t0) / 60, lr))
            t0 = t1
            loss_ = 0.0
    return n_iter


def validate(net, criterion, optimizer, writer, epoch, n_iter):
    net.eval()
    with torch.no_grad():
        anchor_embeddings = np.empty(1)
        positive_embeddings = np.empty(1)
        for batch_idx, (data1, data2, data3) in enumerate(val_loader):
            data1, data2, data3 = data1.cuda().float(
            ), data2.cuda().float(), data3.cuda().float()
            embedded_a, embedded_p, embedded_n = net(data1, data2, data3)
            optimizer.zero_grad()

            dista, distb, loss_triplet, loss_total = criterion(
                embedded_a, embedded_p, embedded_n)
            loss_embedd = embedded_a.norm(
                2) + embedded_p.norm(2) + embedded_n.norm(2)
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
    n_iter = 0
    net.eval()
    with torch.no_grad():
        n_test_data = 3000
        n_test_cache = n_test_data
        ds_test = DataSource(dataset_path, n_test_cache, -1)
        idx = np.array(test_indices['idx'].tolist())
        ds_test.load(n_test_data, idx)
        n_test_data = len(ds_test.anchors)
        test_set = TrainingSet(restore, bandwidth)
        test_set.generateAll(ds_test)
        n_test_set = len(test_set)
        if n_test_set == 0:
            print("Empty test set. Aborting test.")
            return
        print("Total size of the test set: ", n_test_set)
        test_size = n_test_set
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=10, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
        anchor_poses = ds_test.anchor_poses
        positive_poses = ds_test.positive_poses
        assert len(anchor_poses) == len(positive_poses)

        test_accs = AverageMeter()
        test_pos_dist = AverageMeter()
        test_neg_dist = AverageMeter()
        anchor_embeddings = np.empty(1)
        positive_embeddings = np.empty(1)
        for batch_idx, (data1, data2, data3) in enumerate(test_loader):
            embedded_a, embedded_p, embedded_n = net(
                data1.cuda().float(), data2.cuda().float(), data3.cuda().float())
            dist_to_pos, dist_to_neg, loss, loss_total = criterion(
                embedded_a, embedded_p, embedded_n)
            writer.add_scalar('Test/Loss', loss, n_iter)

            acc = accuracy(dist_to_pos, dist_to_neg)
            test_accs.update(acc, data1.size(0))
            test_pos_dist.update(dist_to_pos.cpu().data.numpy().sum())
            test_neg_dist.update(dist_to_neg.cpu().data.numpy().sum())

            writer.add_scalar('Test/Accuracy', test_accs.avg, n_iter)
            writer.add_scalar('Test/Distance/Positive',
                              test_pos_dist.avg, n_iter)
            writer.add_scalar('Test/Distance/Negative',
                              test_neg_dist.avg, n_iter)

            anchor_embeddings = np.append(
                anchor_embeddings, embedded_a.cpu().data.numpy().reshape([1, -1]))
            positive_embeddings = np.append(
                positive_embeddings, embedded_p.cpu().data.numpy().reshape([1, -1]))
            n_iter = n_iter + 1

        desc_anchors = anchor_embeddings[1:].reshape(
            [test_size, descriptor_size])
        desc_positives = positive_embeddings[1:].reshape(
            [test_size, descriptor_size])

        sys.setrecursionlimit(50000)
        tree = spatial.KDTree(desc_positives)
        p_norm = 2
        max_pos_dist = 0.05
        max_loc_dist = 5.0
        max_anchor_dist = 1
        for n_nearest_neighbors in range(1, 21):
            loc_count = 0
            for idx in range(test_size):
                nn_dists, nn_indices = tree.query(
                    desc_anchors[idx, :], p=p_norm, k=n_nearest_neighbors)
                nn_indices = [
                    nn_indices] if n_nearest_neighbors == 1 else nn_indices

                for nn_i in nn_indices:
                    dist = spatial.distance.euclidean(
                        positive_poses[nn_i, 5:8], anchor_poses[idx, 5:8])
                    if (dist <= max_pos_dist):
                        loc_count = loc_count + 1
                        break

            loc_precision = (loc_count * 1.0) / test_size
            writer.add_scalar('Test/Precision/Localization',
                              loc_precision, n_nearest_neighbors)

if not restore:
    train_iter = 0
    val_iter = 0
    loss_ = 0.0
    print(f'Starting training using {n_epochs} epochs')
    for epoch in tqdm(range(n_epochs)):
        lr = adjust_learning_rate_exp(optimizer, epoch_num=epoch, lr=learning_rate)
        t0 = time.time()

        train_iter = train(net, criterion, optimizer, writer, epoch, train_iter, loss_, t0)
        val_iter = validate(net, criterion, optimizer, writer, epoch, val_iter)
        writer.add_scalar('Train/lr', lr, epoch)

    print("Training finished!")
    torch.save(net.state_dict(), model_save)
else:
    net.load_state_dict(torch.load(model_save))

# Test

print("Starting testing...")
torch.cuda.empty_cache()
test(net, criterion, writer)
print("Testing finished!")
writer.close()
