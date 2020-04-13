import sys
import numpy as np
from scipy import spatial

import torch

from evaluation_set import EvaluationSet
from model import Model
from lc_candidate import LcCandidate

class Controller(object):
    def __init__(self, bw = 100, state_dict = './net_params_new_1.pkl', desc_size = 64):
        sys.setrecursionlimit(50000)
        self.descriptors = []
        self.clouds = []
        self.timestamps = []
        self.bw = bw
        self.desc_size = desc_size

        print("Initializing the network...")
        self.net = Model(bw).cuda()
        self.net.load_state_dict(torch.load(state_dict))
        self.net.eval()
        print("Finished initializing the network.")

    def handle_point_cloud(self, ts, cloud):
        # build ts and cloud pairs list
        self.timestamps.append(ts)
        self.clouds.append(cloud)

    def clear_clouds(self):
        self.timestamps = []
        self.clouds = []

    def find_loop_closures(self):
        # build search tree
        print("Building all descriptors...")
        descriptors = self.describe_all_point_clouds(self.clouds, self.bw)
        print("Building the kd tree...")
        tree = spatial.KDTree(descriptors)
        n_nearest_neighbors = 10
        p_norm = 2
        max_distance = 3
        all_candidates = []
        for idx in range(len(descriptors)):
            nn_dists, nn_indices = tree.query(descriptors[idx,:], p = p_norm, k = n_nearest_neighbors, distance_upper_bound = max_distance)
            if nn_indices is None:
                continue

            nn_indices = [nn_indices] if n_nearest_neighbors == 1 else nn_indices
            cand = LcCandidate(self.timestamps[nn_indices], self.clouds[nn_indices])
            all_candidates.append(cand)


        print("Finished loop closure lookup.")

    def describe_all_point_clouds(self, clouds, bw):
        eval_set = EvaluationSet(clouds, bw)
        loader = torch.utils.data.DataLoader(eval_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)

        n_clouds = len(clouds)
        embeddings = np.empty(1)
        for batch_idx, data in enumerate(val_loader):
            data = data.cuda().float()
            embedded = net(data, data, data)
            embeddings = np.append(embeddings, embedded.cpu().data.numpy().reshape([1,-1]))

        descriptors = embeddings[1:].reshape([n_clouds, self.desc_size])
        return descriptors

if __name__ == "__main__":
    ctrl = Controller()
    ctrl.handle_point_cloud(1, None)
