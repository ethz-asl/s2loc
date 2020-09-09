import math
import os
import sys

import numpy as np
from scipy import spatial

import torch
from base_controller import BaseController
from evaluation_set import EvaluationSet
from lc_candidate import LcCandidate
from model import Model


class LocalizationController(BaseController):
    def __init__(self, map_folder, bw=100, state_dict='./net_params_new_1.pkl', desc_size=64):
        super().__init__(bw, state_dict, desc_size)
        self.descriptors, self.timestamps = self.read_descriptors_from_folder(
            map_folder)
        self.trees = self.build_kd_trees(self.descriptors)
        self.n_nearest_neighbors = 10
        self.p_norm = 2
        self.max_distance = 5

    def read_descriptors_from_folder(self, map_folder):
        files = os.listdir(map_folder)
        n_files = int(round(len(files) / 2.0))
        if n_files == 0:
            print("ERROR: Empty map folder found.")

        descriptors = [None] * n_files
        timestamps = [None] * n_files
        for i in range(0, n_files):
            descriptor_filename = f'{map_folder}/descriptors{i}.csv'
            timestamp_filename = f'{map_folder}/timestamps{i}.csv'
            descriptors[i] = np.readtxt(descriptor_filename, delimiter=',')
            timestamps[i] = np.readtxt(timestamps_filename, delimiter=',')
        return descriptors, timestamps

    def build_kd_trees(self, descriptors):
        n_clusters = len(descriptors)
        trees = [None] * n_clusters
        for i in range(0, n_clusters):
            trees[i] = spatial.KDTree(descriptors[i])
        return trees

    def handle_point_cloud(self, ts, cloud):
        descriptor = self.describe_cloud(cloud)

        for idx in range(len(self.trees)):
            nn_dists, nn_indices = self.trees[idx].query(
                descriptors[idx, :], p=self.p_norm, k=self.n_nearest_neighbors, distance_upper_bound=self.max_distance)
            nn_dists, nn_indices = self.fix_nn_output(
                idx, nn_dists, nn_indices)
            if not nn_indices or not nn_dists:
                continue
            nn_indices = [
                nn_indices] if n_nearest_neighbors == 1 else nn_indices
            print(f'Got distances {nn_dists}, with indices {nn_indices}')
            cand = LcCandidate(
                self.timestamps[nn_indices], self.clouds[nn_indices])
            all_candidates.append(cand)

    def describe_cloud(self, cloud):
        eval_set = EvaluationSet([cloud], self.bw)
        loader = torch.utils.data.DataLoader(
            eval_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)

        embeddings = np.empty(1)
        for batch_idx, data in enumerate(loader):
            data = data.cuda().float()
            embedded, _, _ = self.net(data, data, data)
            embeddings = np.append(
                embeddings, embedded.cpu().data.numpy().reshape([1, -1]))

        descriptors = embeddings[1:].reshape([1, self.desc_size])
        return descriptors


if __name__ == "__main__":
    ctrl = LocalizationController("./test_folder")
