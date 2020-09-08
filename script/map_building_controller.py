import math
import os

import numpy as np
from scipy import spatial

import torch
from base_controller import BaseController
from evaluation_set import EvaluationSet
from lc_candidate import LcCandidate
from model import Model


class MapBuildingController(BaseController):
    def __init__(self, export_map_folder="", bw=100, state_dict='./net_params_new_1.pkl', desc_size=64):
        super().__init__(bw, state_dict, desc_size)
        self.descriptors = []
        self.clouds = []
        self.timestamps = []
        self.export_map_folder = export_map_folder

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
        if self.export_map_folder is not "":
            self.export_descriptors_to_folder(
                self.export_map_folder, descriptors)
        print("Building the kd tree...")
        tree = spatial.KDTree(descriptors)
        n_nearest_neighbors = 10
        p_norm = 2
        max_distance = 3
        all_candidates = []
        for idx in range(len(descriptors)):
            nn_dists, nn_indices = tree.query(
                descriptors[idx, :], p=p_norm, k=n_nearest_neighbors, distance_upper_bound=max_distance)
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

        print("Finished loop closure lookup.")

    def fix_nn_output(self, idx, nn_dists, nn_indices):
        self_idx = nn_indices.index(idx)
        del nn_indices[self_idx]
        del nn_dists[self_idx]

        fixed_nn_dists = [dists for dists in nn_dists if not (
            math.isnan(dists) or math.isinf(dists))]
        fixed_nn_indices = [i for i in nn_indices if not (
            math.isnan(i) or math.isinf(i))]
        return fixed_nn_dists, fixed_nn_indices

    def describe_all_point_clouds(self, clouds, bw):
        eval_set = EvaluationSet(clouds, bw)
        loader = torch.utils.data.DataLoader(
            eval_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)

        n_clouds = len(clouds)
        embeddings = np.empty(1)
        for batch_idx, data in enumerate(loader):
            data = data.cuda().float()
            embedded, _, _ = self.net(data, data, data)
            embeddings = np.append(
                embeddings, embedded.cpu().data.numpy().reshape([1, -1]))

        descriptors = embeddings[1:].reshape([n_clouds, self.desc_size])
        return descriptors

    def export_descriptors_to_folder(self, map_folder, descriptors):
        files = os.listdir(map_folder)
        next_index = len(files) + 1
        filename = f'{map_folder}/descriptors{next_index}.csv'

        np.savetxt(filename, descriptors, delimiter=',')


if __name__ == "__main__":
    ctrl = MapBuildingController()
    ctrl.handle_point_cloud(1, None)
