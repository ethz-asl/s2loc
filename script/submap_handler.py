import math
import os

import numpy as np
from scipy import spatial

import torch
from base_controller import BaseController
from evaluation_set import EvaluationSet
from lc_candidate import LcCandidate
from model import Model


class SubmapHandler(object):
    def __init__(self):
        self.pivot_distance = 5
        self.n_nearest_neighbors = 3
        self.p_norm = 2

    def find_close_submap(self, submaps):
        n_submaps = len(submaps)
        if n_submaps == 0:
            return
        submap_positions = self.get_all_positions(submaps)
        tree = spatial.KDTree(submap_positions)

        candidates = np.zeros((n_submaps, n_submaps))
        for i in range(0, n_submaps):
            nn_dists, nn_indices = self.lookup_closest_submap(submap_positions[i, :])
            if not nn_indices or not nn_dists:
                continue

            for nn_i in nn_indices:
                candidates[i, nn_i] = 1

        # filter duplicates by zeroing the lower triangle
        candidates = np.triu(candidates)

    def lookup_closest_submap(self, submap):
        nn_dists, nn_indices = tree.query(
            submap
            p=self.p_norm,
            k=self.n_nearest_neighbors,
            distance_upper_bound=self.pivot_distance)
        nn_dists, nn_indices = Utils.fix_nn_output(
            idx, nn_dists, nn_indices)
        nn_indices = [
            nn_indices] if self.n_nearest_neighbors == 1 else nn_indices

        return nn_dists, nn_indices


    def get_all_positions(self, submaps):
        n_submaps = len(submaps)
        if n_submaps == 0:
            return

        positions = np.empty((n_submaps, 3))
        for i in range(0, n_submaps):
            positions[i, 0:3] = np.transpose(submaps[i].get_pivot_pose()[0:3, 3])

        return positions



if __name__ == "__main__":
    lc_handler = LcHandler()
