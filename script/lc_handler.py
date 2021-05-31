import math
import os

import numpy as np
from scipy import spatial

from lc_candidate import LcCandidate
from model import Model
from utils import Utils


class LcHandler(object):
    def __init__(self):
        self.tree = None

    def find_loop_closures(self, descriptors):
        if descriptors is None:
            print("ERROR: descriptor map is empty.")
            return False

        # build search tree
        print("Building the kd tree...")
        self.tree = spatial.KDTree(descriptors)

        n_nearest_neighbors = 10
        p_norm = 2
        max_distance = 3
        all_candidates = []
        for idx in range(len(descriptors)):
            nn_dists, nn_indices = tree.query(
                descriptors[idx, :], p=p_norm, k=n_nearest_neighbors, distance_upper_bound=max_distance)
            nn_dists, nn_indices = Utils.fix_nn_output(
                idx, nn_dists, nn_indices)
            if not nn_indices or not nn_dists:
                continue
            nn_indices = [
                nn_indices] if n_nearest_neighbors == 1 else nn_indices
            print(f'Got distances {nn_dists}, with indices {nn_indices}')

            # TODO(lbern): fix this here
            #cand = LcCandidate(
                #self.timestamps[nn_indices], self.clouds[nn_indices])
            all_candidates.append(cand)

        print("Finished loop closure lookup.")
        return True



if __name__ == "__main__":
    lc_handler = LcHandler()
