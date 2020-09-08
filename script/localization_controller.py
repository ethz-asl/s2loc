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
        self.descriptors = self.read_descriptors_from_folder(map_folder)

    def read_descriptors_from_folder(self, map_folder):
        files = os.listdir(map_folder)
        n_files = len(files)
        if n_files == 0:
            print("ERROR: Empty map folder found.")

        descriptors = [None] * n_files
        for i in range(0, n_files):
            filename = f'{map_folder}/descriptors{i}.csv'
            descriptors[i] = np.readtxt(filename, delimiter=',')
        return descriptors


if __name__ == "__main__":
    ctrl = LocalizationController("./test_folder")
