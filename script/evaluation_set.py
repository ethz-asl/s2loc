import torch.utils.data
from data_source import DataSource
from dh_grid import DHGrid
from sphere import Sphere
import numpy as np

from tqdm.auto import tqdm, trange
from tqdm.contrib.concurrent import process_map, thread_map
from functools import partial

import pymp

def progresser(sample, grid, auto_position=True, write_safe=False, blocking=True, progress=False):
    sample_sphere = Sphere(sample)
    return sample_sphere.sampleUsingGrid(grid)

class EvaluationSet(torch.utils.data.Dataset):
    def __init__(self, clouds, bw=100):
        self.bw = bw
        self.features = self.__genAllFeatures(bw, clouds)

    def __getitem__(self, index):
        return torch.from_numpy(self.features[index])

    def __len__(self):
        return len(self.features)

    def __genAllFeatures(self, bw, clouds):
        n_ds = len(clouds)
        grid = DHGrid.CreateGrid(bw)
        print("Generating spheres")
        features = process_map(partial(progresser, grid=grid), clouds, max_workers=32)
        return features
