from functools import partial

import numpy as np

import pymp
import torch.utils.data
from data_source import DataSource
from dh_grid import DHGrid
from sphere import Sphere
from tqdm.auto import tqdm, trange
from tqdm.contrib.concurrent import process_map, thread_map

def progresser(submap, grid, auto_position=True, write_safe=False, blocking=True, progress=False):
    dm = submap.get_dense_map()
    np.save('/home/berlukas/Documents/workspace/phaser_ws/src/s2loc/script/playground/foo.npy', dm)
    sample_sphere = Sphere(dm)
    return sample_sphere.sampleUsingGrid(grid)

class EvaluationSet(torch.utils.data.Dataset):
    def __init__(self, clouds, bw=100):
        self.bw = bw
        self.features = self.__genAllFeatures(clouds, bw)

    def __getitem__(self, index):
        return torch.from_numpy(self.features[index])

    def __len__(self):
        return len(self.features)

    def __genAllFeatures(self, clouds, bw):
        n_ds = len(clouds)
        grid = DHGrid.CreateGrid(bw)
        print(f"[EvaluationSet] Generating spheres for {len(clouds)} submaps using a bandwidth of {bw}.")
        print(f'cloud is {clouds[0].get_dense_map().shape}')

        features = process_map(
            partial(progresser, grid=grid), clouds.values(), max_workers=8)
        return features

    def save_features_to_disk(self, filename):
        np.save(filename, self.features)
        print(f"[EvaluationSet] Wrote computed features to {filename}")
