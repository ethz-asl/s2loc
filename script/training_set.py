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

class TrainingSet(torch.utils.data.Dataset):
    def __init__(self, data_source, restore, bw=100):
        self.ds = data_source
        self.bw = bw
        self.is_restoring = restore
        self.test_indices = []
        self.cache = data_source.cache

        if not restore:
            (a,p,n) = self.ds.get_all_cached()
        else:
            (a,p,n) = self.__loadTestSet()

        print(f'Size of cached: {len(a)}')
        self.anchor_features, self.positive_features, self.negative_features = self.__genAllFeatures(bw, a, p, n)
        print(f'Size of cached2: {len(self.anchor_features)}')

    def __getitem__(self, index):
        # isinstance(l[1], str)
        if (index >= self.ds.start_cached) and (index < self.ds.end_cached):
            a, p, n = self.get_and_delete_torch_feature(index)
            return a, p, n

        # We reached the end of the current cached batch.
        # Free the current set and cache the next one.
        prev_end, end = ds.cache_next(index)
        a, p, n = self.ds.get_cached(prev_end, end)
        a, p, n = self.__genAllFeatures(self.bw, a, p, n)
        print(f'Size before {len(self.anchor_features)}.')
        print(f'Appending {len(a)} features.')
        self.anchor_features.extend(a)
        self.positive_features.extend(p)
        self.negative_features.extend(n)
        print(f'Total size {len(self.anchor_features)}.')

        return self.get_and_delete_torch_feature(index)

    def get_and_delete_torch_feature(self, index):
        anchor = torch.from_numpy(self.anchor_features[index])
        positive = torch.from_numpy(self.positive_features[index])
        negative = torch.from_numpy(self.negative_features[index])

        self.anchor_features[index] = None
        self.positive_features[index] = None
        self.negative_features[index] = None

        return anchor, positive, negative

    def __len__(self):
        return self.ds.ds_total_size

    def __genAllFeatures(self, bw, anchors, positives, negatives):
        n_ds = len(anchors)
        grid = DHGrid.CreateGrid(bw)
        print("Generating anchor spheres")
        anchor_features = process_map(partial(progresser, grid=grid), anchors, max_workers=32)
        print("Generating positive spheres")
        positive_features = process_map(partial(progresser, grid=grid), positives, max_workers=32)
        print("Generating negative spheres")
        negative_features = process_map(partial(progresser, grid=grid), negatives, max_workers=32)

        print("Generated features")
        return anchor_features, positive_features, negative_features

    def __loadTestSet(self):
        with open('test_indices.txt','rb') as f:
            self.test_indices = np.loadtxt(f).astype(int)
            #import pdb; pdb.set_trace()

            a = [self.ds.anchors[i] for i in self.test_indices]
            p = [self.ds.positives[i] for i in self.test_indices]
            n = [self.ds.negatives[i] for i in self.test_indices]
            return (a,p,n)


    def isRestoring(self):
        return self.is_restoring


if __name__ == "__main__":
    cache = 10
    ds = DataSource("/mnt/data/datasets/Spherical/training", cache)
    ds.load(100)
    ts = TrainingSet(ds, False)
    print("Total length of trainining set:\t", ts.__len__())

    a,p,n = ts.__getitem__(0)
    print("First anchor:\t", a.shape)
    print("First positive:\t", p.shape)
    print("First negative:\t", n.shape)

    next_idx = cache + 5
    a,p,n = ts.__getitem__(next_idx)
    print(f"{next_idx}th anchor:\t", a.shape)
    print(f"{next_idx}th positive:\t", p.shape)
    print(f"{next_idx}th negative:\t", n.shape)

    a,p,n = ts.__getitem__(1)
    print("Second anchor:\t", a.shape)
    print("Second positive:\t", p.shape)
    print("Second negative:\t", n.shape)
