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

        if not restore:
            (a,p,n) = (self.ds.anchors, self.ds.positives, self.ds.negatives)
        else:
            (a,p,n) = self.__loadTestSet()

        self.anchor_features, self.positive_features, self.negative_features = self.__genAllFeatures(bw, a, p, n)

    def __getitem__(self, index):
        anchor = torch.from_numpy(self.anchor_features[index])
        positive = torch.from_numpy(self.positive_features[index])
        negative = torch.from_numpy(self.negative_features[index])
        return anchor, positive, negative

    def __len__(self):
        return len(self.anchor_features)

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
    ds = DataSource('/home/berlukas/data/spherical/training-set')
    ds.load(100)

    ts = TrainingSet(ds)
    a,p,n = ts.__getitem__(0)
    print("First anchor:\t", a.shape)
    print("First positive:\t", p.shape)
    print("First negative:\t", n.shape)
    print("Total length:\t", ts.__len__())
