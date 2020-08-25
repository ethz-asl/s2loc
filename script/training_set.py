import torch.utils.data
from data_source import DataSource
from dh_grid import DHGrid
from sphere import Sphere
import numpy as np
import pandas as pd

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
        self.grid = DHGrid.CreateGrid(bw)


        if not restore:
            (a,p,n) = self.ds.get_all_cached()
        else:
            (a,p,n) = self.__loadTestSet()

        self.anchor_features = np.zeros((2, 2*bw, 2*bw))
        self.positive_features = np.zeros((2, 2*bw, 2*bw))
        self.negative_features = np.zeros((2, 2*bw, 2*bw))
        
        a_pcl_features, p_pcl_features, n_pcl_features = self.__genAllCloudFeatures(a, p, n)
        

    def __getitem__(self, index):
        # isinstance(l[1], str)
        if (index >= self.ds.start_cached) and (index < self.ds.end_cached):
            a, p, n = self.get_torch_feature(index)
            return a, p, n

        # We reached the end of the current cached batch.
        # Free the current set and cache the next one.
        a, p, n = self.ds.load_clouds_directly(index)
        a, p, n = self.__gen_all_features_single(a, p, n)
        return a, p, n

    def get_and_delete_torch_feature(self, index):
        anchor = torch.from_numpy(self.anchor_features[index])
        positive = torch.from_numpy(self.positive_features[index])
        negative = torch.from_numpy(self.negative_features[index])

        self.anchor_features[index] = None
        self.positive_features[index] = None
        self.negative_features[index] = None

        return anchor, positive, negative

    def get_torch_feature(self, index):
        anchor = torch.from_numpy(self.anchor_features[index])
        positive = torch.from_numpy(self.positive_features[index])
        negative = torch.from_numpy(self.negative_features[index])

        return anchor, positive, negative

    def __len__(self):
        return len(self.ds)

    def __genAllCloudFeatures(self, anchors, positives, negatives):
        print("Generating anchor spheres")
        anchor_features = process_map(partial(progresser, grid=self.grid), anchors, max_workers=32)
        print("Generating positive spheres")
        positive_features = process_map(partial(progresser, grid=self.grid), positives, max_workers=32)
        print("Generating negative spheres")
        negative_features = process_map(partial(progresser, grid=self.grid), negatives, max_workers=32)

        print("Generated features")
        return anchor_features, positive_features, negative_features

    def __gen_all_features_single(self, a, p, n):
        anchor_features = progresser(a, self.grid)
        positive_features = progresser(p, self.grid)
        negative_features = progresser(n, self.grid)
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
    
    def exportGeneratedFeatures(self, export_path):
        n_features = len(self.anchor_features)
        for i in tqdm(range(n_features)):
            cloud = '{:015d}'.format(i+1)
            anchor = self.anchor_features[i]
            positive = self.positive_features[i]
            negative = self.negative_features[i]
            
            d_anchor = {'intensities': list(anchor[0].flatten()), 'ranges': list(anchor[1].flatten())}
            d_positive = {'intensities': list(positive[0].flatten()), 'ranges': list(positive[1].flatten())}
            d_negative = {'intensities': list(negative[0].flatten()), 'ranges': list(negative[1].flatten())}
            df_anchor = pd.DataFrame(d_anchor)
            df_positive = pd.DataFrame(d_positive)
            df_negative = pd.DataFrame(d_negative)                                    

            df_anchor.to_csv(f'{export_path}/anchor/{cloud}.csv', index=False)
            df_positive.to_csv(f'{export_path}/positive/{cloud}.csv', index=False)
            df_negative.to_csv(f'{export_path}/negative/{cloud}.csv', index=False)
    
    def loadExportedFeatures(self, feature_path):
        path_anchor = f'{feature_path}/anchor/*.csv'
        path_positive = f'{feature_path}/positive/*.csv'
        path_negative = f'{feature_path}/negative/*.csv'
        all_anchor_features = sorted(glob.glob(path_anchor))
        all_positive_features = sorted(glob.glob(path_anchor))
        all_negative_features = sorted(glob.glob(path_anchor))
        
        n_features = len(all_anchor_features)
        anchor_features = [None] * n_features
        anchor_features = [None] * n_features
        anchor_features = [None] * n_features
        for i in tqdm(range(n_features)):
            path_anchor_csv = all_anchor_features[i]
            print("Path to anchor csv: ", path_anchor_csv)

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
