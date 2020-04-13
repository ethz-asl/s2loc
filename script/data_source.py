from plyfile import PlyData, PlyElement
import glob
import numpy as np
from os import listdir

class DataSource:
    def __init__(self, path_to_datasource, cache = -1):
        self.datasource = path_to_datasource
        self.anchors = None
        self.positives = None
        self.negatives = None
        self.ds_total_size = 0
        self.cache = cache
        self.start_cached = 0
        self.end_cached = 0
        self.all_files = []
        self.cached_batch = 0

    def load(self, n = -1):
        path_anchor = self.datasource + "/training_anchor/"
        path_positives = self.datasource + "/training_positive/"
        path_negatives = self.datasource + "/training_negative/"

        print("Loading anchors from:\t", path_anchor)
        self.anchors = self.loadDataset(path_anchor, n, self.cache)
        print("Loading positives from:\t", path_positives)
        self.positives = self.loadDataset(path_positives, n, self.cache)
        print("Loading negatives from:\t", path_negatives)
        self.negatives = self.loadDataset(path_negatives, n, self.cache)

        print("Done loading dataset.")
        print(f"\tAnchors total: \t\t{len(self.anchors)}")
        print(f"\tPositives total: \t{len(self.positives)}")
        print(f"\tNegatives total: \t{len(self.negatives)}")

    def loadDataset(self, path_to_dataset, n, cache):
        idx = 0
        self.all_files = sorted(glob.glob(path_to_dataset + '*.ply'))
        self.ds_total_size = len(self.all_files)
        n_ds = min(self.ds_total_size, n) if n > 0 else self.ds_total_size
        dataset = [None] * n_ds
        for ply_file in self.all_files:
            if idx < cache:
                dataset[idx] = self.loadPointCloudFromPath(ply_file)
            else:
                dataset[idx] = ply_file

            idx = idx + 1
            if n != -1 and idx >= n:
                break
        self.end_cached = cache
        return dataset

    def loadDatasetPathOnly(self, path_to_dataset, n):
        all_files = sorted(glob.glob(path_to_dataset + '*.ply'))
        n_ds = min(n_files, n) if n > 0 else n_files
        dataset = all_files[:,n_ds]
        return dataset

    def loadPointCloudFromPath(self, path_to_point_cloud):
        plydata = PlyData.read(path_to_point_cloud)
        x = plydata['vertex']['x']
        y = plydata['vertex']['y']
        z = plydata['vertex']['z']
        i = plydata['vertex']['scalar']
        return np.concatenate((x,y,z,i), axis=0).reshape(4, len(x)).transpose()

    def size(self):
        return self.ds_total_size

    def __len__(self):
        return self.size()

    def cache_next(self):
        self.start_cached = self.end_cached
        self.end_cached = min(self.ds_total_size, self.end_cached + self.cache)
        for idx in range(self.start_cached, self.end_cached):
            dataset[idx] = self.loadPointCloudFromPath(ply_file)
        self.cached_batch = self.cached_batch + 1

    def free_to_start_cached(self):
        for idx in range(0, self.start_cached):
            dataset[idx] = self.all_files[idx]

    def get_cached(self):
        return self.anchors[self.start_cached:self.end_cached],
               self.positives[self.start_cached:self.end_cached],
               self.negatives[self.start_cached:self.end_cached]


if __name__ == "__main__":
    ds = DataSource("/mnt/data/datasets/Spherical/training-set", 10)
    ds.load(100)

    a,p,n = ds.get_cached()
    print(f'len of initial cache {len(a)} of batch {ds.cached_batch}')
    ds.cache_next()
    print(f'len of next cache {len(a)} of batch {ds.cached_batch}')
