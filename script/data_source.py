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
        self.all_anchor_files = []
        self.all_positive_files = []
        self.all_negative_files = []

    def load(self, n = -1):
        path_anchor = self.datasource + "/training_anchor/"
        path_positives = self.datasource + "/training_positive/"
        path_negatives = self.datasource + "/training_negative/"

        print("Loading anchors from:\t", path_anchor)
        self.all_anchor_files = sorted(glob.glob(path_anchor + '*.ply'))
        self.anchors = self.loadDataset(self.all_anchor_files, n, self.cache)
        print("Loading positives from:\t", path_positives)
        self.all_positive_files = sorted(glob.glob(path_positives + '*.ply'))
        self.positives = self.loadDataset(self.all_positive_files, n, self.cache)
        print("Loading negatives from:\t", path_negatives)
        self.all_negative_files = sorted(glob.glob(path_negatives + '*.ply'))
        self.negatives = self.loadDataset(self.all_negative_files, n, self.cache)

        print("Done loading dataset.")
        print(f"\tAnchors total: \t\t{len(self.anchors)}")
        print(f"\tPositives total: \t{len(self.positives)}")
        print(f"\tNegatives total: \t{len(self.negatives)}")

    def loadDataset(self, all_files, n, cache):
        idx = 0
        self.ds_total_size = len(all_files)
        n_ds = min(self.ds_total_size, n) if n > 0 else self.ds_total_size
        dataset = [None] * n_ds
        for ply_file in all_files:
            dataset[idx] = self.loadPointCloudFromPath(ply_file) if idx < cache else ply_file
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
        print(f'Reading PLY from {path_to_point_cloud}')
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

    def cache_next(self, index):
        prev_end = self.end_cached
        self.end_cached = min(self.ds_total_size, index+self.cache)
        for idx in range(prev_end, self.end_cached):
            self.anchors[idx], self.positives[idx], self.negatives[idx] = self.load_clouds_directly(idx)
        return prev_end, self.end_cached

    def free_to_start_cached(self):
        for idx in range(0, self.start_cached):
            self.anchors[idx] = self.all_anchor_files[idx]
            self.positives[idx] = self.all_positive_files[idx]
            self.negatives[idx] = self.all_negative_files[idx]

    def get_all_cached(self):
        return self.get_cached(self.start_cached, self.end_cached)

    def get_cached(self, start, end):
        assert start <= end
        start = max(0, start)
        end = min(self.ds_total_size, end)

        return self.anchors[start:end], \
               self.positives[start:end], \
               self.negatives[start:end]

    def load_clouds_directly(self, idx):
        anchor = self.loadPointCloudFromPath(self.anchors[idx]) if isinstance(self.anchors[idx], str) else self.anchors[idx]
        positive = self.loadPointCloudFromPath(self.positives[idx]) if isinstance(self.positives[idx], str) else self.positives[idx]
        negative = self.loadPointCloudFromPath(self.negatives[idx]) if isinstance(self.negatives[idx], str) else self.negatives[idx]
        return anchor, positive, negative

if __name__ == "__main__":
    ds = DataSource("/mnt/data/datasets/Spherical/training", 10)
    ds.load(100)

    a,p,n = ds.get_all_cached()
    print(f'len of initial cache {len(a)} of batch [{ds.start_cached}, {ds.end_cached}]')
    print("Caching next batch...")
    ds.cache_next(25)
    a,p,n = ds.get_all_cached()
    print(f'len of next cache {len(a)} of batch [{ds.start_cached}, {ds.end_cached}]')
