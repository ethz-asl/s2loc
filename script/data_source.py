from plyfile import PlyData, PlyElement
import glob
import numpy as np
from os import listdir

class DataSource:
    def __init__(self, path_to_datasource, training_split = 0.8):
        self.datasource = path_to_datasource
        self.anchors = None
        self.positives = None
        self.negatives = None
        self.ds_size = 0
        self.training_split = training_split

    def load(self, n = -1):
        path_anchor = self.datasource + "/training_anchor/"
        path_positives = self.datasource + "/training_positive/"
        path_negatives = self.datasource + "/training_negative/"

        print("Loading anchors from:\t", path_anchor)
        self.anchors = self.loadDataset(path_anchor, n)
        print("Loading positives from:\t", path_positives)
        self.positives = self.loadDataset(path_positives, n)
        print("Loading negatives from:\t", path_negatives)
        self.negatives = self.loadDataset(path_negatives, n)

        print("Done loading dataset.")
        print(f"\tAnchors total: \t\t{len(self.anchors)}")
        print(f"\tPositives total: \t{len(self.positives)}")
        print(f"\tNegatives total: \t{len(self.negatives)}")
        self.ds_size = len(self.anchors)

    def loadDataset(self, path_to_dataset, n):
        idx = 0
        all_files = sorted(glob.glob(path_to_dataset + '*.ply'))
        n_files = len(all_files)
        n_ds = min(n_files, n) if n > 0 else n_files
        dataset = [None] * n_ds
        for ply_file in all_files:
            dataset[idx] = self.loadPointCloudFromPath(ply_file)        
            idx = idx + 1
            if n != -1 and idx >= n:
                break
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
        return self.ds_size

    def __len__(self):
        return self.size()

if __name__ == "__main__":
    ds = DataSource("/mnt/data/datasets/Spherical/training-set")
    ds.load()
