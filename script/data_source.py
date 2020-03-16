from plyfile import PlyData, PlyElement
import glob
import numpy as np
import open3d as o3d
from os import listdir

class DataSource:
    def __init__(self, path_to_datasource):
        self.datasource = path_to_datasource
        self.anchors = None
        self.positives = None
        self.negatives = None
        self.ds_size = 0

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
        print("\tAnchors: \t", len(self.anchors))
        print("\tPositives: \t", len(self.positives))
        print("\tNegatives: \t", len(self.negatives))
        self.ds_size = len(self.anchors)

    def loadDataset(self, path_to_dataset, n):
        idx = 0
        all_files = sorted(glob.glob(path_to_dataset + '*.ply'))
        n_files = len(all_files)
        n_ds = min(n_files, n) if n > 0 else n_files
        dataset = [None] * n_ds
        for ply_file in all_files:
            plydata = PlyData.read(ply_file)
            #pcd = o3d.io.read_point_cloud(ply_file)
            x = plydata['vertex']['x']
            y = plydata['vertex']['y']
            z = plydata['vertex']['z']
            i = plydata['vertex']['scalar']
            dataset[idx] = np.concatenate((x,y,z,i), axis=0).reshape(4, len(x)).transpose()
            #dataset[idx] = np.asarray(pcd.points)
            idx = idx + 1
            if n != -1 and idx >= n:
                break
        return dataset

    def size(self):
        return self.ds_size

if __name__ == "__main__":
    ds = DataSource("/mnt/data/datasets/Spherical/training-set")
    ds.load()
