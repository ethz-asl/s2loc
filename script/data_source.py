from plyfile import PlyData, PlyElement
import glob
import numpy as np
import open3d as o3d
from os import listdir

class DataSource:
    def __init__(self, path_to_datasource, training_split = 0.8):
        self.datasource = path_to_datasource
        self.anchors_training = None
        self.positives_training = None
        self.negatives_training = None
        self.anchors_test = None
        self.positives_test = None
        self.negatives_test = None
        self.ds_size = 0
        self.training_split = training_split

    def load(self, n = -1):
        path_anchor = self.datasource + "/training_anchor/"
        path_positives = self.datasource + "/training_positive/"
        path_negatives = self.datasource + "/training_negative/"

        print("Loading anchors from:\t", path_anchor)
        anchors = self.loadDataset(path_anchor, n)
        print("Loading positives from:\t", path_positives)
        positives = self.loadDataset(path_positives, n)
        print("Loading negatives from:\t", path_negatives)
        negatives = self.loadDataset(path_negatives, n)

        print("Splitting up training and testing data.")
        self.splitDataset(anchors, positives, negatives)

        print("Done loading dataset.")
        print(f"\tAnchors total: \t\t{len(anchors)}\t training/test: ({len(self.anchors_training)}/{len(self.anchors_test)})")
        print(f"\tPositives total: \t{len(positives)}\t training/test: ({len(self.positives_training)}/{len(self.positives_test)})")
        print(f"\tNegatives total: \t{len(negatives)}\t training/test: ({len(self.negatives_training)}/{len(self.negatives_test)})")
        self.ds_size = len(self.anchors_training)

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

    def splitDataset(self, anchors, positives, negatives):
        n_total = len(anchors)
        n_training = round(n_total * self.training_split)

        self.anchors_training = anchors[0:n_training]
        self.anchors_test = anchors[n_training:n_total]

        self.positives_training = anchors[0:n_training]
        self.positives_test = anchors[n_training:n_total]

        self.negatives_training = anchors[0:n_training]
        self.negatives_test = anchors[n_training:n_total]

    def size(self):
        return self.ds_size

if __name__ == "__main__":
    ds = DataSource("/mnt/data/datasets/Spherical/training-set")
    ds.load()
