from plyfile import PlyData, PlyElement
import glob
from os import listdir
import numpy as np

class DataSource:
    def __init__(self, path_to_datasource):
        self.datasource = path_to_datasource
        self.anchors = None
        self.positives = None
        self.negatives = None

    def loadAll(self):
        path_anchor = self.datasource + "/training_anchor/"
        path_positives = self.datasource + "/training_positive/"
        path_negatives = self.datasource + "/training_negative/"

        print("Loading anchors from ", path_anchor)
        self.anchors = self.loadDataset(path_anchor)
        print("Loading positives from ", path_positives)
        self.positives = self.loadDataset(path_positives)
        print("Loading negatives from ", path_negatives)
        self.negatives = self.loadDataset(path_negatives)

        print("Done loading dataset.")
        print("\tAnchors: \t", len(self.anchors))
        print("\tPositives: \t", len(self.positives))
        print("\tNegatives: \t", len(self.negatives))

    def loadDataset(self, path_to_dataset):
        idx = 0
        all_files = sorted(glob.glob(path_to_dataset + '*.ply'))
        dataset = [None] * len(all_files)
        for ply_file in all_files:
            plydata = PlyData.read(ply_file)
            x = plydata['vertex']['x']
            y = plydata['vertex']['y']
            z = plydata['vertex']['z']
            i = plydata['vertex']['scalar']
            dataset[idx] = np.concatenate((x,y,z,i), axis=0).reshape(-1, 4)
            idx = idx + 1
        return dataset


if __name__ == "__main__":
    ds = DataSource("/tmp/training")
    ds.loadAll()
