import torch.utils.data
from data_source import DataSource
from dh_grid import DHGrid
from sphere import Sphere

class TrainingSet(torch.utils.data.Dataset):
    def __init__(self, data_source, bw=50):
        self.ds = data_source
        self.bw = bw
        self.anchor_features, self.positive_features, self.negative_features, = self.__genAllFeatures(bw)

    def __getitem__(self, index):
        anchor = torch.from_numpy(self.anchor_features[index])
        positive = torch.from_numpy(self.positive_features[index])
        negative = torch.from_numpy(self.negative_features[index])
        print("anchor size1: ", anchor.size(1))
        #print("anchor size2: ", anchor.size(2))
        #print("anchor size3: ", anchor.size(3))
        return anchor, positive, negative

    def __len__(self):
        return len(self.anchor_features)

    def __genAllFeatures(self, bw):
        n_ds = self.ds.size()
        grid = DHGrid.createGrid(bw)

        anchor_features = [None] * n_ds
        positive_features = [None] * n_ds
        negative_features = [None] * n_ds
        for i in range(n_ds):
            anchor_sphere = Sphere(self.ds.anchors[i])
            positive_sphere = Sphere(self.ds.positives[i])
            negative_sphere = Sphere(self.ds.negatives[i])

            anchor_features[i] = anchor_sphere.sampleUsingGrid(grid)
            positive_features[i] = positive_sphere.sampleUsingGrid(grid)
            negative_features[i] = negative_sphere.sampleUsingGrid(grid)

        return anchor_features, positive_features, negative_features


if __name__ == "__main__":
    ds = DataSource("/mnt/data/datasets/Spherical/training-set")
    ds.load(2)

    ts = TrainingSet(ds)
    a,p,n = ts.__getitem__(0)
    print("First anchor:\t", a.shape)
    print("First positive:\t", p.shape)
    print("First negative:\t", n.shape)
    print("Total length:\t", ts.__len__())
