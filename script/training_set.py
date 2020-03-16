import torch.utils.data
from data_source import DataSource
from dh_grid import DHGrid
from sphere import Sphere

class TrainingSet(torch.utils.data.Dataset):
    def __init__(self, data_source, bw=100, training=True):
        self.ds = data_source
        self.bw = bw
        self.is_training = training

        if training:
            (a,p,n) = (self.ds.anchors_training, self.ds.positives_training, self.ds.negatives_training)
        else:
            (a,p,n) = (self.ds.anchors_test, self.ds.positives_test, self.ds.negatives_test)

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

        anchor_features = [None] * n_ds
        positive_features = [None] * n_ds
        negative_features = [None] * n_ds
        for i in range(n_ds):
            anchor_sphere = Sphere(anchors[i])
            positive_sphere = Sphere(positives[i])
            negative_sphere = Sphere(negatives[i])

            anchor_features[i] = anchor_sphere.sampleUsingGrid(grid)
            positive_features[i] = positive_sphere.sampleUsingGrid(grid)
            negative_features[i] = negative_sphere.sampleUsingGrid(grid)

        return anchor_features, positive_features, negative_features

    def isTraining(self):
        return self.is_training


if __name__ == "__main__":
    ds = DataSource("/mnt/data/datasets/Spherical/training-set")
    ds.load(2)

    ts = TrainingSet(ds)
    a,p,n = ts.__getitem__(0)
    print("First anchor:\t", a.shape)
    print("First positive:\t", p.shape)
    print("First negative:\t", n.shape)
    print("Total length:\t", ts.__len__())
