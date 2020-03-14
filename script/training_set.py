import torch.utils.data
from data_source import DataSource

class TrainingSet(torch.utils.data.Dataset):
    def __init__(self, data_source):
        self.ds = data_source

    def __getitem__(self, index):
        anchor = torch.from_numpy(self.ds.anchors[index])
        positive = torch.from_numpy(self.ds.positives[index])
        negative = torch.from_numpy(self.ds.negatives[index])
        return anchor, positive, negative

    def __len__(self):
        return len(self.ds.anchors)

if __name__ == "__main__":
    ds = DataSource("/tmp/training")
    ds.loadAll()

    ts = TrainingSet(ds)
    a,p,n = ts.__getitem__(0)
    print("First anchor:\t", a.shape)
    print("First positive:\t", p.shape)
    print("First negative:\t", n.shape)
    print("Total length:\t", ts.__len__())
