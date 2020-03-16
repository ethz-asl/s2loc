import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True, batch_all=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        if size_average:
            if batch_all:
                return distance_positive, distance_negative, losses.sum()/(((losses > 1e-16).sum()).float()+1e-16), losses.mean()
            else:
                return distance_positive, distance_negative, losses.mean()
        else:
        	return distance_positive, distance_negative, losses.sum()

class ImprovedTripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin, alpha, margin2):
        super(ImprovedTripletLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.margin2 = margin2

    def forward(self, anchor, positive, negative, size_average=True, batch_all=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin) + self.alpha * F.relu(distance_positive - self.margin2)
        if size_average:
            if batch_all:
                return distance_positive, distance_negative, losses.sum()/(((losses > 1e-16).sum()).float()+1e-16), losses.mean()
            else:
                return distance_positive, distance_negative, losses.mean()
        else:
            return distance_positive, distance_negative, losses.sum()

if __name__ == "__main__":
    from data_source import DataSource
    from training_set import TrainingSet
    ds = DataSource('/tmp/training')
    ds.loadAll()
    ts = TrainingSet(ds)
    a,p,n = ts.__getitem__(0)

    loss = TripletLoss(1.0)
    dp, dn, sum = loss.forward(a, p, n, False)
    print("Distance positive: \t", dp)
    print("Distance negative: \t", dn)
    print("loss sum: \t", sum)
