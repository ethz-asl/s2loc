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
