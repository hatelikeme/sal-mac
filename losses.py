import torch
import torch.nn.functional as F
import torch.nn as nn

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin = 2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        s1 = label * (euclidean_distance ** 2)
        s2 = (1 - label) * (torch.clamp(self.margin - euclidean_distance, min=0.0) ** 2)
        loss = (s1 + s2)/2
        return torch.sum(loss)

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=0.7):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anc, pos, neg):
        return F.triplet_margin_loss(anc, pos, neg, margin = self.margin)