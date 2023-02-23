import torch
import torch.nn as nn

from torch.autograd.function import Function
use_cuda = torch.cuda.is_available() and True
device = torch.device("cuda" if use_cuda else "cpu")
class VariantCenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(VariantCenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        #self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average
        self.num_classes = num_classes
    def forward(self, label, feature):
        batch_size = feature.size(0)
        feature = feature.view(batch_size, -1)
        # To check the dim of centers and features
        if feature.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feature.size(1)))


        expanded_centers = self.centers.expand(batch_size, -1, -1)
        expanded_features = feature.expand(self.num_classes, -1, -1).transpose(1, 0)

        distance_centers = (expanded_features - expanded_centers).pow(2).sum(dim=-1)
        distances_same = distance_centers.gather(1, label.unsqueeze(1))
        intra_distances = distances_same.sum()
        inter_distances = distance_centers.sum().sub(intra_distances)
        epsilon = 1e-6
        # print(expanded_centers)
        labels_center_1 = []
        labels_center_2 = []
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if (i == j) or (j < i):
                    continue
                else:
                    labels_center_1.append(i)
                    labels_center_2.append(j)
        labels_center_1= torch.FloatTensor(labels_center_1).to(device)
        labels_center_2 = torch.FloatTensor(labels_center_2).to(device)
        centers_batch1 = self.centers.index_select(0, labels_center_1.long())
        centers_batch2 = self.centers.index_select(0, labels_center_2.long())

        counter_pair_center_loss = (centers_batch1- centers_batch2).pow(2).sum()
        loss = intra_distances/2.0/batch_size + 0.3 / (inter_distances + epsilon)/batch_size + 0.1 / (counter_pair_center_loss+epsilon) / batch_size
        return loss
