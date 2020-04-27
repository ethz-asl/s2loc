import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from s2cnn import so3_near_identity_grid, S2Convolution, s2_near_identity_grid, SO3Convolution, so3_integrate

class Model(nn.Module):
    def __init__(self, bandwidth=100):
        super().__init__() # call the initialization function of father class (nn.Module)

        self.features = [2, 10, 20, 60, 100, 200]
        self.bandwidths = [bandwidth, 50, 25, 20, 10, 5] 

        assert len(self.bandwidths) == len(self.features)

        grid_s2    =  s2_near_identity_grid(n_alpha=6, max_beta=np.pi/160, n_beta=1)
        grid_so3_1 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/16, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_2 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 8, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_3 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 4, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_4 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 2, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        # grid_so3_4 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 2, n_beta=1, max_gamma=2*np.pi, n_gamma=6)

        self.convolutional = nn.Sequential(
            S2Convolution(
                nfeature_in  = self.features[0],
                nfeature_out = self.features[1],
                b_in  = self.bandwidths[0],
                b_out = self.bandwidths[1],
                grid=grid_s2),
            nn.ReLU(inplace=False),
            nn.BatchNorm3d(self.features[1], affine=True),
            SO3Convolution(
                nfeature_in  = self.features[1],
                nfeature_out = self.features[2],
                b_in  = self.bandwidths[1],
                b_out = self.bandwidths[2],
                grid=grid_so3_1),
            nn.ReLU(inplace=False),
            nn.BatchNorm3d(self.features[2], affine=True),
            SO3Convolution(
                nfeature_in  =  self.features[2],
                nfeature_out = self.features[3],
                b_in  = self.bandwidths[2],
                b_out = self.bandwidths[3],
                grid=grid_so3_2),
            nn.ReLU(inplace=False),
            nn.BatchNorm3d(self.features[3], affine=True),
            SO3Convolution(
                nfeature_in  = self.features[3],
                nfeature_out = self.features[4],
                b_in  = self.bandwidths[3],
                b_out = self.bandwidths[4],
                grid=grid_so3_3),
            nn.BatchNorm3d(self.features[4], affine=True),
            nn.ReLU(inplace=False),
            SO3Convolution(
                nfeature_in  = self.features[4],
                nfeature_out = self.features[5],
                b_in  = self.bandwidths[4],
                b_out = self.bandwidths[5],
                grid=grid_so3_4),
            nn.BatchNorm3d(self.features[5], affine=True),
            nn.ReLU(inplace=False),
            )


        self.linear = nn.Sequential(
            # linear 1
            nn.BatchNorm1d(self.features[5]),
            nn.Linear(in_features=self.features[5],out_features=512),
            nn.ReLU(inplace=False),
            # linear 2
            nn.BatchNorm1d(512),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(inplace=False),
            # linear 3
            nn.BatchNorm1d(256),
            nn.Linear(in_features=256, out_features=128)
        )

    def forward(self, x1, x2, x3):
        x1 = self.convolutional(x1)  # [batch, feature, beta, alpha, gamma]
        output1 = so3_integrate(x1)  # [batch, feature]
        output1 = self.linear(output1)

        x2 = self.convolutional(x2)  # [batch, feature, beta, alpha, gamma]
        output2 = so3_integrate(x2)  # [batch, feature]
        output2 = self.linear(output2)

        x3 = self.convolutional(x3)  # [batch, feature, beta, alpha, gamma]
        output3 = so3_integrate(x3)  # [batch, feature]
        output3 = self.linear(output3)

        return output1,output2,output3
