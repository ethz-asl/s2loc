import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from s2cnn import so3_near_identity_grid, S2Convolution, s2_near_identity_grid, SO3Convolution, so3_integrate

class Model(nn.Module):
    def __init__(self):
        super().__init__() # call the initialization function of father class (nn.Module)

        self.features = [2, 10, 16, 20, 60]
        self.bandwidths = [512, 50, 25, 15, 5] # 256, 64, 32, 16

        assert len(self.bandwidths) == len(self.features)

        grid_s2    =  s2_near_identity_grid(n_alpha=6, max_beta=np.pi/160, n_beta=1)
        grid_so3_1 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/16, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_2 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 8, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_3 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 4, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        # grid_so3_4 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 2, n_beta=1, max_gamma=2*np.pi, n_gamma=6)

        self.convolutional = nn.Sequential(
            S2Convolution(
                nfeature_in  = 2,
                nfeature_out = 10,
                b_in  = 512,
                b_out = 50,
                grid=grid_s2),
            nn.BatchNorm3d(10, affine=True),
            nn.ReLU(inplace=False),
            SO3Convolution(
                nfeature_in  =  10,
                nfeature_out = 20,
                b_in  = 50,
                b_out = 25,
                grid=grid_so3_1),
            nn.ReLU(inplace=False),
            nn.BatchNorm3d(20, affine=True),
            SO3Convolution(
                nfeature_in  = 20,
                nfeature_out = 60,
                b_in  = 25,
                b_out = 5,
                grid=grid_so3_2),
            nn.BatchNorm3d(60, affine=True),
            nn.ReLU(inplace=False),
            )

        self.linear = nn.Sequential(
            # linear 1
            nn.BatchNorm1d(60),
            nn.Linear(in_features=60,out_features=128),
            nn.ReLU(inplace=False),
            # linear 2
            nn.BatchNorm1d(128),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=False),
            # linear 3
            nn.BatchNorm1d(64),
            nn.Linear(in_features=64, out_features=32)
        )
    def forward(self, x1,x2,x3):
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
