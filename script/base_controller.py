import sys

import torch
from model import Model


class BaseController(object):
    def __init__(self, bw=100, state_dict='./net_params_new_1.pkl', desc_size=128):
        sys.setrecursionlimit(50000)
        self.bw = bw
        self.desc_size = desc_size

        print(f'Initializing the network from {state_dict} using a {bw} bandwidth.')
        self.net = Model(bw).cuda()
        self.net.load_state_dict(torch.load(state_dict))
        self.net.eval()
        print("Finished initializing the network.")

    # abstract method
    def handle_point_cloud(self, ts, cloud):
        pass

    def fix_nn_output(self, idx, nn_dists, nn_indices):
        self_idx = nn_indices.index(idx)
        del nn_indices[self_idx]
        del nn_dists[self_idx]

        fixed_nn_dists = [dists for dists in nn_dists if not (
            math.isnan(dists) or math.isinf(dists))]
        fixed_nn_indices = [i for i in nn_indices if not (
            math.isnan(i) or math.isinf(i))]
        return fixed_nn_dists, fixed_nn_indices
