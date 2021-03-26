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
