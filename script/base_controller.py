import sys

import torch
from model import Model


class BaseController(object):
    def __init__(self, bw=100, state_dict='./net_params_new_1.pkl', desc_size=128):
        sys.setrecursionlimit(50000)
        self.bw = bw
        self.desc_size = desc_size
        self.net = None
        self.state_dict = state_dict


    def init_network(self):
        print(f'Initializing the network from {self.state_dict} using a {self.bw} bandwidth.')
        self.net = Model(self.bw).cuda()
        self.net.load_state_dict(torch.load(self.state_dict))
        self.net.eval()
        print("Finished initializing the network.")

    # abstract method
    def handle_point_cloud(self, ts, cloud):
        pass
