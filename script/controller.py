import sys
import numpy as np
from scipy import spatial

from model import Model


class Controller(object):
    def __init__(self, bw = 100, state_dict = './net_params_new_1.pkl'):
        sys.setrecursionlimit(50000)
        self.search_tree = None
        self.descriptors = []
        self.clouds = []
        self.timestamps = []
        self.net = Model(bw).cuda()
        net.load_state_dict(torch.load(state_dict))

    def handle_point_cloud(self, ts, cloud):
        # build descriptor list
        self.timestamps.append(ts)
        self.clouds.append(cloud)

    def find_loop_closures(self):
        # build search tree
        pass

    def describe_all_point_clouds(self, cloud):

        pass

if __name__ == "__main__":
    ctrl = Controller()
    ctrl.handle_point_cloud(1, None)
