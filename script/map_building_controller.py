import math
import os

import numpy as np
from scipy import spatial

import torch
from base_controller import BaseController
from evaluation_set import EvaluationSet
from lc_candidate import LcCandidate
from model import Model
from lc_handler import LcHandler
from submap_handler import SubmapHandler


class MapBuildingController(BaseController):
    def __init__(self, export_map_folder="", bw=100, state_dict='./net_params_new_1.pkl', desc_size=128):
        super().__init__(bw, state_dict, desc_size)
        self.descriptors = None
        self.submaps = {}
        self.export_map_folder = export_map_folder

        self.alignment_engine = SubmapHandler()
        self.lc_engine = LcHandler()

    def add_submap(self, submap):
        id = submap.id
        self.submaps[id] = submap

    def clear_clouds(self):
        self.submaps = {}

    def get_submaps(self):
        return list(self.submaps.values())

    # --- SUBMAP CONSTRAINTS -------------------------------------------------

    def compute_submap_constraints(self, submaps):
        n_submaps = len(submaps)
        if n_submaps == 0:
            return None
        print(f"Computing constraints for {n_submaps} submaps.")
        return self.alignment_engine.compute_constraints(submaps)

    def publish_all_submaps(self, submaps):
        self.alignment_engine.publish_submaps(submaps)

    # --- SUBMAP DESCRIPTORS --------------------------------------------------

    def build_descriptor_map(self):
        print("Building feature...")
        eval_set = EvaluationSet(submaps, self.bw)

        '''
        self.descriptors = self.describe_all_point_clouds(self.submaps)
        if self.export_map_folder is not "":
            self.export_descriptors_to_folder(
                self.export_map_folder, descriptors)
        '''

    def find_loop_closures(self):
        if self.descriptors is None:
            print("ERROR: descriptor map is empty.")
            return False
        return self.lc_engine.find_loop_closures(self.descriptors)

    def describe_all_point_clouds(self, submaps):
        eval_set = EvaluationSet(submaps, self.bw)
        loader = torch.utils.data.DataLoader(
            eval_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)

        n_clouds = len(clouds)
        embeddings = np.empty(1)
        for batch_idx, data in enumerate(loader):
            data = data.cuda().float()
            embedded, _, _ = self.net(data, data, data)
            embeddings = np.append(
                embeddings, embedded.cpu().data.numpy().reshape([1, -1]))

        descriptors = embeddings[1:].reshape([n_clouds, self.desc_size])
        return descriptors

    def export_descriptors_to_folder(self, map_folder, descriptors):
        files = os.listdir(map_folder)
        next_index = len(files) + 1
        descriptor_filename = f'{map_folder}/descriptors{next_index}.csv'
        timestamp_filename = f'{map_folder}/timestamps{next_index}.csv'
        ts_arr = np.array(self.timestamps)

        np.savetxt(descriptor_filename, descriptors, delimiter=',')
        np.savetxt(timestamp_filename, ts_arr, delimiter=',')


if __name__ == "__main__":
    ctrl = MapBuildingController()
    ctrl.handle_point_cloud(None)
