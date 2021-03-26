import math
import os

import rospy
import numpy as np
from scipy import spatial

import torch
from base_controller import BaseController
from evaluation_set import EvaluationSet
from lc_candidate import LcCandidate
from model import Model
from reg_box import RegBox

from maplab_msgs.msg import SubmapConstraint


class SubmapHandler(object):
    def __init__(self):
        self.pivot_distance = 5
        self.n_nearest_neighbors = 3
        self.p_norm = 2
        self.reg_box = RegBox()

    def compute_constraints(self, submaps):
        candidates = self.find_close_submaps(submaps)
        if np.count_nonzero(candidates) == 0:
            rospy.logerr("Unable to find any close submaps.")
            return;
        return self.evaluate_candidates(submaps, candidates)

    def find_close_submaps(self, submaps):
        n_submaps = len(submaps)
        if n_submaps == 0:
            rospy.logerr("Empty submap array given.")
            return
        submap_positions = self.get_all_positions(submaps)
        tree = spatial.KDTree(submap_positions)

        candidates = np.zeros((n_submaps, n_submaps))
        for i in range(0, n_submaps):
            nn_dists, nn_indices = self.lookup_closest_submap(submap_positions[i, :])
            if not nn_indices or not nn_dists:
                continue

            for nn_i in nn_indices:
                candidates[i, nn_i] = 1

        # filter duplicates by zeroing the lower triangle
        return np.triu(candidates)

    def lookup_closest_submap(self, submap):
        nn_dists, nn_indices = tree.query(
            submap
            p=self.p_norm,
            k=self.n_nearest_neighbors,
            distance_upper_bound=self.pivot_distance)
        nn_dists, nn_indices = Utils.fix_nn_output(
            idx, nn_dists, nn_indices)
        nn_indices = [
            nn_indices] if self.n_nearest_neighbors == 1 else nn_indices

        return nn_dists, nn_indices

    def get_all_positions(self, submaps):
        n_submaps = len(submaps)
        if n_submaps == 0:
            return
        positions = np.empty((n_submaps, 3))
        for i in range(0, n_submaps):
            positions[i, 0:3] = np.transpose(submaps[i].get_pivot_pose_IMU()[0:3, 3])
        return positions

    def evaluate_candidates(self, submaps, candidates):
        n_submaps = len(submaps)
        if n_submaps == 0 or len(candidates) == 0:
            return
        all_constraints = []
        for i in range(0, n_submaps):
            submap_msgs = self.evaluate_neighbors(submaps, candidates, i)
            all_constraints.extend(sub_msgs)
        return all_constraints

    def evaluate_neighbors_for(self, submaps, candidates, i):
        neighbors = candidates[i,:]
        nnz = np.count_nonzero(neighbors)
        if nnz == 0:
            rospy.logerr(f"Found no neighbors for submap {i}")

        candidate_a = submaps[i]
        submap_msgs = []
        for j in range(0, len(neighbors)):
            if neighbors[j] > 0:
                candidate_a = submaps[j]
                T_L_a_L_b = self.compute_alignment(candidate_a, candidate_b)
                msg = self.create_submap_constraint_msg(candidate_a, candidate_b, T_L_a_L_b)
                submap_msgs.append(msg)
        return submap_msgs

    def compute_alignment(self, candidate_a, candidate_b):
        if len(submaps) == 0:
            return
        point_a = candidate_a.compute_dense_map()
        point_b = candidate_b.compute_dense_map()

        # Compute prior transformation.
        T_L_G_a = np.linalg.inv(T_candidate_a.get_pivot_pose_LiDAR())
        T_G_L_b = T_candidate_b.get_pivot_pose_LiDAR()
        T_L_a_L_b = np.matmul(T_L_G_a, T_G_L_b)

        # Register the submaps.
        return self.reg_box.register(points_a, points_b, T_L_a_L_b)

    def create_submap_constraint_msg(self, candidate_a, candidate_b, T_L_a_L_b):
        msg = SubmapConstraint()
        msg.id_from = candidate_a.id
        msg.timestamp_from = candidate_a.get_pivot_timestamp_ros()

        msg.id_to = candidate_b.id
        msg.timestamp_to = candidate_b.get_pivot_timestamp_ros()

        msg.T_a_b = T_L_a_L_b
        msg.header.stamp = rospy.get_rostime()
        return msg

if __name__ == "__main__":
    submap_handler = SubmapHandler()
