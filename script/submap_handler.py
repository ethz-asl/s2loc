import math
import os

import rospy
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from scipy import spatial

import torch
from base_controller import BaseController
from evaluation_set import EvaluationSet
from lc_candidate import LcCandidate
from model import Model
from reg_box import RegBox
from utils import Utils

from maplab_msgs.msg import SubmapConstraint
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField

# The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
FIELDS_XYZI = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name='i', offset=12, datatype=PointField.FLOAT32, count=1),
]

class SubmapHandler(object):
    def __init__(self):
        self.pivot_distance = 15
        self.n_nearest_neighbors = 3
        self.p_norm = 2
        self.reg_box = RegBox()

        #submap_topic = rospy.get_param("~submap_constraint_topic")
        map_topic = '/s2loc/map'
        self.map_pub = rospy.Publisher(map_topic, PointCloud2, queue_size=10)

    def publish_submaps(self, submaps):
        n_submaps = len(submaps)
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'
        map_points = np.zeros((1,4))

        for i in range(0, n_submaps):
            T_G_L = submaps[i].get_pivot_pose_LiDAR()
            submap = submaps[i].compute_dense_map()
            submap[:,3] = i
            submap_points = Utils.transform_pointcloud(submap, T_G_L)
            map_points = np.append(map_points, submap_points, axis=0)

        n_points = map_points.shape[0]
        if n_points > 1:
            map_points = map_points[1:,:]
            map_pointcloud_ros = pc2.create_cloud(header, FIELDS_XYZI, map_points)
            self.map_pub.publish(map_pointcloud_ros)
            rospy.loginfo(f"Published map with {n_points} points.")

    def compute_constraints(self, submaps):
        candidates = self.find_close_submaps(submaps)
        if np.count_nonzero(candidates) == 0:
            rospy.logerr("Unable to find any close submaps.")
            return
        return self.evaluate_candidates(submaps, candidates)

    def find_close_submaps(self, submaps):
        n_submaps = len(submaps)
        candidates = np.zeros((n_submaps, n_submaps))
        if n_submaps <= 1:
            rospy.logerr("Not enough submaps to find candidates.")
            return candidates
        submap_positions = self.get_all_positions(submaps)
        tree = spatial.KDTree(submap_positions)

        print(f"number of submaps {n_submaps}, candidates {candidates.shape}")
        print(f"submap positions {submap_positions}")
        for i in range(0, n_submaps):
            nn_dists, nn_indices = self.lookup_closest_submap(submap_positions, tree, i)
            if not nn_indices or not nn_dists:
                continue

            for nn_i in nn_indices:
                candidates[i, nn_i] = 1

        # filter duplicates by zeroing the lower triangle
        return np.triu(candidates)

    def lookup_closest_submap(self, submaps, tree, idx):
        current_submap = submaps[idx, :]
        n_neighbors = min(len(submaps), self.n_nearest_neighbors)
        nn_dists, nn_indices = tree.query(
            current_submap,
            p=self.p_norm,
            k=n_neighbors,
            distance_upper_bound=self.pivot_distance)

        # Remove self and fix output.
        nn_dists, nn_indices = Utils.fix_nn_output(n_neighbors, idx, nn_dists, nn_indices)

        print(f"nn_indices = {nn_indices}, nn_dists = {nn_dists}")

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
            submap_msgs = self.evaluate_neighbors_for(submaps, candidates, i)
            all_constraints.extend(submap_msgs)
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
                candidate_b = submaps[j]
                T_L_a_L_b = self.compute_alignment(candidate_a, candidate_b)
                msg = self.create_submap_constraint_msg(candidate_a, candidate_b, T_L_a_L_b)
                submap_msgs.append(msg)
        return submap_msgs

    def compute_alignment(self, candidate_a, candidate_b):
        rospy.loginfo("[SubmapHandler] Computing alignment")
        points_a = candidate_a.compute_dense_map()
        points_b = candidate_b.compute_dense_map()

        # Compute prior transformation.
        T_L_G_a = np.linalg.inv(candidate_a.get_pivot_pose_LiDAR())
        T_G_L_b = candidate_b.get_pivot_pose_LiDAR()
        T_L_a_L_b = np.matmul(T_L_G_a, T_G_L_b)

        # Register the submaps.
        print(f"Prior transformation: \n{T_L_a_L_b}")
        T = self.reg_box.register(points_a, points_b, T_L_a_L_b)
        print(f"Computed transformation: \n{T}")
        #self.reg_box.draw_registration_result(points_a, points_b, T)
        #self.reg_box.draw_registration_result(points_a, points_b, T_L_a_L_b)
        if self.reg_box.verify_registration_result(T, T_L_a_L_b):
            return T
        else:
            return T_L_a_L_b

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
