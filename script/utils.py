#! /usr/bin/env python3

import math
import numpy as np
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2

class Utils(object):
    @staticmethod
    def convert_pointcloud2_msg_to_array(cloud_msg):
        points_list = []
        for data in pc2.read_points(cloud_msg, skip_nans=True):
            points_list.append([data[0], data[1], data[2], data[3]])
        return np.array(points_list)

    @staticmethod
    def convert_pose_stamped_msg_to_array(pose_msg):
       position = np.array([pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z])
       orientation = np.array([pose_msg.pose.orientation.w, pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z])
       return position, orientation

    @staticmethod
    def convert_pos_quat_to_transformation(pos, quat):
        R = o3d.geometry.get_rotation_matrix_from_quaternion(quat)
        T = np.empty((4, 4))
        T[0:3, 0:3] = R
        T[0:3, 3] = pos
        T[3, :] = [0, 0, 0, 1]
        return T

    @staticmethod
    def convert_pointcloud2_msg_to_array(cloud_msg):
        points_list = []
        for data in pc2.read_points(cloud_msg, skip_nans=True):
            points_list.append([data[0], data[1], data[2], data[3]])
        return np.array(points_list)

    @staticmethod
    def transform_pointcloud(cloud, T):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud[:, 0:3])
        pcd.transform(T)
        dst = np.asarray(pcd.points)
        return np.column_stack((dst, cloud[:, 3]))

    @staticmethod
    def fix_nn_output(n_neighbors, idx, nn_dists, nn_indices):
        self_idx = np.where(nn_indices == idx)[0][0]
        nn_dists = np.delete(nn_dists, [self_idx])
        nn_indices = np.delete(nn_indices, [self_idx])

        nn_indices = [nn_indices] if n_neighbors == 1 else nn_indices
        nn_dists = [nn_dists] if n_neighbors == 1 else nn_dists

        mask = np.isnan(nn_indices.astype(float))
        mask = mask | np.isnan(nn_dists.astype(float))
        mask = mask | np.isinf(nn_indices.astype(float))
        mask = mask | np.isinf(nn_dists.astype(float))
        print(f"mask is {mask}")
        mask = np.logical_not(mask)
        print(f"not mask is {mask}")
        print(f"mask2 is {np.isinf(nn_dists.astype(float))}")
        return nn_dists[mask], nn_indices[mask]
