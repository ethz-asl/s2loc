#! /usr/bin/env python3

import rospy

import argparse
import sys

import numpy as np
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2

from sensor_msgs.msg import PointCloud2, PointField
from maplab_msgs.msg import Submap, DenseNode
from std_srvs.srv import Empty

from localization_controller import LocalizationController
from map_building_controller import MapBuildingController
from visualize import Visualize

class S2LocNode(object):
    def __init__(self):
        self.ctrl = None
        mode = rospy.get_param("~mode")
        bw = rospy.get_param("~bw")
        net = rospy.get_param("~net")
        descriptor_size = rospy.get_param("~descriptor_size")

        # Configure for the selected mode.
        if mode == "localization":
            self.setup_localization_mode()
        elif mode == "map-building":
            self.setup_map_building_mode()
        else:
            print("ERROR: no valid mode specified: ", mode)
            sys.exit(-1)

        pc_topic = rospy.get_param("~pc_topic")
        self.pc_sub = rospy.Subscriber(
            pc_topic, Submap, self.submap_callback)

    def setup_localization_mode(self):
        map_folder = rospy.get_param("~map_folder")
        if map_folder == "":
            print("ERROR: --map_folder must be specified in localization mode.")
            sys.exit(-1)
        self.ctrl = LocalizationController(
            map_folder, bw, net, descritpor_size)

    def setup_map_building_mode(self):
        export_map_folder = rospy.get_param("~export_map_folder")
        #self.ctrl = MapBuildingController(
            #export_map_folder, bw, net, descriptor_size)
        self.is_detecting = False
        self.map_builder_service = rospy.Service(
            's2loc_build_map', Empty, self.build_descriptor_map)
        self.lc_service = rospy.Service(
            's2loc_detect', Empty, self.detect_lc)
        self.clear_map_service = rospy.Service(
            's2loc_clear_map', Empty, self.clear_descriptor_map)
        print("Listening for submap messages.")


    def submap_callback(self, submap_msg):
        if self.is_detecting:
            return
        cloud = self.__convert_submap_msg_to_array(submap_msg)
        print(f'Received submap from {submap_msg.robot_name} with {len(submap_msg.nodes)} nodes.')
        #self.ctrl.handle_point_cloud(cloud_msg.header.stamp, cloud)

    def detect_lc(self, request):
        self.is_detecting = True
        self.ctrl.find_loop_closures()
        self.ctrl.clear_clouds()
        self.is_detecting = False

    def build_descriptor_map(self, request):
        self.ctrl.build_descriptor_map()

    def clear_descriptor_map(self, request):
        self.ctrl.clear_descriptor_map()


    def __convert_submap_msg_to_array(self, cloud_msg):
        n_nodes = len(cloud_msg.nodes)
        if n_nodes == 0:
            return

        T_B_L = np.array(
            [[1, 0, 0, 0.005303],
             [0, 1, 0, 0.037340],
             [0, 0, 1, 0.063319],
             [0, 0, 0, 1]])

        pivot = n_nodes // 2
        pivot_node = cloud_msg.nodes[pivot]
        pivot_pos, pivot_quat = self.__convert_pose_stamped_msg_to_array(pivot_node.pose)
        T_G_B_pivot = self.__convert_pos_quat_to_transformation(pivot_pos, pivot_quat)
        T_G_L_pivot = T_G_B_pivot * T_B_L
        T_L_pivot_G = np.linalg.inv(T_G_L_pivot)
        pivot_points = self.__convert_pointcloud2_msg_to_array(pivot_node.cloud)
        print(f"pivot points shape {pivot_points.shape}")
        print(f"T_L_pivot_G {T_L_pivot_G}")

        viz = Visualize()
        viz.visualizeRawPointCloud(pivot_points)

        '''
        # Nodes to the left of the pivot.
        for i in range(0, n_nodes):
            if i == pivot:
                continue

            node = cloud_msg.nodes[i]
            pos, quat = self.__convert_pose_stamped_msg_to_array(node.pose)
            T_G_B = self.__convert_pos_quat_to_transformation(pos, quat)
            T_G_L = T_G_B * T_B_L
            T_L_pivot_L = T_L_pivot_G * np.linalg.inv(T_G_L)

            points = self.__convert_pointcloud2_msg_to_array(node.cloud)
            points = self.transform_pointcloud(points, T_L_pivot_L)
        '''



    def __convert_pose_stamped_msg_to_array(self, pose_msg):
       position = np.array([pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z])
       orientation = np.array([pose_msg.pose.orientation.w, pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z])
       return position, orientation

    def __convert_pos_quat_to_transformation(self, pos, quat):
        R = o3d.geometry.get_rotation_matrix_from_quaternion(quat)
        T = np.empty((4, 4))
        T[0:3, 0:3] = R
        T[0:3, 3] = pos
        T[3, :] = [0, 0, 0, 1]
        return T

    def __convert_pointcloud2_msg_to_array(self, cloud_msg):
        points_list = []
        for data in pc2.read_points(cloud_msg, skip_nans=True):
            points_list.append([data[0], data[1], data[2], data[3]])
        return np.array(points_list)

    def transform_pointcloud(cloud, T):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud[:, 0:3])
        pcd.transform(T)
        dst = np.asarray(pcd.points)
        return np.column_stack((dst, cloud[:, 3]))


if __name__ == "__main__":
    rospy.init_node('S2LocNode')
    print("=== Running S2Loc Node ====================")
    try:
        s2loc = S2LocNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
