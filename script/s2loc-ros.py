#! /usr/bin/env python2

import rospy

import argparse
import sys

import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_srvs.srv import Empty

from localization_controller import LocalizationController
from map_building_controller import MapBuildingController

class S2LocNode(object):
    def __init__(self):
        self.ctrl = None
        mode = rospy.get_param("~mode")
        bw = rospy.get_param("~bw")
        net = rospy.get_param("~net")
        descriptor_size = rospy.get_param("~descriptor_size")

        if mode == "localization":
            map_folder = rospy.get_param("~map_folder")
            if map_folder == "":
                print("ERROR: --map_folder must be specified in localization mode.")
                sys.exit(-1)
            self.ctrl = LocalizationController(
                map_folder, bw, net, descritpor_size)
        elif mode == "map-building":
            export_map_folder = rospy.get_param("~export_map_folder")
            self.ctrl = MapBuildingController(
                export_map_folder, bw, net, descriptor_size)
            self.is_detecting = False

            self.map_builder_service = rospy.Service(
                's2loc_build_map', Empty, self.build_descriptor_map)
            self.lc_service = rospy.Service(
                's2loc_detect', Empty, self.detect_lc)
            self.clear_map_service = rospy.Service(
                's2loc_clear_map', Empty, self.clear_descriptor_map)
            print("Waiting for map point clouds.")
        else:
            print("ERROR: no valid mode specified: ", mode)
            sys.exit(-1)

        pc_topic = rospy.get_param("~pc_topic")
        self.pc_sub = rospy.Subscriber(
            pc_topic, PointCloud2, self.laser_callback)

    def laser_callback(self, cloud_msg):
        if self.is_detecting:
            return
        cloud = self.__convert_msg_to_array(cloud_msg)
        print(f'Received pc with size {cloud.size}  and shape {cloud.shape}')
        self.ctrl.handle_point_cloud(cloud_msg.header.stamp, cloud)

    def detect_lc(self, request):
        self.is_detecting = True
        self.ctrl.find_loop_closures()
        self.ctrl.clear_clouds()
        self.is_detecting = False

    def build_descriptor_map(self, request):
        self.ctrl.build_descriptor_map()

    def clear_descriptor_map(self, request):
        self.ctrl.clear_descriptor_map()

    def __convert_msg_to_array(self, cloud_msg):
        points_list = []
        for data in pc2.read_points(cloud_msg, skip_nans=True):
            print('Point cloud length: ', len(data))
            points_list.append([data[0], data[1], data[2], data[3]])
        return np.array(points_list)

if __name__ == "__main__":
    rospy.init_node('S2LocNode')
    print("=== Running S2Loc Node ====================")
    try:
        s2loc = S2LocNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
