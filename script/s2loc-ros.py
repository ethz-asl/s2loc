#! /usr/bin/env python3

import rospy

import argparse
import sys

import numpy as np
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2

from sensor_msgs.msg import PointCloud2, PointField
from maplab_msgs.msg import Submap, DenseNode, SubmapConstraint
from std_srvs.srv import Empty

from localization_controller import LocalizationController
from map_building_controller import MapBuildingController
from visualize import Visualize
from submap_model import SubmapModel

class S2LocNode(object):
    def __init__(self):
        # General settings.
        self.ctrl = None
        mode = rospy.get_param("~mode")

        # Network specific settings.
        bw = rospy.get_param("~bw")
        net = rospy.get_param("~net")
        descriptor_size = rospy.get_param("~descriptor_size")

        # Configure for the selected mode.
        if mode == "localization":
            self.setup_localization_mode()
        elif mode == "map-building":
            self.setup_map_building_mode(bw, net, descriptor_size)
        else:
            print("ERROR: no valid mode specified: ", mode)
            sys.exit(-1)

        pc_topic = rospy.get_param("~pc_topic")
        self.pc_sub = rospy.Subscriber(
            pc_topic, Submap, self.submap_callback)
        submap_topic = rospy.get_param("~submap_constraint_topic")
        self.submap_pub = rospy.Publisher(submap_topic, SubmapConstraint, queue_size=10)

    def setup_localization_mode(self):
        map_folder = rospy.get_param("~map_folder")
        if map_folder == "":
            print("ERROR: --map_folder must be specified in localization mode.")
            sys.exit(-1)
        self.ctrl = LocalizationController(
            map_folder, bw, net, descritpor_size)

    def setup_map_building_mode(self, bw, net, descriptor_size):
        export_map_folder = rospy.get_param("~export_map_folder")
        self.ctrl = MapBuildingController(
            export_map_folder, bw, net, descriptor_size)
        self.is_detecting = False
        self.map_builder_service = rospy.Service('s2loc_build_map', Empty, self.build_descriptor_map)
        self.lc_service = rospy.Service('s2loc_detect', Empty, self.detect_lc)
        self.clear_map_service = rospy.Service('s2loc_clear_map', Empty, self.clear_descriptor_map)
        print("Listening for submap messages.")


    def submap_callback(self, submap_msg):
        if self.is_detecting:
            return
        submap = SubmapModel()
        submap.construct_data(submap_msg)
        submap.compute_dense_map()
        print(f'Received submap from {submap_msg.robot_name} with {len(submap_msg.nodes)} nodes.')
        self.ctrl.add_submap(submap)

        self.ctrl.publish_all_submaps()
        msgs = self.ctrl.compute_submap_constraints()
        #for msg in msgs:
            #self.submap_pub(msg)

    def detect_lc(self, request):
        self.is_detecting = True
        self.ctrl.find_loop_closures()
        self.ctrl.clear_clouds()
        self.is_detecting = False

    def build_descriptor_map(self, request):
        self.ctrl.build_descriptor_map()

    def clear_descriptor_map(self, request):
        self.ctrl.clear_descriptor_map()

if __name__ == "__main__":
    rospy.init_node('S2LocNode')
    print("=== Running S2Loc Node ====================")
    try:
        s2loc = S2LocNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
