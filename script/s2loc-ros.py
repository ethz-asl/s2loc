#! /usr/bin/env python3

import rospy
import copy
import argparse
import sys
from multiprocessing import Lock
import numpy as np
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2

from sensor_msgs.msg import PointCloud2, PointField
from maplab_msgs.msg import Submap, DenseNode, SubmapConstraint
from maplab_msgs.srv import PlaceLookup
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
        self.rate = rospy.Rate(0.1)

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

        # Input pointclouds
        pc_topic = rospy.get_param("~pc_topic")
        self.pc_sub = rospy.Subscriber(pc_topic, Submap, self.submap_callback)

        # Input place lookup requests
        place_lookup_topic = rospy.get_param("~place_lookup_topic")
        self.place_lookup = rospy.Service(place_lookup_topic, PlaceLookup, place_lookup_request)

        # Submap constraint output
        submap_topic = rospy.get_param("~submap_constraint_topic")
        self.submap_pub = rospy.Publisher(submap_topic, SubmapConstraint, queue_size=10)

        self.mutex = Lock()

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
        print(f'Received submap from {submap_msg.robot_name} with {len(submap_msg.nodes)} nodes and id {submap_msg.id}.')
        self.mutex.acquire()
        self.ctrl.add_submap(submap)
        self.mutex.release()

    def place_lookup_request(self, place_lookup_req):
        rospy.loginfo(f"[S2Loc-Ros] Received a place lookup request with {place_lookup_req.n_neighbors} neighbors and threshold of {place_lookup_req.confidence_threshold}.")

    def update(self):
        rospy.loginfo("Checking for updates.")
        self.mutex.acquire()
        submaps = copy.deepcopy(self.ctrl.get_submaps())
        self.mutex.release()
        self.ctrl.publish_all_submaps(submaps)
        msg = self.ctrl.compute_submap_constraints(submaps)
        if msg is not None:
            self.submap_pub.publish(msg)

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

    node = S2LocNode()
    while not rospy.is_shutdown():
        node.update()
        node.rate.sleep()
