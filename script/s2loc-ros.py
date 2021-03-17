import argparse
import sys

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_srvs.srv import Empty

from localization_controller import LocalizationController
from map_building_controller import MapBuildingController


class S2LocNode(object):
    def __init__(self, args):
        self.ctrl = None
        if args.mode == "localization":
            if args.map_folder == "":
                print("ERROR: --map_folder must be specified in localization mode.")
                sys.exit(-1)
            self.ctrl = LocalizationController(
                args.map_folder, args.bw, args.net, args.descritpor_size)
        elif args.mode == "map-building":
            self.ctrl = MapBuildingController(
                args.export_map_folder, args.bw, args.net, args.descriptor_size)
            self.is_detecting = False
            self.map_builder_service = rospy.Service(
                's2loc_build_map', Empty, self.build_descriptor_map)
            self.lc_service = rospy.Service(
                's2loc_detect', Empty, self.detect_lc)
            self.clear_map_service = rospy.Service(
                's2loc_clear_map', Empty, self.clear_descriptor_map)
            print("Waiting for map point clouds.")
        else:
            print("ERROR: no valid mode specified: ", args.mode)
            sys.exit(-1)

        rospy.init_node('S2LocNode', anonymous=True)
        self.pc_sub = rospy.Subscriber(
            args.pc_topic, PointCloud2, self.laser_callback)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, default="map-building",
                        help="Defines the mode of the node: map-building or localization.")
    parser.add_argument("--verbosity", type=int,
                        default=3, help="Verbosity level.")
    parser.add_argument("--map_folder", type=str, default="",
                        help="Defines the location of the descriptor map.")
    parser.add_argument("--pc_topic", type=str, default="/point_cloud",
                        help="Sets the topic name for the input point clouds.")
    parser.add_argument("--net", type=str, default="net_params_new_1.pkl",
                        help="Sets the file name for the network.")
    parser.add_argument("--bw", type=int, default=100,
                        help="Defines the used spherical bandwidth.")
    parser.add_argument("--descriptor_size", type=int, default=128,
                        help="Defines the size of the descriptor.")
    parser.add_argument("--export_map_folder", type=str, default="",
                        help="If defined the created descriptor map will be exported to this folder.")
    args = parser.parse_args()
    print("=== Running S2Loc Node ====================")
    try:
        s2loc = S2LocNode(args)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
