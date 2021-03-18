#! /usr/bin/env python3

import rospy
import numpy as np
from maplab_msgs.msg import Submap, DenseNode
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

class SubmapModel(object):
    def __init__(self):
        self.submap_ts = 0
        self.ts = []
        self.seq_nr = 0
        self.robot_name = ""
        self.id = 0
        self.poses = []
        self.pointclouds = []

    def construct_data(self, msg):
        n_data = len(msg.nodes)

        # Parse general information
        self.parse_information(msg)

        for i in range(0, n_data):
            pose, cloud = self.parse_node(msg.nodes[i])
            self.poses.append(pose)
            self.pointclouds.append(cloud)

    def parse_information(self, msg):
        ts = msg.header.stamp
        seq = msg.header.seq
        robot_name = msg.robot_name
        id = msg.id

        self.set_submap_information(ts, seq, robot_name, id)

    def set_submap_information(self, ts, seq_nr, robot_name, id):
        self.submap_ts = ts
        self.seq_nr = seq_nr
        self.robot_name = robot_name
        self.id = id

    def parse_node(self, dense_node):
        pose = self.__convert_pose_stamped_msg_to_array(dense_node.pose)
        cloud = self.__convert_pointcloud2_msg_to_array(dense_node.cloud)
        return pose,cloud


    def __convert_pointcloud2_msg_to_array(self, cloud_msg):
        points_list = []
        for data in pc2.read_points(cloud_msg, skip_nans=True):
            points_list.append([data[0], data[1], data[2], data[3]])
        return np.array(points_list)

    def __convert_pose_stamped_msg_to_array(self, pose_msg):
        poses = []
        poses.append([
            pose_msg.pose.position.x,
            pose_msg.pose.position.y,
            pose_msg.pose.position.z,
            pose_msg.pose.orientation.w,
            pose_msg.pose.orientation.x,
            pose_msg.pose.orientation.y,
            pose_msg.pose.orientation.z])
        return np.array(poses)

if __name__ == "__main__":
    rospy.init_node('foo')

    submap_msg = Submap()
    submap_msg.header.stamp = rospy.get_rostime()
    submap_msg.header.seq = 0

    pose_msg = PoseStamped()
    pose_msg.header.stamp = rospy.get_rostime()
    pose_msg.header.seq = 0
    pose_msg.pose.position.x = 1
    pose_msg.pose.position.y = 2
    pose_msg.pose.position.z = 3
    pose_msg.pose.orientation.w = 1
    pose_msg.pose.orientation.x = 0
    pose_msg.pose.orientation.y = 0
    pose_msg.pose.orientation.z = 0

    cloud_msg = PointCloud2()
    points = np.random.random_sample((3, 3))
    cloud_msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)]
    cloud_msg.data = points.tostring()

    dense_node = DenseNode()
    dense_node.pose = pose_msg
    dense_node.cloud = cloud_msg
    submap_msg.nodes.append(dense_node)

    model = SubmapModel()
    model.construct_data(submap_msg)

    print(f"Model for robot {model.robot_name} contains {len(model.poses)} poses and {len(model.pointclouds)} clouds.")
