import copy
import open3d as o3d
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

import rospy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA

from data_source import DataSource
from sphere import Sphere

class Visualize:
    def __init__(self):
        self.line_id = 0

        # Line Marker
        self.pub_line_marker = rospy.Publisher("/s2loc/submap_constraint", MarkerArray, queue_size=10)
        self.line_marker = Marker()
        self.line_marker.header.frame_id = "map"
        self.line_marker.ns = "Line" # unique ID
        self.line_marker.action = Marker().ADD
        self.line_marker.type = Marker().LINE_STRIP
        self.line_marker.lifetime = rospy.Duration(0.0)
        self.line_marker.scale.x = 0.05
        self.line_marker.id = 0

        self.submap_constraints = MarkerArray()

    def visualizeRawPointCloudFromSphere(self, sphere, jupyter = False):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sphere.point_cloud[:, 0:3])
        colors = self.__mapIntensityToRGB(sphere.intensity)
        pcd.colors = o3d.utility.Vector3dVector(colors[:,0:3])

        if jupyter:
            self.__visualizeJupyter(pcd)
        else:
            o3d.visualization.draw_geometries([pcd])

    def visualizeRawPointCloud(self, cloud, jupyter = False):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud[:, 0:3])
        colors = self.__mapIntensityToRGB(cloud[:, 3])
        pcd.colors = o3d.utility.Vector3dVector(colors[:,0:3])

        if jupyter:
            self.__visualizeJupyter(pcd)
        else:
            o3d.visualization.draw_geometries([pcd])

    def visualizeSphere(self, sphere, jupyter = False):
        pcd = o3d.geometry.PointCloud()
        cart_sphere = sphere.getProjectedInCartesian()
        self.visualizeCartesianSphere(np.column_stack((cart_sphere, sphere.intensity)))

    def visualizeCartesianSphere(self, cart_sphere, jupyter = False):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cart_sphere[:, 0:3])
        colors = self.__mapIntensityToRGB(cart_sphere[:,3])
        pcd.colors = o3d.utility.Vector3dVector(colors[:,0:3])

        if jupyter:
            self.__visualizeJupyter(pcd)
        else:
            o3d.visualization.draw_geometries([pcd])

    def __visualizeJupyter(self, pcd):
        visualizer = o3d.JVisualizer()
        visualizer.add_geometry(pcd)
        visualizer.show()

    def __mapIntensityToRGB(self, i):
        return cm.jet(plt.Normalize(min(i), max(i))(i))

    def getLineColor(self):
        result = ColorRGBA()
        result.a = 1
        result.r = 0.8
        result.g = 0.1
        result.b = 0.1
        return result

    def visualizeCandidates(self, candidate_a, candidate_b, T_a_b):
        T_G_L_a = candidate_a.get_pivot_pose_LiDAR()
        T_G_L_b = candidate_b.get_pivot_pose_LiDAR()
        #T_G_L_b = T_G_L_a * T_a_b
        point_a = T_G_L_a[0:3, 3]
        point_b = T_G_L_b[0:3, 3]
        self.visualizeLine(point_a, point_b)

    def visualizeLine(self, point_a, point_b):
        line_point_a = Point(point_a[0], point_a[1], point_a[2])
        line_point_b = Point(point_b[0], point_b[1], point_b[2])

        self.line_marker.id += 1
        line_marker = copy.deepcopy(self.line_marker)
        line_marker.header.stamp = rospy.Time.now()
        line_marker.color = self.getLineColor()

        line_marker.points[:] = []
        line_marker.points.append(line_point_a)
        line_marker.points.append(line_point_b)

        self.submap_constraints.markers.append(line_marker)
        self.pub_line_marker.publish(self.submap_constraints)

if __name__ == "__main__":
    ds = DataSource('/tmp/training')
    ds.loadAll()
    viz = Visualize()
    sph = Sphere(ds.anchors[0])

    viz.visualizeRawPointCloud(sph)
