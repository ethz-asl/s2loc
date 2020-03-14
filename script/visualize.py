import open3d as o3d
import numpy as np
from data_source import DataSource
from sphere import Sphere

class Visualize:

    def visualizeRawPointCloud(self, sphere, jupyter = False):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sphere.point_cloud[:, 0:3])
        if jupyter:
            self.__visualizeJupyter(pcd)
        else:
            o3d.visualization.draw_geometries([pcd])
    def visualizeSphere(self, sphere, jupyter = False):
        pcd = o3d.geometry.PointCloud()
        cart_sphere = sphere.getProjectedInCartesian()
        pcd.points = o3d.utility.Vector3dVector(cart_sphere[:, 0:3])
        if jupyter:
            self.__visualizeJupyter(pcd)
        else:
            o3d.visualization.draw_geometries([pcd])


    def __visualizeJupyter(self, pcd):
        visualizer = o3d.JVisualizer()
        visualizer.add_geometry(pcd)
        visualizer.show()

if __name__ == "__main__":
    ds = DataSource('/tmp/training')
    ds.loadAll()
    viz = Visualize()
    sph = Sphere(ds.anchors[0])

    viz.visualizeRawPointCloud(sph)
