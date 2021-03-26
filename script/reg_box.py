#! /usr/bin/env python3

import numpy as np
import open3d as o3d

class RegBox(object):

    def __init__(self):
        self.threshold = 0.02

    def register(self, points_a, points_b, T_prior=None):
        if T_prior is None:
            T_prior = np.eye(4,4)
        cloud_a = self.create_point_cloud(points_a)
        cloud_b = self.create_point_cloud(points_b)
        return self.apply_point_to_point(cloud_a, cloud_b, T_prior)


    def create_point_cloud(self, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])
        return pcd

    def apply_point_to_point(self, source, target, T_prior):
        result = o3d.registration.registration_icp(
            source, target, self.threshold, T_prior,
            o3d.registration.TransformationEstimationPointToPoint())

        return result.transformation

    def apply_point_to_plane(self, source, target, T_prior):
        result = o3d.registration.registration_icp(
            source, target, self.threshold, T_prior,
            o3d.registration.TransformationEstimationPointToPlane())

        return result.transformation

def random_test(regbox):
    points_a = np.random.rand(100,4)
    points_b = np.random.rand(100,4)
    T_reg = regbox.register(points_a, points_b)
    print(f"Registration result:\n {T_reg}")


if __name__ == "__main__":
    regbox = RegBox()
    random_test(regbox)
