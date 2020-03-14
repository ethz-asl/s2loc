from data_source import DataSource
import numpy as np

class Sphere:
    def __init__(self, point_cloud):
        self.point_cloud = point_cloud
        self.sphere = self.__projectPointCloud(point_cloud)
        self.intensity = point_cloud[:,3]

    def getProjectedInCartesian(self):
        cart_sphere = np.empty(self.sphere.shape)
        cart_sphere[:,0] = np.multiply(np.sin(self.sphere[:,0]), np.cos(self.sphere[:,1]))
        cart_sphere[:,1] = np.multiply(np.sin(self.sphere[:,0]), np.sin(self.sphere[:,1]))
        cart_sphere[:,2] = np.cos(self.sphere[:,0])

        return cart_sphere

    def __projectPointCloud(self, cloud):
        # sqrt(x^2+y^2+z^2)
        dist = np.sqrt(cloud[:,0]**2 + cloud[:,1]**2 + cloud[:,2]**2)
        projected = np.empty([len(cloud), 3])

        # Some values might be zero or NaN, lets ignore them for now.
        with np.errstate(divide='ignore', invalid='ignore'):
            projected[:,0] = np.arccos(cloud[:,2] / dist)
            projected[:,1] = np.mod(np.arctan2(cloud[:,1], cloud[:,0]) + 2*np.pi, 2*np.pi)
            projected[:,2] = dist
        return projected


if __name__ == "__main__":
    ds = DataSource("/tmp/training")
    ds.loadAll()

    sph = Sphere(ds.anchors[0])
