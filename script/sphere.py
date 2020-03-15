from data_source import DataSource
from dh_grid import DHGrid
import numpy as np
import open3d as o3d

class Sphere:
    def __init__(self, point_cloud):
        self.point_cloud = point_cloud
        (self.sphere, self.ranges) = self.__projectPointCloudOnSphere(point_cloud)
        self.intensity = point_cloud[:,3]

    def getProjectedInCartesian(self):
        return self.__convertSphericalToEuclidean(self.sphere)

    def sampleUsingGrid(self, grid):
        cart_sphere = self.__convertSphericalToEuclidean(self.sphere)
        cart_grid = self.__convertSphericalToEuclidean(grid)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cart_sphere[:, 0:3])
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        kNearestNeighbors = 1
        n_sample_points = len(cart_grid)
        features = np.zeros([2, n_sample_points])
        for i in range(n_sample_points):
            [k, idx, _] = pcd_tree.search_knn_vector_3d(cart_grid[i,:], kNearestNeighbors)

            for nn in range(kNearestNeighbors):
                cur_idx = idx[nn]
                features[0, i] = self.ranges[cur_idx]
                features[1, i] = self.intensity[cur_idx]

        return features


    def __projectPointCloudOnSphere(self, cloud):
        # sqrt(x^2+y^2+z^2)
        dist = np.sqrt(cloud[:,0]**2 + cloud[:,1]**2 + cloud[:,2]**2)
        projected = np.empty([len(cloud), 3])
        ranges = np.empty([len(cloud), 1])

        # Some values might be zero or NaN, lets ignore them for now.
        with np.errstate(divide='ignore', invalid='ignore'):
            projected[:,0] = np.arccos(cloud[:,2] / dist)
            projected[:,1] = np.mod(np.arctan2(cloud[:,1], cloud[:,0]) + 2*np.pi, 2*np.pi)
            ranges[:,0] = dist
        return projected, ranges

    def __convertSphericalToEuclidean(self, spherical):
        cart_sphere = np.empty([len(spherical), 3])
        cart_sphere[:,0] = np.multiply(np.sin(spherical[:,0]), np.cos(spherical[:,1]))
        cart_sphere[:,1] = np.multiply(np.sin(spherical[:,0]), np.sin(spherical[:,1]))
        cart_sphere[:,2] = np.cos(spherical[:,0])
        return cart_sphere



if __name__ == "__main__":
    ds = DataSource("/tmp/training")
    ds.loadAll()

    sph = Sphere(ds.anchors[0])
    grid = DHGrid.createGrid(50)
    features = sph.sampleUsingGrid(grid)
