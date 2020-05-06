from data_source import DataSource
from dh_grid import DHGrid
import numpy as np
from scipy import spatial
import sys
from tqdm.auto import tqdm

class Sphere:
    def __init__(self, point_cloud):
        self.point_cloud = point_cloud
        (self.sphere, self.ranges) = self.__projectPointCloudOnSphere(point_cloud)
        self.intensity = point_cloud[:,3]

    def getProjectedInCartesian(self):
        return self.__convertSphericalToEuclidean(self.sphere)
    
    def sampleUsingGrid(self, grid):
        cart_sphere = self.__convertSphericalToEuclidean(self.sphere)
        cart_grid = DHGrid.ConvertGridToEuclidean(grid)

        sys.setrecursionlimit(50000)
        sphere_tree = spatial.cKDTree(cart_sphere[:,0:3])
        p_norm = 2
        n_nearest_neighbors = 1
        features = np.zeros([2, grid.shape[1], grid.shape[2]])
        for i in range(grid.shape[1]):
            for j in range(grid.shape[2]):
                nn_dists, nn_indices = sphere_tree.query(cart_grid[:, i, j], p = p_norm, k = n_nearest_neighbors)            
                nn_indices = [nn_indices] if n_nearest_neighbors == 1 else nn_indices
                
                # TODO(lbern): Average over all neighbors
                for cur_idx in nn_indices:
                    features[0, i, j] = self.ranges[cur_idx]
                    features[1, i, j] = self.intensity[cur_idx]

        return features
    
    def sampleUsingGrid2(self, grid):
        cart_sphere = self.__convertSphericalToEuclidean(self.sphere)
        cart_grid = DHGrid.ConvertGridToEuclidean(grid)

        sys.setrecursionlimit(50000)
        sphere_tree = spatial.cKDTree(cart_sphere[:,0:3])
        p_norm = 2
        n_nearest_neighbors = 1
        features = np.zeros([2, grid.shape[1], grid.shape[2]])
        for i in range(grid.shape[1]):
            for j in range(grid.shape[2]):
                nn_dists, nn_indices = sphere_tree.query(cart_grid[:, i, j], p = p_norm, k = n_nearest_neighbors)            
                nn_indices = [nn_indices] if n_nearest_neighbors == 1 else nn_indices
                
                # TODO(lbern): Average over all neighbors
                for cur_idx in nn_indices:
                    features[0, i, j] = self.ranges[cur_idx]
                    features[1, i, j] = self.intensity[cur_idx]

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
        cart_sphere = np.zeros([len(spherical), 3])
        cart_sphere[:,0] = np.multiply(np.sin(spherical[:,0]), np.cos(spherical[:,1]))
        cart_sphere[:,1] = np.multiply(np.sin(spherical[:,0]), np.sin(spherical[:,1]))
        cart_sphere[:,2] = np.cos(spherical[:,0])
        mask = np.isnan(cart_sphere)
        cart_sphere[mask] = 0
        return cart_sphere




if __name__ == "__main__":
    ds = DataSource("/media/scratch/berlukas/spherical/training")
    ds.load(10)

    sph = Sphere(ds.anchors[0])
    grid = DHGrid.CreateGrid(50)
    features = sph.sampleUsingGrid2(grid)
    print("features: ", features.shape)
