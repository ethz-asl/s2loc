from functools import partial

import numpy as np
import open3d as o3d
import pymp
import torch.utils.data
from data_source import DataSource
from dh_grid import DHGrid
from sphere import Sphere
from tqdm.auto import tqdm, trange
from tqdm.contrib.concurrent import process_map, thread_map

T_B_L = np.array(
        [[0.999776464807781,  -0.016285963261510,  0.013460141210110, -0.029098378563024],
         [0.016299962125963,   0.999865603816677,  -0.000875084243449, 0.121665163511970],
         [-0.013444131722031,   0.001094290840472,   0.999909050000742, -0.157908708175463],
         [0, 0, 0, 1]])

def transformCloudToIMU(cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud[:, 0:3])
    pcd.transform(T_B_L)
    dst = np.asarray(pcd.points)
    return np.column_stack((dst, cloud[:,3]))
    

def progresser(sample, grid, auto_position=True, write_safe=False, blocking=True, progress=False):
    sample_in_B = transformCloudToIMU(sample)
    sample_sphere = Sphere(sample_in_B)
    return sample_sphere.sampleUsingGrid(grid)


class TrainingSet(torch.utils.data.Dataset):
    def __init__(self, data_source, restore, bw=100):
        self.ds = data_source
        self.bw = bw
        self.is_restoring = restore
        self.test_indices = []
        self.cache = data_source.cache
        self.grid = DHGrid.CreateGrid(bw)

        # Generate features from clouds.
        if not restore:
            (a, p, n) = self.ds.get_all_cached_clouds()
        else:
            (a, p, n) = self.__loadTestSet()
        a_pcl_features, p_pcl_features, n_pcl_features = self.__genAllCloudFeatures(
            a, p, n)

        # Copy all features to the data structure.
        double_bw = 2 * bw
        n_clouds = self.ds.size()
        self.anchor_features = [np.zeros((3, double_bw, double_bw))] * n_clouds
        self.positive_features = [
            np.zeros((3, double_bw, double_bw))] * n_clouds
        self.negative_features = [
            np.zeros((3, double_bw, double_bw))] * n_clouds

        a_img_features, p_img_features, n_img_features = self.ds.get_all_cached_images()
        for i in range(self.ds.start_cached, self.ds.end_cached):
            self.anchor_features[i], self.positive_features[i], self.negative_features[i] = self.createFeature(
                a_pcl_features[i], a_img_features[i], p_pcl_features[i], p_img_features[i], n_pcl_features[i], n_img_features[i])

    def __getitem__(self, index):
        # isinstance(l[1], str)
        if (index >= self.ds.start_cached) and (index < self.ds.end_cached):
            a, p, n = self.get_and_delete_torch_feature(index)
            return a, p, n

        # We reached the end of the current cached batch.
        # Free the current set and cache the next one.
        a_cloud, p_cloud, n_cloud = self.ds.load_clouds_directly(index)
        a_cloud, p_cloud, n_cloud = self.__gen_all_features_single(
            a_cloud, p_cloud, n_cloud)
        a_img, p_img, n_img = self.ds.load_images_directly(index)

        a, p, n = self.createFeature(
            a_cloud, a_img, p_cloud, p_img, n_cloud, n_img)

        return a, p, n

    def createFeature(self, a_cloud, a_img, p_cloud, p_img, n_cloud, n_img):
        double_bw = 2 * self.bw
        anchor_features = np.zeros((3, double_bw, double_bw))
        positive_features = np.zeros((3, double_bw, double_bw))
        negative_features = np.zeros((3, double_bw, double_bw))

        a_img_feat = np.reshape(a_img.transpose(), (4, double_bw, double_bw))
        anchor_features[0, :, :] = a_cloud[0, :, :]
        anchor_features[1, :, :] = a_cloud[1, :, :]
        anchor_features[2, :, :] = a_img_feat[3, :, :]

        p_img_feat = np.reshape(p_img.transpose(), (4, double_bw, double_bw))
        positive_features[0, :, :] = p_cloud[0, :, :]
        positive_features[1, :, :] = p_cloud[1, :, :]
        positive_features[2, :, :] = p_img_feat[3, :, :]

        n_img_feat = np.reshape(n_img.transpose(), (4, double_bw, double_bw))
        negative_features[0, :, :] = n_cloud[0, :, :]
        negative_features[1, :, :] = n_cloud[1, :, :]
        negative_features[2, :, :] = n_img_feat[3, :, :]

        return anchor_features, positive_features, negative_features

    def get_and_delete_torch_feature(self, index):
        anchor = torch.from_numpy(self.anchor_features[index])
        positive = torch.from_numpy(self.positive_features[index])
        negative = torch.from_numpy(self.negative_features[index])

        self.anchor_features[index] = None
        self.positive_features[index] = None
        self.negative_features[index] = None

        return anchor, positive, negative

    def __len__(self):
        return len(self.ds)

    def __genAllCloudFeatures(self, anchors, positives, negatives):
        print("Generating anchor spheres")
        anchor_features = process_map(
            partial(progresser, grid=self.grid), anchors, max_workers=32)
        print("Generating positive spheres")
        positive_features = process_map(
            partial(progresser, grid=self.grid), positives, max_workers=32)
        print("Generating negative spheres")
        negative_features = process_map(
            partial(progresser, grid=self.grid), negatives, max_workers=32)

        print("Generated features")
        return anchor_features, positive_features, negative_features

    def __gen_all_features_single(self, a, p, n):
        anchor_features = progresser(a, self.grid)
        positive_features = progresser(p, self.grid)
        negative_features = progresser(n, self.grid)
        return anchor_features, positive_features, negative_features

    def __loadTestSet(self):
        with open('test_indices.txt', 'rb') as f:
            self.test_indices = np.loadtxt(f).astype(int)
            #import pdb; pdb.set_trace()

            a = [self.ds.anchors[i] for i in self.test_indices]
            p = [self.ds.positives[i] for i in self.test_indices]
            n = [self.ds.negatives[i] for i in self.test_indices]
            return (a, p, n)

    def isRestoring(self):
        return self.is_restoring


if __name__ == "__main__":
    cache = 10
    #ds = DataSource("/mnt/data/datasets/Spherical/training", cache)
    ds = DataSource("/tmp/training", 10)
    ds.load(100)
    ts = TrainingSet(ds, False)
    print("Total length of trainining set:\t", ts.__len__())

    a, p, n = ts.__getitem__(0)
    print("First anchor:\t", a.shape)
    print("First positive:\t", p.shape)
    print("First negative:\t", n.shape)

    next_idx = cache + 5
    a, p, n = ts.__getitem__(next_idx)
    print(f"{next_idx}th anchor:\t", a.shape)
    print(f"{next_idx}th positive:\t", p.shape)
    print(f"{next_idx}th negative:\t", n.shape)

    a, p, n = ts.__getitem__(1)
    print("Second anchor:\t", a.shape)
    print("Second positive:\t", p.shape)
    print("Second negative:\t", n.shape)
