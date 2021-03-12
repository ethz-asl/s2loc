import os
from functools import partial

import numpy as np
import gc
import time

import open3d as o3d
import pymp
import torch.utils.data
from data_source import DataSource
from dh_grid import DHGrid
from sphere import Sphere
from tqdm.auto import tqdm, trange
from tqdm.contrib.concurrent import process_map, thread_map

# LiDARMace
T_B_L_mace = np.array(
    [[0.999776464807781,  -0.016285963261510,  0.013460141210110, -0.029098378563024],
     [0.016299962125963,   0.999865603816677,  -0.000875084243449, 0.121665163511970],
     [-0.013444131722031,   0.001094290840472,  0.999909050000742, -0.157908708175463],
     [0, 0, 0, 1]])
# LiDARStick
T_B_L_stick = np.array(
    [[0.699172, -0.7149, 0.00870821, -0.100817],
     [-0.714841, -0.699226, -0.00925541, -0.00368789],
     [0.0127057, 0.000246142, -0.999919, -0.0998847],
     [0, 0, 0, 1]])

def transformStickCloudToIMU(cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud[:, 0:3])
    pcd.transform(T_B_L_stick)
    dst = np.asarray(pcd.points)
    return np.column_stack((dst, cloud[:, 3]))

def transformMaceCloudToIMU(cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud[:, 0:3])
    pcd.transform(T_B_L_mace)
    dst = np.asarray(pcd.points)
    return np.column_stack((dst, cloud[:, 3]))


def progresser_low_res(sample, grid, auto_position=True, write_safe=False, blocking=True, progress=False):
    sample_in_B = transformStickCloudToIMU(sample)
    sample_sphere = Sphere(sample_in_B)
    return sample_sphere.sampleUsingGrid(grid)

def progresser_high_res(sample, grid, auto_position=True, write_safe=False, blocking=True, progress=False):
    sample_in_B = transformMaceCloudToIMU(sample)
    sample_sphere = Sphere(sample_in_B)
    return sample_sphere.sampleUsingGrid(grid)


class TrainingSet(torch.utils.data.Dataset):
    def __init__(self, restore, bw=100):
        self.ds = None
        self.bw = bw
        self.is_restoring = restore
        self.test_indices = []
        self.cache = None
        self.grid = DHGrid.CreateGrid(bw)
        self.anchor_features = []
        self.positive_features = []
        self.negative_features = []
    

    def generateAll(self, datasource):
        self.ds = datasource
        self.cache = datasource.cache
        print(f'Generating features from {self.ds.start_cached} to {self.ds.end_cached}')

        # Generate features from clouds.
        a_pcl_features=None
        p_pcl_features=None
        n_pcl_features=None
        if self.ds.load_negatives:
            (a, p, n) = self.ds.get_all_cached_clouds()
            a_pcl_features, p_pcl_features, n_pcl_features = self.__genAllCloudFeatures(
                    a, p, n)
        else:
            (a, p) = self.ds.get_all_cached_clouds()
            a_pcl_features, p_pcl_features, _ = self.__genAllCloudFeatures(a, p, None)

        # Copy all features to the data structure.
        double_bw = 2 * self.bw
        n_clouds = len(a_pcl_features)
        self.anchor_features = [np.zeros((3, double_bw, double_bw))] * n_clouds
        self.positive_features = [
            np.zeros((3, double_bw, double_bw))] * n_clouds
        self.negative_features = [
            np.zeros((3, double_bw, double_bw))] * n_clouds
        
        if self.ds.load_negatives:
            a_img_features, p_img_features, n_img_features = self.ds.get_all_cached_images()
            for i in range(self.ds.start_cached, self.ds.end_cached):
                self.anchor_features[i], self.positive_features[i], self.negative_features[i] = self.createFeature(
                    a_pcl_features[i], a_img_features[i], p_pcl_features[i], p_img_features[i], n_pcl_features[i], n_img_features[i])
            self.negative_features = np.array(self.negative_features)
        else:
            a_img_features, p_img_features = self.ds.get_all_cached_images()
            for i in range(self.ds.start_cached, self.ds.end_cached):
                self.anchor_features[i], self.positive_features[i] = self.createFeatureForTest(
                    a_pcl_features[i], a_img_features[i], p_pcl_features[i], p_img_features[i])

        self.anchor_features = np.array(self.anchor_features)
        self.positive_features = np.array(self.positive_features)

    def __getitem__(self, index):
        # isinstance(l[1], str)
        if (self.ds is not None):
            return self.loadFromDatasource(index)
        else:
            return self.loadFromFeatures(index)

    def loadFromDatasource(self, index):
        if (index >= self.ds.start_cached) and (index < self.ds.end_cached):
            return self.get_and_delete_torch_feature(index)

        # We reached the end of the current cached batch.
        # Free the current set and cache the next one.
        a_cloud, p_cloud, n_cloud = self.ds.load_clouds_directly(index)
        a_cloud, p_cloud, n_cloud = self.__gen_all_features_single(
            a_cloud, p_cloud, n_cloud)
        a_img, p_img, n_img = self.ds.load_images_directly(index)

        a, p, n = self.createFeature(
            a_cloud, a_img, p_cloud, p_img, n_cloud, n_img)

        return a, p, n

    def loadFromFeatures(self, index):
        return self.anchor_features[index], self.positive_features[index], self.negative_features[index]

    def createFeature(self, a_cloud, a_img, p_cloud, p_img, n_cloud, n_img):
        double_bw = 2 * self.bw
        anchor_features = np.zeros((3, double_bw, double_bw))
        positive_features = np.zeros((3, double_bw, double_bw))
        negative_features = np.zeros((3, double_bw, double_bw))

        a_img_feat = np.reshape(a_img.transpose(), (4, double_bw, double_bw))
        anchor_features[0, :, :] = np.nan_to_num(a_cloud[0, :, :])
        anchor_features[1, :, :] = np.nan_to_num(a_cloud[1, :, :])
        anchor_features[2, :, :] = np.nan_to_num(a_img_feat[3, :, :])

        p_img_feat = np.reshape(p_img.transpose(), (4, double_bw, double_bw))
        positive_features[0, :, :] = np.nan_to_num(p_cloud[0, :, :])
        positive_features[1, :, :] = np.nan_to_num(p_cloud[1, :, :])
        positive_features[2, :, :] = np.nan_to_num(p_img_feat[3, :, :])

        n_img_feat = np.reshape(n_img.transpose(), (4, double_bw, double_bw))
        negative_features[0, :, :] = np.nan_to_num(n_cloud[0, :, :])
        negative_features[1, :, :] = np.nan_to_num(n_cloud[1, :, :])
        negative_features[2, :, :] = np.nan_to_num(n_img_feat[3, :, :])
        

        return anchor_features, positive_features, negative_features
    
    def createFeatureForTest(self, a_cloud, a_img, p_cloud, p_img):
        double_bw = 2 * self.bw
        anchor_features = np.zeros((3, double_bw, double_bw))
        positive_features = np.zeros((3, double_bw, double_bw))

        a_img_feat = np.reshape(a_img.transpose(), (4, double_bw, double_bw))
        anchor_features[0, :, :] = np.nan_to_num(a_cloud[0, :, :])
        anchor_features[1, :, :] = np.nan_to_num(a_cloud[1, :, :])
        anchor_features[2, :, :] = np.nan_to_num(a_img_feat[3, :, :])

        p_img_feat = np.reshape(p_img.transpose(), (4, double_bw, double_bw))
        positive_features[0, :, :] = np.nan_to_num(p_cloud[0, :, :])
        positive_features[1, :, :] = np.nan_to_num(p_cloud[1, :, :])
        positive_features[2, :, :] = np.nan_to_num(p_img_feat[3, :, :])

        return anchor_features, positive_features

    def get_and_delete_torch_feature(self, index):
        anchor = torch.from_numpy(self.anchor_features[index])
        positive = torch.from_numpy(self.positive_features[index])
        
        if self.ds.load_negatives:
            negative = torch.from_numpy(self.negative_features[index])
            return anchor, positive, negative
        return anchor, positive

    def __len__(self):
        return len(self.anchor_features)

    def __genAllCloudFeatures(self, anchors, positives, negatives):
        elapsed_s = 0
        processed = 0
        
        print("Generating anchor spheres")
        start = time.time()
        anchor_features = process_map(
            partial(progresser_high_res, grid=self.grid), anchors, max_workers=32)
        end = time.time()
        elapsed_s = elapsed_s + (end - start)
        processed = processed + len(anchors)
        print(f'Processing time in total {elapsed_s} for {processed} anchors.')
        np.save('/home/berlukas/data/spherical/arche_low_res_big/anchor_cloud_features.npy', anchor_features)
        anchors = []

        print("Generating positive spheres")
        start = time.time()
        positive_features = process_map(
            partial(progresser_high_res, grid=self.grid), positives, max_workers=32)
        end = time.time()
        elapsed_s = elapsed_s + (end - start)
        processed = processed + len(positives)
        print(f'Processing time in total {elapsed_s} for {processed} positives.')
        np.save('/home/berlukas/data/spherical/arche_low_res_big/positive_cloud_features.npy', positive_features)
        positives = []

        if self.ds.load_negatives:
            print("Generating negative spheres")
            start = time.time()
            negative_features = process_map(
                partial(progresser_high_res, grid=self.grid), negatives, max_workers=32)
            end = time.time()
            elapsed_s = elapsed_s + (end - start)
            processed = processed + len(negatives)
            print(f'Processing time in total {elapsed_s} for {processed} negatives.')
            np.save('/home/berlukas/data/spherical/arche_low_res_big/negative_cloud_features.npy', negative_features)
            negatives = []
        else:
            negative_features = None
        print("Generated all pcl features")
        
        #anchor_features = np.load('/home/berlukas/data/spherical/arche_low_res_big/anchor_cloud_features.npy');
        #positive_features = np.load('/home/berlukas/data/spherical/arche_low_res_big/positive_cloud_features.npy');
        #negative_features = np.load('/home/berlukas/data/spherical/arche_low_res_big/negative_cloud_features.npy');
        
        if processed > 0:
            print(f'Processing time in total {elapsed_s} for {processed} items.')
            avg_s = elapsed_s / (processed)
            print(f'Processing time avg is {avg_s:.5f}')

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

    def exportGeneratedFeatures(self, export_path):
        k_anchor_features_path = export_path + '/anchor_features.npy'
        k_positive_features_path = export_path + '/positiv_featurese.npy'
        k_negative_features_path = export_path + '/negativ_featurese.npy'
        k_anchor_poses_path = export_path + '/anchor_poses.npy'
        k_positive_poses_path = export_path + '/positiv_poses.npy'
        k_negative_poses_path = export_path + '/negativ_poses.npy'

        np.save(k_anchor_features_path, self.anchor_features)
        np.save(k_positive_features_path, self.positive_features)
        np.save(k_negative_features_path, self.negative_features)

        np.save(k_anchor_poses_path, np.array(self.ds.anchor_poses))
        np.save(k_positive_poses_path, np.array(self.ds.positive_poses))
        np.save(k_negative_poses_path, np.array(self.ds.negative_poses))

    def loadFeatures(self, export_path):
        k_anchor_features_path = export_path + '/anchor_features.npy'
        k_positive_features_path = export_path + '/positiv_featurese.npy'
        k_negative_features_path = export_path + '/negativ_featurese.npy'
        k_anchor_poses_path = export_path + '/anchor_poses.npy'
        k_positive_poses_path = export_path + '/positiv_poses.npy'
        k_negative_poses_path = export_path + '/negativ_poses.npy'

        # load features
        self.anchor_features = np.load(k_anchor_features_path)
        self.positive_features = np.load(k_positive_features_path)
        self.negative_features = np.load(k_negative_features_path)

        # load poses
        anchor_poses = np.load(k_anchor_poses_path)
        positive_poses = np.load(k_positive_poses_path)
        negative_poses = np.load(k_negative_poses_path)


        return anchor_poses, positive_poses, negative_poses

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
