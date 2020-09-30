from functools import partial

import numpy as np
import open3d as o3d
import pymp
import torch.utils.data
import os
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

        # Generate features from clouds.
        if not self.is_restoring:
            (a, p, n) = self.ds.get_all_cached_clouds()
        else:
            (a, p, n) = self.__loadTestSet()
        a_pcl_features, p_pcl_features, n_pcl_features = self.__genAllCloudFeatures(
            a, p, n)

        # Copy all features to the data structure.
        double_bw = 2 * self.bw
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
        if (self.ds is not None):
            return self.loadFromDatasource(index)
        else:
            return self.loadFromTransformedFeatures(index)
    
    def loadFromDatasource(self, index):
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
    
    def loadFromTransformedFeatures(self, index):
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

    def get_and_delete_torch_feature(self, index):
        anchor = torch.from_numpy(self.anchor_features[index])
        positive = torch.from_numpy(self.positive_features[index])
        negative = torch.from_numpy(self.negative_features[index])

        self.anchor_features[index] = None
        self.positive_features[index] = None
        self.negative_features[index] = None

        return anchor, positive, negative

    def __len__(self):
        return len(self.anchor_features)

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
    
    def exportGeneratedFeatures(self, export_path):
        k_anchor_path = export_path + '/anchor/'
        k_positive_path = export_path + '/positive/'
        k_negative_path = export_path + '/negative/'
        
        n_features = len(self.anchor_features)
        assert n_features == len(self.positive_features)
        assert n_features == len(self.negative_features)
        
        print(f'Exporting {n_features} generated features:')
        for i in tqdm(range(0, n_features)):
            self.exportAllRangeFeatures(k_anchor_path, k_positive_path, k_negative_path, i)
            self.exportAllIntensityFeatures(k_anchor_path, k_positive_path, k_negative_path, i)
            self.exportAllVisualFeatures(k_anchor_path, k_positive_path, k_negative_path, i)
        
    
    def exportAllRangeFeatures(self, anchor_feature_path, positive_feature_path, negative_feature_path, i_feature):
        anchor_range_feature_path = f'{anchor_feature_path}range{i_feature}.csv'
        positive_range_feature_path = f'{positive_feature_path}range{i_feature}.csv'
        negative_range_feature_path = f'{negative_feature_path}range{i_feature}.csv'
        self.exportFeature(anchor_range_feature_path, positive_range_feature_path, negative_range_feature_path, i_feature, 0)
    
    def exportAllIntensityFeatures(self, anchor_feature_path, positive_feature_path, negative_feature_path, i_feature):
        anchor_intensity_feature_path = f'{anchor_feature_path}intensity{i_feature}.csv'
        positive_intensity_feature_path = f'{positive_feature_path}intensity{i_feature}.csv'
        negative_intensity_feature_path = f'{negative_feature_path}intensity{i_feature}.csv'
        self.exportFeature(anchor_intensity_feature_path, positive_intensity_feature_path, negative_intensity_feature_path, i_feature, 1)
        
    def exportAllVisualFeatures(self, anchor_feature_path, positive_feature_path, negative_feature_path, i_feature):
        anchor_visual_feature_path = f'{anchor_feature_path}visual{i_feature}.csv'
        positive_visual_feature_path = f'{positive_feature_path}visual{i_feature}.csv'
        negative_visual_feature_path = f'{negative_feature_path}visual{i_feature}.csv'
        self.exportFeature(anchor_visual_feature_path, positive_visual_feature_path, negative_visual_feature_path, i_feature, 2)
            
    def exportFeature(self, anchor_feature_path, positive_feature_path, negative_feature_path, i_feature, feature_idx):
        anchor_features = self.anchor_features[i_feature][feature_idx,:,:]
        positive_features = self.positive_features[i_feature][feature_idx,:,:]
        negative_features = self.negative_features[i_feature][feature_idx,:,:]
        
        np.savetxt(anchor_feature_path, anchor_features, delimiter=',')
        np.savetxt(positive_feature_path, positive_features, delimiter=',')
        np.savetxt(negative_feature_path, negative_features, delimiter=',')
        
    def loadFeatures(self, feature_path):
        k_anchor_path = feature_path + '/anchor/'
        k_positive_path = feature_path + '/positive/'
        k_negative_path = feature_path + '/negative/'
        
        anchor_files = os.listdir(k_anchor_path)
        positive_files = os.listdir(k_positive_path)
        negative_files = os.listdir(k_negative_path)
        
        n_files = len(anchor_files)
        assert n_files == len(positive_files)
        assert n_files == len(negative_files)
        
        n_files_per_feature = int(round(n_files / 3))
        self.anchor_features = [np.zeros((3, 2*self.bw, 2*self.bw))] * n_files_per_feature
        self.positive_features = [np.zeros((3, 2*self.bw, 2*self.bw))] * n_files_per_feature
        self.negative_features = [np.zeros((3, 2*self.bw, 2*self.bw))] * n_files_per_feature
        print(f'Loading {n_files_per_feature} features from: ')
        print(f'\t Anchor: {k_anchor_path}')
        print(f'\t Positive: {k_positive_path}')
        print(f'\t Negative: {k_negative_path}')
        for i in tqdm(range(0,n_files_per_feature)):
            self.loadAllRangeFeatures(k_anchor_path, k_positive_path, k_negative_path, i)
            self.loadAllIntensityFeatures(k_anchor_path, k_positive_path, k_negative_path, i)
            self.loadAllVisualFeatures(k_anchor_path, k_positive_path, k_negative_path, i)

    def loadTransformedFeatures(self, transformed_feature_path):
        k_anchor_path = transformed_feature_path + '/anchor/'
        k_positive_path = transformed_feature_path + '/positive/'
        k_negative_path = transformed_feature_path + '/negative/'
        
        anchor_files = os.listdir(k_anchor_path)
        positive_files = os.listdir(k_positive_path)
        negative_files = os.listdir(k_negative_path)
        
        n_files = len(anchor_files)
        assert n_files == len(positive_files)
        assert n_files == len(negative_files)
        
        n_files_per_feature = int(round(n_files / 3))
        self.anchor_features = [np.zeros((3, self.bw, self.bw))] * n_files_per_feature
        self.positive_features = [np.zeros((3, self.bw, self.bw))] * n_files_per_feature
        self.negative_features = [np.zeros((3, self.bw, self.bw))] * n_files_per_feature
        
        print(f'Loading {n_files_per_feature} transformed features:')
        for i in tqdm(range(0,n_files_per_feature)):
            self.loadAllRangeTransformedFeatures(k_anchor_path, k_positive_path, k_negative_path, i)
            self.loadAllIntensityTransformedFeatures(k_anchor_path, k_positive_path, k_negative_path, i)
            self.loadAllVisualTransformedFeatures(k_anchor_path, k_positive_path, k_negative_path, i)
        
    def loadAllRangeTransformedFeatures(self, anchor_path, positive_path, negative_path, i_feature):
        anchor_range_transformed_path = f'{anchor_path}range_transformed{i_feature}.csv'
        positive_range_transformed_path = f'{positive_path}range_transformed{i_feature}.csv'
        negative_range_transformed_path = f'{negative_path}range_transformed{i_feature}.csv'
        self.loadFeature(anchor_range_transformed_path, positive_range_transformed_path, negative_range_transformed_path, i_feature, 0)
        
    def loadAllIntensityTransformedFeatures(self, anchor_path, positive_path, negative_path, i_feature):
        anchor_intensity_transformed_path = f'{anchor_path}intensity_transformed{i_feature}.csv'
        positive_intensity_transformed_path = f'{positive_path}intensity_transformed{i_feature}.csv'
        negative_intensity_transformed_path = f'{negative_path}intensity_transformed{i_feature}.csv'
        self.loadFeature(anchor_intensity_transformed_path, positive_intensity_transformed_path, negative_intensity_transformed_path, i_feature, 1)
        
    def loadAllVisualTransformedFeatures(self, anchor_path, positive_path, negative_path, i_feature):
        anchor_visual_transformed_path = f'{anchor_path}visual_transformed{i_feature}.csv'
        positive_visual_transformed_path = f'{positive_path}visual_transformed{i_feature}.csv'
        negative_visual_transformed_path = f'{negative_path}visual_transformed{i_feature}.csv'
        self.loadFeature(anchor_visual_transformed_path, positive_visual_transformed_path, negative_visual_transformed_path, i_feature, 2)
        
    def loadAllRangeFeatures(self, anchor_path, positive_path, negative_path, i_feature):
        anchor_range_path = f'{anchor_path}range{i_feature}.csv'
        positive_range_path = f'{positive_path}range{i_feature}.csv'
        negative_range_path = f'{negative_path}range{i_feature}.csv'
        self.loadFeature(anchor_range_path, positive_range_path, negative_range_path, i_feature, 0)
        
    def loadAllIntensityFeatures(self, anchor_path, positive_path, negative_path, i_feature):
        anchor_intensity_path = f'{anchor_path}intensity{i_feature}.csv'
        positive_intensity_path = f'{positive_path}intensity{i_feature}.csv'
        negative_intensity_path = f'{negative_path}intensity{i_feature}.csv'
        self.loadFeature(anchor_intensity_path, positive_intensity_path, negative_intensity_path, i_feature, 1)
        
    def loadAllVisualFeatures(self, anchor_path, positive_path, negative_path, i_feature):
        anchor_visual_path = f'{anchor_path}visual{i_feature}.csv'
        positive_visual_path = f'{positive_path}visual{i_feature}.csv'
        negative_visual_path = f'{negative_path}visual{i_feature}.csv'
        self.loadFeature(anchor_visual_path, positive_visual_path, negative_visual_path, i_feature, 2)
        
    def loadFeature(self, anchor_transformed_path, positive_transformed_path, negative_transformed_path, i_feature, feature_idx):
        self.anchor_features[i_feature][feature_idx, :, :] = np.loadtxt(anchor_transformed_path, delimiter=',')
        self.positive_features[i_feature][feature_idx, :, :] = np.loadtxt(positive_transformed_path, delimiter=',')
        self.negative_features[i_feature][feature_idx, :, :] = np.loadtxt(negative_transformed_path, delimiter=',')

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
