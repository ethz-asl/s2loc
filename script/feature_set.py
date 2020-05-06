import torch.utils.data
from data_source import DataSource
from dh_grid import DHGrid
from sphere import Sphere
import numpy as np
import pandas as pd
import glob

from tqdm.auto import tqdm, trange
from tqdm.contrib.concurrent import process_map, thread_map

class FeatureSet(torch.utils.data.Dataset):
    def __init__(self, features_path, bw=100):        
        self.bw = bw
        self.anchor_features, self.positive_features, self.negative_features = self.__loadAllFeatures(features_path)

    def __getitem__(self, index):
        anchor = torch.from_numpy(self.anchor_features[index])
        positive = torch.from_numpy(self.positive_features[index])
        negative = torch.from_numpy(self.negative_features[index])

        return anchor, positive, negative
    
    def __len__(self):
        return len(self.anchor_features)

    def __loadAllFeatures(self, features_path):
        
        path_anchor = f'{features_path}/anchor/*.csv'
        path_positive = f'{features_path}/positive/*.csv'
        path_negative = f'{features_path}/negative/*.csv'
        
        print("Loading anchor features from: ", path_anchor)
        anchor_features = self.__loadExportedFeatures(path_anchor)
        print("Loading positive features from: ", path_positive)
        positive_features = self.__loadExportedFeatures(path_positive)
        print("Loading negative features from: ", path_negative)
        negative_features = self.__loadExportedFeatures(path_negative)
        
        return anchor_features, positive_features, negative_features
        
    def __loadExportedFeatures(self, feature_path):               
        all_features = sorted(glob.glob(feature_path))
        
        n_features = len(all_features)
        features = [None] * n_features
        double_bw = 2*self.bw
        n_features = 2
        for i in tqdm(range(n_features)):
            path_csv = all_features[i]
            df = pd.read_csv(path_csv)
            features[i] = np.zeros([n_features, double_bw, double_bw])
            features[i][0] = np.array(df.intensities).reshape([double_bw,double_bw])
            features[i][1] = np.array(df.ranges).reshape([double_bw,double_bw])


        return features            