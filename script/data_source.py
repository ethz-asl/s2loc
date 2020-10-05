import csv
import glob
from functools import partial
from os import listdir, path

import numpy as np

from plyfile import PlyData, PlyElement
from tqdm.auto import tqdm, trange
from tqdm.contrib.concurrent import process_map, thread_map


def progresser(ply_file, auto_position=True, write_safe=False, blocking=True, progress=False):
    try:
        plydata = PlyData.read(ply_file)
        vertex = plydata['vertex']
        x = vertex['x']
        y = vertex['y']
        z = vertex['z']
        if 'scalar' in vertex._property_lookup:
            i = vertex['scalar']
        elif 'intensity' in vertex._property_lookup:
            i = vertex['intensity']
        else:
            i = plydata['vertex'][plydata.elements[0].properties[3].name]
        return np.concatenate((x, y, z, i), axis=0).reshape(4, len(x)).transpose()
    except Exception as e:
        print(f'ERROR reading file {ply_file}.')
        print(e)
        return np.empty(4, 1)


class DataSource:
    def __init__(self, path_to_datasource, cache=-1, skip_nth=-1):
        self.datasource = path_to_datasource
        self.anchors = None
        self.anchor_sph_images = None
        self.positives = None
        self.positive_sph_images = None
        self.negatives = None
        self.negative_sph_images = None
        self.ds_total_size = 0
        self.cache = cache
        self.start_cached = 0
        self.end_cached = 0
        self.skip_nth = skip_nth
        self.all_anchor_files = []
        self.all_anchor_image_files = []
        self.all_positive_files = []
        self.all_positive_image_files = []
        self.all_negative_files = []
        self.all_negative_image_files = []
        self.anchor_poses = []
        self.positive_poses = []
        self.negative_poses = []

    def load(self, n=-1, indices=None, filter_clusters=False):
        path_anchor = self.datasource + "/training_anchor_pointclouds/"
        path_anchor_images = self.datasource + "/training_anchor_sph_images/"
        path_positives = self.datasource + "/training_positive_pointclouds/"
        path_positive_images = self.datasource + "/training_positive_sph_images/"
        path_negatives = self.datasource + "/training_negative_pointclouds/"
        path_negative_images = self.datasource + "/training_negative_sph_images/"
        path_anchor_poses = self.datasource + "/anchor-poses.csv"
        path_positive_poses = self.datasource + "/positive-poses.csv"
        path_negative_poses = self.datasource + "/negative-poses.csv"

        # Prepare paths to PLYs and poses.
        self.all_anchor_files = sorted(glob.glob(path_anchor + '*.ply'))
        self.all_anchor_image_files = sorted(
            glob.glob(path_anchor_images + '*.ply'))
        self.all_positive_files = sorted(glob.glob(path_positives + '*.ply'))
        self.all_positive_image_files = sorted(
            glob.glob(path_positive_images + '*.ply'))
        self.all_negative_files = sorted(glob.glob(path_negatives + '*.ply'))
        self.all_negative_image_files = sorted(
            glob.glob(path_negative_images + '*.ply'))

        # Reads [ts qw qx qy qz, x y z].
        self.anchor_poses = self.loadPoses(path_anchor_poses, n)
        self.positive_poses = self.loadPoses(path_positive_poses, n)
        self.negative_poses = self.loadPoses(path_negative_poses, n)
        if indices is not None:
            self.all_anchor_files, self.all_anchor_image_files = self.filterFiles(
                self.all_anchor_files, self.all_anchor_image_files, n, indices)
            self.all_positive_files, self.all_positive_image_files = self.filterFiles(
                self.all_positive_files, self.all_positive_image_files, n, indices)
            self.all_negative_files, self.all_negative_image_files = self.filterFiles(
                self.all_negative_files, self.all_negative_image_files, n, indices)
            self.anchor_poses, self.positive_poses, self.negative_poses = self.filterPoses(
                self.anchor_poses, self.positive_poses, self.negative_poses, n, indices)
        if filter_clusters:
            non_clustered_indices = self.filterClusters(
                self.anchor_poses, self.positive_poses)
            self.all_anchor_files, self.all_anchor_image_files = self.filterFiles(
                self.all_anchor_files, self.all_anchor_image_files, n, non_clustered_indices)
            self.all_positive_files, self.all_positive_image_files = self.filterFiles(
                self.all_positive_files, self.all_positive_image_files, n, non_clustered_indices)
            self.all_negative_files, self.all_negative_image_files = self.filterFiles(
                self.all_negative_files, self.all_negative_image_files, n, non_clustered_indices)
            self.anchor_poses, self.positive_poses, self.negative_poses = self.filterPoses(
                self.anchor_poses, self.positive_poses, self.negative_poses, n, non_clustered_indices)

        print(f"Loading anchors from:\t{path_anchor} and {path_anchor_images}")
        self.anchors = self.loadDataset(self.all_anchor_files, n, self.cache)
        self.anchor_sph_images = self.loadDataset(
            self.all_anchor_image_files, n, self.cache)

        print(f"Loading positives from:\t{path_positives} and {path_positive_images}")
        self.positives = self.loadDataset(
            self.all_positive_files, n, self.cache)
        self.positive_sph_images = self.loadDataset(
            self.all_positive_image_files, n, self.cache)

        print(f"Loading negatives from:\t{path_negatives} and {path_negative_images}")
        self.negatives = self.loadDataset(
            self.all_negative_files, n, self.cache)
        self.negative_sph_images = self.loadDataset(
            self.all_negative_image_files, n, self.cache)

        print("Done loading dataset.")
        print(f"\tAnchor point clouds total: \t{len(self.anchors)}")
        print(f"\tAnchor images total: \t\t{len(self.anchor_sph_images)}")
        print(f"\tAnchor poses total: \t\t{len(self.anchor_poses)}")
        print(f"\tPositive point clouds total: \t{len(self.positives)}")
        print(f"\tPositive images total: \t\t{len(self.positive_sph_images)}")
        print(f"\tPositive poses total: \t\t{len(self.positive_poses)}")
        print(f"\tNegative point clouds total: \t{len(self.negatives)}")
        print(f"\tNegative images total: \t\t{len(self.negative_sph_images)}")
        print(f"\tNegative poses total: \t\t{len(self.negative_poses)}")

    def filterClusters(self, anchor_poses, positive_poses):
        n_poses = len(anchor_poses)
        assert n_poses == len(positive_poses)
        k_dist_threshold = 4.5
        non_clustered = [anchor_poses[0, 5:8]]
        non_clustered_indices = [0]
        for i in range(1, n_poses):
            curr_pose = anchor_poses[i, 5:8]
            diff = np.subtract(np.array(non_clustered), curr_pose)
            distances = np.linalg.norm(diff, axis=1)
            min_dist = np.amin(distances)
            if min_dist < k_dist_threshold:
                continue
            min_idx = np.where(distances == min_dist)
            print(f'Minimum distance is {min_dist} at index {min_idx[0][0]} for sample {i}')
            non_clustered.append(curr_pose)
            non_clustered_indices.append(i)
        assert len(non_clustered) == len(non_clustered_indices)
        return np.array(non_clustered_indices)

    def filterPoses(self, a_poses, p_poses, n_poses, n_data, indices):
        idx = indices[indices < n_data]
        return a_poses[idx], p_poses[idx], n_poses[idx]

    def filterFiles(self, cloud_files, img_files, n_data, indices):
        k = 0
        n_indices = len(indices)
        cloud_files_filtered = [None] * n_indices
        img_files_filtered = [None] * n_indices
        for i in indices:
            if i >= n_data:
                continue
            cloud_files_filtered[k] = cloud_files[i]
            img_files_filtered[k] = img_files[i]
            k = k + 1

        return list(filter(None, cloud_files_filtered)), list(filter(None, img_files_filtered))

    def loadDataset(self, all_files, n, cache):
        idx = 0
        self.ds_total_size = len(all_files)
        n_ds = min(self.ds_total_size, n) if n > 0 else self.ds_total_size
        cache = min(n_ds, cache)
        dataset = [None] * n_ds
        skipping = 0
        '''
        for i in tqdm(range(0, n_ds)):
            ply_file = all_files[i]

            if self.skip_nth != -1 and skipping > 0 and skipping <= self.skip_nth:
                skipping = skipping + 1
                continue

            dataset[idx] = self.loadPointCloudFromPath(
                ply_file) if idx < cache else ply_file
            idx = idx + 1
            skipping = 1
        self.end_cached = cache
        if self.skip_nth != -1:
            dataset = list(filter(None.__ne__, dataset))
        '''
        dataset = process_map(
            partial(progresser), all_files[0:n_ds], max_workers=32)
        self.end_cached = n_ds

        return dataset

    def loadPoses(self, path_to_poses, n):
        if not path.exists(path_to_poses):
            return []
        poses = np.genfromtxt(path_to_poses, delimiter=',')
        n_poses_read = len(poses)
        n_poses_to_process = min(n_poses_read, n) if n > 0 else n_poses_read
        return poses[0:n_poses_to_process, :]

    def loadDatasetPathOnly(self, path_to_dataset, n):
        all_files = sorted(glob.glob(path_to_dataset + '*.ply'))
        n_ds = min(n_files, n) if n > 0 else n_files
        dataset = all_files[:, n_ds]
        return dataset

    def loadPointCloudFromPath(self, path_to_point_cloud):
        # print(f'Loading point cloud from {path_to_point_cloud}')
        plydata = PlyData.read(path_to_point_cloud)
        vertex = plydata['vertex']
        x = vertex['x']
        y = vertex['y']
        z = vertex['z']
        if 'scalar' in vertex._property_lookup:
            i = vertex['scalar']
        elif 'intensity' in vertex._property_lookup:
            i = vertex['intensity']
        else:
            i = plydata['vertex'][plydata.elements[0].properties[3].name]

        return np.concatenate((x, y, z, i), axis=0).reshape(4, len(x)).transpose()

    def writePointCloudToPath(self, cloud, path_to_point_cloud):
        types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('i', 'f4')]
        vertex = np.array(cloud, types)
        import pdb
        pdb.set_trace()
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el], text=True).write(path_to_point_cloud)

    def writeFeatureCloudToPath(self, cloud, path_to_point_cloud):
        types = [('x', 'f4'), ('y', 'f4')]
        vertex = np.array(cloud, types)
        import pdb
        pdb.set_trace()
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el], text=True).write(path_to_point_cloud)

    def size(self):
        return len(self.anchors)

    def __len__(self):
        return self.size()

    def cache_next(self, index):
        prev_end = self.end_cached
        self.end_cached = min(self.size(), index + self.cache)
        for idx in range(prev_end, self.end_cached):
            self.anchors[idx], self.positives[idx], self.negatives[idx] = self.load_clouds_directly(
                idx)
            self.anchor_sph_images[idx], self.positive_sph_images[idx], self.negative_sph_images[idx] = self.load_images_directly(
                idx)
        return prev_end, self.end_cached

    def free_to_start_cached(self):
        for idx in range(0, self.start_cached):
            self.anchors[idx] = self.all_anchor_files[idx]
            self.positives[idx] = self.all_positive_files[idx]
            self.negatives[idx] = self.all_negative_files[idx]
            self.anchor_sph_images[idx] = self.all_anchor_image_files[idx]
            self.positive_sph_images[idx] = self.all_positive_image_files[idx]
            self.negative_sph_images[idx] = self.all_negative_image_files[idx]

    def get_all_cached_clouds(self):
        return self.get_cached_clouds(self.start_cached, self.end_cached)

    def get_cached_clouds(self, start, end):
        assert start <= end
        start = max(0, start)
        end = min(self.ds_total_size, end)

        return self.anchors[start:end], \
            self.positives[start:end], \
            self.negatives[start:end]

    def get_all_cached_images(self):
        return self.get_cached_images(self.start_cached, self.end_cached)

    def get_cached_images(self, start, end):
        assert start <= end
        start = max(0, start)
        end = min(self.ds_total_size, end)
        return self.anchor_sph_images[start:end], \
            self.positive_sph_images[start:end], \
            self.negative_sph_images[start:end]

    def load_clouds_directly(self, idx):
        # print(f'Requesting direct index {idx} of size {len(self.anchors)}')
        anchor = self.loadPointCloudFromPath(self.anchors[idx]) if isinstance(
            self.anchors[idx], str) else self.anchors[idx]
        positive = self.loadPointCloudFromPath(self.positives[idx]) if isinstance(
            self.positives[idx], str) else self.positives[idx]
        negative = self.loadPointCloudFromPath(self.negatives[idx]) if isinstance(
            self.negatives[idx], str) else self.negatives[idx]
        return anchor, positive, negative

    def load_images_directly(self, idx):
        # print(f'Requesting direct index {idx} of size {len(self.anchors)}')
        anchor = self.loadPointCloudFromPath(self.anchor_sph_images[idx]) if isinstance(
            self.anchor_sph_images[idx], str) else self.anchor_sph_images[idx]
        positive = self.loadPointCloudFromPath(self.positive_sph_images[idx]) if isinstance(
            self.positive_sph_images[idx], str) else self.positive_sph_images[idx]
        negative = self.loadPointCloudFromPath(self.negative_sph_images[idx]) if isinstance(
            self.negative_sph_images[idx], str) else self.negative_sph_images[idx]
        return anchor, positive, negative


if __name__ == "__main__":
    n_data = 100
    n_cache = n_data
    ds = DataSource("/mnt/data/datasets/Spherical/test_training/", n_cache)
    #ds = DataSource("/tmp/training", 10)
    #ds = DataSource("/media/scratch/berlukas/spherical", 10)
    ds.load(n_data, filter_clusters=True)

    a, p, n = ds.get_all_cached_clouds()
    print(f'len of initial cache {len(a)} of batch [{ds.start_cached}, {ds.end_cached}]')
    print("Caching next batch...")
    ds.cache_next(25)
    a, p, n = ds.get_all_cached_clouds()
    print(f'len of next cache {len(a)} of batch [{ds.start_cached}, {ds.end_cached}]')
    a_img, p_img, n_img = ds.get_all_cached_images()
    print(f'anchor image len {len(a_img)}')
    if len(ds.anchor_poses) > 0:
        print(f'first anchor pose: {ds.anchor_poses[0,:]}')
    print(f'anchor image mage shape: {a_img[0].shape}')
