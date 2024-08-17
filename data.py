import os
import random
import numpy as np
import deepdish as dd
import matplotlib

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

import pytorch_lightning as pl

class PointCloudDataModule(pl.LightningDataModule):
    def __init__(self, point_clouds, batch_size=32, train_val_split=0.8):
        """
        Initialize the PointCloudDataModule.
        
        Args:
            point_clouds: List of point cloud data
            batch_size: Batch size for dataloaders
            train_val_split: Ratio of training data to validation data
        """
        super().__init__()
        self.point_clouds = point_clouds
        self.batch_size = batch_size
        self.train_val_split = train_val_split

    def setup(self, stage=None):
        """
        Prepare data for training and validation stages.
        """
        dataset = TensorDataset(torch.FloatTensor(self.point_clouds))
        train_size = int(self.train_val_split * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        """
        Create and return the training data loader.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """
        Create and return the validation data loader.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
class PointCloudDataset(Dataset):
    def __init__(self, data_dir, num_points=2048, transform=None, input_mode='voxels', 
                 output_mode='voxels', normalize=True, jitter=True, rotate=False, resolution=32, relevant_object_categories=None):
        """
        Initialize the PointCloudDataset.
        
        Args:
            data_dir: Directory containing the data files
            num_points: Number of points in each point cloud, used when output_mode is 'point_clouds'
            transform: Optional transform to be applied on a sample
            input_mode: 'point_clouds' or 'voxels'
            output_mode: 'point_clouds' or 'voxels'
            normalize: Whether to normalize the point cloud
            jitter: Whether to apply jitter augmentation
            rotate: Whether to apply rotation augmentation
            resolution: Resolution of voxel grid
            relevant_object_categories: List of relevant object shape categories to include in the dataset
        """
        self.data_dir = data_dir
        self.transform = transform
        self.num_points = num_points
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.dd')]
        self.input_mode = input_mode
        self.output_mode = output_mode
        self.normalize = normalize 
        self.jitter = jitter
        self.rotate = rotate
        self.resolution = 32
        if relevant_object_categories is None:
            self.relevant_object_categories = ['all']  # Can be expanded: E.g. ['chair'],['table'],['airplane'], 
        else:                                          #['chair','table'] or ['all'] categories
            self.relevant_object_categories = relevant_object_categories
        self.shapenet_id_to_category = {
            '02691156': 'airplane',
            '02747177': 'ashcan',
            '02773838': 'bag',
            '02801938': 'basket',
            '02808440': 'bathtub',
            '02818832': 'bed',
            '02828884': 'bench',
            '02843684': 'birdhouse',
            '02871439': 'bookshelf',
            '02876657': 'bottle',
            '02880940': 'bowl',
            '02924116': 'bus',
            '02933112': 'cabinet',
            '02942699': 'camera',
            '02946921': 'can',
            '02954340': 'cap',
            '02958343': 'car',
            '02992529': 'cellular_telephone',
            '03001627': 'chair',
            '03046257': 'clock',
            '03085013': 'computer_keyboard',
            '03207941': 'dishwasher',
            '03211117': 'display',
            '03261776': 'earphone',
            '03325088': 'faucet',
            '03337140': 'file',
            '03467517': 'guitar',
            '03513137': 'helmet',
            '03593526': 'jar',
            '03624134': 'knife',
            '03636649': 'lamp',
            '03642806': 'laptop',
            '03691459': 'loudspeaker',
            '03710193': 'mailbox',
            '03759954': 'microphone',
            '03761084': 'microwave',
            '03790512': 'motorcycle',
            '03797390': 'mug',
            '03928116': 'piano',
            '03938244': 'pillow',
            '03948459': 'pistol',
            '03991062': 'pot',
            '04004475': 'printer',
            '04074963': 'remote_control',
            '04090263': 'rifle',
            '04099429': 'rocket',
            '04225987': 'skateboard',
            '04256520': 'sofa',
            '04330267': 'stove',
            '04379243': 'table',
            '04401088': 'telephone',
            '04460130': 'tower',
            '04468005': 'train',
            '04530566': 'vessel',
            '04554684': 'washer'
        }
        self.filter_file_list()

    def filter_file_list(self):
        """
        Filter the file list based on relevant object categories.
        """
        if self.input_mode != 'voxels' or self.relevant_object_categories == ['all']:
            return
        else:
            relevant_file_list = [f for f in self.file_list if self.shapenet_id_to_category[f.split('_')[4]] in self.relevant_object_categories]
            
            # Sanity check to see if filenames are correct
            # relevant_file_list2 = [f for f in self.file_list if dd.io.load(os.path.join(self.data_dir, f))['object_type'] in self.relevant_object_categories]
            # assert relevant_file_list.sort() == relevant_file_list2.sort()
            
            self.file_list = relevant_file_list

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Retrieve and process a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
        
        Returns:
            Processed point cloud or voxel data
        """
        file_path = os.path.join(self.data_dir, self.file_list[idx])

        if self.input_mode == 'voxels':
            voxels = dd.io.load(file_path)['data']

            self.resolution = voxels.shape[0] # assert voxels.shape[0] == voxels.shape[1] == voxels.shape[2]

            # Normalize voxel values
            if np.min(voxels) == np.max(voxels):
                voxels = np.full_like(voxels, np.min(voxels))
            else:
                voxels = (voxels - np.min(voxels)) / (np.max(voxels) - np.min(voxels))
            
            if self.output_mode == 'voxels' and self.transform is None and not any([self.jitter, self.rotate]):
                return torch.FloatTensor(np.expand_dims(voxels, axis=0))

            point_cloud = self.voxel_to_point_cloud(voxels)
        elif self.input_mode == 'point_clouds':
            point_cloud = dd.io.load(file_path)['data']
        else:
            raise ValueError("Invalid input_mode for PointCloudDataset")
 
        if self.transform:
            point_cloud = self.transform(point_cloud)
        if self.rotate:
            point_cloud = self.normalize_point_cloud(point_cloud)
            point_cloud = self.rotate_around_vertical_axis(point_cloud)
        if self.jitter:
            point_cloud = self.jitter_points(point_cloud) 

        if self.output_mode == 'voxels':
            output = self.point_cloud_to_voxel(point_cloud, self.resolution) 
            output = np.expand_dims(output, axis=0)
        elif self.output_mode == 'point_clouds':
            if self.normalize:
                point_cloud = self.normalize_point_cloud(point_cloud)
            point_cloud = self.sample_point_cloud(point_cloud, self.num_points)
            output = point_cloud
        else:
            raise ValueError("Invalid output_mode for PointCloudDataset")

        return torch.FloatTensor(output)

    def voxel_to_point_cloud(self, voxels, threshold=0.5):
        """
        Convert voxel data to point cloud.
        """
        coords = np.array(np.where(voxels > threshold)).T
        return coords
    
    def point_cloud_to_voxel(self, point_cloud, resolution):
        """
        Convert a point cloud to a voxel grid.
        """
        points = (point_cloud + 1) * (resolution - 1) / 2
        points = np.clip(points, 0, resolution - 1).astype(int)
        voxel_grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
        voxel_grid[points[:, 2], points[:, 1], points[:, 0]] = 1
        return voxel_grid

    def normalize_point_cloud(self, point_cloud):
        """
        Normalize the point cloud to fit in a unit sphere.
        """
        centroid = np.mean(point_cloud, axis=0)
        point_cloud = point_cloud - centroid
        furthest_distance = np.max(np.sqrt(np.sum(point_cloud**2, axis=1)))
        point_cloud = point_cloud / furthest_distance
        return point_cloud

    def sample_point_cloud(self, point_cloud, num_points):
        """
        Sample a fixed number of points from the point cloud.
        """
        if len(point_cloud) == num_points:
            return point_cloud
        elif len(point_cloud) > num_points:
            indices = random.sample(range(len(point_cloud)), num_points)
            return point_cloud[indices]
        else:
            # If we don't have enough points, use all indices first and then sample the rest with replacement
            all_indices = list(range(len(point_cloud)))
            additional_indices = np.random.choice(len(point_cloud), num_points - len(point_cloud), replace=True)
            indices = all_indices + additional_indices.tolist()
            return point_cloud[indices]
        
    def farthest_point_sample(self, point_cloud, num_points):
        """
        Performs furthest point sampling on a given point cloud to select a subset of points.
        Warning: This sampling method makes dataloading very slow
        """

        if len(point_cloud) == num_points:  ## early return if point cloud size is equal to 'num_points' already
            return point_cloud

        N, D = point_cloud.shape
        xyz = point_cloud[:,:3]
        centroids = np.zeros((num_points,))
        distance = np.ones((N,)) * 1e10
        farthest = np.random.randint(0, N)
        
        for i in range(num_points):
            centroids[i] = farthest
            centroid = xyz[farthest, :]
            dist = np.sum((xyz - centroid) ** 2, axis=-1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance, axis=-1)
        
        point_cloud = point_cloud[centroids.astype(np.int32)]
        
        if len(point_cloud) < num_points:
            # If we don't have enough points, we'll duplicate some
            duplicates = np.random.choice(len(point_cloud), num_points - len(point_cloud), replace=True)
            point_cloud = np.concatenate([point_cloud, point_cloud[duplicates]])
        
        return point_cloud

    def jitter_points(self, points, sigma=0.01, clip=0.05):
        """
        Apply random jitter to the points.
        """
        jittered_data = np.clip(sigma * np.random.randn(*points.shape), -1*clip, clip)
        jittered_data += points
        return jittered_data

    def rotate_around_vertical_axis(self, point_cloud):
        """
        Rotate the point cloud around the vertical axis.
        """
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        rotated_data = np.dot(point_cloud, rotation_matrix)
        return rotated_data  

class PointCloudDataDirectoryModule(pl.LightningDataModule):
    def __init__(self, data_dir, num_points=2048, batch_size=32, num_workers=4, train_val_split=0.8, 
                 file_mode='voxels', output_mode='point_clouds', augmentations=True, normalization=True, relevant_object_categories=None):
        """
        Initialize the PointCloudDataDirectoryModule.
        
        Args:
            data_dir: Directory containing the data files
            num_points: Number of points in each point cloud, used when output mode is 'point_clouds'
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes for data loading
            train_val_split: Ratio of training data to validation data
            file_mode: 'voxels' or 'point_clouds'
            output_mode: 'voxels' or 'point_clouds'
            augmentations: Whether to apply data augmentations
            normalization: Whether to normalize the point clouds
            relevant_object_categories: List of relevant object shape categories to include in the dataset
        """
        super().__init__()
        self.data_dir = data_dir
        self.num_points = num_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.file_mode = file_mode
        self.output_mode = output_mode
        self.augmentations = augmentations
        self.normalization = normalization
        self.relevant_object_categories = relevant_object_categories

    def setup(self, stage=None):
        """
        Prepare data for training and validation stages.
        """
        if self.augmentations:
            full_dataset = PointCloudDataset(self.data_dir, num_points=self.num_points, 
                                             input_mode=self.file_mode, output_mode=self.output_mode, normalize=self.normalization,
                                             relevant_object_categories=self.relevant_object_categories)
        else:
            full_dataset = PointCloudDataset(self.data_dir, num_points=self.num_points, 
                                             input_mode=self.file_mode, output_mode=self.output_mode, rotate=False, jitter=False,
                                             normalize=self.normalization, relevant_object_categories=self.relevant_object_categories)

        # Calculate lengths of splits
        total_size = len(full_dataset)
        train_size = int(self.train_val_split * total_size)
        val_size = total_size - train_size
        
        # Split the dataset
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        """
        Create and return the training data loader.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        """
        Create and return the validation data loader.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)