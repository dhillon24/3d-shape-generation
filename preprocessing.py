import os
import numpy as np
import deepdish as dd
from tqdm import tqdm
from utils import save_point_cloud_comparison, save_three_point_cloud_comparison

def furthest_point_sample_numpy(points, npoint):
    """
    Perform Furthest Point Sampling (FPS) on a point cloud.
    
    Input:
        points: pointcloud data, [N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint]
    """
    N, _ = points.shape
    centroids = np.zeros(npoint, dtype=int)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = points[farthest, :]
        dist = np.sum((points - centroid) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    return centroids

def voxel_to_point_cloud(voxels, threshold=0.5):
    """
    Convert a voxel grid to a point cloud.
    
    Args:
        voxels: 3D numpy array representing the voxel grid
        threshold: Minimum value to consider a voxel as occupied
    
    Returns:
        coords: [N, 3] array of point coordinates
    """
    coords = np.array(np.where(voxels > threshold)).T
    return coords

def normalize_point_cloud(point_cloud):
    """
    Normalize the point cloud to fit within a unit sphere centered at the origin.
    
    Args:
        point_cloud: [N, 3] array of point coordinates
    
    Returns:
        normalized_point_cloud: [N, 3] array of normalized point coordinates
    """
    centroid = np.mean(point_cloud, axis=0)
    point_cloud = point_cloud - centroid
    furthest_distance = np.max(np.sqrt(np.sum(point_cloud**2, axis=1)))
    point_cloud = point_cloud / furthest_distance
    return point_cloud

def preprocess_data_fps_only(input_dir, output_dir, vis_dir, num_points=2048, max_visualizations=100):
    """
    Preprocess voxel data to point clouds using Furthest Point Sampling (FPS).
    
    Args:
        input_dir: Directory containing input voxel data
        output_dir: Directory to save processed point cloud data
        vis_dir: Directory to save visualizations
        num_points: Number of points to sample for each point cloud
        max_visualizations: Maximum number of visualizations to generate
    """
    # Create output directories if they don't exist
    for directory in [output_dir, vis_dir]:
        os.makedirs(directory, exist_ok=True)

    idx = 0
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith('.dd'):
            file_path = os.path.join(input_dir, filename)
            voxel_data = dd.io.load(file_path)['data']
            point_cloud = voxel_to_point_cloud(voxel_data)
            
            original_point_cloud = normalize_point_cloud(point_cloud)
            
            # Perform FPS or random sampling if not enough points
            if len(point_cloud) >= num_points:
                sampled_indices = furthest_point_sample_numpy(point_cloud, num_points)
                sampled_point_cloud = point_cloud[sampled_indices]
            else:
                # If we don't have enough points, use all indices first and then sample the rest with replacement
                all_indices = np.arange(len(point_cloud))
                additional_indices = np.random.choice(len(point_cloud), num_points - len(point_cloud), replace=True)
                sampled_indices = np.concatenate((all_indices, additional_indices))
    
            sampled_point_cloud = point_cloud[sampled_indices]
            
            sampled_point_cloud = normalize_point_cloud(sampled_point_cloud)
            
            # Save the sampled point cloud
            output_filename = filename
            output_path = os.path.join(output_dir, output_filename)
            dd.io.save(output_path, {'data': sampled_point_cloud})

            ## Note on using dd.io.save:
            ## np.object is deprecated in numpy >= 1.20 but is used by deepdish to save ndarrays making any deepdish pip version incompatible with numpy >= 1.20 
            ## Modify library code of deepdish library as given by this commit on official repo:https://github.com/uchicago-cs/deepdish/commit/b4f399a6ab1ad86fdb532d78893bb2f6af568e60
            ## pip doesn't have this version, you will have to build deepdish from source, or just modify this file in your library code:
            ## 'deepdish/io/hdf5io.py:125'

            # Visualize and save the comparison plot if less than max visualization
            if idx < max_visualizations:
                vis_filename = os.path.splitext(filename)[0] + '_comparison.png'
                vis_path = os.path.join(vis_dir, vis_filename)
                save_point_cloud_comparison(original_point_cloud, sampled_point_cloud, vis_path, f"Point Cloud Comparison - {filename}")
                idx += 1

def preprocess_data_fps_and_random(input_dir, fps_output_dir, random_output_dir, vis_dir, num_points=2048, max_visualizations=100):
    """
    Preprocess voxel data to point clouds using both FPS and random sampling.
    
    Args:
        input_dir: Directory containing input voxel data
        fps_output_dir: Directory to save FPS processed point cloud data
        random_output_dir: Directory to save randomly sampled point cloud data
        vis_dir: Directory to save visualizations
        num_points: Number of points to sample for each point cloud
        max_visualizations: Maximum number of visualizations to generate
    """
    # Create output directories if they don't exist
    for directory in [fps_output_dir, random_output_dir, vis_dir]:
        os.makedirs(directory, exist_ok=True)

    idx = 0
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith('.dd'):
            file_path = os.path.join(input_dir, filename)
            voxel_data = dd.io.load(file_path)['data']
            point_cloud = voxel_to_point_cloud(voxel_data)
            
            original_point_cloud = normalize_point_cloud(point_cloud)

            num_original_points = len(point_cloud)
            
            # Perform FPS and random sampling
            if num_original_points >= num_points:
                fps_indices = furthest_point_sample_numpy(point_cloud, num_points)
                fps_point_cloud = point_cloud[fps_indices]
                
                random_indices = np.random.choice(len(point_cloud), num_points, replace=False)
                random_point_cloud = point_cloud[random_indices]
            else:
                # If we don't have enough points, use all indices first and then sample the rest with replacement
                all_indices = list(range(len(point_cloud)))
                additional_indices = np.random.choice(len(point_cloud), num_points - len(point_cloud), replace=True)
                fps_indices = all_indices + additional_indices.tolist()
                fps_point_cloud = point_cloud[fps_indices]
                random_point_cloud = point_cloud[fps_indices]  # Use the same indices for consistency
            
            fps_point_cloud = normalize_point_cloud(fps_point_cloud)
            random_point_cloud = normalize_point_cloud(random_point_cloud)
            
            # Save the FPS sampled point cloud
            fps_output_path = os.path.join(fps_output_dir, filename)
            dd.io.save(fps_output_path, {'data': fps_point_cloud})

            ## Note on using dd.io.save:
            ## np.object is deprecated in numpy >= 1.20 but is used by deepdish to save ndarrays making any deepdish pip version incompatible with numpy >= 1.20 
            ## Modify library code of deepdish library as given by this commit on official repo:https://github.com/uchicago-cs/deepdish/commit/b4f399a6ab1ad86fdb532d78893bb2f6af568e60
            ## pip doesn't have this version, you will have to build deepdish from source, or just modify this file in your library code:
            ## 'deepdish/io/hdf5io.py:125'
            
            # Save the randomly sampled point cloud
            random_output_path = os.path.join(random_output_dir, filename)
            dd.io.save(random_output_path, {'data': random_point_cloud})
            
            # Visualize and save the comparison plot if less than max visualization
            if idx < max_visualizations:
                vis_filename = os.path.splitext(filename)[0] + '_comparison.png'
                vis_path = os.path.join(vis_dir, vis_filename)
                save_three_point_cloud_comparison(original_point_cloud, fps_point_cloud, random_point_cloud, 
                                            vis_path, f"Point Cloud Comparison - {filename}", num_original_points, num_points)
                idx += 1

def main():
    # Set input and output directories
    input_dir = os.path.join('data','shape_net_voxel_data_v1')
    fps_output_dir = os.path.join('data','fps_sampled_shape_net_point_clouds_v1')
    random_output_dir = os.path.join('data','random_sampled_shape_net_point_clouds_v1')
    vis_dir = 'data/point_cloud_visualizations'

    # Preprocess data using both FPS and random sampling
    preprocess_data_fps_and_random(input_dir, fps_output_dir, random_output_dir, vis_dir, num_points=2048)

if __name__ == '__main__':
    main()