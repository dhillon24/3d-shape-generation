import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from plyfile import PlyData, PlyElement
import logging
import torch

def get_coords(dims):
    """
    Generate 3D coordinates for a given dimension.
    
    Args:
        dims (int or np.array): Dimensions of the 3D space
    
    Returns:
        np.array: Stacked array of x, y, z coordinates
    """
    if isinstance(dims, int):
        dims = np.array([dims, dims, dims])

    x = np.linspace(-1, 1, dims[0])
    y = np.linspace(-1, 1, dims[1])
    z = np.linspace(-1, 1, dims[2])

    x_1, y_1, z_1 = np.meshgrid(x, y, z)
    return np.stack([x_1, y_1, z_1])

def save_to_ply(filename, points):
    """
    Save points to a PLY file.
    
    Args:
        filename (str): Output filename
        points (np.array): Nx3 array of point coordinates
    """
    assert points.ndim == 2 and points.shape[1] == 3, "Points should be a Nx3 array"

    vertex = np.array(
        [(p[0], p[1], p[2]) for p in points],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    )

    ply_element = PlyElement.describe(vertex, 'vertex')
    PlyData([ply_element]).write(filename)

def plot_single_voxel_graph(ax, coords, voxels):
    """
    Plot a single voxel graph.
    
    Args:
        ax (matplotlib.axes.Axes): Matplotlib axes for plotting
        coords (np.array): Coordinates of the voxels
        voxels (np.array): Voxel occupancy grid
    
    Returns:
        np.array: Array of plotted points
    """
    mask = voxels[:, :, :] > 0
    points = np.array([coords[0, mask], coords[1, mask], coords[2, mask]]).T
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, vmin=-1, vmax=1, c='k')
    return points

def plot_voxels(voxels, save_ply=False):
    """
    Plot voxels and optionally save as PLY file.
    
    Args:
        voxels (np.array): Voxel occupancy grid
        save_ply (bool): Whether to save the plot as a PLY file
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    coords = get_coords(32)

    points = plot_single_voxel_graph(ax, coords, voxels)
    plt.show()

    if save_ply:
        save_to_ply('object.ply', points)

def voxel_to_point_cloud(voxels, dims=None, threshold=0.5):
    """
    Convert a voxel grid to a point cloud.
    
    Args:
        voxels (np.array): Voxel occupancy grid
        dims (np.array, optional): Dimensions of the voxel grid
        threshold (float): Threshold for considering a voxel as occupied
    
    Returns:
        np.array: Point cloud representation
    """
    if dims is None:
        dims = np.array([voxels.shape[0], voxels.shape[1], voxels.shape[2]])

    coords = get_coords(dims)
    coords_flat = coords.reshape(3, -1).T
    points = coords_flat[voxels.flatten() > threshold]
    
    return points

def point_cloud_to_voxel(points, dims=32, padding=1e-4):
    """
    Convert a point cloud to a voxel grid.
    
    Args:
        points (np.array): Point cloud data
        dims (int): Dimensions of the output voxel grid
        padding (float): Padding value for occupied voxels
    
    Returns:
        np.array: Voxel grid representation
    """
    voxels = np.zeros((dims, dims, dims))
    
    points = (points + 1) * (dims - 1) / 2
    points = np.round(points).astype(int)
    
    mask = np.all((points >= 0) & (points < dims), axis=1)
    points = points[mask]
    
    voxels[points[:, 0], points[:, 1], points[:, 2]] = 1
    
    voxels = ndimage.maximum_filter(voxels, size=3)
    
    return voxels

def plot_3d(data, is_voxel=True):
    """
    Plot 3D data (voxels or point cloud).
    
    Args:
        data (np.array): 3D data to plot
        is_voxel (bool): Whether the data is a voxel grid or point cloud
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if is_voxel:
        points = voxel_to_point_cloud(data)
    else:
        points = data
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    
    plt.title('3D Object Visualization')
    plt.show()

def plot_multiple_3d(data_list, is_voxel=True, rows=1, cols=1):
    """
    Plot multiple 3D objects.
    
    Args:
        data_list (list): List of 3D data to plot
        is_voxel (bool): Whether the data is voxel grids or point clouds
        rows (int): Number of rows in the plot grid
        cols (int): Number of columns in the plot grid
    """
    fig = plt.figure(figsize=(6*cols, 6*rows))
    
    for i, data in enumerate(data_list):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        
        if is_voxel:
            points = voxel_to_point_cloud(data)
        else:
            points = data
        
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        
        ax.set_title(f'Object {i+1}')
    
    plt.tight_layout()
    plt.show()

def plot_comparison_point_cloud(ax, points, title=''):
    """
    Plot a single point cloud for comparison.
    
    Args:
        ax (matplotlib.axes.Axes): Matplotlib axes for plotting
        points (np.array): Point cloud data
        title (str): Title for the plot
    """
    if len(points) == 0:
        return  

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    max_range = np.max(points) - np.min(points)
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

def plot_comparison_point_clouds(original, sampled, title, title1='Original Point Cloud', title2='Sampled Point Cloud'):
    """
    Plot two point clouds for comparison.
    
    Args:
        original (np.array): Original point cloud data
        sampled (np.array): Sampled point cloud data
        title (str): Main title for the plot
        title1 (str): Title for the original point cloud
        title2 (str): Title for the sampled point cloud
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    plot_comparison_point_cloud(ax1, original)
    ax1.set_title(title1)
    
    plot_comparison_point_cloud(ax2, sampled)
    ax2.set_title(title2)
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def save_point_cloud_comparison(original, sampled, output_path, title, title1='Original Point Cloud', title2='Sampled Point Cloud'):
    """
    Save a comparison of two point clouds to a file.
    
    Args:
        original (np.array): Original point cloud data
        sampled (np.array): Sampled point cloud data
        output_path (str): Path to save the comparison plot
        title (str): Main title for the plot
        title1 (str): Title for the original point cloud
        title2 (str): Title for the sampled point cloud
    """
    fig = plot_comparison_point_clouds(original, sampled, title, title1, title2)
    plt.savefig(output_path)
    plt.close(fig)

def save_three_point_cloud_comparison(original, fps_sampled, random_sampled, output_path, title, num_original_points, num_points):
    """
    Save a comparison of three point clouds to a file.
    
    Args:
        original (np.array): Original point cloud data
        fps_sampled (np.array): FPS sampled point cloud data
        random_sampled (np.array): Randomly sampled point cloud data
        output_path (str): Path to save the comparison plot
        title (str): Main title for the plot
        num_original_points (int): Number of points in the original point cloud
        num_points (int): Number of points in the sampled point clouds
    """
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    
    plot_comparison_point_cloud(ax1, original, f'Original Point Cloud, n={num_original_points}')
    plot_comparison_point_cloud(ax2, fps_sampled, f'FPS Sampled Point Cloud, n={num_points}')
    plot_comparison_point_cloud(ax3, random_sampled, f'Random Sampled Point Cloud, n={num_points}')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

def plot_point_cloud_3d(point_cloud):
    """
    Plot a 3D point cloud.
    
    Args:
        point_cloud (np.array or torch.Tensor): Point cloud data
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if not isinstance(point_cloud, np.ndarray):
        point_cloud = point_cloud.detach().cpu().numpy()
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if len(point_cloud) == 0:  
        return fig 
    
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1)
    
    max_range = np.max(point_cloud) - np.min(point_cloud)
    mid_x = (point_cloud[:, 0].max() + point_cloud[:, 0].min()) * 0.5
    mid_y = (point_cloud[:, 1].max() + point_cloud[:, 1].min()) * 0.5
    mid_z = (point_cloud[:, 2].max() + point_cloud[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return fig

def plot_point_cloud_2d(point_cloud):
    """
    Plot 2D projections of a point cloud.
    
    Args:
        point_cloud (np.array or torch.Tensor): Point cloud data
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if not isinstance(point_cloud, np.ndarray):
        point_cloud = point_cloud.detach().cpu().numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.scatter(point_cloud[:, 0], point_cloud[:, 1], s=1)
    ax1.set_title('XY Projection')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    ax2.scatter(point_cloud[:, 0], point_cloud[:, 2], s=1)
    ax2.set_title('XZ Projection')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    
    ax3.scatter(point_cloud[:, 1], point_cloud[:, 2], s=1)
    ax3.set_title('YZ Projection')
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Z')
    
    plt.tight_layout()
    return fig

def setup_logger(log_file, name):
    """
    Set up a logger for the project.
    
    Args:
        log_file (str): Path to the log file
        name (str): Name of the logger
    
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def save_point_cloud(point_cloud, filename):
    """
    Save a point cloud to a file.
    
    Args:
        point_cloud (torch.Tensor): Point cloud data
        filename (str): Path to save the point cloud
    """
    np.savetxt(filename, point_cloud.cpu().numpy(), delimiter=',')

def index_points(points, idx):
    """
    Index points with given indices.
    
    Args:
        points (torch.Tensor): Input points data, [B, N, C]
        idx (torch.Tensor): Sample index data, [B, S, K]
    
    Returns:
        torch.Tensor: Indexed points data, [B, S, K, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Perform farthest point sampling on point cloud data.
    
    Args:
        xyz (torch.Tensor): Pointcloud data, [B, N, 3]
        npoint (int): Number of samples
    
    Returns:
        torch.Tensor: Sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def square_distance(src, dst):
    """
    Calculate Euclidean distance between each two points.
    
    Args:
        src (torch.Tensor): Source points, [B, N, C]
        dst (torch.Tensor): Target points, [B, M, C]
    
    Returns:
        torch.Tensor: Squared distances, [B, N, M]
    """
    return torch.cdist(src, dst, p=2.0).pow(2)

def knn_square_distance(src, dst, k=3):
    """
    Compute k-nearest neighbors and their squared distances.
    
    Args:
        src (torch.Tensor): Source points, [B, N, C]
        dst (torch.Tensor): Target points, [B, M, C]
        k (int): Number of nearest neighbors
    
    Returns:
        tuple: Squared distances [B, N, k] and indices [B, N, k]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    
    src2 = (src ** 2).sum(dim=-1, keepdim=True)  # [B, N, 1]
    dst2 = (dst ** 2).sum(dim=-1).unsqueeze(1)   # [B, 1, M]

    dists = torch.zeros((B, N, M), device=src.device)

    chunk_size = min(1000, N)
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        dists[:, i:end] = src2[:, i:end] + dst2 - 2 * torch.bmm(src[:, i:end], dst.transpose(1, 2))
    
    torch.cuda.synchronize()
    
    dists, indices = dists.topk(k=k, dim=-1, largest=False)

    return dists, indices

def voxelize(points, voxel_resolution=32):
    """
    Convert point cloud to voxel grid.
    
    Args:
        points (torch.Tensor): Point cloud data, [B, N, 3]
        voxel_resolution (int): Resolution of the voxel grid
    
    Returns:
        torch.Tensor: Voxel grid, [B, voxel_resolution, voxel_resolution, voxel_resolution]
    """
    points = points.unsqueeze(0) if points.dim() == 2 else points

    points = (points + 1) * (voxel_resolution - 1) / 2
    points = points.long().clamp(0, voxel_resolution - 1)
    batch_size = points.size(0)
    voxels = torch.zeros(batch_size, voxel_resolution, voxel_resolution, voxel_resolution).to(points.device)
    
    for i in range(batch_size):
        voxels[i, points[i, :, 0], points[i, :, 1], points[i, :, 2]] = 1
    
    return voxels

def voxel_tensor_to_point_clouds(voxel_grid, threshold=0.5):
    """
    Convert a voxel grid to a point cloud.
    
    Args:
        voxel_grid (torch.Tensor): Voxel grid tensor of shape [batch_size, 1, depth, height, width]
        threshold (float): Threshold for considering a voxel as occupied
    
    Returns:
        list: List of point clouds, one for each sample in the batch
    """
    device = voxel_grid.device
    batch_size, _, depth, height, width = voxel_grid.shape
    point_clouds = []

    for i in range(batch_size):
        z, y, x = torch.where(voxel_grid[i, 0] > threshold)
        
        if len(z) > 0:
            points = torch.stack([x, y, z], dim=1).float()
            
            # Normalize to [-1, 1]
            points = 2 * points / torch.tensor([width-1, height-1, depth-1], device=device) - 1
        else:
            points = torch.empty((0, 3), device=device)
        
        point_clouds.append(points)

    return point_clouds