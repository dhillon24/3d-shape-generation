import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

from utils import voxelize

def normalize_to_cube(points):
    """
    Normalize a set of 3D points to fit within a unit cube centered at the origin.
    
    Args:
        points (torch.Tensor): The input points of shape [batch_size, num_points, 3].
        
    Returns:
        torch.Tensor: The normalized points of the same shape as the input.
    """
    center = (points.max(dim=1, keepdim=True)[0] + points.min(dim=1, keepdim=True)[0]) / 2
    points = points - center
    scale = points.abs().max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    points = points / scale
    return points

def chamfer_distance(x, y, scaling_factor=1e+3):
    """
    Compute the Chamfer Distance between two point clouds.
    
    Args:
        x (torch.Tensor): First point cloud of shape [batch_size, num_points_1, 3] or [num_points_1, 3]
        y (torch.Tensor): Second point cloud of shape [batch_size, num_points_2, 3] or [num_points_2, 3]
        scaling_factor (float): Scaling factor for the Chamfer Distance (default: 1e+3)
    
    Returns:
        torch.Tensor: Computed Chamfer Distance
    """
    x = x.unsqueeze(0) if x.dim() == 2 else x
    y = y.unsqueeze(0) if y.dim() == 2 else y

    x = normalize_to_cube(x)
    y = normalize_to_cube(y)

    dist = torch.cdist(x, y)
    
    min_dist_xy = torch.min(dist, dim=2)[0]
    min_dist_yx = torch.min(dist, dim=1)[0]
    
    cd = torch.mean(min_dist_xy) + torch.mean(min_dist_yx)
    return cd*scaling_factor

def earth_mover_distance_cpu(x, y, scaling_factor=1):
    """
    Compute the exact Earth Mover's Distance between two 3D point clouds using CPU.
    
    Args:
        x (torch.Tensor): First point cloud of shape [batch_size, num_points_1, 3] or [num_points_1, 3]
        y (torch.Tensor): Second point cloud of shape [batch_size, num_points_2, 3] or [num_points_2, 3]
        scaling_factor (float): Scaling factor for the Chamfer Distance (default: 1e+3)
    
    Returns:
        torch.Tensor: Computed Earth Mover's Distance
    """

    x = x.unsqueeze(0) if x.dim() == 2 else x
    y = y.unsqueeze(0) if y.dim() == 2 else y

    assert x.shape[0] == y.shape[0], "Batch sizes must be the same"
    assert x.shape[2] == y.shape[2], "Point clouds must have the same number of dimensions"

    x = normalize_to_cube(x)
    y = normalize_to_cube(y)

    x = x.unsqueeze(1) if x.dim() == 2 else x
    y = y.unsqueeze(1) if y.dim() == 2 else y
    
    emd_dist = []
    for x_pc, y_pc in zip(x, y):
        num_points_x = x_pc.shape[1]
        num_points_y = y_pc.shape[1]
        
        x_np = x_pc.cpu().numpy()
        y_np = y_pc.cpu().numpy()
        
        # Compute pairwise distances
        distances = np.linalg.norm(x_np[:, None] - y_np[None, :], axis=-1)
        
        # Solve the assignment problem
        row_ind, col_ind = linear_sum_assignment(distances)
        
        # Compute EMD
        emd = distances[row_ind, col_ind].sum() / max(num_points_x, num_points_y)
        emd_dist.append(emd)
    
    return torch.tensor(emd_dist, device=x.device).mean()*scaling_factor

def earth_mover_distance_gpu(x, y, epsilon=1e-2, thresh=1e-5, max_iter=100, scaling_factor=1):
    """
    Compute the approximate Earth Mover's Distance using Sinkhorn iterations on GPU.
    
    Args:
        x (torch.Tensor): First point cloud of shape [batch_size, num_points_1, 3] or [num_points, 3]
        y (torch.Tensor): Second point cloud of shape [batch_size, num_points_2, 3] or [num_points, 3]
        epsilon (float): Regularization parameter
        thresh (float): Convergence threshold for Sinkhorn iterations
        max_iter (int): Maximum number of Sinkhorn iterations
        scaling_factor (float): Scaling factor for the Chamfer Distance (default: 1e+3)
    
    Returns:
        torch.Tensor: Computed approximate Earth Mover's Distance
    """

    x = x.unsqueeze(0) if x.dim() == 2 else x
    y = y.unsqueeze(0) if y.dim() == 2 else y

    x = normalize_to_cube(x)
    y = normalize_to_cube(y)

    batch_size, n, _ = x.shape
    _, m, _ = y.shape

    # Compute cost matrix
    C = torch.cdist(x, y, p=2)

    # Normalize the cost matrix
    C = C / C.max()

    # Sinkhorn scaling factor
    lambda_val = 1 / epsilon

    # Initialize dual variables
    alpha = torch.zeros(batch_size, n, 1, device=x.device)
    beta = torch.zeros(batch_size, m, 1, device=x.device)

    # Initialize marginals
    mu = torch.ones(batch_size, n, 1, device=x.device) / n
    nu = torch.ones(batch_size, m, 1, device=x.device) / m

    # Sinkhorn iterations
    for _ in range(max_iter):
        alpha_prev, beta_prev = alpha, beta

        # Update alpha
        alpha = epsilon * (torch.log(mu + 1e-10) - torch.logsumexp(-lambda_val * C + beta.transpose(1,2), dim=2, keepdim=True))
        
        # Update beta
        beta = epsilon * (torch.log(nu + 1e-10) - torch.logsumexp(-lambda_val * C.transpose(1,2) + alpha.transpose(1,2), dim=2, keepdim=True))

        # Check for convergence
        err_alpha = torch.abs(alpha - alpha_prev).max()
        err_beta = torch.abs(beta - beta_prev).max()
        if err_alpha < thresh and err_beta < thresh:
            break

    # Compute transport plan
    P = torch.exp(-lambda_val * C + alpha + beta.transpose(1,2))

    # Compute EMD
    emd = torch.sum(P * C, dim=(1, 2))

    return emd.mean()*scaling_factor

def compute_metrics(generated_samples, reference_samples, use_approximate_gpu_emd=False):
    """
    Compute various metrics between generated and reference point cloud samples.
    
    Args:
        generated_samples (torch.Tensor): Generated point cloud samples
        reference_samples (torch.Tensor): Reference point cloud samples
    
    Returns:
        tuple: Computed Chamfer Distance, Earth Mover's Distance, and Reconstruction Loss
    """
    # Compute Chamfer Distances
    avg_cd = chamfer_distance(generated_samples, reference_samples)

    # Compute Earth Mover's Distances
    if use_approximate_gpu_emd:
        avg_emd = earth_mover_distance_gpu(generated_samples, reference_samples)
    else:
        avg_emd = earth_mover_distance_cpu(generated_samples, reference_samples)

    # Compute Reconstruction Loss
    recon_loss = torch.nn.functional.binary_cross_entropy(voxelize(generated_samples), voxelize(reference_samples)) 

    return avg_cd, avg_emd, recon_loss

def voxel_focal_loss(pred, target, focal_alpha=0.25, focal_gamma=2):
    """
    Compute the Focal Loss for voxel-based predictions.
    
    Args:
        pred (torch.Tensor): Predicted voxel occupancy probabilities
        target (torch.Tensor): Ground truth voxel occupancy
        focal_alpha (float): Weighting factor in range (0,1) to balance positive vs negative examples
        focal_gamma (float): Focusing parameter for modulating factor (1-p)
    
    Returns:
        torch.Tensor: Computed Focal Loss
    """
    # Clamp predictions to avoid log(0)
    pred = torch.clamp(pred, min=1e-7, max=1-1e-7)
    
    bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
    
    pt = torch.where(target == 1, pred, 1 - pred)
    focal_term = (1 - pt) ** focal_gamma
    focal_loss = focal_term * bce
    
    # Weigh occupied voxels in target more than unoccupied
    alpha_factor = torch.where(target == 1, focal_alpha, 1 - focal_alpha)
    focal_loss = alpha_factor * focal_loss
    
    # Return the mean focal loss
    return focal_loss.mean()