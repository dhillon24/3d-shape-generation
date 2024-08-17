
import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Use non-interactive backend for matplotlib
matplotlib.use('Agg')

import pytorch_lightning as pl
# Set random seed for reproducibility
pl.seed_everything(24)

from diffusion import PointCloudDiffusion
from data import PointCloudDataDirectoryModule
from metrics import compute_metrics
from utils import plot_point_cloud_2d, plot_point_cloud_3d, save_point_cloud_comparison
import logging
from datetime import datetime

from utils import setup_logger, save_point_cloud

def test_ddpm_generation(model, model_name, num_samples=10, num_points=2048):
    """
    Generate samples using the specified model and save them as visualizations and point cloud files.

    Parameters:
    - model: The model used for generating samples.
    - model_name: The name of the model.
    - num_samples: The number of samples to generate (default: 10).
    - num_points: The number of points in each sample (default: 2048).
    """
    # Generate samples
    with torch.no_grad():
        generated_samples = model.sample(num_samples=num_samples, num_points=num_points, num_steps=1000)

    # Create output directories
    os.makedirs(os.path.join('test', 'visualizations', model_name), exist_ok=True)
    # os.makedirs(os.path.join('test', 'point_clouds', model_name), exist_ok=True)

    # Save samples as visualizations and point cloud files
    for i, sample in enumerate(generated_samples):
        # Visualization
        fig1 = plot_point_cloud_3d(sample)
        plt.savefig(os.path.join('test', 'visualizations', model_name, f'sample_{i}_3d.png'))
        plt.close(fig1)
        fig2 = plot_point_cloud_2d(sample)
        plt.savefig(os.path.join('test', 'visualizations', model_name, f'sample_{i}_2d.png'))
        plt.close(fig2)
        
        # Point cloud file
        # save_point_cloud(sample, os.path.join('test', 'point_clouds', model_name, f'sample_{i}.txt'))

    logger = logging.getLogger('test_logger_point_ddpm')
    logger.info(f"Generated and saved {num_samples} samples.")

def test_ddpm_reconstruction(model, model_name, data_module, num_samples=10, initial_t=0.010):
    """
    Test the reconstruction performance of a deep density estimation model.

    Args:
        model (torch.nn.Module): The deep density estimation model.
        model_name (str): The name of the model.
        data_module: The data module containing the validation dataset.
        num_samples (int, optional): The number of samples to use for testing. Defaults to 10.
        initial_t (float, optional): The initial timestamp for denoising before reconstruction. Defaults to 0.010.
    """
 
    # Get samples from the validation set
    val_dataloader = data_module.val_dataloader()
    original_samples = next(iter(val_dataloader))[:num_samples]

    original_samples = original_samples.to(model.device)

    # Generate reconstructed samples
    with torch.no_grad():
        t = torch.ones(num_samples, device=model.device)*initial_t
        noisy_samples, _, _, _ = model.add_noise(original_samples, t)
        reconstructed_samples = model.sample3(num_samples=num_samples, num_points=original_samples.shape[1], x=noisy_samples, start_t=t)

        # Compute metrics
        cds, emds, recon_losses = [], [], []
        avg_cd, avg_emd, avg_recon_loss = 0.0, 0.0, 0.0
        for original_sample, reconstructed_sample in zip(original_samples, reconstructed_samples):
            cd, emd, recon_loss = compute_metrics(original_sample, reconstructed_sample)
            avg_cd += cd
            avg_emd += emd
            avg_recon_loss += recon_loss
            cds.append(cd)
            emds.append(emd)
            recon_losses.append(recon_loss)

        num_samples = len(original_samples)
        avg_cd /= num_samples
        avg_emd /= num_samples
        avg_recon_loss /= num_samples

    # Compute metrics
    # avg_cd, avg_emd, avg_recon_loss = compute_metrics(original_samples, reconstructed_samples) ## compute metrics in a batch

    logger = logging.getLogger('test_logger_point_ddpm')
    logger.info(f"Average Chamfer Distance: {avg_cd:.3f}")
    logger.info(f"Average Earth Mover's Distance: {avg_emd:.3f}")
    logger.info(f"Average Reconstruction Loss: {avg_recon_loss:.3f}")
    # Create output directories
    os.makedirs(os.path.join('test', 'visualizations', model_name), exist_ok=True)
    # os.makedirs(os.path.join('test', 'point_clouds', model_name), exist_ok=True)

    # Save original and reconstructed samples
    for i, (orig, recon) in enumerate(zip(original_samples, reconstructed_samples)):
        # Visualizations
        vis_path = os.path.join('test', 'visualizations', model_name, f'comparison_{i}.png')
        title = f"Point Cloud Comparison, Sample: CD (x10^3) = {cds[i]:.3f}, EMD = {emds[i]:.3f}, RE = {recon_losses[i]:.3f}"
        # title = f"Point Cloud Comparison, Batch: CD (x10^3) = {avg_cd:.3f}, EMD = {avg_emd:.3f}, RE = {avg_recon_loss:.3f}"
        save_point_cloud_comparison(orig.cpu().numpy(), recon.cpu().numpy(), vis_path, title=title, title1='Original', title2='Reconstructed')

        # Point cloud files
        # save_point_cloud(orig, os.path.join('test', 'point_clouds', model_name, f'original_{i}.txt'))
        # save_point_cloud(recon, os.path.join('test', 'point_clouds', model_name, f'reconstructed_{i}.txt'))

    logger.info(f"Reconstructed and saved {num_samples} samples.")

if __name__ == "__main__":
    augmentations = False
    relevant_object_categories=['airplane'] ## ['chair'], ['table'], ['airplane'] ## change this and model name, only affects the reconstruction test
    subdirectory_name = f"{relevant_object_categories[0]}_from_scratch_no_augs_2048"  ## change this to the subdirectory that you want to test, all ddpm models in it will be tested
    checkpoint_dir = os.path.join('checkpoints', 'best_run', 'point_ddpm', subdirectory_name)
    data_dir = os.path.join('data', 'shape_net_voxel_data_v1')  
    num_samples = 16
    num_points = 2048  # Ensure this matches model's configuration

    # Setup logger
    os.makedirs(os.path.join('test', 'logs'), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join('test', 'logs', f'test_point_ddpm_log_{timestamp}.log')
    logger = setup_logger(log_file, 'test_logger_point_ddpm')

    logger.info("Starting test script")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Number of samples: {num_samples}")
    logger.info(f"Number of points: {num_points}")

    # Create data module
    data_module = PointCloudDataDirectoryModule(data_dir, num_points=num_points, batch_size=num_samples, file_mode='voxels', 
                                                augmentations=augmentations, relevant_object_categories=relevant_object_categories)
    data_module.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load all models in the checkpoint directory
    for file_name in os.listdir(checkpoint_dir):
        if file_name.endswith(".ckpt"):
            checkpoint_path = os.path.join(checkpoint_dir, file_name)
            model_name = f'{subdirectory_name}-{file_name[:-5]}'

            logger.info(f"Testing model: {model_name}")

            # Load the model
            model = PointCloudDiffusion.load_from_checkpoint(checkpoint_path)
            model.eval()
            model = model.to(device)

            test_ddpm_generation(model, model_name, num_samples, num_points)
            test_ddpm_reconstruction(model, model_name, data_module, num_samples) ## num_of_samples here should be less than batch_size of data_module

    