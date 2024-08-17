
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

from networks import VAE3DLarge as VAE3D
from data import PointCloudDataDirectoryModule
from metrics import compute_metrics
from utils import plot_point_cloud_2d, plot_point_cloud_3d, save_point_cloud_comparison, voxel_tensor_to_point_clouds
import logging
from datetime import datetime

from utils import setup_logger, save_point_cloud

def test_vae_generation(model, model_name, num_samples=10, threshold=0.5):
    """
    Test the generation capability of a Variational Autoencoder (VAE) model.

    Args:
        model (VAE): The VAE model to test.
        model_name (str): The name of the model.
        num_samples (int, optional): The number of samples to generate. Defaults to 10.
        threshold (float, optional): The threshold value for generating samples. Defaults to 0.5.
    """

    # Generate samples
    with torch.no_grad():
        generated_samples = model.sample(num_samples=num_samples, threshold=threshold)

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

    logger = logging.getLogger('test_logger_point_ldm')
    logger.info(f"Generated and saved {num_samples} samples.")

def test_vae_reconstruction(model, model_name, data_module, num_samples=10, initial_t=0.010, threshold=0.5):
    """
    Test the VAE model's reconstruction performance on a given dataset.

    Args:
        model (torch.nn.Module): The VAE model to be tested.
        model_name (str): The name of the model.
        data_module: The data module containing the validation dataset.
        num_samples (int, optional): The number of samples to be used for testing. Defaults to 10.
        initial_t (float, optional): The initial timestamp for denoising before reconstruction. Defaults to 0.010.
        threshold (float, optional): The threshold value for converting voxel tensors to point clouds. Defaults to 0.4.
    """

    # Get samples from the validation set
    val_dataloader = data_module.val_dataloader()
    original_samples = next(iter(val_dataloader))[:num_samples]

    # Generate reconstructed samples
    with torch.no_grad():
        original_samples = original_samples.to(model.device)
        reconstructed_voxels, _, _ = model(original_samples)
        original_samples = voxel_tensor_to_point_clouds(original_samples.detach(), threshold)
        reconstructed_samples = voxel_tensor_to_point_clouds(reconstructed_voxels.detach(), threshold)
        
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

    logger = logging.getLogger('test_logger_point_ldm')
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
        save_point_cloud_comparison(orig.cpu().numpy(), recon.cpu().numpy(), vis_path, title=title, title1='Original', title2='Reconstructed')

        # Point cloud files
        # save_point_cloud(orig, os.path.join('test', 'point_clouds', model_name, f'original_{i}.txt'))
        # save_point_cloud(recon, os.path.join('test', 'point_clouds', model_name, f'reconstructed_{i}.txt'))

    logger.info(f"Reconstructed and saved {num_samples} samples.")

if __name__ == "__main__":
    augmentations = False
    relevant_object_categories=['table'] ## ['chair'], ['table'], ['airplane']  ## change this and model name, only affects the reconstruction test
    subdirectory_name = f"{relevant_object_categories[0]}_from_scratch_no_augs_voxel_simoid_bce_kl_mean_beta_warmup_annealed_upto_100"  ## change this to the subdirectory that you want to test, all vae models in it will be tested
    checkpoint_dir = os.path.join('checkpoints', 'best_run', 'point_ldm', subdirectory_name)
    data_dir = os.path.join('data', 'shape_net_voxel_data_v1')  
    num_samples = 16
    threshold = 0.5

    # Setup logger
    os.makedirs(os.path.join('test', 'logs'), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join('test', 'logs', f'test_point_ldm_log_{timestamp}.log')
    logger = setup_logger(log_file, 'test_logger_point_ldm')

    logger.info("Starting test script")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Number of samples: {num_samples}")

    # Create data module 
    data_module = PointCloudDataDirectoryModule(data_dir, num_points=2048, batch_size=16, file_mode='voxels', output_mode='voxels', # For 3DVAE, 'voxels' output_mode
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
            model = VAE3D.load_from_checkpoint(checkpoint_path)
            model.eval()
            model = model.to(device)

            test_vae_generation(model, model_name, num_samples=num_samples, threshold=threshold)
            test_vae_reconstruction(model, model_name, data_module, num_samples=num_samples, threshold=threshold) ## num_of_samples here should be less than batch_size of data_module

    