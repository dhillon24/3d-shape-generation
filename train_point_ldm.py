import os
import matplotlib
import matplotlib.pyplot as plt

from utils import plot_point_cloud_3d, setup_logger

# Use non-interactive backend for matplotlib
matplotlib.use('Agg') 

from datetime import datetime
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Set random seed for reproducibility
pl.seed_everything(24)

from data import PointCloudDataDirectoryModule
from diffusion import LatentDiffusion as LatentDiffusion
from networks import VAE3DLarge as VAE

import logging

def train_vae(data_module, checkpoint_path=None, timestamp=None, num_points=2048, is_voxel_based=True):
    """
    Train the VAE model.
    
    Args:
        data_module: PyTorch Lightning data module
        checkpoint_path: Path to load a pre-trained VAE model
        timestamp: Timestamp for saving checkpoints
        num_points: Number of points in each point cloud
    
    Returns:
        Trained VAE model
    """
    logger = logging.getLogger('train_point_ldm')

    # Load model from checkpoint or create a new one
    if checkpoint_path:
        logger.info(f"Loading VAE model from checkpoint: {checkpoint_path}")
        vae = VAE.load_from_checkpoint(checkpoint_path)
    else:
        if is_voxel_based:
            vae = VAE(input_shape=(32, 32, 32))  # 3DVAE
        else:
            vae = VAE(num_points=num_points)   # PointNetVAE
    
    # Set up TensorBoard logger
    tensorboard_logger = TensorBoardLogger("lightning_logs", name="vae")

    # Create timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Set up model checkpoint callbacks

    val_checkpoint = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join('checkpoints', 'point_ldm', timestamp),
        filename='vae-{epoch:02d}-{val_loss:.2f}',
        save_top_k=10,
        mode='min',
    )

    train_checkpoint = ModelCheckpoint(
        monitor='train_loss',
        dirpath=os.path.join('checkpoints', 'point_ldm', timestamp),
        filename='vae-{epoch:02d}-{train_loss:.2f}',
        save_top_k=10,
        mode='min',
    )
    
    # Create PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=500,
        devices=1 if torch.cuda.is_available() else "auto",
        logger=tensorboard_logger,
        callbacks=[val_checkpoint, train_checkpoint],
    )
    
    # Train the model
    trainer.fit(vae, data_module, ckpt_path=checkpoint_path)
    
    return vae

def train_diffusion(data_module, vae, checkpoint_path=None, timestamp=None):
    """
    Train the Latent Diffusion model.
    
    Args:
        data_module: PyTorch Lightning data module
        vae: Trained VAE model
        checkpoint_path: Path to load a pre-trained Diffusion model
        timestamp: Timestamp for saving checkpoints
    
    Returns:
        Trained Latent Diffusion model
    """
    logger = logging.getLogger('train_point_ldm')
    
    # Load model from checkpoint or create a new one
    if checkpoint_path:
        logger.info(f"Loading Diffusion model from checkpoint: {checkpoint_path}")
        diffusion = LatentDiffusion.load_from_checkpoint(checkpoint_path, vae=vae)
    else:
        diffusion = LatentDiffusion(vae)
    
    # Set up TensorBoard logger
    tensorboard_logger = TensorBoardLogger("lightning_logs", name="latent_diffusion")

    # Create timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Set up model checkpoint callbacks
    
    val_checkpoint = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join('checkpoints', 'point_ldm', timestamp),
        filename='latent_diffusion-{epoch:02d}-{val_loss:.2f}',
        save_top_k=10,
        mode='min',
    )

    train_checkpoint = ModelCheckpoint(
        monitor='train_loss',
        dirpath=os.path.join('checkpoints', 'point_ldm', timestamp),
        filename='latent_diffusion-{epoch:02d}-{train_loss:.2f}',
        save_top_k=10,
        mode='min',
    )
    
    # Create PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=500,
        devices=1 if torch.cuda.is_available() else "auto",
        logger=tensorboard_logger,
        callbacks=[val_checkpoint, train_checkpoint],
    )
    
    # Train the model
    trainer.fit(diffusion, data_module, ckpt_path=checkpoint_path)
    
    return diffusion

def main():
    # Create timestamp and logger
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(os.path.join('train', 'logs'), exist_ok=True)
    log_file = os.path.join('train', 'logs', f'train_point_ldm_log_{timestamp}.log')
    logger = setup_logger(log_file, 'train_point_ldm')

    # Set the path to your data directory
    data_dir = os.path.join('data', 'shape_net_voxel_data_v1')
    
    num_points = 2048
    augmentations = False  # Set whether to use rotation (around z axis) or jitter augmentations

    is_voxel_based = True # Set whether want to use voxel-based VAE (recommended) or point cloud-based VAE 

    # Create data module
    relevant_object_categories=['airplane'] ## ['chair'], ['table'], ['airplane'], ['all']  # Set to which shapes want to train on
    if is_voxel_based:
        data_module = PointCloudDataDirectoryModule(data_dir, num_points=num_points, batch_size=16, file_mode='voxels', output_mode='voxels',
                                                augmentations=augmentations,  # For 3DVAE, 'voxels' output_mode
                                                relevant_object_categories=relevant_object_categories) 
    else:                                                          
        data_module = PointCloudDataDirectoryModule(data_dir, num_points=num_points, batch_size=16, file_mode='voxels', output_mode='point_clouds',
                                                augmentations=augmentations,  # For PointNetVAE, 'point_clouds' output_mode
                                                relevant_object_categories=relevant_object_categories)    
    # Set training flags
    perform_vae_training = True
    perform_diffusion_training = False

    # Set vae checkpoint paths
    vae_checkpoint = None # Set to None if starting from scratch, otherwise to vae checkpoint path
    # subdirectory_name = 'airplane_from_scratch_no_augs_voxel_simoid_bce_kl_mean_beta_warmup_annealed_upto_100'
    # model_name = 'vae-epoch=56-val_loss=0.04.ckpt'
    # vae_checkpoint = os.path.join('checkpoints', 'best_run', 'point_ldm', subdirectory_name, model_name)

    # Train or load VAE
    if perform_vae_training:
        logger.info("Starting VAE Training")
        vae = train_vae(data_module, checkpoint_path=vae_checkpoint, timestamp=timestamp, is_voxel_based=is_voxel_based)
    else:
        assert vae_checkpoint is not None
        logger.info(f"Loading VAE model from checkpoint: {vae_checkpoint}")
        vae = VAE.load_from_checkpoint(vae_checkpoint)

    vae.eval()

    # Generate and save VAE samples
    num_samples = 10
    samples_vae = vae.sample(num_samples=num_samples)
    
    for i, sample in enumerate(samples_vae):
        fig = plot_point_cloud_3d(sample)
        plt.savefig(f'generated_vae_sample_{i}.png')
        plt.close(fig)

    logger.info(f"Generated {num_samples} VAE samples")

    # Set diffusion checkpoint path
    diffusion_checkpoint = None  # Set to None if starting from scratch, otherwise to diffusion checkpoint path    
    # subdirectory_name = 'no_augs_2048'
    # model_name = 'latent_diffusion=99-val_loss=0.29.ckpt'
    # vae_checkpoint = os.path.join('checkpoints', 'best_run', 'point_ldm', subdirectory_name, model_name)

    # Check if diffusion training or inference should be performed
    if not perform_diffusion_training and diffusion_checkpoint is None:
        logger.info("Skipping diffusion training and/or inference")
        return
    
    # Train or load Diffusion model
    if perform_diffusion_training:
        logger.info("Starting Latent Diffusion Training")
        diffusion = train_diffusion(data_module, vae, checkpoint_path=diffusion_checkpoint, timestamp=timestamp)
    else:
        logger.info(f"Loading Diffusion model from checkpoint: {diffusion_checkpoint}")
        diffusion = LatentDiffusion.load_from_checkpoint(diffusion_checkpoint, vae=vae, is_voxel_based=is_voxel_based)

    # Generate and save Diffusion samples
    num_samples = 10
    samples_diffusion = diffusion.sample(num_samples=num_samples)
    
    for i, sample in enumerate(samples_diffusion):
        fig = plot_point_cloud_3d(sample)
        plt.savefig(f'generated_latent_diffusion_sample_{i}.png')
        plt.close(fig)

    logging.info(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Generated {num_samples} diffusion denoised samples")

if __name__ == "__main__":
    main()