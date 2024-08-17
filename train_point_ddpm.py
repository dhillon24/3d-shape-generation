import os
import matplotlib.pyplot as plt
import matplotlib

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
from diffusion import PointCloudDiffusion

import logging

def main():
    # Create timestamp and logger
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(os.path.join('train', 'logs'), exist_ok=True)
    log_file = os.path.join('train', 'logs', f'train_point_ldm_log_{timestamp}.log')
    logger = setup_logger(log_file, 'train_point_ddpm')

    # Specify the path to a previous model checkpoint to resume training, or set to None to start training from scratch
    model_checkpoint = None
    # subdirectory_name = 'with_augs_2048'
    # model_name = 'point_cloud_diffusion-epoch=99-val_loss=0.19.ckpt'
    # mode_checkpoint = os.path.join('checkpoints', 'best_run', 'point_ddpm', subdirectory_name, model_name)

    # Set the path to your data directory
    data_dir = os.path.join('data', 'shape_net_voxel_data_v1')

    num_points = 2048  # Number of points in each point cloud
    augmentations = False # Set whether to use rotation (around z axis) or jitter augmentations

    # Create data module
    relevant_object_categories=['chair'] ## ['chair'], ['table'], ['airplane'], ['all']  # Set to which shapes want to train on
    data_module = PointCloudDataDirectoryModule(data_dir, num_points=num_points, batch_size=16, file_mode='voxels', 
                                                output_mode='point_clouds',augmentations=augmentations, 
                                                relevant_object_categories=relevant_object_categories)

    # Create new model or load from checkpoint
    if model_checkpoint:
        logger.info(f"Loading Diffusion model from checkpoint: {model_checkpoint}")
        model = PointCloudDiffusion.load_from_checkpoint(model_checkpoint)
        assert model.num_points == num_points
    else:
        model = PointCloudDiffusion(num_points=num_points)

    # Create TensorBoard logger
    tesnorboard_logger = TensorBoardLogger("lightning_logs", name="point_cloud_diffusion")

    # Create checkpoint callbacks

    val_checkpoint = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join('checkpoints', 'point_ddpm', timestamp),
        filename='point_cloud_diffusion-{epoch:02d}-{val_loss:.2f}',
        save_top_k=10,
        mode='min',
    )

    train_checkpoint = ModelCheckpoint(
        monitor='train_loss',
        dirpath=os.path.join('checkpoints', 'point_ddpm', timestamp),
        filename='point_cloud_diffusion-{epoch:02d}-{train_loss:.2f}',
        save_top_k=10,
        mode='min',
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=500,
        devices=1 if torch.cuda.is_available() else "auto",
        logger=tesnorboard_logger,
        callbacks=[val_checkpoint, train_checkpoint],
    )

    # Train the model
    logger.info("Starting Diffusion Training")
    trainer.fit(model, data_module)

    # Generate multiple samples after training
    num_samples = 10  # Number of point clouds to generate
    samples = model.sample(num_samples=num_samples, num_points=num_points)

    # Save or visualize the samples
    for i, sample in enumerate(samples):
        fig = plot_point_cloud_3d(sample)
        plt.savefig(f'generated_diffusion_sample_{i}.png')
        plt.close(fig)

    logging.info(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Generated {num_samples} samples")

if __name__ == "__main__":
    main()