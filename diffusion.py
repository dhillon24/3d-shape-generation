import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from utils import plot_comparison_point_clouds, plot_point_cloud_3d, plot_point_cloud_2d, voxel_tensor_to_point_clouds

from networks import UNetPointNetLarge as UNetPointNetLarge 
from networks import SimpleLatentUNetPointNet as SimpleLatentUNetPointNet

class PointCloudDiffusion(pl.LightningModule):
    def __init__(self, num_points, dim=256, time_dim=256, lr=1e-4, noise_schedule = 'cosine'): ## 'cosine'
        """
        Initialize the PointCloudDiffusion model.
        
        Args:
            num_points: Number of points in each point cloud
            dim: Dimension of the model's hidden layers
            time_dim: Dimension of the time embedding
            lr: Learning rate
            noise_schedule: Type of noise schedule ('cosine' or 'linear')
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = UNetPointNetLarge(dim, time_dim)
        self.num_points = num_points
        self.lr = lr
        self.noise_schedule = noise_schedule
        self.linear_min_rate = 0.0001           # Beta values - 0.0001 (original)
        self.linear_max_rate = 0.02             # Beta values - 0.02 (original)
        self.cosine_min_signal_rate = 0.02      # 0.02
        self.cosine_max_signal_rate = 0.95      # 0.95
        self.diffusion_schedule = self.offset_cosine_diffusion_schedule if self.noise_schedule == 'cosine' \
                                    else self.linear_diffusion_schedule
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the model.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
    
    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.
        
        Args:
            batch: Input batch of point clouds
            batch_idx: Index of the current batch
        
        Returns:
            Computed loss for the batch
        """
        x_0 = batch
        batch_size = x_0.shape[0]
        t = torch.rand(batch_size, device=self.device)   
        loss = self.diffusion_loss(x_0, t)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.
        
        Args:
            batch: Input batch of point clouds
            batch_idx: Index of the current batch
        """
        x_0 = batch
        batch_size = x_0.shape[0]
        t = torch.rand(batch_size, device=self.device)  
        loss = self.diffusion_loss(x_0, t)
        self.log('val_loss', loss)

        # sampling_interval = self.trainer.num_val_batches[0] // 5
        # img_idx = batch_idx // sampling_interval

        # if batch_idx % sampling_interval == 0:
        #     fig_3d = plot_point_cloud_3d(batch[0])
        #     self.logger.experiment.add_figure(f'input_point_cloud_3d_{img_idx}_0', fig_3d, global_step=self.current_epoch)
        #     plt.close(fig_3d)
            
        #     fig_2d = plot_point_cloud_2d(batch[0])
        #     self.logger.experiment.add_figure(f'input_point_cloud_2d_{img_idx}_0', fig_2d, global_step=self.current_epoch)
        #     plt.close(fig_2d)

        sampling_interval = self.trainer.num_val_batches[0] // 5
        img_idx = batch_idx // sampling_interval
        if batch_idx % sampling_interval == 0:

            sample_idx = (0 + len(batch)) // 2

            input_point_cloud = batch[sample_idx].detach()

            initial_t = 0.01
            with torch.no_grad():
                t = torch.ones(1, device=self.device)*initial_t
                noisy_samples, _, _, _ = self.add_noise(batch[sample_idx].detach(), t)
                reconstructed_point_clouds = self.sample3(num_samples=1, num_points=batch.shape[1], x=noisy_samples, start_t=t)
                reconstructed_point_cloud = reconstructed_point_clouds[0].detach()
            
            fig_3d = plot_comparison_point_clouds(input_point_cloud.cpu().numpy(), reconstructed_point_cloud.cpu().numpy(), 
                                                  f"Point Cloud Comparison", "Input", f"Reconstructed")
            self.logger.experiment.add_figure(f'input_vs_reconstructed_point_cloud_3d_{img_idx}_{sample_idx}', fig_3d, global_step=self.current_epoch)
            plt.close(fig_3d)
            
            fig_2d = plot_point_cloud_2d(input_point_cloud)
            self.logger.experiment.add_figure(f'input_point_cloud_2d_{img_idx}_{sample_idx}', fig_2d, global_step=self.current_epoch)
            plt.close(fig_2d)

    def add_noise(self, x_0, t):
        """
        Add noise to the input point cloud.
        
        Args:
            x_0: Input point cloud
            t: Time step
        
        Returns:
            Noisy point cloud, noise, noise rates, and signal rates
        """
        noise = torch.randn_like(x_0)
        noise_rates, signal_rates = self.diffusion_schedule(t)
        x_t = signal_rates.view(-1, 1, 1) * x_0 + noise_rates.view(-1, 1, 1) * noise
        return x_t, noise, noise_rates, signal_rates

    def remove_noise(self, x_t, predicted_noise, noise_rates, signal_rates):
        """
        Remove noise from the noisy point cloud.
        
        Args:
            x_t: Noisy point cloud
            predicted_noise: Predicted noise by the model
            noise_rates: Noise rates
            signal_rates: Signal rates
        
        Returns:
            Denoised point cloud
        """
        x_0 = (x_t - noise_rates.view(-1,1,1) * predicted_noise) / signal_rates.view(-1, 1, 1)
        return x_0 

    def diffusion_loss(self, x_0, t):
        """
        Compute the diffusion loss.
        
        Args:
            x_0: Input point cloud
            t: Time step
        
        Returns:
            Computed loss
        """
        x_t, noise, _, _ = self.add_noise(x_0, t)
        predicted_noise = self.model(x_t, t)
        loss = F.l1_loss(noise, predicted_noise)     ## Note: Use MAE loss instead of MSE
        # loss = F.mse_loss(noise, predicted_noise)

        return loss
    
    # @lru_cache(maxsize=None)
    def linear_diffusion_schedule(self, diffusion_times):
        """
        Compute the linear diffusion schedule.
        
        Args:
            diffusion_times: Time steps
        
        Returns:
            Noise rates and signal rates
        """
        betas = self.linear_min_rate + torch.tensor(diffusion_times, device=diffusion_times.device) \
                                                * (self.linear_max_rate - self.linear_min_rate)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        signal_rates = alpha_bars
        noise_rates = 1 - alpha_bars
        return noise_rates, signal_rates

    # @lru_cache(maxsize=None)
    def offset_cosine_diffusion_schedule(self, diffusion_times):
        """
        Compute the offset cosine diffusion schedule.
        
        Args:
            diffusion_times: Time steps
        
        Returns:
            Noise rates and signal rates
        """
        start_angle = torch.acos(torch.tensor(self.cosine_max_signal_rate, device=diffusion_times.device))
        end_angle = torch.acos(torch.tensor(self.cosine_min_signal_rate, device=diffusion_times.device))
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
        signal_rates = torch.cos(diffusion_angles)
        noise_rates = torch.sin(diffusion_angles)
        return noise_rates, signal_rates

    @torch.no_grad()
    def sample2(self, num_samples, num_points, num_steps=1000):
        """
        Generate samples using pure DDPM sampling
        
        Args:
            num_samples: Number of samples to generate
            num_points: Number of points in each point cloud
            num_steps: Number of denoising steps
        
        Returns:
            Generated point cloud samples
        """
        self.eval()
        x_t = torch.randn(num_samples, num_points, 3, device=self.device)
        
        for i in reversed(range(num_steps)):
            t = torch.ones(num_samples, device=self.device) * i / num_steps
            noise_rates, signal_rates = self.diffusion_schedule(t)
            
            predicted_noise = self.model(x_t, t)
            x_0 = self.remove_noise(x_t, predicted_noise, noise_rates, signal_rates)
            
            if i > 0:  # Don't add noise at the last step
                t_prev = torch.ones(num_samples, device=self.device) * (i - 1) / num_steps
                noise_rates_prev, signal_rates_prev = self.diffusion_schedule(t_prev)
                
                # DDPM update
                coefficient = torch.sqrt(noise_rates_prev / noise_rates)
                noise = torch.randn_like(x_t)
                x_t = signal_rates_prev.view(-1, 1, 1) * x_0 + coefficient.view(-1, 1, 1) * noise_rates.view(-1, 1, 1) * noise
            else:
                x_t = x_0
        
        return x_t
    
    @torch.no_grad()
    def sample(self, num_samples, num_points, num_steps=1000):        
        """
        Generate samples using DDIM sampling
        
        Args:
            num_samples: Number of samples to generate
            num_points: Number of points in each point cloud
            num_steps: Number of denoising steps
        
        Returns:
            Generated point cloud samples
        """
        self.eval()
        x_t = torch.randn(num_samples, num_points, 3, device=self.device)

        step_size = 1.0 / num_steps

        for step in range(num_steps):
            t = torch.ones(num_samples, device=self.device) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(t)
            predicted_noise = self.model(x_t, t)
            x_0 = self.remove_noise(x_t, predicted_noise, noise_rates, signal_rates)
            
            next_t = t - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_t)
            x_t = next_signal_rates.view(-1, 1, 1) * x_0 + next_noise_rates.view(-1, 1, 1) * predicted_noise          ## DDIM
            
        return x_0
    
    @torch.no_grad()
    def sample3(self, num_samples, num_points, x=None, start_t=None, num_steps=1000):
        """
        Generate samples using DDIM sampling with intiail x and time step specification
        
        Args:
            num_samples: Number of samples to generate
            num_points: Number of points in each point cloud
            x: Initial point cloud (optional)
            start_t: Starting time step (optional)
            num_steps: Number of denoising steps
        
        Returns:
            Generated point cloud samples
        """
        self.eval()
        device = self.device

        # If no initial x is provided, start from random noise
        if x is None:
            x = torch.randn(num_samples, num_points, 3, device=device)
            start_t = torch.ones(num_samples, device=device)
        else:
            # Ensure x is on the correct device
            x = x.to(device)
            # If start_t is not provided, assume we're starting from the most noisy state
            if start_t is None:
                start_t = torch.ones(num_samples, device=device)
            else:
                start_t = start_t.to(device)

        # Calculate the actual number of steps
        end_t = torch.zeros(num_samples, device=device)
        steps = torch.linspace(start_t[0], end_t[0], num_steps, device=device)

        for i in range(num_steps):
            t = steps[i]
            noise_rates, signal_rates = self.diffusion_schedule(t)
            predicted_noise = self.model(x, t.expand(num_samples))
            x_0 = self.remove_noise(x, predicted_noise, noise_rates, signal_rates)
            
            if i < num_steps - 1:  # Skip this step for the last iteration
                next_t = steps[i+1]
                next_noise_rates, next_signal_rates = self.diffusion_schedule(next_t)
                x = next_signal_rates.view(-1, 1, 1) * x_0 + next_noise_rates.view(-1, 1, 1) * predicted_noise

        return x_0
    
    def on_validation_epoch_end(self, num_samples=4):
        """
        Generate and log samples at the end of each validation epoch.
        
        Args:
            num_samples: Number of samples to generate
        """
        if not self.logger:
            return

        samples = self.sample(num_samples, self.hparams.num_points)
    
        for i, sample in enumerate(samples):
            fig_3d = plot_point_cloud_3d(sample)
            self.logger.experiment.add_figure(f'diffusion_generated_sample_3d_{i}', fig_3d, global_step=self.current_epoch)
            plt.close(fig_3d)
            
            fig_2d = plot_point_cloud_2d(sample)
            self.logger.experiment.add_figure(f'diffusion_generated_sample_2d_{i}', fig_2d, global_step=self.current_epoch)
            plt.close(fig_2d)


class LatentDiffusion(pl.LightningModule):
    def __init__(self, vae, latent_dim=256, dim=512, time_dim=256, lr=1e-4, noise_schedule='cosine', is_voxel_based=True):
        """
        Initialize the LatentDiffusion model.
        
        Args:
            vae: Variational Autoencoder model
            latent_dim: Dimension of the latent space
            dim: Dimension of the model's hidden layers
            time_dim: Dimension of the time embedding
            lr: Learning rate
            noise_schedule: Type of noise schedule ('cosine' or 'linear')
        """
        super().__init__()
        self.save_hyperparameters(ignore=['vae'])
        self.vae = vae
        for param in self.vae.parameters():
            param.requires_grad = False
        
        self.model = SimpleLatentUNetPointNet(latent_dim, dim, time_dim)
        
        self.lr = lr
        self.noise_schedule = noise_schedule
        self.linear_min_rate = 0.0001
        self.linear_max_rate = 0.02 
        self.cosine_min_signal_rate = 0.02
        self.cosine_max_signal_rate = 0.95
        self.diffusion_schedule = self.offset_cosine_diffusion_schedule if self.noise_schedule == 'cosine' \
                                    else self.linear_diffusion_schedule
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the model.
        """
        for name, module in self.named_children():
            if name != 'vae':  # Skip VAE initialization
                for m in self.modules():
                    if isinstance(m, (nn.Conv1d, nn.Linear)):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.LayerNorm):
                        nn.init.constant_(m.weight, 1.0)
                        nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm1d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
    
    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.
        
        Args:
            batch: Input batch of point clouds
            batch_idx: Index of the current batch
        
        Returns:
            Computed loss for the batch
        """
        x = batch
        with torch.no_grad():
            mu, logvar = self.vae.encode(x)
            z = self.vae.reparameterize(mu, logvar)
        
        t = torch.rand(z.shape[0], device=self.device)
        loss = self.diffusion_loss(z, t)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.
        
        Args:
            batch: Input batch of point clouds
            batch_idx: Index of the current batch
        """
        x = batch
        with torch.no_grad():
            mu, logvar = self.vae.encode(x)
            z = self.vae.reparameterize(mu, logvar)
        
        t = torch.rand(z.shape[0], device=self.device)
        loss = self.diffusion_loss(z, t)
        self.log('val_loss', loss)

        sampling_interval = self.trainer.num_val_batches[0] // 5
        img_idx = batch_idx // sampling_interval
        if batch_idx % sampling_interval == 0:

            sample_idx = (0 + len(batch)) // 2

            if self.hparams.is_voxel_based:
                input_point_cloud = voxel_tensor_to_point_clouds(batch[sample_idx].detach().unsqueeze(0))[0]   
            else:
                input_point_cloud = batch[sample_idx].detach()   

            initial_t = 0.01
            with torch.no_grad():
                t = torch.ones(1, device=self.device)*initial_t
                noisy_samples, _, _, _ = self.add_noise(z[0].unsqueeze(0), t)
                reconstructed_point_clouds = self.sample3(num_samples=1, z=noisy_samples, start_t=t)
                reconstructed_point_cloud = reconstructed_point_clouds[0]
            
            fig_3d = plot_comparison_point_clouds(input_point_cloud.cpu().numpy(), reconstructed_point_cloud.cpu().numpy(), 
                                                  f"Point Cloud Comparison", "Input", f"Reconstructed")
            self.logger.experiment.add_figure(f'input_vs_reconstructed_point_cloud_3d_{img_idx}_{sample_idx}', fig_3d, global_step=self.current_epoch)
            plt.close(fig_3d)
            
            fig_2d = plot_point_cloud_2d(input_point_cloud)
            self.logger.experiment.add_figure(f'input_point_cloud_2d_{img_idx}_{sample_idx}', fig_2d, global_step=self.current_epoch)
            plt.close(fig_2d)    

    def add_noise(self, z_0, t):
        """
        Add noise to the input latent vector.
        
        Args:
            z_0: Input latent vector
            t: Time step
        
        Returns:
            Noisy latent vector, noise, noise rates, and signal rates
        """
        noise = torch.randn_like(z_0)
        noise_rates, signal_rates = self.diffusion_schedule(t)
        z_t = signal_rates.view(-1, 1) * z_0 + noise_rates.view(-1, 1) * noise
        return z_t, noise, noise_rates, signal_rates

    def remove_noise(self, z_t, predicted_noise, noise_rates, signal_rates):
        """
        Remove noise from the noisy latent vector.
        
        Args:
            z_t: Noisy latent vector
            predicted_noise: Predicted noise by the model
            noise_rates: Noise rates
            signal_rates: Signal rates
        
        Returns:
            Denoised latent vector
        """
        z_0 = (z_t - noise_rates.view(-1,1) * predicted_noise) / signal_rates.view(-1, 1)
        return z_0 

    def diffusion_loss(self, z_0, t):
        """
        Compute the diffusion loss.
        
        Args:
            z_0: Input latent vector
            t: Time step
        
        Returns:
            Computed loss
        """
        z_t, noise, _, _ = self.add_noise(z_0, t)
        predicted_noise = self.model(z_t, t)
        loss = F.l1_loss(noise, predicted_noise)
        return loss
    
    # @lru_cache(maxsize=None)
    def linear_diffusion_schedule(self, diffusion_times):
        """
        Compute the linear diffusion schedule.
        
        Args:
            diffusion_times: Time steps
        
        Returns:
            Noise rates and signal rates
        """
        betas = self.linear_min_rate + torch.tensor(diffusion_times, device=diffusion_times.device) \
                                                * (self.linear_max_rate - self.linear_min_rate)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        signal_rates = alpha_bars
        noise_rates = 1 - alpha_bars
        return noise_rates, signal_rates

    # @lru_cache(maxsize=None)
    def offset_cosine_diffusion_schedule(self, diffusion_times):
        """
        Compute the offset cosine diffusion schedule.
        
        Args:
            diffusion_times: Time steps
        
        Returns:
            Noise rates and signal rates
        """
        start_angle = torch.acos(torch.tensor(self.cosine_max_signal_rate, device=diffusion_times.device))
        end_angle = torch.acos(torch.tensor(self.cosine_min_signal_rate, device=diffusion_times.device))
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
        signal_rates = torch.cos(diffusion_angles)
        noise_rates = torch.sin(diffusion_angles)
        return noise_rates, signal_rates

    @torch.no_grad()
    def sample2(self, num_samples, num_steps=1000, threshold=0.4):
        """
        Generate samples using pure DDPM sampling in latent space
        
        Args:
            num_samples: Number of samples to generate
            num_steps: Number of denoising steps
            threshold: Threshold for voxelization if voxel based VAE
        
        Returns:
            Generated point cloud samples
        """
        self.eval()
        z_t = torch.randn(num_samples, self.hparams.latent_dim, device=self.device)
        
        for i in reversed(range(num_steps)):
            t = torch.ones(num_samples, device=self.device) * i / num_steps
            noise_rates, signal_rates = self.diffusion_schedule(t)
            
            predicted_noise = self.model(z_t, t)
            z_0 = self.remove_noise(z_t, predicted_noise, noise_rates, signal_rates)
            
            if i > 0:  # Don't add noise at the last step
                t_prev = torch.ones(num_samples, device=self.device) * (i - 1) / num_steps
                noise_rates_prev, signal_rates_prev = self.diffusion_schedule(t_prev)
                
                # DDPM update
                coefficient = torch.sqrt(noise_rates_prev / noise_rates)
                noise = torch.randn_like(z_t)
                z_t = signal_rates_prev.view(-1, 1) * z_0 + coefficient.view(-1, 1) * noise_rates.view(-1, 1) * noise
            else:
                z_t = z_0
        
        x_0 = self.vae.decode(z_t)

        if self.hparams.is_voxel_based:
            point_clouds = voxel_tensor_to_point_clouds(x_0, threshold=threshold)
        else:
            point_clouds = x_0  # Assuming x_0 is already in point cloud format if not voxel-based

        return point_clouds

    
    @torch.no_grad()
    def sample(self, num_samples, num_steps=1000, threshold=0.4):
        """
        Generate samples using DDIM sampling
        
        Args:
            num_samples: Number of samples to generate
            num_steps: Number of denoising steps
            threshold: Threshold for voxelization if voxel based VAE
        
        Returns:
            Generated point cloud samples
        """
        self.eval()
        z_t = torch.randn(num_samples, self.hparams.latent_dim, device=self.device)

        step_size = 1.0 / num_steps

        for step in range(num_steps):
            t = torch.ones(num_samples, device=self.device) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(t)
            predicted_noise = self.model(z_t, t)
            z_0 = self.remove_noise(z_t, predicted_noise, noise_rates, signal_rates)
            
            next_t = t - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_t)
            z_t = next_signal_rates.view(-1, 1) * z_0 + next_noise_rates.view(-1, 1) * predicted_noise
        
        # Decode the final latent vectors
        x_0 = self.vae.decode(z_0)

        if self.hparams.is_voxel_based:
            point_clouds = voxel_tensor_to_point_clouds(x_0, threshold=threshold)

        return point_clouds
    
    @torch.no_grad()
    def sample3(self, num_samples, z=None, start_t=None, num_steps=1000, threshold=0.4):
        """
        Generate samples using DDIM sampling with initial latent vector or time step
        
        Args:
            num_samples: Number of samples to generate
            num_points: Number of points in each point cloud
            z: Initial point cloud (optional)
            start_t: Starting time step (optional)
            num_steps: Number of denoising steps
            threshold: Threshold for voxelization if voxel based VAE
        
        Returns:
            Generated point cloud samples
        """
        self.eval()
        device = self.device

        # If no initial z is provided, start from random noise
        if z is None:
            z = torch.randn(num_samples, self.hparams.latent_dim, device=device)
            start_t = torch.ones(num_samples, device=device)
        else:
            # Ensure z is on the correct device
            z = z.to(device)
            # If start_t is not provided, assume we're starting from the most noisy state
            if start_t is None:
                start_t = torch.ones(num_samples, device=device)
            else:
                start_t = start_t.to(device)

        # Calculate the actual number of steps
        end_t = torch.zeros(num_samples, device=device)
        steps = torch.linspace(start_t[0], end_t[0], num_steps, device=device)

        for i in range(num_steps):
            t = steps[i]
            noise_rates, signal_rates = self.diffusion_schedule(t)
            predicted_noise = self.model(z, t.expand(num_samples))
            z_0 = self.remove_noise(z, predicted_noise, noise_rates, signal_rates)
            
            if i < num_steps - 1:  # Skip this step for the last iteration
                next_t = steps[i+1]
                next_noise_rates, next_signal_rates = self.diffusion_schedule(next_t)
                z = next_signal_rates.view(-1, 1) * z_0 + next_noise_rates.view(-1, 1) * predicted_noise

        x_0 = self.vae.decode(z_0)
        
        if self.hparams.is_voxel_based:
            point_clouds = voxel_tensor_to_point_clouds(x_0, threshold=threshold)

        return point_clouds
    
    def on_train_epoch_end(self):
        """
        Callback method called at the end of each training epoch.
        """
        pass
    
    def on_validation_epoch_end(self, num_samples=4):
        """
        Generate and log samples at the end of each validation epoch.
        
        Args:
            num_samples: Number of samples to generate
        """
        if not self.logger:
            return

        samples = self.sample(num_samples)
    
        for i, sample in enumerate(samples):
            fig_3d = plot_point_cloud_3d(sample)
            self.logger.experiment.add_figure(f'latent_diffusion_generated_sample_3d_{i}', fig_3d, global_step=self.current_epoch)
            plt.close(fig_3d)
            
            fig_2d = plot_point_cloud_2d(sample)
            self.logger.experiment.add_figure(f'latent_diffusion_generated_sample_2d_{i}', fig_2d, global_step=self.current_epoch)
            plt.close(fig_2d)
    

        

