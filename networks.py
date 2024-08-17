import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from metrics import chamfer_distance, earth_mover_distance_gpu, voxel_focal_loss
from utils import farthest_point_sample, index_points, plot_comparison_point_clouds, plot_point_cloud_3d, plot_point_cloud_2d
from utils import square_distance, voxel_tensor_to_point_clouds, voxelize, voxel_to_point_cloud

############# Layers #################

class PointNetLayer(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim = None):
        """
        Initialize a PointNet layer.
        
        Args:
            in_dim (int): Input dimension
            mid_dim (int): Middle dimension
            out_dim (int, optional): Output dimension. If None, set to mid_dim
        """
        super().__init__()
        if out_dim is None:
            out_dim = mid_dim
        self.conv1 = nn.Conv1d(in_dim, mid_dim, 1)
        self.bn1 = nn.BatchNorm1d(mid_dim)
        self.conv2 = nn.Conv1d(mid_dim, mid_dim, 1)
        self.bn2 = nn.BatchNorm1d(mid_dim)
        self.conv3 = nn.Conv1d(mid_dim, out_dim, 1)
        self.bn3 = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        """
        Forward pass of the PointNet layer.
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output tensor after passing through the layer
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x
    
class SetAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):
        """
        Initialize a Set Attention Block.
        
        Args:
            dim (int): Dimension of the input
            num_heads (int): Number of attention heads
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        self.ln2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        """
        Forward pass of the Set Attention Block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C)
        
        Returns:
            torch.Tensor: Output tensor of shape (B, N, C)
        """
        x = x.transpose(0, 1)  # (N, B, C) for nn.MultiheadAttention
        x = x + self.attention(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.ff(self.ln2(x))
        return x.transpose(0, 1)  # (B, N, C)

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim_x, dim_y, num_heads=4):
        """
        Initialize a Cross Attention Block.
        
        Args:
            dim_x (int): Dimension of the first input
            dim_y (int): Dimension of the second input
            num_heads (int): Number of attention heads
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(dim_x, num_heads)
        self.ln_x = nn.LayerNorm(dim_x)
        self.ln_y = nn.LayerNorm(dim_y)
        self.ff = nn.Sequential(
            nn.Linear(dim_x, dim_x * 4),
            nn.ReLU(),
            nn.Linear(dim_x * 4, dim_x)
        )
        self.ln_out = nn.LayerNorm(dim_x)
        self.proj_y = nn.Linear(dim_y, dim_x)
        
    def forward(self, x, y):
        """
        Forward pass of the Cross Attention Block.
        
        Args:
            x (torch.Tensor): First input tensor of shape (B, C_x, N)
            y (torch.Tensor): Second input tensor of shape (B, C_y, N)
        
        Returns:
            torch.Tensor: Output tensor of shape (B, C_x, N)
        """
        B, C_x, N = x.shape
        _, C_y, _ = y.shape
        
        x = self.ln_x(x.transpose(1, 2))  # (B, N, C_x)
        y = self.ln_y(y.transpose(1, 2))  # (B, N, C_y)
        
        y = self.proj_y(y)  # (B, N, C_x)
        
        x = x.transpose(0, 1)  # (N, B, C_x)
        y = y.transpose(0, 1)  # (N, B, C_x)
        
        attn_output, _ = self.attention(x, y, y)
        
        attn_output = attn_output.transpose(0, 1)  # (B, N, C_x)
        x = x.transpose(0, 1) + attn_output  # (B, N, C_x)
        
        x = x + self.ff(self.ln_out(x))
        
        return x.transpose(1, 2)  # (B, C_x, N)

class PointNetLayerWithAttention(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim=None):
        """
        Initialize a PointNet layer with attention.
        
        Args:
            in_dim (int): Input dimension
            mid_dim (int): Middle dimension
            out_dim (int, optional): Output dimension. If None, set to mid_dim
        """
        super().__init__()
        if out_dim is None:
            out_dim = mid_dim
        self.conv1 = nn.Conv1d(in_dim, mid_dim, 1)
        self.bn1 = nn.BatchNorm1d(mid_dim)
        self.conv2 = nn.Conv1d(mid_dim, mid_dim, 1)
        self.bn2 = nn.BatchNorm1d(mid_dim)
        self.conv3 = nn.Conv1d(mid_dim, out_dim, 1)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.attention = nn.MultiheadAttention(out_dim, num_heads=4)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x):
        """
        Forward pass of the PointNet layer with attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, N)
        
        Returns:
            torch.Tensor: Output tensor of shape (B, C, N)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        # Self-attention with residual connection
        x_attn = x.transpose(1, 2)  # (B, C, N) -> (B, N, C)
        x_attn, _ = self.attention(self.ln(x_attn), self.ln(x_attn), self.ln(x_attn))
        x = x + x_attn.transpose(1, 2)  # Add residual connection
        
        x = F.relu(x)
        return x

class SetAbstraction(nn.Module):  ## PointNet++ Element, refer original paper
    def __init__(self, npoint, radius, nsample, in_channel, mlp, first_layer=False):
        """
        Initialize a Set Abstraction layer for PointNet++.
        
        Args:
            npoint (int): Number of points to sample
            radius (float): Radius of the ball query
            nsample (int): Maximum number of points in each ball query
            in_channel (int): Number of input channels
            mlp (list): List of output channels for each MLP layer
            first_layer (bool): Whether this is the first layer
        """
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + (3 if not first_layer else 0)
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Forward pass of the Set Abstraction layer.
        
        Args:
            xyz (torch.Tensor): Input points position data, [B, C, N]
            points (torch.Tensor): Input points data, [B, D, N]
        
        Returns:
            new_xyz (torch.Tensor): Sampled points position data, [B, C, S]
            new_points_concat (torch.Tensor): Sample points feature data, [B, D', S]
        """
        if self.npoint is None and self.radius is None and self.nsample is None:
            return self.forward_global(xyz, points)
        else:
            return self.forward_local(xyz, points)

    def forward_local(self, xyz, points):
        """
        Local forward pass of the Set Abstraction layer.
        
        Args:
            xyz (torch.Tensor): Input points position data, [B, C, N]
            points (torch.Tensor): Input points data, [B, D, N]
        
        Returns:
            new_xyz (torch.Tensor): Sampled points position data, [B, C, S]
            new_points_concat (torch.Tensor): Sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = min(self.npoint, N)  # Ensure we don't sample more points than we have

        # Select a subset of points as centroids using Furthest Point Sampling
        fps_idx = farthest_point_sample(xyz, S)  # [B, S]
        new_xyz = index_points(xyz, fps_idx)  # [B, S, C]

        # Find neighboring points for each centroid using K-Nearest Neighbors. TODO: Use self.radius
        idx = square_distance(new_xyz, xyz).argsort()[:, :, :self.nsample]  # [B, S, nsample]

        # Group the neighboring points and their features
        grouped_xyz = index_points(xyz, idx)  # [B, S, nsample, C]
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)  # Normalize coordinates
        
        if points is not None:
            grouped_points = index_points(points, idx)  # [B, S, nsample, D]
            grouped_points = torch.cat([grouped_points, grouped_xyz_norm], dim=-1)  # [B, S, nsample, D+C]
        else:
            grouped_points = grouped_xyz_norm

        # Apply MLPs to extract features and aggregate them
        grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D+C, nsample, S]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_points = F.relu(bn(conv(grouped_points)))

        # Aggregate features across each group using max pooling
        new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]

        new_xyz = new_xyz.permute(0, 2, 1)  # [B, C, S]
        return new_xyz, new_points

    def forward_global(self, xyz, points):
        """
        Global forward pass of the Set Abstraction layer.
        
        Args:
            xyz (torch.Tensor): Input points position data, [B, C, N]
            points (torch.Tensor): Input points data, [B, D, N]
        
        Returns:
            new_xyz (torch.Tensor): Input points position data, [B, C, N]
            new_points (torch.Tensor): Global feature, [B, D', 1]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape

        # Use all points
        new_xyz = xyz

        # Combine xyz and points features
        if points is not None:
            new_points = torch.cat([xyz, points], dim=-1)
        else:
            new_points = xyz

        # Apply MLPs to extract features
        new_points = new_points.permute(0, 2, 1).unsqueeze(2)  # [B, D+C, 1, N]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # Global max pooling
        new_points = torch.max(new_points, -1)[0]  # [B, D', 1]

        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points
    
class FeaturePropagation(nn.Module):    ## PointNet++ Element, refer original paper
    def __init__(self, in_channel, mlp):
        """
        Initialize a Feature Propagation layer for PointNet++.
        
        Args:
            in_channel (int): Number of input channels
            mlp (list): List of output channels for each MLP layer
        """
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Forward pass of the Feature Propagation layer.
        
        Args:
            xyz1 (torch.Tensor): Points position data of set 1, [B, N, C]
            xyz2 (torch.Tensor): Points position data of set 2, [B, S, C]
            points1 (torch.Tensor): Points feature data of set 1, [B, N, D]
            points2 (torch.Tensor): Points feature data of set 2, [B, S, D]
        
        Returns:
            new_points (torch.Tensor): Upsampled points feature data, [B, D', N]
        """
        # Reshape inputs
        xyz1 = xyz1.permute(0, 2, 1)  # [B, N, C]
        xyz2 = xyz2.permute(0, 2, 1)  # [B, S, C]
        points2 = points2.permute(0, 2, 1)  # [B, S, D]

        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if points2.shape[1] == 1:
            points2 = points2.repeat(1, S, 1)  # Repeat along the second dimension to match S

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            # Compute pairwise distances
            dists = square_distance(xyz1, xyz2)
            
            # Find k (k=3) nearest neighbors
            dists, idx = dists.topk(k=3, dim=-1, largest=False)  # [B, N, 3]

            # Compute weights for interpolation
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm  # [B, N, 3]

            # Perform weighted interpolation
            indexed_points = index_points(points2, idx)
            interpolated_points = torch.sum(indexed_points * weight.unsqueeze(-1), dim=2)

        # Concatenate with existing features (if any)
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)  # [B, N, D]
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        # Apply MLP to refine features
        new_points = new_points.permute(0, 2, 1)  # [B, D', N]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        return new_points

class FoldingLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Initialize a Folding layer.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super(FoldingLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 1)
        )

    def forward(self, x):
        """
        Forward pass of the Folding layer.
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output tensor after folding operation
        """
        return self.layer(x)
    

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Initialize a 3D Convolutional Block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int): Stride for the convolution
        """
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the 3D Convolutional Block.
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output tensor after convolution, batch normalization, and ReLU
        """
        return self.relu(self.bn(self.conv(x)))

class Deconv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, output_padding=0):
        """
        Initialize a 3D Deconvolutional Block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int): Stride for the deconvolution
            output_padding (int): Additional size added to one side of the output shape
        """
        super().__init__()
        self.deconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=output_padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the 3D Deconvolutional Block.
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output tensor after deconvolution, batch normalization, and ReLU
        """
        return self.relu(self.bn(self.deconv(x)))


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Initialize a 3D Residual Block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        """
        Forward pass of the 3D Residual Block.
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output tensor after residual connection and ReLU
        """
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

############# Diffusion Architectures #################

class SimpleUNetPointNet(nn.Module):
    def __init__(self, num_points, dim=256, time_dim=256):
        """
        Initialize a Simple UNet-like PointNet architecture.
        
        Args:
            num_points (int): Number of points in the point cloud
            dim (int): Dimension of hidden layers
            time_dim (int): Dimension of time embedding
        """
        super().__init__()
        self.time_dim = time_dim
        
        # Encoder
        self.enc1 = PointNetLayer(3 + time_dim, 64)
        self.enc2 = PointNetLayer(64, 128)
        self.enc3 = PointNetLayer(128, 256)
        
        # Global features
        self.global_feat = nn.Sequential(
            nn.Conv1d(256, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        
        # Decoder
        self.dec3 = PointNetLayer(1024 + 256, 256)
        self.dec2 = PointNetLayer(256 + 128, 128)
        self.dec1 = PointNetLayer(128 + 64, 64)

        # Output layer
        self.output = nn.Conv1d(64, 3, 1)
        
    def forward(self, x, t):
        """
        Forward pass of the Simple UNet-like PointNet.
        
        Args:
            x (torch.Tensor): Input point cloud tensor of shape (B, N, 3)
            t (torch.Tensor): Time embedding tensor
        
        Returns:
            torch.Tensor: Output point cloud tensor of shape (B, N, 3)
        """
        # Time embedding
        t_embed = self.get_timestep_embedding(t, self.time_dim)
        
        # Concatenate time embedding with input
        x = x.transpose(2, 1)  # (B, N, 3) -> (B, 3, N)
        t_embed = t_embed.unsqueeze(2).expand(-1, -1, x.shape[2])  # (B, time_dim, N)
        x = torch.cat([x, t_embed], dim=1)  # (B, 3 + time_dim, N)

        # Encoder
        x1 = self.enc1(x)  # (B, 64, N)
        x2 = self.enc2(x1)  # (B, 128, N)
        x3 = self.enc3(x2)  # (B, 256, N)

        # Global features
        global_feat = self.global_feat(x3)  # (B, 1024, N)
        global_feat = torch.max(global_feat, 2, keepdim=True)[0]  # (B, 1024, 1)
        global_feat = global_feat.repeat(1, 1, x.shape[2])  # (B, 1024, N)

        # Decoder
        x = self.dec3(torch.cat([global_feat, x3], dim=1))  # (B, 256, N)
        x = self.dec2(torch.cat([x, x2], dim=1))  # (B, 128, N)
        x = self.dec1(torch.cat([x, x1], dim=1))  # (B, 64, N)

        x = self.output(x)  # (B, 3, N)

        return x.transpose(2, 1)  # (B, N, 3)
    
    def get_timestep_embedding(self, timesteps, embedding_dim):
        """
        Create sinusoidal time embeddings.
        
        Args:
            timesteps (torch.Tensor): Tensor of timestep values
            embedding_dim (int): Dimension of the embedding
        
        Returns:
            torch.Tensor: Time embedding tensor
        """
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0, device=timesteps.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb.float()

class UNetAttentionPointExperimental(nn.Module): ## Experimental architecture not used currently
    def __init__(self, num_points, dim=256, num_heads=4, num_blocks=3, time_dim=256):
        """
        Initialize an experimental UNet-like PointNet architecture with attention.
        
        Args:
            num_points (int): Number of points in the point cloud
            dim (int): Dimension of hidden layers
            num_heads (int): Number of attention heads
            num_blocks (int): Number of attention blocks
            time_dim (int): Dimension of time embedding
        """
        super().__init__()
        self.time_dim = time_dim
        
        # Embedding layers for time
        self.emb1 = nn.Linear(time_dim, 3)          # Embedding layer for enc1
        self.emb2 = nn.Linear(time_dim, 64)         # Embedding layer for enc2
        self.emb3 = nn.Linear(time_dim, 128)        # Embedding layer for enc3
        self.emb_dec3 = nn.Linear(time_dim, 256)    # Embedding layer for dec3
        self.emb_dec2 = nn.Linear(time_dim, 128)    # Embedding layer for dec2
        self.emb_dec1 = nn.Linear(time_dim, 64)     # Embedding layer for dec1

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        
        # Encoder
        self.enc1 = PointNetLayer(3, 64)            # Input: (B, 3, N) -> Output: (B, 64, N)
        self.att1 = SetAttentionBlock(64, num_heads)      # Input: (B, N, 64) -> Output: (B, N, 64)
        
        self.enc2 = PointNetLayer(64, 128)          # Input: (B, 64, N) -> Output: (B, 128, N)
        self.att2 = SetAttentionBlock(128, num_heads)     # Input: (B, N, 128) -> Output: (B, N, 128)
        
        self.enc3 = PointNetLayer(128, 256)         # Input: (B, 128, N) -> Output: (B, 256, N)
        self.att3 = SetAttentionBlock(256, num_heads)     # Input: (B, N, 256) -> Output: (B, N, 256)
        
        # Bottleneck
        self.bottleneck = SetAttentionBlock(256, num_heads)  # Input: (B, N, 256) -> Output: (B, N, 256)
        
        # Decoder
        self.att_dec3 = SetAttentionBlock(256, num_heads)     # Input: (B, N, 256) -> Output: (B, N, 256)
        self.dec3 = PointNetLayer(256 + 256, 128)       # Input: (B, 256 + 256, N) -> Output: (B, 128, N)
        
        self.att_dec2 = SetAttentionBlock(128, num_heads)     # Input: (B, N, 128) -> Output: (B, N, 128)
        self.dec2 = PointNetLayer(128 + 128, 64)        # Input: (B, 128 + 128, N) -> Output: (B, 64, N)
        
        self.att_dec1 = SetAttentionBlock(64, num_heads)      # Input: (B, N, 64) -> Output: (B, N, 64)
        self.dec1 = PointNetLayer(64 + 64, 3)           # Input: (B, 64 + 64, N) -> Output: (B, 3, N)

        # Output layer
        self.output = nn.Conv1d(3, 3, 1)
        
    def forward(self, x, t):
        """
        Forward pass of the UNetAttentionPointExperimental.
        
        Args:
            x (torch.Tensor): Input point cloud tensor of shape (B, N, 3)
            t (torch.Tensor): Time embedding tensor
        
        Returns:
            torch.Tensor: Output point cloud tensor of shape (B, N, 3)
        """
        # Time embedding
        t_embed = self.get_timestep_embedding(t, self.time_dim)
        t_embed = self.time_mlp(t_embed)                     

        # Encoder
        t_emb1 = self.emb1(t_embed).unsqueeze(2)  # (B, 3, 1)
        x = x.transpose(2, 1) + t_emb1  # (B, 3, N) + (B, 3, 1) -> (B, 3, N)
        x1 = self.enc1(x)  # (B, 64, N)
        x1 = self.att1(x1.transpose(2, 1)).transpose(2, 1)  # (B, 64, N)

        t_emb2 = self.emb2(t_embed).unsqueeze(2)  # (B, 64, 1)
        x1 = x1 + t_emb2  # (B, 64, N)
        x2 = self.enc2(x1)  # (B, 128, N)
        x2 = self.att2(x2.transpose(2, 1)).transpose(2, 1)  # (B, 128, N)

        t_emb3 = self.emb3(t_embed).unsqueeze(2)  # (B, 128, 1)
        x2 = x2 + t_emb3  # (B, 128, N)
        x3 = self.enc3(x2)  # (B, 256, N)
        x3 = self.att3(x3.transpose(2, 1)).transpose(2, 1)  # (B, 256, N)

        # Bottleneck
        x_b = self.bottleneck(x3.transpose(2, 1)).transpose(2, 1)  # (B, 256, N)

        # Decoder
        t_emb_dec3 = self.emb_dec3(t_embed).unsqueeze(2)  # (B, 256, 1)
        x_b = x_b + t_emb_dec3  # (B, 256, N)
        x_b = self.att_dec3(x_b.transpose(2, 1)).transpose(2, 1)  # (B, 256, N)
        x = self.dec3(torch.cat([x_b, x3], dim=1))  # (B, 128, N)

        t_emb_dec2 = self.emb_dec2(t_embed).unsqueeze(2)  # (B, 128, 1)
        x = x + t_emb_dec2  # (B, 128, N)
        x = self.att_dec2(x.transpose(2, 1)).transpose(2, 1)  # (B, 128, N)
        x = self.dec2(torch.cat([x, x2], dim=1))  # (B, 64, N)

        t_emb_dec1 = self.emb_dec1(t_embed).unsqueeze(2)  # (B, 64, 1)
        x = x + t_emb_dec1  # (B, 64, N)
        x = self.att_dec1(x.transpose(2, 1)).transpose(2, 1)  # (B, 64, N)
        x = self.dec1(torch.cat([x, x1], dim=1))  # (B, 3, N)

        x = self.output(x).transpose(2, 1)  # (B, N, 3)

        return x  # (B, N, 3)
    
    def get_timestep_embedding(self, timesteps, embedding_dim):
        """
        Create sinusoidal time embeddings.
        
        Args:
            timesteps (torch.Tensor): Tensor of timestep values
            embedding_dim (int): Dimension of the embedding
        
        Returns:
            torch.Tensor: Time embedding tensor
        """
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0, device=timesteps.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb.float()

class UNetPointNetLarge(nn.Module):
    def __init__(self, dim=512, time_dim=256):
        """
        Initialize a large UNet-like PointNet architecture.
        
        Args:
            dim (int): Dimension of hidden layers
            time_dim (int): Dimension of time embedding
        """
        super().__init__()
        self.time_dim = time_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        
        # Encoder
        self.enc1 = PointNetLayer(3 + time_dim, 64, 128)
        self.enc2 = PointNetLayer(128, 128, 256)
        self.enc3 = PointNetLayer(256, 256, 512)
        self.enc4 = PointNetLayer(512, 512, 1024)
        
        # Global features
        self.global_feat = nn.Sequential(
            nn.Conv1d(1024, 2048, 1),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Conv1d(2048, 4096, 1),
            nn.BatchNorm1d(4096),
            nn.ReLU()
        )
        
        # Decoder
        self.dec4 = PointNetLayer(4096 + 1024, 1024, 512)
        self.dec3 = PointNetLayer(512 + 512, 512, 256)
        self.dec2 = PointNetLayer(256 + 256, 256, 128)
        self.dec1 = PointNetLayer(128 + 128, 128, 64)

        # Output layers
        self.output = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )
        
        # Skip connection refinement
        self.refine1 = nn.Conv1d(128, 128, 1)
        self.refine2 = nn.Conv1d(256, 256, 1)
        self.refine3 = nn.Conv1d(512, 512, 1)
        self.refine4 = nn.Conv1d(1024, 1024, 1)

    def forward(self, x, t):
        """
        Forward pass of the UNetPointNetLarge.
        
        Args:
            x (torch.Tensor): Input point cloud tensor of shape (B, N, 3)
            t (torch.Tensor): Time embedding tensor
        
        Returns:
            torch.Tensor: Output point cloud tensor of shape (B, N, 3)
        """
        # Time embedding
        t_emb = self.get_timestep_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        
        # Concatenate time embedding with input
        x = x.transpose(2, 1)  # (B, N, 3) -> (B, 3, N)
        t_emb = t_emb.unsqueeze(2).expand(-1, -1, x.shape[2])  # (B, time_dim, N)
        x = torch.cat([x, t_emb], dim=1)  # (B, 3 + time_dim, N)

        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        # Global features
        global_feat = self.global_feat(x4)
        global_feat = torch.max(global_feat, 2, keepdim=True)[0]  # (B, 4096, 1)
        global_feat = global_feat.repeat(1, 1, x.shape[2])  # (B, 4096, N)

        # Decoder with skip connections
        x = self.dec4(torch.cat([global_feat, self.refine4(x4)], dim=1))
        x = self.dec3(torch.cat([x, self.refine3(x3)], dim=1))
        x = self.dec2(torch.cat([x, self.refine2(x2)], dim=1))
        x = self.dec1(torch.cat([x, self.refine1(x1)], dim=1))

        x = self.output(x)  # (B, 3, N)

        return x.transpose(2, 1)  # (B, N, 3)

    def get_timestep_embedding(self, timesteps, embedding_dim):
        """
        Create sinusoidal time embeddings.
        
        Args:
            timesteps (torch.Tensor): Tensor of timestep values
            embedding_dim (int): Dimension of the embedding
        
        Returns:
            torch.Tensor: Time embedding tensor
        """
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0, device=timesteps.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        if embedding_dim % 2 == 1:  # zero pad if dimension is odd
            emb = F.pad(emb, (0, 1), mode='constant')
        return emb
    
class UNetPointNetLargeWithAttentionExperimental(nn.Module): ## Experimental architecture not used currently
    def __init__(self, dim=512, time_dim=256):
        """
        Initialize a large UNet-like PointNet architecture with attention mechanisms.
        
        Args:
            dim (int): Dimension of hidden layers
            time_dim (int): Dimension of time embedding
        """
        super().__init__()
        self.time_dim = time_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        
        # Encoder
        self.enc1 = PointNetLayerWithAttention(3 + time_dim, 64, 128)
        self.enc2 = PointNetLayerWithAttention(128, 128, 256)
        self.enc3 = PointNetLayerWithAttention(256, 256, 512)
        self.enc4 = PointNetLayerWithAttention(512, 512, 1024)

        
        # Global features
        self.global_feat = nn.Sequential(
            nn.Conv1d(1024, 2048, 1),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Conv1d(2048, 4096, 1),
            nn.BatchNorm1d(4096),
            nn.ReLU()
        )
        
        # Decoder
        self.dec4 = PointNetLayerWithAttention(4096, 1024, 512)
        self.dec3 = PointNetLayerWithAttention(512, 512, 256)
        self.dec2 = PointNetLayerWithAttention(256, 256, 128)
        self.dec1 = PointNetLayerWithAttention(128, 128, 64)

        # Output layers
        self.output = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )
        
        # Skip connection attention
        self.skip_attention4 = CrossAttentionBlock(512, 1024)
        self.skip_attention3 = CrossAttentionBlock(256, 512)
        self.skip_attention2 = CrossAttentionBlock(128, 256)
        self.skip_attention1 = CrossAttentionBlock(64, 128)

    def forward(self, x, t):
        """
        Forward pass of the UNetPointNetLargeWithAttention.
        
        Args:
            x (torch.Tensor): Input point cloud tensor of shape (B, N, 3)
            t (torch.Tensor): Time embedding tensor
        
        Returns:
            torch.Tensor: Output point cloud tensor of shape (B, N, 3)
        """
        # Time embedding
        t_emb = self.get_timestep_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        
        # Concatenate time embedding with input
        x = x.transpose(2, 1)  # (B, N, 3) -> (B, 3, N)
        t_emb = t_emb.unsqueeze(2).expand(-1, -1, x.shape[2])  # (B, time_dim, N)
        x = torch.cat([x, t_emb], dim=1)  # (B, 3 + time_dim, N)

        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        # Global features
        global_feat = self.global_feat(x4)

        # Decoder with skip connections and attention
        x = self.dec4(global_feat)
        x = self.skip_attention4(x, x4)
        
        x = self.dec3(x)
        x = self.skip_attention3(x, x3)
        
        x = self.dec2(x)
        x = self.skip_attention2(x, x2)
        
        x = self.dec1(x)
        x = self.skip_attention1(x, x1)

        x = self.output(x)  # (B, 3, N)

        return x.transpose(2, 1)  # (B, N, 3)

    def get_timestep_embedding(self, timesteps, embedding_dim):
        """
        Create sinusoidal time embeddings.
        
        Args:
            timesteps (torch.Tensor): Tensor of timestep values
            embedding_dim (int): Dimension of the embedding
        
        Returns:
            torch.Tensor: Time embedding tensor
        """
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        if embedding_dim % 2 == 1:  # zero pad if dimension is odd
            emb = F.pad(emb, (0, 1), mode='constant')
        return emb   

class SimpleLatentUNetPointNet(nn.Module):
    def __init__(self, latent_dim, dim=512, time_dim=256, dropout_rate=0.1):
        """
        Initialize a simple UNet-like architecture for latent space point clouds.
        
        Args:
            latent_dim (int): Dimension of the latent space
            dim (int): Dimension of hidden layers
            time_dim (int): Dimension of time embedding
            dropout_rate (float): Dropout rate
        """
        super().__init__()
        self.time_dim = time_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Linear(latent_dim + time_dim, dim // 4),
            nn.GroupNorm(8, dim // 4),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Linear(dim // 4, dim // 2),
            nn.GroupNorm(8, dim // 2),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Linear(dim // 2, dim),
            nn.GroupNorm(8, dim),
            nn.ReLU()
        )
        self.enc4 = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GroupNorm(8, dim * 2),
            nn.ReLU()
        )
        
        # Global features
        self.global_feat = nn.Sequential(
            nn.Linear(dim * 2, dim * 4),
            nn.GroupNorm(8, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim * 8),
            nn.GroupNorm(8, dim * 8),
            nn.ReLU()
        )
        
        # Decoder
        self.dec4 = nn.Sequential(
            nn.Linear(dim * 8 + dim * 2, dim * 2),
            nn.GroupNorm(8, dim * 2),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.Linear(dim * 2 + dim, dim),
            nn.GroupNorm(8, dim),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.Linear(dim + dim // 2, dim // 2),
            nn.GroupNorm(8, dim // 2),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.Linear(dim // 2 + dim // 4, dim // 4),
            nn.GroupNorm(8, dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate)  # Only dropout here
        )

        # Output layer
        self.output = nn.Sequential(
            nn.Linear(dim // 4, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, latent_dim)
        )
        
        # Skip connection refinement
        self.refine1 = nn.Linear(dim // 4, dim // 4)
        self.refine2 = nn.Linear(dim // 2, dim // 2)
        self.refine3 = nn.Linear(dim, dim)
        self.refine4 = nn.Linear(dim * 2, dim * 2)

    def forward(self, z, t):
        """
        Forward pass of the SimpleLatentUNetPointNet.
        
        Args:
            z (torch.Tensor): Input latent tensor
            t (torch.Tensor): Time embedding tensor
        
        Returns:
            torch.Tensor: Output latent tensor
        """
        # Time embedding
        t_emb = self.get_timestep_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        
        # Concatenate time embedding with input
        z = torch.cat([z, t_emb], dim=1)

        # Encoder
        z1 = self.enc1(z)
        z2 = self.enc2(z1)
        z3 = self.enc3(z2)
        z4 = self.enc4(z3)

        # Global features
        global_feat = self.global_feat(z4)

        # Decoder with skip connections
        z = self.dec4(torch.cat([global_feat, self.refine4(z4)], dim=1))
        z = self.dec3(torch.cat([z, self.refine3(z3)], dim=1))
        z = self.dec2(torch.cat([z, self.refine2(z2)], dim=1))
        z = self.dec1(torch.cat([z, self.refine1(z1)], dim=1))

        z = self.output(z)

        return z

    def get_timestep_embedding(self, timesteps, embedding_dim):
        """
        Create sinusoidal time embeddings.
        
        Args:
            timesteps (torch.Tensor): Tensor of timestep values
            embedding_dim (int): Dimension of the embedding
        
        Returns:
            torch.Tensor: Time embedding tensor
        """
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0, device=timesteps.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        if embedding_dim % 2 == 1:  # zero pad if dimension is odd
            emb = F.pad(emb, (0, 1), mode='constant')
        return emb
    
##################### VAE Encoders and Decoders #############################

class SimplePointNetVAE(pl.LightningModule):
    def __init__(self, num_points, latent_dim=256, hidden_dim=512, lr=1e-4, beta=1e-1, dropout_rate=0.1, 
                 chamfer_lambda = 1, voxel_lambda = 1, focal_alpha=0.25, focal_gamma=2.00):
        """
        Initialize a simple PointNet-based Variational Autoencoder.
        
        Args:
            num_points (int): Number of points in each point cloud
            latent_dim (int): Dimension of the latent space
            hidden_dim (int): Dimension of hidden layers
            lr (float): Learning rate
            beta (float): Beta parameter for KL divergence weight
            dropout_rate (float): Dropout rate
            chamfer_lambda (float): Weight for Chamfer distance in loss
            voxel_lambda (float): Weight for voxel loss
            focal_alpha (float): Alpha parameter for focal loss
            focal_gamma (float): Gamma parameter for focal loss
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.encoder = nn.Sequential(
            PointNetLayer(3, 64),
            PointNetLayer(64, 128),
            PointNetLayer(128, 256),
            PointNetLayer(256, hidden_dim),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_points * 3),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.output_layer = nn.Linear(num_points * 3, num_points * 3)

        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the network.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Special initialization for latent space
        nn.init.xavier_normal_(self.fc_mu.weight, gain=0.01)
        nn.init.xavier_normal_(self.fc_logvar.weight, gain=0.01)

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

    def encode(self, x):
        """
        Encode the input point cloud to latent space.
        
        Args:
            x (torch.Tensor): Input point cloud tensor of shape (B, N, 3)
        
        Returns:
            tuple: Mean and log variance of the latent distribution
        """
        x = x.transpose(2, 1)  # (B, N, 3) -> (B, 3, N)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from the latent distribution.
        
        Args:
            mu (torch.Tensor): Mean of the latent distribution
            logvar (torch.Tensor): Log variance of the latent distribution
        
        Returns:
            torch.Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decode the latent vector to a point cloud.
        
        Args:
            z (torch.Tensor): Latent vector
        
        Returns:
            torch.Tensor: Reconstructed point cloud
        """
        decoded = self.decoder(z)
        output = self.output_layer(decoded)
        return output.view(-1, self.hparams.num_points, 3)

    def forward(self, x):
        """
        Forward pass of the VAE.
        
        Args:
            x (torch.Tensor): Input point cloud tensor
        
        Returns:
            tuple: Reconstructed point cloud, mean, and log variance of the latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def voxel_focal_loss(self, pred, target):
        """
        Compute the focal loss for voxel reconstruction.
        
        Args:
            pred (torch.Tensor): Predicted voxel grid
            target (torch.Tensor): Target voxel grid
        
        Returns:
            torch.Tensor: Computed focal loss
        """
        # Clamp predictions to avoid log(0)
        pred = torch.clamp(pred, min=1e-7, max=1-1e-7)
        
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_term = (1 - pt) ** self.hparams.focal_gamma
        focal_loss = focal_term * bce
        
        # Weigh occupied voxels in target more than unoccupied
        alpha_factor = torch.where(target == 1, self.hparams.focal_alpha, 1 - self.hparams.focal_alpha)
        focal_loss = alpha_factor * focal_loss
        
        # Return the mean focal loss
        return focal_loss.mean()

    def voxel_loss(self, x, y):
        """
        Compute the voxel reconstruction loss.
        
        Args:
            x (torch.Tensor): Predicted point cloud
            y (torch.Tensor): Target point cloud
        
        Returns:
            torch.Tensor: Computed voxel loss
        """
        x_voxels = voxelize(x)
        y_voxels = voxelize(y)
        return F.binary_cross_entropy(x_voxels, y_voxels)

    def reconstruction_loss(self, x, y):
        """
        Compute the reconstruction loss.
        
        Args:
            x (torch.Tensor): Predicted point cloud
            y (torch.Tensor): Target point cloud
        
        Returns:
            tuple: Combined loss, Chamfer distance, and voxel loss
        """
        chamfer_loss = chamfer_distance(x, y)
        chamfer_lambda = self.hparams.chamfer_lambda
        voxel_lambda = self.hparams.voxel_lambda
        voxel_loss = self.voxel_loss(x, y)
        combined_loss = chamfer_lambda * chamfer_loss + voxel_lambda * voxel_loss
        return combined_loss, chamfer_loss, voxel_loss
    
    def calculate_loss(self, batch, mode):
        """
        Calculate the total loss for a batch.
        
        Args:
            batch (torch.Tensor): Input batch of point clouds
            mode (str): 'train' or 'val'
        
        Returns:
            torch.Tensor: Computed total loss
        """
        x = batch
        recon_x, mu, logvar = self(x)
        recon_loss, chamfer_loss, voxel_loss = self.reconstruction_loss(recon_x, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.hparams.beta * kl_loss
        self.log(f'{mode}_loss', loss)
        self.log(f'{mode}_recon_loss', recon_loss)
        self.log(f'{mode}_chamfer_loss', chamfer_loss)
        self.log(f'{mode}_voxel_loss', voxel_loss)
        self.log(f'{mode}_kl_loss', kl_loss)
        return loss

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.
        
        Args:
            batch (torch.Tensor): Input batch of point clouds
            batch_idx (int): Index of the current batch
        
        Returns:
            torch.Tensor: Computed loss for the batch
        """
        return self.calculate_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.
        
        Args:
            batch (torch.Tensor): Input batch of point clouds
            batch_idx (int): Index of the current batch
        """
        self.calculate_loss(batch, mode='val')

        sampling_interval = self.trainer.num_val_batches[0] // 5
        img_idx = batch_idx // sampling_interval
        if batch_idx % sampling_interval == 0:
            fig_3d = plot_point_cloud_3d(batch[0])
            self.logger.experiment.add_figure(f'input_point_cloud_3d_{img_idx}_0', fig_3d, global_step=self.current_epoch)
            plt.close(fig_3d)
            
            fig_2d = plot_point_cloud_2d(batch[0])
            self.logger.experiment.add_figure(f'input_point_cloud_2d_{img_idx}_0', fig_2d, global_step=self.current_epoch)
            plt.close(fig_2d)

    @torch.no_grad()
    def sample(self, num_samples=4, temp=1.0, z=None):
        """
        Sample point clouds from the latent space.
        
        Args:
            num_samples (int): Number of samples to generate
            temp (float): Temperature for sampling
            z (torch.Tensor, optional): Latent vectors to use for sampling
        
        Returns:
            torch.Tensor: Generated point cloud samples
        """
        self.eval()
        
        if z is None:
            z = torch.randn(num_samples, self.hparams.latent_dim, device=self.device) * temp
        else:
            num_samples = z.shape[0]
        
        x = self.decode(z)

        return x

    def on_train_epoch_end(self):
        """
        Callback at the end of each training epoch.
        """
        pass

    def on_validation_epoch_end(self):
        """
        Callback at the end of each validation epoch. Generates and logs sample point clouds.
        """
        if not self.logger:
            return

        samples = self.sample(num_samples=4)
    
        for i, sample in enumerate(samples):
            fig_3d = plot_point_cloud_3d(sample)
            self.logger.experiment.add_figure(f'vae_generated_sample_3d_{i}', fig_3d, global_step=self.current_epoch)
            plt.close(fig_3d)
            
            fig_2d = plot_point_cloud_2d(sample)
            self.logger.experiment.add_figure(f'vae_generated_sample_2d_{i}', fig_2d, global_step=self.current_epoch)
            plt.close(fig_2d)

class PointNetPPEncoder(nn.Module):
    def __init__(self, latent_dim):
        """
        Initialize a PointNet++ encoder.
        
        Args:
            latent_dim (int): Dimension of the latent space
        """
        super(PointNetPPEncoder, self).__init__()
        self.sa1 = SetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128])
        self.sa2 = SetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128, mlp=[128, 128, 256])
        self.sa3 = SetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256, mlp=[256, 512, 1024])
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        """
        Forward pass of the PointNet++ encoder.
        
        Args:
            x (torch.Tensor): Input point cloud tensor of shape [B, N, 3]
        
        Returns:
            tuple: Mean and log variance of the latent distribution
        """
        x = x.permute(0, 2, 1)  # shape: [B, 3, N]
        l1_xyz, l1_points = self.sa1(x, x)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(l3_points.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class FoldingDecoder(nn.Module):
    def __init__(self, latent_dim, num_points):
        """
        Initialize a Folding-based decoder for point cloud generation.
        
        Args:
            latent_dim (int): Dimension of the latent space
            num_points (int): Number of points to generate
        """
        super(FoldingDecoder, self).__init__()
        self.num_points = num_points
        self.latent_dim = latent_dim

        # Create 2D grid (32x32)
        range_x = torch.linspace(-1, 1, 32)
        range_y = torch.linspace(-1, 1, 32)
        x_coord, y_coord = torch.meshgrid(range_x, range_y)
        self.grid = torch.stack([x_coord, y_coord], dim=-1).view(-1, 2).transpose(0, 1)

        # Folding layers
        self.fold1 = nn.Sequential(
            FoldingLayer(latent_dim + 2, 512),
            FoldingLayer(512, 512),
            FoldingLayer(512, 3)
        )
        
        self.fold2 = nn.Sequential(
            FoldingLayer(latent_dim + 3, 512),
            FoldingLayer(512, 512),
            FoldingLayer(512, 3)
        )

        self.upsample = nn.Linear(1024, num_points)

    def forward(self, z):
        """
        Forward pass of the Folding decoder.
        
        Args:
            z (torch.Tensor): Latent vector
        
        Returns:
            torch.Tensor: Generated point cloud
        """
        batch_size = z.size(0)
        grid = self.grid.to(z.device).unsqueeze(0).repeat(batch_size, 1, 1)
        z = z.unsqueeze(2).repeat(1, 1, grid.size(2))
        
        # First folding
        inputs = torch.cat([z, grid], dim=1)
        fold1_out = self.fold1(inputs)
        
        # Second folding
        inputs = torch.cat([z, fold1_out], dim=1)  # [B, latent_dim+3, 1024]
        fold2_out = self.fold2(inputs)  # [B, 3, 1024]
        
        # Upsample to desired number of points
        fold2_out = fold2_out.transpose(1, 2)  # [B, 1024, 3]
        upsampled = self.upsample(fold2_out.transpose(1, 2)).transpose(1, 2) # [B, num_points, 3]
        
        return upsampled

class PointNetVAE(pl.LightningModule):
    def __init__(self, num_points=2048, latent_dim=256, lr=1e-4, beta=1e-1):
        """
        Initialize a PointNet-based Variational Autoencoder.
        
        Args:
            num_points (int): Number of points in each point cloud
            latent_dim (int): Dimension of the latent space
            lr (float): Learning rate
            beta (float): Beta parameter for KL divergence weight
        """
        super().__init__()
        self.save_hyperparameters()
        self.num_points = num_points
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = PointNetPPEncoder(latent_dim)

        # Decoder
        self.decoder = FoldingDecoder(latent_dim, num_points)

        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the network.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Special initialization for latent space
        nn.init.xavier_normal_(self.encoder.fc_mu.weight, gain=0.01)
        nn.init.xavier_normal_(self.encoder.fc_logvar.weight, gain=0.01)

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

    def encode(self, x):
        """
        Encode the input point cloud to latent space.
        
        Args:
            x (torch.Tensor): Input point cloud tensor
        
        Returns:
            tuple: Mean and log variance of the latent distribution
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Decode the latent vector to a point cloud.
        
        Args:
            z (torch.Tensor): Latent vector
        
        Returns:
            torch.Tensor: Reconstructed point cloud
        """
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from the latent distribution.
        
        Args:
            mu (torch.Tensor): Mean of the latent distribution
            logvar (torch.Tensor): Log variance of the latent distribution
        
        Returns:
            torch.Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass of the VAE.
        
        Args:
            x (torch.Tensor): Input point cloud tensor
        
        Returns:
            tuple: Reconstructed point cloud, mean, and log variance of the latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def vae_loss(self, x):
        """
        Compute the VAE loss.
        
        Args:
            x (torch.Tensor): Input point cloud tensor
        
        Returns:
            tuple: Total loss, reconstruction loss, and KL divergence loss
        """
        recon_x, mu, logvar = self.forward(x)
        recon_loss = chamfer_distance(recon_x, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.hparams.beta * kl_loss
        return loss, recon_loss, kl_loss

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.
        
        Args:
            batch (torch.Tensor): Input batch of point clouds
            batch_idx (int): Index of the current batch
        
        Returns:
            torch.Tensor: Computed loss for the batch
        """
        loss, recon_loss, kl_loss = self.vae_loss(batch)
        self.log('train_loss', loss)
        self.log('train_recon_loss', recon_loss)
        self.log('train_kl_loss', kl_loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.
        
        Args:
            batch (torch.Tensor): Input batch of point clouds
            batch_idx (int): Index of the current batch
        """
        loss, recon_loss, kl_loss = self.vae_loss(batch)
        self.log('val_loss', loss)
        self.log('val_recon_loss', recon_loss)
        self.log('val_kl_loss', kl_loss)
        
        sampling_interval = self.trainer.num_val_batches[0] // 5
        img_idx = batch_idx // sampling_interval
        if batch_idx % sampling_interval == 0:
            fig_3d = plot_point_cloud_3d(batch[0])
            self.logger.experiment.add_figure(f'input_point_cloud_3d_{img_idx}_0', fig_3d, global_step=self.current_epoch)
            plt.close(fig_3d)
            
            fig_2d = plot_point_cloud_2d(batch[0])
            self.logger.experiment.add_figure(f'input_point_cloud_2d_{img_idx}_0', fig_2d, global_step=self.current_epoch)
            plt.close(fig_2d)

    def on_train_epoch_end(self):
        """
        Callback at the end of each training epoch.
        """
        pass

    def on_validation_epoch_end(self, num_samples=4):
        """
        Callback at the end of each validation epoch. Generates and logs sample point clouds.
        """
        if not self.logger:
            return

        samples = self.sample(num_samples)
    
        for i, sample in enumerate(samples):
            fig_3d = plot_point_cloud_3d(sample)
            self.logger.experiment.add_figure(f'vae_generated_sample_3d_{i}', fig_3d, global_step=self.current_epoch)
            plt.close(fig_3d)
            
            fig_2d = plot_point_cloud_2d(sample)
            self.logger.experiment.add_figure(f'vae_generated_sample_2d_{i}', fig_2d, global_step=self.current_epoch)
            plt.close(fig_2d)

    @torch.no_grad()
    def sample(self, num_samples=4, temp=1.0, z=None):
        """
        Sample point clouds from the latent space.
        
        Args:
            num_samples (int): Number of samples to generate
            temp (float): Temperature for sampling
            z (torch.Tensor, optional): Latent vectors to use for sampling
        
        Returns:
            torch.Tensor: Generated point cloud samples
        """
        self.eval()

        if z is None:
            # Sample random latent vectors from the prior distribution
            z = torch.randn(num_samples, self.hparams.latent_dim, device=self.device) * temp
        else:
            # Use provided latent vectors (e.g., from diffusion process)
            num_samples = z.shape[0]

        x = self.decode(z)

        return x
    
class PointNetVAEExperimental(pl.LightningModule):      ## Experimental architecture not used currently
    def __init__(self, num_points=2048, latent_dim=256, lr=1e-4, beta=1e-1):
        """
        Initialize an experimental PointNet-based Variational Autoencoder.
        
        Args:
            num_points (int): Number of points in each point cloud
            latent_dim (int): Dimension of the latent space
            lr (float): Learning rate
            beta (float): Beta parameter for KL divergence weight
        """
        super().__init__()
        self.save_hyperparameters()
        self.num_points = num_points

        # Encoder (PointNet++ backbone)
        self.sa1 = SetAbstraction(npoint=1024, radius=0.1, nsample=32, in_channel=3, mlp=[32, 32, 64], first_layer=True)
        self.sa2 = SetAbstraction(npoint=256, radius=0.2, nsample=32, in_channel=64, mlp=[64, 64, 128])
        self.sa3 = SetAbstraction(npoint=64, radius=0.4, nsample=32, in_channel=128, mlp=[128, 128, 256])
        self.sa4 = SetAbstraction(npoint=16, radius=0.8, nsample=32, in_channel=256, mlp=[256, 256, 512])

        # Latent space
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # Decoder (Feature Propagation)
        self.fp4 = FeaturePropagation(in_channel=latent_dim, mlp=[256, 256])
        self.fp3 = FeaturePropagation(in_channel=256, mlp=[256, 256])
        self.fp2 = FeaturePropagation(in_channel=256, mlp=[256, 128])
        self.fp1 = FeaturePropagation(in_channel=128, mlp=[128, 128, 3])

        # Upscale to num_points
        self.output = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(3, 3, kernel_size=1, bias=True),
            nn.Conv1d(3, 3, kernel_size=1, bias=True),
            nn.Conv1d(3, 3, kernel_size=1, bias=True),
        )

        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the network.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Special initialization for latent space
        nn.init.xavier_normal_(self.fc_mu.weight, gain=0.01)
        nn.init.xavier_normal_(self.fc_logvar.weight, gain=0.01)
            

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

    def encode(self, x):
        """
        Encode the input point cloud to latent space.
        
        Args:
            x (torch.Tensor): Input point cloud tensor of shape [B, N, 3]
        
        Returns:
            tuple: Mean and log variance of the latent distribution, and list of intermediate features
        """
        # x shape: [B, N, 3]
        x = x.permute(0, 2, 1)  # shape: [B, 3, N]
        l1_xyz, l1_points = self.sa1(x, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        x = l4_points.mean(dim=2)  # Global pooling, shape: [B, 512]

        mu = self.fc_mu(x)  # shape: [B, latent_dim]
        logvar = self.fc_logvar(x)  # shape: [B, latent_dim]
        return mu, logvar, (l1_xyz, l2_xyz, l3_xyz, l4_xyz)

    def decode(self, z, xyz_list):
        """
        Decode the latent vector to a point cloud.
        
        Args:
            z (torch.Tensor): Latent vector
            xyz_list (list): List of intermediate feature tensors from the encoder
        
        Returns:
            torch.Tensor: Reconstructed point cloud
        """
        l1_xyz, l2_xyz, l3_xyz, l4_xyz = xyz_list
        l3_points = self.fp4(l3_xyz, l4_xyz, None, z.unsqueeze(2))
        l2_points = self.fp3(l2_xyz, l3_xyz, None, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, None, l2_points)
        l0_points = self.fp1(l1_xyz, l1_xyz, None, l1_points)
        output = self.output(l0_points)
        return output

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from the latent distribution.
        
        Args:
            mu (torch.Tensor): Mean of the latent distribution
            logvar (torch.Tensor): Log variance of the latent distribution
        
        Returns:
            torch.Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass of the VAE.
        
        Args:
            x (torch.Tensor): Input point cloud tensor
        
        Returns:
            tuple: Reconstructed point cloud, mean, and log variance of the latent distribution
        """
        mu, logvar, xyz_list = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, xyz_list), mu, logvar

    def vae_loss(self, x):
        """
        Compute the VAE loss.
        
        Args:
            x (torch.Tensor): Input point cloud tensor
        
        Returns:
            tuple: Total loss, reconstruction loss, and KL divergence loss
        """
        recon_x, mu, logvar = self.forward(x)
        recon_loss = chamfer_distance(recon_x.permute(0, 2, 1), x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.hparams.beta * kl_loss
        return loss, recon_loss, kl_loss

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.
        
        Args:
            batch (torch.Tensor): Input batch of point clouds
            batch_idx (int): Index of the current batch
        
        Returns:
            torch.Tensor: Computed loss for the batch
        """
        loss, recon_loss, kl_loss = self.vae_loss(batch)
        self.log('train_loss', loss)
        self.log('train_recon_loss', recon_loss)
        self.log('train_kl_loss', kl_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.
        
        Args:
            batch (torch.Tensor): Input batch of point clouds
            batch_idx (int): Index of the current batch
        """
        loss, recon_loss, kl_loss = self.vae_loss(batch)
        self.log('val_loss', loss)
        self.log('val_recon_loss', recon_loss)
        self.log('val_kl_loss', kl_loss)
        
        sampling_interval = self.trainer.num_val_batches[0] // 5
        img_idx = batch_idx // sampling_interval
        if batch_idx % sampling_interval == 0:
            fig_3d = plot_point_cloud_3d(batch[0])
            self.logger.experiment.add_figure(f'input_point_cloud_3d_{img_idx}_0', fig_3d, global_step=self.current_epoch)
            plt.close(fig_3d)
            
            fig_2d = plot_point_cloud_2d(batch[0])
            self.logger.experiment.add_figure(f'input_point_cloud_2d_{img_idx}_0', fig_2d, global_step=self.current_epoch)
            plt.close(fig_2d)

    @torch.no_grad()
    def sample(self, num_samples=4, temp=1.0, z=None):
        """
        Sample point clouds from the latent space.
        
        Args:
            num_samples (int): Number of samples to generate
            temp (float): Temperature for sampling
            z (torch.Tensor, optional): Latent vectors to use for sampling
        
        Returns:
            torch.Tensor: Generated point cloud samples
        """
        self.eval()

        if z is None:
            # Sample random latent vectors from the prior distribution
            z = torch.randn(num_samples, self.hparams.latent_dim, device=self.device) * temp
        else:
            # Use provided latent vectors (e.g., from diffusion process)
            num_samples = z.shape[0]

        # Generate a dummy point cloud to get the xyz_list
        dummy_pc = torch.randn(num_samples, self.num_points, 3, device=self.device)

        _, _, xyz_list = self.encode(dummy_pc)
        x = self.decode(z, xyz_list)
        x = x.permute(0, 2, 1)  # [B, N, 3]

        return x
    
    def on_train_epoch_end(self):
        """
        Callback at the end of each training epoch.
        """
        pass

    def on_validation_epoch_end(self, num_samples=4):
        """
        Callback at the end of each validation epoch. Generates and logs sample point clouds.
        """
        if not self.logger:
            return

        samples = self.sample(num_samples)
    
        for i, sample in enumerate(samples):
            fig_3d = plot_point_cloud_3d(sample)
            self.logger.experiment.add_figure(f'vae_generated_sample_3d_{i}', fig_3d, global_step=self.current_epoch)
            plt.close(fig_3d)
            
            fig_2d = plot_point_cloud_2d(sample)
            self.logger.experiment.add_figure(f'vae_generated_sample_2d_{i}', fig_2d, global_step=self.current_epoch)
            plt.close(fig_2d)

class VAE3D(pl.LightningModule):
    def __init__(self, input_shape=(32, 32, 32), latent_dim=256, beta=1e-1):
        """
        Initialize a 3D Variational Autoencoder.
        
        Args:
            input_shape (tuple): Shape of the input voxel grid
            latent_dim (int): Dimension of the latent space
            beta (float): Beta parameter for KL divergence weight
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Encoder
        self.encoder = nn.Sequential(
            Conv3DBlock(1, 32, stride=2),  # 32x16x16x16
            Conv3DBlock(32, 64, stride=2),  # 64*8x8x8
            Conv3DBlock(64, 128, stride=2),  # 128*4x4x4
            Conv3DBlock(128, 256, stride=2),  # 256*2x2x2
            nn.Flatten(),
            nn.Linear(256 * 2 * 2 * 2, 512),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 256 * 2 * 2 * 2)

        self.decoder = nn.Sequential(
            Deconv3DBlock(256, 128, stride=2, output_padding=1),  # 128*4x4x4
            Deconv3DBlock(128, 64, stride=2, output_padding=1),  # 64*8x8x8
            Deconv3DBlock(64, 32, stride=2, output_padding=1),  # 32*16x16x16
            nn.ConvTranspose3d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 1*32x32x32
            nn.Sigmoid()
        )

        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the network.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Special initialization for latent space
        nn.init.xavier_normal_(self.fc_mu.weight, gain=0.01)
        nn.init.xavier_normal_(self.fc_logvar.weight, gain=0.01)

    def configure_optimizers(self):
        """
        Configure the optimizer for training.
        """
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def encode(self, x):
        """
        Encode the input voxel grid to latent space.
        
        Args:
            x (torch.Tensor): Input voxel grid tensor
        
        Returns:
            tuple: Mean and log variance of the latent distribution
        """
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from the latent distribution.
        
        Args:
            mu (torch.Tensor): Mean of the latent distribution
            logvar (torch.Tensor): Log variance of the latent distribution
        
        Returns:
            torch.Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decode the latent vector to a voxel grid.
        
        Args:
            z (torch.Tensor): Latent vector
        
        Returns:
            torch.Tensor: Reconstructed voxel grid
        """
        z = self.decoder_input(z)
        z = z.view(-1, 256, 2, 2, 2)
        z = self.decoder(z)
        return z

    def forward(self, x):
        """
        Forward pass of the VAE.
        
        Args:
            x (torch.Tensor): Input voxel grid tensor
        
        Returns:
            tuple: Reconstructed voxel grid, mean, and log variance of the latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def calculate_loss(self, batch, mode):
        """
        Calculate the VAE loss.
        
        Args:
            batch (torch.Tensor): Input batch of voxel grids
            mode (str): 'train' or 'val'
        
        Returns:
            torch.Tensor: Computed loss
        """
        x = batch
        recon_x, mu, logvar = self(x)
        recon_loss = F.binary_cross_entropy(recon_x.squeeze(1), x.squeeze(1), reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.hparams.beta * kl_loss
        self.log(f'{mode}_loss', loss)
        self.log(f'{mode}_recon_loss', recon_loss)
        self.log(f'{mode}_kl_loss', kl_loss)
        return loss

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.
        
        Args:
            batch (torch.Tensor): Input batch of voxel grids
            batch_idx (int): Index of the current batch
        
        Returns:
            torch.Tensor: Computed loss for the batch
        """
        return self.calculate_loss(batch, mode='train')
    
    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.
        
        Args:
            batch (torch.Tensor): Input batch of voxel grids
            batch_idx (int): Index of the current batch
        """
        self.calculate_loss(batch, mode='val')

        sampling_interval = self.trainer.num_val_batches[0] // 5
        img_idx = batch_idx // sampling_interval
        if batch_idx % sampling_interval == 0:
            point_cloud = voxel_tensor_to_point_clouds(batch[0].detach().unsqueeze(0))[0]
            fig_3d = plot_point_cloud_3d(point_cloud)
            self.logger.experiment.add_figure(f'input_point_cloud_3d_{img_idx}_0', fig_3d, global_step=self.current_epoch)
            plt.close(fig_3d)
            
            fig_2d = plot_point_cloud_2d(point_cloud)
            self.logger.experiment.add_figure(f'input_point_cloud_2d_{img_idx}_0', fig_2d, global_step=self.current_epoch)
            plt.close(fig_2d)

    @torch.no_grad()
    def sample(self, num_samples, threshold=0.4):
        """
        Sample voxel grids from the latent space.
        
        Args:
            num_samples (int): Number of samples to generate
            threshold (float): Threshold for converting voxel probabilities to binary values
        
        Returns:
            list: Generated point cloud samples
        """
        self.eval()
        z = torch.randn(num_samples, self.hparams.latent_dim).to(self.device)
        samples = self.decode(z)
        samples = voxel_tensor_to_point_clouds(samples, threshold)
        return samples

    def on_train_epoch_end(self):
        """
        Callback at the end of each training epoch.
        """
        pass

    def on_validation_epoch_end(self, num_samples=4):
        """
        Callback at the end of each validation epoch. Generates and logs sample point clouds.
        """
        if not self.logger:
            return

        samples = self.sample(num_samples)
    
        for i, sample in enumerate(samples):
            
            if len(sample) == 0:
                continue
            
            fig_3d = plot_point_cloud_3d(sample)
            self.logger.experiment.add_figure(f'vae_generated_sample_3d_{i}', fig_3d, global_step=self.current_epoch)
            plt.close(fig_3d)
            
            fig_2d = plot_point_cloud_2d(sample)
            self.logger.experiment.add_figure(f'vae_generated_sample_2d_{i}', fig_2d, global_step=self.current_epoch)
            plt.close(fig_2d)

class VAE3DLarge(pl.LightningModule):
    def __init__(self, input_shape=(32, 32, 32), latent_dim=256, lr=1e-4, kl_warmup_epochs=10, kl_warmup_max_beta=0.1, 
                 kl_annealing_epochs=100):
        """
        Initialize a large 3D Variational Autoencoder with KL divergence warmup.
        
        Args:
            input_shape (tuple): Shape of the input voxel grid
            latent_dim (int): Dimension of the latent space
            lr (float): Learning rate
            kl_warmup_epochs (int): Number of epochs for KL divergence warmup
            kl_warmup_max_beta (float): Maximum beta value for KL divergence after warmup
        """
        super().__init__()
        self.save_hyperparameters()

        # Encoder                                                  # Input: [1, 32, 32, 32]     
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1),  # Output: [32, 32, 32, 32]
            nn.ReLU(inplace=True),
            ResidualBlock3D(32, 64),  # Output: [64, 32, 32, 32]
            nn.Conv3d(64, 64, kernel_size=4, stride=2, padding=1),  # Output: [64, 16, 16, 16]
            nn.ReLU(inplace=True),
            ResidualBlock3D(64, 128),  # Output: [128, 16, 16, 16]
            nn.Conv3d(128, 128, kernel_size=4, stride=2, padding=1),  # Output: [128, 8, 8, 8]
            nn.ReLU(inplace=True),
            ResidualBlock3D(128, 256),  # Output: [256, 8, 8, 8]
            nn.Conv3d(256, 256, kernel_size=4, stride=2, padding=1),  # Output: [256, 4, 4, 4]
            nn.ReLU(inplace=True),
            ResidualBlock3D(256, 512),  # Output: [512, 4, 4, 4]
            nn.Conv3d(512, 512, kernel_size=4, stride=1, padding=0),  # Output: [512, 1, 1, 1]
            nn.ReLU(inplace=True),
            nn.Flatten()  # Output: [512]
        )

        self.fc_mu = nn.Linear(512, latent_dim)  # Output: [latent_dim]
        self.fc_logvar = nn.Linear(512, latent_dim)  # Output: [latent_dim]

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 512 * 4 * 4 * 4)  # Output: [512 * 4 * 4 * 4]

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),  # Output: [256, 8, 8, 8]
            nn.ReLU(inplace=True),
            ResidualBlock3D(256, 256),  # Output: [256, 8, 8, 8]
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: [128, 16, 16, 16]
            nn.ReLU(inplace=True),
            ResidualBlock3D(128, 128),  # Output: [128, 16, 16, 16]
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: [64, 32, 32, 32]
            nn.ReLU(inplace=True),
            ResidualBlock3D(64, 64),  # Output: [64, 32, 32, 32]
            nn.Conv3d(64, 32, kernel_size=3, padding=1),  # Output: [32, 32, 32, 32]
            nn.ReLU(inplace=True),
            ResidualBlock3D(32, 32),  # Output: [32, 32, 32, 32]
            nn.Conv3d(32, 1, kernel_size=3, padding=1),  # Output: [1, 32, 32, 32]
            nn.Sigmoid()
        )

        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the network.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Special initialization for latent space
        nn.init.xavier_normal_(self.fc_mu.weight, gain=0.01)
        nn.init.xavier_normal_(self.fc_logvar.weight, gain=0.01)

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def encode(self, x):
        """
        Encode the input voxel grid to latent space.
        
        Args:
            x (torch.Tensor): Input voxel grid tensor
        
        Returns:
            tuple: Mean and log variance of the latent distribution
        """
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from the latent distribution.
        
        Args:
            mu (torch.Tensor): Mean of the latent distribution
            logvar (torch.Tensor): Log variance of the latent distribution
        
        Returns:
            torch.Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decode the latent vector to a voxel grid.
        
        Args:
            z (torch.Tensor): Latent vector
        
        Returns:
            torch.Tensor: Reconstructed voxel grid
        """
        z = self.decoder_input(z)
        z = z.view(-1, 512, 4, 4, 4)
        return self.decoder(z)

    def forward(self, x):
        """
        Forward pass of the VAE.
        
        Args:
            x (torch.Tensor): Input voxel grid tensor
        
        Returns:
            tuple: Reconstructed voxel grid, mean, and log variance of the latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def get_kl_weight(self):
        """
        Calculate the KL divergence weight based on the current epoch.
        
        Returns:
            float: KL divergence weight
        """
        annealing_epochs = min(self.trainer.max_epochs, self.hparams.kl_annealing_epochs)

        max_warmup_beta = self.hparams.kl_warmup_max_beta
        warmup_epochs = self.hparams.kl_warmup_epochs
        if self.current_epoch < 10:
            return (self.current_epoch + 1) / warmup_epochs * max_warmup_beta
        else:
            return min(max_warmup_beta + (self.current_epoch - warmup_epochs + 1) / 
                       (annealing_epochs - warmup_epochs) * (1.0 - max_warmup_beta), 1.0)

    def calculate_loss(self, batch, mode):
        """
        Calculate the VAE loss.
        
        Args:
            batch (torch.Tensor): Input batch of voxel grids
            mode (str): 'train' or 'val'
        
        Returns:
            tuple: Computed loss and reconstructed voxel grid
        """
        x = batch
        recon_x, mu, logvar = self(x)
        
        # Reconstruction loss (Binary Cross-Entropy)
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='mean')
        
        # KL divergence
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # KL annealing
        kl_annealed_weight = self.get_kl_weight() if mode =='train' else 1.0
        
        # Total loss
        loss = recon_loss + kl_annealed_weight * kl_div
        
        # Logging
        self.log(f'{mode}_loss', loss)
        self.log(f'{mode}_recon_loss', recon_loss)
        self.log(f'{mode}_kl_div', kl_div)
        
        return loss, recon_x

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.
        
        Args:
            batch (torch.Tensor): Input batch of voxel grids
            batch_idx (int): Index of the current batch
        
        Returns:
            torch.Tensor: Computed loss for the batch
        """
        return self.calculate_loss(batch, mode='train')[0]
    
    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.
        
        Args:
            batch (torch.Tensor): Input batch of voxel grids
            batch_idx (int): Index of the current batch
        """
        _, recon_x = self.calculate_loss(batch, mode='val')
    
        sampling_interval = self.trainer.num_val_batches[0] // 5
        img_idx = batch_idx // sampling_interval
        if batch_idx % sampling_interval == 0:

            sample_idx = (0 + len(batch)) // 2

            input_point_cloud = voxel_tensor_to_point_clouds(batch[sample_idx].detach().unsqueeze(0))[0]
            reconstructed_point_cloud = voxel_tensor_to_point_clouds(recon_x[sample_idx].detach().unsqueeze(0))[0]
            
            fig_3d = plot_comparison_point_clouds(input_point_cloud.cpu().numpy(), reconstructed_point_cloud.cpu().numpy(), 
                                                  f"Point Cloud Comparison", "Input", f"Reconstructed")
            self.logger.experiment.add_figure(f'input_vs_reconstructed_point_cloud_3d_{img_idx}_{sample_idx}', fig_3d, global_step=self.current_epoch)
            plt.close(fig_3d)
            
            fig_2d = plot_point_cloud_2d(input_point_cloud)
            self.logger.experiment.add_figure(f'input_point_cloud_2d_{img_idx}_{sample_idx}', fig_2d, global_step=self.current_epoch)
            plt.close(fig_2d)

    @torch.no_grad()
    def sample(self, num_samples, threshold=0.4):
        """
        Sample voxel grids from the latent space.
        
        Args:
            num_samples (int): Number of samples to generate
            threshold (float): Threshold for converting voxel probabilities to binary values
        
        Returns:
            list: Generated point cloud samples
        """
        self.eval()
        z = torch.randn(num_samples, self.hparams.latent_dim).to(self.device)
        samples = self.decode(z)
        samples = voxel_tensor_to_point_clouds(samples, threshold)
        return samples
    
    def on_train_epoch_end(self):
        """
        Callback at the end of each training epoch.
        """
        pass

    def on_validation_epoch_end(self, num_samples=4):
        """
        Callback at the end of each validation epoch. Generates and logs sample point clouds.
        """
        if not self.logger:
            return

        samples = self.sample(num_samples)
    
        for i, sample in enumerate(samples):
            
            if len(sample) == 0:
                continue
            
            fig_3d = plot_point_cloud_3d(sample)
            self.logger.experiment.add_figure(f'vae_generated_sample_3d_{i}', fig_3d, global_step=self.current_epoch)
            plt.close(fig_3d)
            
            fig_2d = plot_point_cloud_2d(sample)
            self.logger.experiment.add_figure(f'vae_generated_sample_2d_{i}', fig_2d, global_step=self.current_epoch)
            plt.close(fig_2d)