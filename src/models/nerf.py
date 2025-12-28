"""Core NeRF and DynamicNeRF models for scene representation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for 3D coordinates and viewing directions."""
    
    def __init__(self, num_freqs: int = 10, include_input: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.register_buffer(
            'freq_bands',
            torch.linspace(0, num_freqs - 1, num_freqs) * math.pi
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode position using positional encoding.
        
        Args:
            x: Input tensor of shape (..., D)
            
        Returns:
            Encoded tensor of shape (..., 2*D*num_freqs) or (..., D + 2*D*num_freqs)
        """
        encoded = []
        if self.include_input:
            encoded.append(x)
        
        for freq in self.freq_bands:
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        
        return torch.cat(encoded, dim=-1)


class NeRF(nn.Module):
    """Static Neural Radiance Field.
    
    MLP-based NeRF that predicts RGB and density from 3D positions and view directions.
    """
    
    def __init__(
        self,
        pos_freqs: int = 10,
        dir_freqs: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 8,
    ):
        super().__init__()
        
        self.pos_encoding = PositionalEncoding(pos_freqs)
        self.dir_encoding = PositionalEncoding(dir_freqs)
        
        # Calculate encoding dimensions
        pos_dim = 3 + 2 * 3 * pos_freqs  # Input + encoded
        dir_dim = 3 + 2 * 3 * dir_freqs
        
        # Main MLP
        layers = []
        in_dim = pos_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        self.main_mlp = nn.Sequential(*layers)
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
        
        # Color prediction head
        self.color_mlp = nn.Sequential(
            nn.Linear(hidden_dim + dir_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )
    
    def forward(self, pos: torch.Tensor, direction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict RGB and density.
        
        Args:
            pos: Position tensor of shape (N, 3)
            direction: View direction of shape (N, 3)
            
        Returns:
            Tuple of (rgb, sigma) with shapes (N, 3) and (N, 1)
        """
        pos_encoded = self.pos_encoding(pos)
        dir_encoded = self.dir_encoding(direction)
        
        # Get latent features from main MLP
        features = self.main_mlp(pos_encoded)
        
        # Predict sigma (density)
        sigma = self.sigma_head(features)
        
        # Predict RGB with view direction
        color_input = torch.cat([features, dir_encoded], dim=-1)
        rgb = self.color_mlp(color_input)
        
        return rgb, sigma


class DynamicNeRF(nn.Module):
    """Dynamic Neural Radiance Field for scene changes.
    
    Extends NeRF with temporal dynamics for handling moving objects.
    """
    
    def __init__(
        self,
        pos_freqs: int = 10,
        dir_freqs: int = 4,
        time_freqs: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 8,
    ):
        super().__init__()
        
        self.static_nerf = NeRF(pos_freqs, dir_freqs, hidden_dim, num_layers)
        self.time_encoding = PositionalEncoding(time_freqs)
        
        # Deformation field for temporal dynamics
        time_dim = 1 + 2 * 1 * time_freqs
        self.deformation_mlp = nn.Sequential(
            nn.Linear(hidden_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 3D displacement
        )
    
    def forward(
        self,
        pos: torch.Tensor,
        direction: torch.Tensor,
        time: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with temporal information.
        
        Args:
            pos: Position of shape (N, 3)
            direction: View direction of shape (N, 3)
            time: Time embedding of shape (N, 1)
            
        Returns:
            Tuple of (rgb, sigma, deformation)
        """
        rgb, sigma = self.static_nerf(pos, direction)
        
        # Get features from static NeRF for deformation
        pos_encoded = self.static_nerf.pos_encoding(pos)
        features = self.static_nerf.main_mlp(pos_encoded)
        
        # Predict deformation based on time
        time_encoded = self.time_encoding(time)
        deformation_input = torch.cat([features, time_encoded], dim=-1)
        deformation = self.deformation_mlp(deformation_input)
        
        return rgb, sigma, deformation
