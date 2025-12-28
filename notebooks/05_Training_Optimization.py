"""
Training Optimization and Learning Strategies for NeRF

Comprehensive training pipeline with learning rate scheduling, regularization, and metrics.

Author: Rishav Raj
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class NeRFTrainer:
    """
    Complete training loop for NeRF models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.0,
        scheduler_type: str = 'exponential'
    ):
        self.model = model
        self.device = next(model.parameters()).device
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        if scheduler_type == 'exponential':
            self.scheduler = ExponentialLR(self.optimizer, gamma=0.999)
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10000)
        else:
            self.scheduler = None
        
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'psnr': [],
            'ssim': []
        }
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int
    ) -> Dict[str, float]:
        """
        Single training iteration.
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Extract batch data
        rays_o = batch['rays_o'].to(self.device)
        rays_d = batch['rays_d'].to(self.device)
        target_rgb = batch['target_rgb'].to(self.device)
        
        # Forward pass
        predicted_rgb = self.model(rays_o, rays_d)
        
        # Loss computation
        mse_loss = nn.MSELoss()(predicted_rgb, target_rgb)
        
        # Regularization
        l2_loss = torch.tensor(0.0, device=self.device)
        for param in self.model.parameters():
            l2_loss += torch.norm(param)
        
        total_loss = mse_loss + 0.0001 * l2_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        if self.scheduler:
            self.scheduler.step()
        
        return {
            'loss': total_loss.item(),
            'mse': mse_loss.item(),
            'l2': l2_loss.item(),
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def validation_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Validation iteration without gradient computation.
        """
        self.model.eval()
        
        with torch.no_grad():
            rays_o = batch['rays_o'].to(self.device)
            rays_d = batch['rays_d'].to(self.device)
            target_rgb = batch['target_rgb'].to(self.device)
            
            predicted_rgb = self.model(rays_o, rays_d)
            
            # Loss
            val_loss = nn.MSELoss()(predicted_rgb, target_rgb)
            
            # PSNR
            psnr = 20 * torch.log10(1.0 / torch.sqrt(val_loss))
            
            # SSIM (simplified)
            ssim = self._compute_ssim(predicted_rgb, target_rgb)
        
        return {
            'val_loss': val_loss.item(),
            'psnr': psnr.item(),
            'ssim': ssim.item()
        }
    
    def _compute_ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        window_size: int = 11,
        sigma: float = 1.5
    ) -> torch.Tensor:
        """
        Compute Structural Similarity Index (SSIM).
        """
        # Simplified SSIM computation
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        # Mean
        mu1 = img1.mean()
        mu2 = img2.mean()
        
        # Variance
        sigma1_sq = ((img1 - mu1) ** 2).mean()
        sigma2_sq = ((img2 - mu2) ** 2).mean()
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
        
        # SSIM
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return ssim


class Loss weighted Combination(nn.Module):
    """
    Adaptive weighted loss combination.
    """
    
    def __init__(self, initial_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        
        self.weights = initial_weights or {
            'photometric': 1.0,
            'regularization': 0.01,
            'perceptual': 0.1
        }
    
    def forward(
        self,
        photometric_loss: torch.Tensor,
        regularization_loss: torch.Tensor,
        perceptual_loss: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Combine weighted losses.
        """
        total_loss = self.weights['photometric'] * photometric_loss + \
                     self.weights['regularization'] * regularization_loss
        
        if perceptual_loss is not None:
            total_loss += self.weights['perceptual'] * perceptual_loss
        
        return total_loss


class WarmupScheduler:
    """
    Learning rate warmup scheduler.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int = 1000,
        base_lr: float = 5e-4
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.step_count = 0
    
    def step(self) -> float:
        """
        Update learning rate with warmup.
        """
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.step_count / self.warmup_steps)
        else:
            # Exponential decay after warmup
            decay_steps = self.step_count - self.warmup_steps
            lr = self.base_lr * (0.999 ** decay_steps)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class NeRFDataset(Dataset):
    """
    Dataset wrapper for NeRF training.
    """
    
    def __init__(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        target_rgb: torch.Tensor,
        batch_size: int = 1024
    ):
        self.rays_o = rays_o
        self.rays_d = rays_d
        self.target_rgb = target_rgb
        self.batch_size = batch_size
        
        assert len(rays_o) == len(rays_d) == len(target_rgb), \
            "Inconsistent data lengths"
    
    def __len__(self) -> int:
        return len(self.rays_o)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'rays_o': self.rays_o[idx],
            'rays_d': self.rays_d[idx],
            'target_rgb': self.target_rgb[idx]
        }


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Dummy model
    model = nn.Linear(6, 3).to(device)
    
    # Create trainer
    trainer = NeRFTrainer(model, learning_rate=5e-4)
    
    # Dummy data
    batch = {
        'rays_o': torch.randn(1024, 3).to(device),
        'rays_d': torch.randn(1024, 3).to(device),
        'target_rgb': torch.rand(1024, 3).to(device)
    }
    
    # Training step
    metrics = trainer.training_step(batch, step=0)
    print('Training metrics:', metrics)
    
    # Validation step
    val_metrics = trainer.validation_step(batch)
    print('Validation metrics:', val_metrics)
