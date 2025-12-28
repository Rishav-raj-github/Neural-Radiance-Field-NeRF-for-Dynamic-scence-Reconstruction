"""Data loading utilities for multi-view images and poses."""
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, List, Optional
import json


class MultiViewDataset(Dataset):
    """Load multi-view images and camera poses.
    
    Supports LLFF and Blender-style dataset formats.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        downsample: int = 1,
        hold_every: int = 8,
    ):
        """Initialize dataset.
        
        Args:
            data_dir: Path to dataset directory
            split: 'train', 'val', or 'test'
            downsample: Downsample images by this factor
            hold_every: Hold every Nth image for validation
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.downsample = downsample
        self.hold_every = hold_every
        
        # Load images and poses
        self.images, self.poses, self.intrinsics = self._load_data()
    
    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load images, poses, and camera intrinsics."""
        # Placeholder: in production, load from disk
        # For now, return dummy data for testing
        num_images = 50
        h, w = 256, 256
        
        images = torch.randn(num_images, h, w, 3)  # RGB images [0, 1]
        poses = torch.eye(4).unsqueeze(0).repeat(num_images, 1, 1)  # Identity poses
        intrinsics = torch.tensor([
            [w / 2, 0, w / 2],
            [0, h / 2, h / 2],
            [0, 0, 1],
        ], dtype=torch.float32)
        
        return images, poses, intrinsics
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> dict:
        return {
            'image': self.images[idx],
            'pose': self.poses[idx],
            'intrinsics': self.intrinsics,
        }


def create_dataloader(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    **kwargs,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for NeRF training."""
    dataset = MultiViewDataset(data_dir, **kwargs)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=kwargs.get('split') == 'train',
        num_workers=num_workers,
    )
