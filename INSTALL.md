# Installation Guide

This guide provides detailed instructions for installing and setting up the Neural Radiance Field (NeRF) for Dynamic Scene Reconstruction on Edge GPUs.

## Table of Contents

- [Prerequisites](#prerequisites)
- [System Requirements](#system-requirements)
- [Installation Steps](#installation-steps)
- [Edge Device Setup](#edge-device-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Software Requirements

- Python 3.8 or higher
- pip (Python package manager)
- git
- CUDA 11.8+ (for GPU support)
- cuDNN 8.6+ (for GPU support)

### Hardware Requirements

**Minimum (Development):**
- CPU: 4+ cores
- RAM: 8GB
- GPU: NVIDIA GPU with 4GB+ VRAM (Compute Capability 6.0+)

**Recommended (Production):**
- CPU: 8+ cores
- RAM: 16GB+
- GPU: NVIDIA RTX series or Jetson Xavier/Orin

## System Requirements

### For Desktop/Server

- Ubuntu 20.04/22.04 or Windows 10/11
- NVIDIA Driver 520+ (for CUDA 11.8)
- Docker (optional, for containerized deployment)

### For Edge Devices (Jetson)

- NVIDIA Jetson Nano, Xavier NX, AGX Xavier, or Orin
- JetPack 5.0+
- Minimum 32GB storage

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Rishav-raj-github/Neural-Radiance-Field-NeRF-for-Dynamic-scence-Reconstruction.git
cd Neural-Radiance-Field-NeRF-for-Dynamic-scence-Reconstruction
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install PyTorch

**For CUDA 11.8:**
```bash
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only:**
```bash
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e .[dev,full]
```

### 5. Install tiny-cuda-nn (Optional, for faster training)

```bash
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

## Edge Device Setup

### NVIDIA Jetson Installation

1. **Flash JetPack 5.0+** to your Jetson device
2. **Increase swap space:**
```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

3. **Install PyTorch for Jetson:**
```bash
wget https://nvidia.box.com/shared/static/ssf2v7pf5i245fk4i0q926hy4imzs2ph.whl -O torch-2.0.0-cp38-cp38m-linux_aarch64.whl
pip install torch-2.0.0-cp38-cp38m-linux_aarch64.whl
```

4. **Install other dependencies:**
```bash
pip install -r requirements.txt
```

## Verification

Verify your installation:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Run tests:
```bash
pytest tests/
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in config
- Lower image resolution
- Enable gradient checkpointing

### Import Errors
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

### Jetson Performance Issues
- Enable maximum performance mode: `sudo nvpmodel -m 0`
- Set maximum clock speeds: `sudo jetson_clocks`

### Missing CUDA/cuDNN
- Verify CUDA installation: `nvcc --version`
- Check cuDNN: `ldconfig -p | grep cudnn`

## Additional Resources

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [NVIDIA Jetson Documentation](https://developer.nvidia.com/embedded/jetson)
- [Project Documentation](./docs/)

## Support

For issues and questions:
- Open an [Issue](https://github.com/Rishav-raj-github/Neural-Radiance-Field-NeRF-for-Dynamic-scence-Reconstruction/issues)
- Check existing issues and discussions
