# Neural Radiance Field (NeRF) - Documentation

Welcome to the documentation for Real-Time Neural Radiance Field (NeRF) for Dynamic Scene Reconstruction on Edge GPUs.

## Overview

This project brings Neural Radiance Fields to edge devices, enabling real-time 3D scene reconstruction on hardware like NVIDIA Jetson.

## Documentation Structure

### Getting Started
- [Installation Guide](../INSTALL.md) - Complete setup instructions
- [Configuration Guide](configuration.md) - Configure your NeRF setup
- [Quick Start Tutorial](quickstart.md) - Get up and running quickly

### Core Concepts
- [Architecture Overview](architecture.md) - System design and components
- [Neural Radiance Fields](nerf-basics.md) - Understanding NeRF fundamentals
- [Dynamic Scene Handling](dynamic-scenes.md) - Temporal modeling approach
- [Edge Optimization](edge-optimization.md) - Making NeRF fast on edge devices

### API Reference
- [Core API](api/core.md) - Main API functions
- [Model API](api/models.md) - Model architectures
- [Training API](api/training.md) - Training utilities
- [Rendering API](api/rendering.md) - Rendering functions

### Tutorials
- [Training Your First NeRF](tutorials/first-training.md)
- [Real-time Rendering](tutorials/realtime-rendering.md)
- [Deploying on Jetson](tutorials/jetson-deployment.md)
- [Custom Scenes](tutorials/custom-scenes.md)

### Performance
- [Benchmarks](benchmarks.md) - Performance metrics
- [Optimization Guide](optimization.md) - Tips for better performance
- [Profiling](profiling.md) - Analyzing bottlenecks

### Advanced Topics
- [Multi-GPU Training](advanced/multi-gpu.md)
- [Custom Network Architectures](advanced/custom-networks.md)
- [Advanced Rendering Techniques](advanced/rendering.md)
- [Research Extensions](advanced/research.md)

### Development
- [Contributing Guidelines](../SECURITY.md#contributing)
- [Development Setup](development.md)
- [Testing Guide](testing.md)
- [Code Style Guide](code-style.md)

## Quick Links

- **Repository**: [GitHub](https://github.com/Rishav-raj-github/Neural-Radiance-Field-NeRF-for-Dynamic-scence-Reconstruction)
- **Issues**: [Bug Reports & Feature Requests](https://github.com/Rishav-raj-github/Neural-Radiance-Field-NeRF-for-Dynamic-scence-Reconstruction/issues)
- **Changelog**: [Version History](../CHANGELOG.md)
- **License**: [MIT License](../LICENSE)

## Need Help?

If you have questions or run into issues:

1. Check the [Installation Guide](../INSTALL.md) for setup help
2. Read the [Troubleshooting Guide](troubleshooting.md)
3. Search existing [GitHub Issues](https://github.com/Rishav-raj-github/Neural-Radiance-Field-NeRF-for-Dynamic-scence-Reconstruction/issues)
4. Open a new issue if needed

## Project Status

- **Version**: 0.1.0 (Alpha)
- **Status**: Active Development
- **Python**: 3.8+
- **GPU**: CUDA 11.8+
- **Edge Devices**: Jetson Nano, Xavier, Orin

---

*Last updated: October 9, 2025*
