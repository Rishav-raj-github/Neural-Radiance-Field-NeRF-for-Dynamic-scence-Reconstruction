# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of Neural-Radiance-Field-NeRF seriously. If you believe you have found a security vulnerability, please report it to us as described below.

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to the repository maintainer.

### What to Include

Please include the following information in your report:

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### Response Timeline

You can expect:

- Acknowledgment of your report within 48 hours
- Regular updates about our progress
- Notification when the issue is fixed

## Security Best Practices

When using this project:

1. **Keep Dependencies Updated**: Regularly update PyTorch, CUDA drivers, and other dependencies
2. **Validate Input Data**: Always validate and sanitize input scene data and camera parameters
3. **Secure Model Weights**: Protect trained model weights from unauthorized access
4. **Resource Limits**: Implement resource limits to prevent denial-of-service on edge devices
5. **Network Security**: If deploying over network, use encrypted connections (HTTPS/TLS)

## Known Security Considerations

- **GPU Memory Exhaustion**: Large scenes may exhaust GPU memory. Implement appropriate bounds checking
- **Model Poisoning**: When using pre-trained models, verify their source and integrity
- **Data Privacy**: Rendered scenes may contain sensitive information. Handle with appropriate data protection measures
