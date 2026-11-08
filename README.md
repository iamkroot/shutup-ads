# shutup-ads

Real-time basketball broadcast commercial break detector and stream attenuator powered by Google's YAMNet neural network.

## Features
- **Low-Latency YAMNet Inference**: Uses ONNX Runtime (~6ms inference time per chunk).
- **Basketball Profile**: Detects basketball bounces, sneaker squeaks, referee whistles, arena buzzers, crowd cheering, and stadium applause, discriminating against commercial studio music and jingles.
- **Smart Attenuation**: Automatically reduces volume of target streams (e.g., Firefox, Sunshine) to 15% during commercial breaks and restores full volume when the game returns.
- **Rich Dashboard**: Real-time terminal interface displaying live VU meter, detected sound classes, state status, and session stats.
- **Interactive Calibration**: Fast wizard to record live game vs ad samples and fit a custom broadcast embedding classifier.

## Quick Start
```bash
# Start live detection and attenuation (auto-targeting Firefox & Sunshine)
shutup-ads run

# Dry-run mode (monitoring only, no volume changes)
shutup-ads run --dry-run

# List active audio devices and application streams
shutup-ads list-devices
```
