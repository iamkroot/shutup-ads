# shutup-ads

Real-time sports broadcast commercial break detector and stream attenuator powered by Google's YAMNet neural network.

## Features
- **Low-Latency YAMNet Inference**: Uses ONNX Runtime (~6ms inference time per chunk).
- **Sport Presets**: Built-in acoustic presets for Basketball (sneaker squeaks, court bounce, referee whistle, buzzer), Football, Soccer, and General Sports. Custom presets easily configurable in `config.toml`.
- **Hardware Agnostic**: Automatically discovers active PulseAudio/PipeWire physical sinks and monitors, with full support for locking to specific hardware DACs or PCI audio devices via `config.toml`.
- **Smart Attenuation**: Automatically reduces volume of target streams (e.g., Firefox, Sunshine) to 15% (or full mute) during commercial breaks and restores full volume when the game returns.
- **Rich Dashboard**: Real-time terminal interface displaying live VU meter, detected sound classes, state status, and session stats.
- **Interactive Calibration**: Fast wizard to record live game vs ad samples and fit a custom broadcast embedding classifier.

## Quick Start
```bash
# Start live detection and attenuation (auto-targeting Firefox & Sunshine)
shutup-ads run

# Run with a specific sport preset
shutup-ads run --preset football

# List available sport presets
shutup-ads list-presets

# Start with debug screenshots (captures screenshot on every commercial / game transition)
shutup-ads run --debug

# Dry-run mode (monitoring only, no volume changes)
shutup-ads run --dry-run

# List active audio devices, sinks, and application streams
shutup-ads list-devices
```

## Configuration (`config.toml`)
`shutup-ads` looks for configuration in:
1. `--config /path/to/config.toml`
2. Environment variable `$SHUTUP_ADS_CONFIG`
3. `./config.toml` (in the current directory)
4. `~/.config/shutup-ads/config.toml`

Copy `config.example.toml` to get started:
```bash
cp config.example.toml config.toml
```

### Example `config.toml`
```toml
# Active broadcast preset: "basketball", "football", "soccer", "general_sports", or custom
preset = "basketball"

[audio]
# PulseAudio / PipeWire hardware sink name (leave empty for auto-detection)
default_sink = ""

# Fallback sink used during cleanup if discovery fails
fallback_sink = "@DEFAULT_SINK@"

# Applications to watch and attenuate during commercial breaks
target_apps = ["firefox", "firefox developer edition", "sunshine", "ff"]

# Attenuation mode: "attenuate" (reduce volume), "mute", or "dry_run"
action_mode = "attenuate"
ad_volume = 15.0

[presets.basketball]
name = "Basketball"
ad_thresh = 0.40
sports_thresh = 0.06
ad_delay = 3.5
sports_delay = 1.0
min_ad_duration = 15.0
max_ad_duration = 270.0
```
