"""shutup-ads: Real-time sports commercial break detector."""

from shutup_ads.config import (
    AppConfig,
    AudioConfig,
    PresetConfig,
    get_preset,
    load_config,
)

__version__ = "0.1.0"
__all__ = ["AppConfig", "AudioConfig", "PresetConfig", "get_preset", "load_config"]
