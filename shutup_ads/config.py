"""Configuration loader and preset definitions for shutup-ads."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        import tomllib  # type: ignore[no-redef]

from shutup_ads.detector.classes import (
    ARENA_ATMOSPHERE_CLASSES,
    BASKETBALL_COURT_CLASSES,
    COMMERCIAL_FOLEY_CLASSES,
    COMMERCIAL_MUSIC_CLASSES,
)

logger = logging.getLogger(__name__)


@dataclass
class PresetConfig:
    """Acoustic and timing parameters for a sport or broadcast preset."""

    name: str = "Basketball"
    description: str = "Basketball broadcast preset (court squeaks, whistle, buzzer, crowd cheering)"
    ad_thresh: float = 0.40
    sports_thresh: float = 0.06
    ad_delay: float = 3.5
    sports_delay: float = 1.0
    min_ad_duration: float = 15.0
    max_ad_duration: float = 270.0
    alpha_attack: float = 0.35
    alpha_release: float = 0.10
    court_classes: list[int] = field(default_factory=lambda: list(BASKETBALL_COURT_CLASSES))
    crowd_classes: list[int] = field(default_factory=lambda: list(ARENA_ATMOSPHERE_CLASSES))
    music_classes: list[int] = field(default_factory=lambda: list(COMMERCIAL_MUSIC_CLASSES))
    foley_classes: list[int] = field(default_factory=lambda: list(COMMERCIAL_FOLEY_CLASSES))


@dataclass
class AudioConfig:
    """Hardware audio devices, bridge configuration, and stream targets."""

    default_sink: str = ""
    fallback_sink: str = "@DEFAULT_SINK@"
    source: str = ""
    bridge_sink_name: str = "shutup_ads_bridge"
    use_bridge: bool = True
    target_apps: list[str] = field(
        default_factory=lambda: ["firefox", "firefox developer edition", "sunshine", "ff"]
    )
    action_mode: str = "attenuate"
    ad_volume: float = 15.0


@dataclass
class AppConfig:
    """Global configuration for shutup-ads."""

    preset: str = "basketball"
    debug: bool = False
    debug_dir: str = "debug_captures"
    audio: AudioConfig = field(default_factory=AudioConfig)
    presets: dict[str, PresetConfig] = field(default_factory=dict)


def get_builtin_presets() -> dict[str, PresetConfig]:
    """Return dictionary of built-in sport and broadcast presets."""
    return {
        "basketball": PresetConfig(
            name="Basketball",
            description="NBA / College basketball (court squeaks, whistle, buzzer, crowd cheering)",
            ad_thresh=0.40,
            sports_thresh=0.06,
            ad_delay=3.5,
            sports_delay=1.0,
            min_ad_duration=15.0,
            max_ad_duration=270.0,
            alpha_attack=0.35,
            alpha_release=0.10,
            court_classes=list(BASKETBALL_COURT_CLASSES),
            crowd_classes=list(ARENA_ATMOSPHERE_CLASSES),
            music_classes=list(COMMERCIAL_MUSIC_CLASSES),
            foley_classes=list(COMMERCIAL_FOLEY_CLASSES),
        ),
        "football": PresetConfig(
            name="American Football",
            description="NFL / College football (referee whistles, crowd cheering, band music)",
            ad_thresh=0.45,
            sports_thresh=0.08,
            ad_delay=3.5,
            sports_delay=1.2,
            min_ad_duration=15.0,
            max_ad_duration=270.0,
            alpha_attack=0.35,
            alpha_release=0.10,
            court_classes=[396, 392],  # Whistle, Buzzer
            crowd_classes=list(ARENA_ATMOSPHERE_CLASSES),
            music_classes=list(COMMERCIAL_MUSIC_CLASSES),
            foley_classes=list(COMMERCIAL_FOLEY_CLASSES),
        ),
        "soccer": PresetConfig(
            name="Soccer",
            description="Soccer / Football match (continuous crowd cheering, referee whistle, stadium chants)",
            ad_thresh=0.45,
            sports_thresh=0.08,
            ad_delay=4.0,
            sports_delay=1.2,
            min_ad_duration=15.0,
            max_ad_duration=270.0,
            alpha_attack=0.35,
            alpha_release=0.10,
            court_classes=[396],  # Whistle
            crowd_classes=list(ARENA_ATMOSPHERE_CLASSES),
            music_classes=list(COMMERCIAL_MUSIC_CLASSES),
            foley_classes=list(COMMERCIAL_FOLEY_CLASSES),
        ),
        "general_sports": PresetConfig(
            name="General Sports",
            description="Generic sports broadcast with referee whistle and arena crowd cheering",
            ad_thresh=0.42,
            sports_thresh=0.08,
            ad_delay=3.5,
            sports_delay=1.2,
            min_ad_duration=15.0,
            max_ad_duration=270.0,
            alpha_attack=0.35,
            alpha_release=0.10,
            court_classes=[396, 392, 459, 355],
            crowd_classes=list(ARENA_ATMOSPHERE_CLASSES),
            music_classes=list(COMMERCIAL_MUSIC_CLASSES),
            foley_classes=list(COMMERCIAL_FOLEY_CLASSES),
        ),
    }


def find_config_file(explicit_path: Path | str | None = None) -> Path | None:
    """Locate the configuration file across standard search paths."""
    if explicit_path is not None:
        p = Path(explicit_path)
        return p if p.is_file() else None

    # Check environment variable
    env_cfg = os.environ.get("SHUTUP_ADS_CONFIG")
    if env_cfg:
        p = Path(env_cfg)
        if p.is_file():
            return p

    # Check working directory
    local_cfg = Path("config.toml")
    if local_cfg.is_file():
        return local_cfg

    # Check XDG user config directory
    user_cfg = Path.home() / ".config" / "shutup-ads" / "config.toml"
    if user_cfg.is_file():
        return user_cfg

    return None


def load_config(path: Path | str | None = None) -> AppConfig:
    """Load configuration from TOML file, falling back to sensible defaults.

    If path is specified explicitly and not found, raises FileNotFoundError.
    """
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        config_file = p
    else:
        config_file = find_config_file()

    builtin_presets = get_builtin_presets()
    app_config = AppConfig(presets=builtin_presets)

    if config_file is None:
        return app_config

    logger.info("Loading configuration from %s", config_file)
    try:
        with open(config_file, "rb") as f:
            data: dict[str, Any] = tomllib.load(f)
    except Exception as e:
        logger.error("Failed to parse config file %s: %s", config_file, e)
        raise

    if "preset" in data:
        app_config.preset = str(data["preset"])
    if "debug" in data:
        app_config.debug = bool(data["debug"])
    if "debug_dir" in data:
        app_config.debug_dir = str(data["debug_dir"])

    if "audio" in data and isinstance(data["audio"], dict):
        audio_data = data["audio"]
        app_config.audio.default_sink = str(audio_data.get("default_sink", app_config.audio.default_sink))
        app_config.audio.fallback_sink = str(audio_data.get("fallback_sink", app_config.audio.fallback_sink))
        app_config.audio.source = str(audio_data.get("source", app_config.audio.source))
        app_config.audio.bridge_sink_name = str(audio_data.get("bridge_sink_name", app_config.audio.bridge_sink_name))
        app_config.audio.use_bridge = bool(audio_data.get("use_bridge", app_config.audio.use_bridge))
        if "target_apps" in audio_data:
            app_config.audio.target_apps = [str(x).strip() for x in audio_data["target_apps"]]
        if "action_mode" in audio_data:
            app_config.audio.action_mode = str(audio_data["action_mode"])
        if "ad_volume" in audio_data:
            app_config.audio.ad_volume = float(audio_data["ad_volume"])

    if "presets" in data and isinstance(data["presets"], dict):
        for preset_key, p_data in data["presets"].items():
            if not isinstance(p_data, dict):
                continue
            base = app_config.presets.get(preset_key, PresetConfig())
            app_config.presets[preset_key] = PresetConfig(
                name=str(p_data.get("name", base.name)),
                description=str(p_data.get("description", base.description)),
                ad_thresh=float(p_data.get("ad_thresh", base.ad_thresh)),
                sports_thresh=float(p_data.get("sports_thresh", base.sports_thresh)),
                ad_delay=float(p_data.get("ad_delay", base.ad_delay)),
                sports_delay=float(p_data.get("sports_delay", base.sports_delay)),
                min_ad_duration=float(p_data.get("min_ad_duration", base.min_ad_duration)),
                max_ad_duration=float(p_data.get("max_ad_duration", base.max_ad_duration)),
                alpha_attack=float(p_data.get("alpha_attack", base.alpha_attack)),
                alpha_release=float(p_data.get("alpha_release", base.alpha_release)),
                court_classes=list(p_data.get("court_classes", base.court_classes)),
                crowd_classes=list(p_data.get("crowd_classes", base.crowd_classes)),
                music_classes=list(p_data.get("music_classes", base.music_classes)),
                foley_classes=list(p_data.get("foley_classes", base.foley_classes)),
            )

    return app_config


def get_preset(config: AppConfig, name: str | None = None) -> PresetConfig:
    """Retrieve the specified preset from config, with case-insensitive lookup."""
    target_name = (name or config.preset).lower().strip()
    if target_name in config.presets:
        return config.presets[target_name]

    for p in config.presets.values():
        if p.name.lower().strip() == target_name:
            return p

    logger.warning("Preset '%s' not found. Falling back to 'basketball'.", target_name)
    return config.presets.get("basketball", PresetConfig())
