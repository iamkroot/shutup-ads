"""Unit tests for configuration loader and presets."""

import pytest

from shutup_ads.config import (
    AppConfig,
    PresetConfig,
    get_builtin_presets,
    get_preset,
    load_config,
)
from shutup_ads.detector.engine import BasketballDetector, SportsDetector
from shutup_ads.detector.state_machine import BroadcastStateMachine


def test_builtin_presets_exist():
    presets = get_builtin_presets()
    assert "basketball" in presets
    assert "football" in presets
    assert "soccer" in presets
    assert "general_sports" in presets

    bb = presets["basketball"]
    assert bb.ad_thresh == 0.40
    assert bb.sports_thresh == 0.06
    assert bb.ad_delay == 3.5
    assert bb.sports_delay == 1.0


def test_load_default_config_when_no_file(monkeypatch, tmp_path):
    monkeypatch.setenv("SHUTUP_ADS_CONFIG", str(tmp_path / "nonexistent.toml"))
    monkeypatch.chdir(tmp_path)
    # When explicit file doesn't exist, load_config(path) raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent.toml")

    # When no explicit path is passed, returns default config
    monkeypatch.delenv("SHUTUP_ADS_CONFIG", raising=False)
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.preset == "basketball"
    assert cfg.audio.default_sink == ""
    assert cfg.audio.fallback_sink == "@DEFAULT_SINK@"
    assert "firefox" in cfg.audio.target_apps


def test_load_custom_config_toml(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
preset = "soccer"
debug = true
debug_dir = "/tmp/custom_captures"

[audio]
default_sink = "alsa_output.pci-0000_00_1f.3.analog-stereo"
fallback_sink = "my_backup_sink"
source = "custom.monitor"
bridge_sink_name = "my_custom_bridge"
use_bridge = false
target_apps = ["vlc", "mpv"]
action_mode = "mute"
ad_volume = 0.0

[presets.soccer]
name = "World Cup Soccer"
ad_thresh = 0.55
sports_thresh = 0.12
ad_delay = 5.0
sports_delay = 2.0

[presets.tennis]
name = "Tennis"
description = "Tennis match broadcast"
ad_thresh = 0.48
sports_thresh = 0.10
""")

    cfg = load_config(config_file)
    assert cfg.preset == "soccer"
    assert cfg.debug is True
    assert cfg.debug_dir == "/tmp/custom_captures"

    # Audio config
    assert cfg.audio.default_sink == "alsa_output.pci-0000_00_1f.3.analog-stereo"
    assert cfg.audio.fallback_sink == "my_backup_sink"
    assert cfg.audio.source == "custom.monitor"
    assert cfg.audio.bridge_sink_name == "my_custom_bridge"
    assert cfg.audio.use_bridge is False
    assert cfg.audio.target_apps == ["vlc", "mpv"]
    assert cfg.audio.action_mode == "mute"
    assert cfg.audio.ad_volume == 0.0

    # Overridden preset
    soccer = get_preset(cfg, "soccer")
    assert soccer.name == "World Cup Soccer"
    assert soccer.ad_thresh == 0.55
    assert soccer.sports_thresh == 0.12
    assert soccer.ad_delay == 5.0
    assert soccer.sports_delay == 2.0

    # Custom newly added preset
    tennis = get_preset(cfg, "tennis")
    assert tennis.name == "Tennis"
    assert tennis.description == "Tennis match broadcast"
    assert tennis.ad_thresh == 0.48
    assert tennis.sports_thresh == 0.10


def test_get_preset_case_insensitive_and_fallback():
    cfg = AppConfig(presets=get_builtin_presets())

    # Case insensitive
    assert get_preset(cfg, "BASKETBALL").name == "Basketball"
    assert get_preset(cfg, "Basketball").name == "Basketball"
    assert get_preset(cfg, "football").name == "American Football"

    # Nonexistent fallback
    fallback = get_preset(cfg, "unknown_sport")
    assert fallback.name == "Basketball"


def test_detector_and_state_machine_with_preset():
    preset = PresetConfig(
        name="Custom Basketball",
        ad_thresh=0.60,
        sports_thresh=0.15,
        ad_delay=2.5,
        sports_delay=1.5,
        min_ad_duration=20.0,
        max_ad_duration=180.0,
    )

    detector = BasketballDetector(preset=preset)
    assert detector.ad_threshold == 0.60
    assert detector.sports_threshold == 0.15

    # Test alias
    detector2 = SportsDetector(preset=preset)
    assert detector2.ad_threshold == 0.60

    sm = BroadcastStateMachine(preset=preset)
    assert sm.ad_threshold == 0.60
    assert sm.sports_threshold == 0.15
    assert sm.ad_trigger_seconds == 2.5
    assert sm.sports_trigger_seconds == 1.5
    assert sm.min_ad_duration == 20.0
    assert sm.max_ad_duration == 180.0


def test_pulse_controller_hw_names():
    from shutup_ads.audio.controller import PulseController

    ctrl = PulseController(
        default_sink="alsa_output.pci-custom",
        fallback_sink="fallback_sink_name",
        bridge_sink_name="test_bridge",
        use_bridge=False,
    )
    assert ctrl.default_sink_name == "alsa_output.pci-custom"
    assert ctrl.fallback_sink == "fallback_sink_name"
    assert ctrl.bridge_sink_name == "test_bridge"
    assert ctrl.use_bridge is False
