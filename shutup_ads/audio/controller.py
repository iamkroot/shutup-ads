"""PulseAudio / PipeWire stream discovery, volume attenuation, and muting."""

from __future__ import annotations

import atexit
import logging
import signal
from dataclasses import dataclass
from typing import Any, Sequence

import pulsectl

logger = logging.getLogger(__name__)

DEFAULT_TARGET_APPS = ["firefox", "firefox developer edition", "sunshine", "ff"]


@dataclass
class StreamInfo:
    """Information about a tracked audio stream."""
    index: int
    app_name: str
    binary: str
    baseline_volume: float
    current_volume: float
    is_attenuated: bool = False
    is_muted: bool = False


class PulseController:
    """Manages volume attenuation and muting for target application streams."""

    def __init__(
        self,
        target_apps: Sequence[str] = DEFAULT_TARGET_APPS,
        attenuate_volume: float = 0.15,
        action_mode: str = "attenuate",  # "attenuate", "mute", or "dry_run"
    ):
        self.target_apps = [app.lower().strip() for app in target_apps]
        self.attenuate_volume = float(attenuate_volume)
        self.action_mode = action_mode
        self._pulse: pulsectl.Pulse | None = None
        self._tracked_streams: dict[int, StreamInfo] = {}

        # Register cleanup on program exit
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _get_pulse(self) -> pulsectl.Pulse:
        """Get or initialize pulsectl client connection."""
        if self._pulse is None:
            self._pulse = pulsectl.Pulse("shutup-ads-controller")
        return self._pulse

    def _matches_target(self, app_name: str, binary: str) -> bool:
        """Check if stream properties match target application list."""
        if not self.target_apps:
            return True
        app_name_lower = app_name.lower()
        binary_lower = binary.lower()
        for target in self.target_apps:
            if target in app_name_lower or target in binary_lower:
                return True
        return False

    def scan_streams(self) -> list[StreamInfo]:
        """Scan active sink inputs and return matching target streams."""
        matched: list[StreamInfo] = []
        try:
            pulse = self._get_pulse()
            sink_inputs = pulse.sink_input_list()
            for si in sink_inputs:
                app_name = si.proplist.get("application.name", "")
                binary = si.proplist.get("application.process.binary", "")
                if self._matches_target(app_name, binary):
                    vol = float(si.volume.value_flat)
                    if si.index not in self._tracked_streams:
                        # Record stream with its baseline volume
                        stream_info = StreamInfo(
                            index=si.index,
                            app_name=app_name,
                            binary=binary,
                            baseline_volume=vol if vol > 0.05 else 1.0,
                            current_volume=vol,
                            is_attenuated=False,
                            is_muted=bool(si.mute),
                        )
                        self._tracked_streams[si.index] = stream_info
                    else:
                        stream_info = self._tracked_streams[si.index]
                        stream_info.current_volume = vol
                    matched.append(stream_info)
        except Exception as e:
            logger.warning("Error scanning sink inputs: %s", e)
            self._pulse = None
        return matched

    def list_all_streams(self) -> list[dict[str, Any]]:
        """List all active sink inputs on the system (for cli list-devices)."""
        result = []
        try:
            pulse = self._get_pulse()
            sink_inputs = pulse.sink_input_list()
            for si in sink_inputs:
                app_name = si.proplist.get("application.name", "Unknown")
                binary = si.proplist.get("application.process.binary", "")
                media_name = si.proplist.get("media.name", "")
                vol = float(si.volume.value_flat)
                is_target = self._matches_target(app_name, binary)
                result.append({
                    "index": si.index,
                    "app_name": app_name,
                    "binary": binary,
                    "media_name": media_name,
                    "volume": vol,
                    "mute": bool(si.mute),
                    "is_target": is_target,
                })
        except Exception as e:
            logger.error("Failed to list streams: %s", e)
            self._pulse = None
        return result

    def get_monitor_source(self) -> str:
        """Resolve the monitor source name for the default audio sink."""
        try:
            pulse = self._get_pulse()
            server_info = pulse.server_info()
            default_sink = pulse.get_sink_by_name(server_info.default_sink_name)
            return default_sink.monitor_source_name
        except Exception as e:
            logger.warning("Could not query default sink monitor: %s, falling back to default.monitor", e)
            return "default.monitor"

    def apply_commercial_state(self, is_commercial: bool) -> list[StreamInfo]:
        """Apply attenuation/muting if in commercial, or restore volume if live sports.

        Returns list of modified streams.
        """
        if self.action_mode == "dry_run":
            return self.scan_streams()

        active_targets = self.scan_streams()
        try:
            pulse = self._get_pulse()
            for stream in active_targets:
                try:
                    si = pulse.sink_input_info(stream.index)
                except Exception:
                    # Stream may have closed
                    continue

                if is_commercial:
                    if not stream.is_attenuated:
                        logger.info("Commercial detected -> Attenuating %s (#%d)", stream.app_name, stream.index)
                        if self.action_mode == "mute":
                            pulse.sink_input_mute(stream.index, 1)
                            stream.is_muted = True
                        else:
                            # Attenuate volume to target pct (relative to baseline)
                            target_vol = min(stream.baseline_volume, self.attenuate_volume)
                            pulse.volume_set_all_flat(si, target_vol)
                            stream.current_volume = target_vol
                        stream.is_attenuated = True
                else:
                    if stream.is_attenuated:
                        logger.info("Live sports detected -> Restoring %s (#%d) volume to %.0f%%",
                                    stream.app_name, stream.index, stream.baseline_volume * 100)
                        if self.action_mode == "mute":
                            pulse.sink_input_mute(stream.index, 0)
                            stream.is_muted = False
                        # Restore original baseline volume
                        pulse.volume_set_all_flat(si, stream.baseline_volume)
                        stream.current_volume = stream.baseline_volume
                        stream.is_attenuated = False
        except Exception as e:
            logger.warning("Error applying stream volume state: %s", e)
            self._pulse = None

        return active_targets

    def cleanup(self) -> None:
        """Restore all streams to baseline volume before exit."""
        if not self._tracked_streams:
            return
        logger.info("Restoring all streams to baseline volumes on exit...")
        try:
            pulse = self._get_pulse()
            for idx, stream in list(self._tracked_streams.items()):
                try:
                    si = pulse.sink_input_info(idx)
                    pulse.volume_set_all_flat(si, stream.baseline_volume)
                    pulse.sink_input_mute(idx, 0)
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            if self._pulse:
                try:
                    self._pulse.close()
                except Exception:
                    pass
                self._pulse = None
            self._tracked_streams.clear()

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Signal handler to ensure cleanup on SIGINT or SIGTERM."""
        self.cleanup()
        raise SystemExit(0)
