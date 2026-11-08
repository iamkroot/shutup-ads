"""PulseAudio / PipeWire stream discovery, virtual bridge routing, and volume attenuation."""

from __future__ import annotations

import atexit
import logging
import signal
import subprocess
import time
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
    """Manages virtual audio bridge routing and volume attenuation.

    Uses a virtual null-sink + loopback architecture:
    1. Target applications (Firefox / Sunshine) output into a virtual null sink ('shutup_ads_bridge').
    2. A loopback routes sound from 'shutup_ads_bridge.monitor' to the user's physical speakers.
    3. YAMNet captures directly from 'shutup_ads_bridge.monitor', which ALWAYS has the 100% original,
       unattenuated broadcast audio.
    4. When commercial breaks trigger, shutup-ads attenuates or mutes ONLY the loopback stream to
       the user's speakers, leaving the audio analysis completely unmuted and unaffected.
    5. When shutup-ads exits, streams are seamlessly moved back to the physical speakers and
       virtual modules are unloaded.
    """

    def __init__(
        self,
        target_apps: Sequence[str] = DEFAULT_TARGET_APPS,
        attenuate_volume: float = 0.15,
        action_mode: str = "attenuate",  # "attenuate", "mute", or "dry_run"
        use_bridge: bool = True,
    ):
        self.target_apps = [app.lower().strip() for app in target_apps]
        self.attenuate_volume = float(attenuate_volume)
        self.action_mode = action_mode
        self.use_bridge = use_bridge

        self._pulse: pulsectl.Pulse | None = None
        self._tracked_streams: dict[int, StreamInfo] = {}

        self.default_sink_name: str = ""
        self._bridge_null_mod: str | None = None
        self._bridge_loop_mod: str | None = None
        self._loopback_si_id: int | None = None
        self._is_attenuated: bool = False

        # Register cleanup on program exit
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        if self.use_bridge and self.action_mode != "dry_run":
            self.setup_bridge()

    def _get_pulse(self) -> pulsectl.Pulse:
        """Get or initialize pulsectl client connection."""
        if self._pulse is None:
            self._pulse = pulsectl.Pulse("shutup-ads-ctrl")
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

    def setup_bridge(self) -> None:
        """Create virtual null sink and loopback for clean unattenuated audio capture."""
        try:
            pulse = self._get_pulse()
            server_info = pulse.server_info()
            self.default_sink_name = server_info.default_sink_name
        except Exception as e:
            logger.warning("Failed to query default sink via pulsectl: %s", e)
            self.default_sink_name = "alsa_output.pci-0000_63_00.6.pro-output-0"

        # Unload any stale leftover bridge modules
        try:
            res = subprocess.run(["pactl", "list", "short", "modules"], capture_output=True, text=True).stdout
            for line in res.splitlines():
                if "shutup_ads_bridge" in line:
                    mod_id = line.split()[0]
                    subprocess.run(["pactl", "unload-module", mod_id], capture_output=True)
        except Exception:
            pass

        logger.info("Setting up Virtual Audio Bridge with output to: %s", self.default_sink_name)

        # 1. Load null sink
        r1 = subprocess.run(
            ["pactl", "load-module", "module-null-sink", "sink_name=shutup_ads_bridge", "sink_properties=device.description=ShutupAdsBridge"],
            capture_output=True, text=True,
        )
        if r1.returncode == 0 and r1.stdout.strip().isdigit():
            self._bridge_null_mod = r1.stdout.strip()
        else:
            logger.warning("Could not load module-null-sink: %s", r1.stderr)

        # 2. Load loopback to default physical speakers
        r2 = subprocess.run(
            ["pactl", "load-module", "module-loopback", "source=shutup_ads_bridge.monitor", f"sink={self.default_sink_name}"],
            capture_output=True, text=True,
        )
        if r2.returncode == 0 and r2.stdout.strip().isdigit():
            self._bridge_loop_mod = r2.stdout.strip()
        else:
            logger.warning("Could not load module-loopback: %s", r2.stderr)

        # 3. Route target streams to bridge
        self.scan_and_route_streams()

    def get_monitor_source(self) -> str:
        """Resolve the monitor source name for audio analysis."""
        if self.use_bridge and self._bridge_null_mod:
            return "shutup_ads_bridge.monitor"

        try:
            pulse = self._get_pulse()
            server_info = pulse.server_info()
            default_sink = pulse.get_sink_by_name(server_info.default_sink_name)
            return default_sink.monitor_source_name
        except Exception as e:
            logger.warning("Could not query default sink monitor: %s, falling back to default.monitor", e)
            return "default.monitor"

    def scan_and_route_streams(self) -> list[StreamInfo]:
        """Scan active sink inputs, route targets into the bridge, and return stream list."""
        matched: list[StreamInfo] = []
        try:
            pulse = self._get_pulse()
            sink_inputs = pulse.sink_input_list()
            for si in sink_inputs:
                app_name = si.proplist.get("application.name", "")
                binary = si.proplist.get("application.process.binary", "")

                # Skip our own loopback stream
                if "shutup_ads" in app_name.lower() or "loopback" in si.proplist.get("media.name", "").lower():
                    continue

                if self._matches_target(app_name, binary):
                    vol = float(si.volume.value_flat)
                    if si.index not in self._tracked_streams:
                        stream_info = StreamInfo(
                            index=si.index,
                            app_name=app_name,
                            binary=binary,
                            baseline_volume=vol if vol > 0.05 else 1.0,
                            current_volume=vol,
                            is_attenuated=self._is_attenuated,
                            is_muted=bool(si.mute),
                        )
                        self._tracked_streams[si.index] = stream_info
                    else:
                        stream_info = self._tracked_streams[si.index]
                        stream_info.is_attenuated = self._is_attenuated

                    # If virtual bridge is active, ensure target stream is routed to shutup_ads_bridge
                    if self.use_bridge and self._bridge_null_mod:
                        # Check if it needs to be moved to the bridge
                        try:
                            bridge_sink = pulse.get_sink_by_name("shutup_ads_bridge")
                            if si.sink != bridge_sink.index:
                                subprocess.run(["pactl", "move-sink-input", str(si.index), "shutup_ads_bridge"], capture_output=True)
                                subprocess.run(["pactl", "set-sink-input-volume", str(si.index), "100%"], capture_output=True)
                                subprocess.run(["pactl", "set-sink-input-mute", str(si.index), "0"], capture_output=True)
                        except Exception:
                            subprocess.run(["pactl", "move-sink-input", str(si.index), "shutup_ads_bridge"], capture_output=True)

                    matched.append(stream_info)
        except Exception as e:
            logger.debug("Error scanning sink inputs: %s", e)

        return matched

    def _find_loopback_sink_input_id(self) -> int | None:
        """Find the sink-input ID of the loopback routing audio to user speakers."""
        if not self._bridge_loop_mod:
            return None

        try:
            res = subprocess.run(["pactl", "list", "sink-inputs"], capture_output=True, text=True).stdout
            for block in res.split("Sink Input #")[1:]:
                lines = block.splitlines()
                si_id = int(lines[0].strip())
                if f"pulse.module.id = \"{self._bridge_loop_mod}\"" in block:
                    self._loopback_si_id = si_id
                    return si_id
        except Exception:
            pass
        return self._loopback_si_id

    def apply_commercial_state(self, is_commercial: bool) -> list[StreamInfo]:
        """Apply volume attenuation to loopback speakers (or target streams in direct mode)."""
        if self.action_mode == "dry_run":
            return self.scan_and_route_streams()

        active_targets = self.scan_and_route_streams()

        if self.use_bridge and self._bridge_loop_mod:
            loop_id = self._find_loopback_sink_input_id()
            if loop_id is not None:
                if is_commercial:
                    if not self._is_attenuated:
                        logger.info("Commercial detected -> Attenuating speaker loopback #%d to %.0f%%",
                                    loop_id, self.attenuate_volume * 100)
                        if self.action_mode == "mute":
                            subprocess.run(["pactl", "set-sink-input-mute", str(loop_id), "1"], capture_output=True)
                        else:
                            vol_pct = f"{int(round(self.attenuate_volume * 100))}%"
                            subprocess.run(["pactl", "set-sink-input-volume", str(loop_id), vol_pct], capture_output=True)
                        self._is_attenuated = True
                else:
                    if self._is_attenuated:
                        logger.info("Live sports detected -> Restoring speaker loopback #%d to 100%%", loop_id)
                        subprocess.run(["pactl", "set-sink-input-mute", str(loop_id), "0"], capture_output=True)
                        subprocess.run(["pactl", "set-sink-input-volume", str(loop_id), "100%"], capture_output=True)
                        self._is_attenuated = False

            for s in active_targets:
                s.is_attenuated = self._is_attenuated
                s.current_volume = self.attenuate_volume if self._is_attenuated else s.baseline_volume

        return active_targets

    def list_all_streams(self) -> list[dict[str, Any]]:
        """List all active sink inputs on the system."""
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
        return result

    def cleanup(self) -> None:
        """Restore all streams to default sink and unload virtual bridge modules."""
        logger.info("Restoring audio streams and unloading virtual bridge...")
        # 1. Un-mute / restore loopback if present
        if self._loopback_si_id:
            subprocess.run(["pactl", "set-sink-input-mute", str(self._loopback_si_id), "0"], capture_output=True)
            subprocess.run(["pactl", "set-sink-input-volume", str(self._loopback_si_id), "100%"], capture_output=True)

        # 2. Move tracked streams back to default physical sink
        target_sink = self.default_sink_name or "alsa_output.pci-0000_63_00.6.pro-output-0"
        for idx in list(self._tracked_streams.keys()):
            try:
                subprocess.run(["pactl", "move-sink-input", str(idx), target_sink], capture_output=True)
                subprocess.run(["pactl", "set-sink-input-volume", str(idx), "100%"], capture_output=True)
                subprocess.run(["pactl", "set-sink-input-mute", str(idx), "0"], capture_output=True)
            except Exception:
                pass

        # 3. Unload modules
        if self._bridge_loop_mod:
            subprocess.run(["pactl", "unload-module", self._bridge_loop_mod], capture_output=True)
            self._bridge_loop_mod = None
        if self._bridge_null_mod:
            subprocess.run(["pactl", "unload-module", self._bridge_null_mod], capture_output=True)
            self._bridge_null_mod = None

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
