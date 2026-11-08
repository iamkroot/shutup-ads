"""Asynchronous screenshot capture for debugging broadcast state transitions."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from shutup_ads.detector.engine import DetectionMetrics
from shutup_ads.detector.state_machine import StateTransitionEvent

logger = logging.getLogger(__name__)


class ScreenshotCapturer:
    """Takes background screenshots during state transitions for offline verification."""

    def __init__(
        self,
        output_dir: Path | str | None = None,
        enabled: bool = False,
    ):
        self.enabled = enabled
        if output_dir is None:
            self.output_dir = Path("debug_captures")
        else:
            self.output_dir = Path(output_dir)

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = self.output_dir / "transitions.jsonl"
        else:
            self.log_file = None

        # Thread pool to avoid blocking audio streaming
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="screenshot-worker")

    def _take_screenshot(self, dest_path: Path) -> bool:
        """Attempt to capture full-screen screenshot using available system tools."""
        # 1. Try spectacle (KDE Wayland / X11 default)
        if shutil.which("spectacle"):
            cmd = ["spectacle", "-b", "-n", "-o", str(dest_path)]
            try:
                res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=4.0)
                if res.returncode == 0 and dest_path.exists() and dest_path.stat().st_size > 1000:
                    return True
            except Exception as e:
                logger.debug("spectacle capture failed: %s", e)

        # 2. Try grim (wlroots Wayland)
        if shutil.which("grim"):
            cmd = ["grim", str(dest_path)]
            try:
                res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=4.0)
                if res.returncode == 0 and dest_path.exists() and dest_path.stat().st_size > 1000:
                    return True
            except Exception as e:
                logger.debug("grim capture failed: %s", e)

        # 3. Try ImageMagick import (X11)
        if shutil.which("import"):
            cmd = ["import", "-window", "root", str(dest_path)]
            try:
                res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=4.0)
                if res.returncode == 0 and dest_path.exists() and dest_path.stat().st_size > 1000:
                    return True
            except Exception as e:
                logger.debug("import capture failed: %s", e)

        # 4. Try ffmpeg x11grab fallback
        if shutil.which("ffmpeg"):
            display = ":1"  # standard fallback
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "x11grab", "-i", display,
                "-vframes", "1", "-y", str(dest_path),
            ]
            try:
                res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=4.0)
                if res.returncode == 0 and dest_path.exists() and dest_path.stat().st_size > 1000:
                    return True
            except Exception as e:
                logger.debug("ffmpeg capture failed: %s", e)

        return False

    def capture_transition(
        self,
        event: StateTransitionEvent,
        metrics: DetectionMetrics,
    ) -> None:
        """Asynchronously capture screenshot and log transition metadata."""
        if not self.enabled:
            return

        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        ad_pct = int(round(metrics.ad_probability * 100))
        sport_pct = int(round(metrics.sports_score * 100))
        filename = f"{ts_str}_{event.from_state.value}_to_{event.to_state.value}_ad{ad_pct}_sports{sport_pct}.png"
        dest_path = self.output_dir / filename

        def _worker():
            success = self._take_screenshot(dest_path)
            meta = {
                "timestamp": datetime.now().isoformat(),
                "from_state": event.from_state.value,
                "to_state": event.to_state.value,
                "reason": event.reason,
                "metrics": {
                    "ad_probability": metrics.ad_probability,
                    "sports_score": metrics.sports_score,
                    "music_score": metrics.music_score,
                    "speech_score": metrics.speech_score,
                    "court_score": metrics.court_score,
                    "crowd_score": metrics.crowd_score,
                    "rms_energy": metrics.rms_energy,
                },
                "screenshot": str(dest_path) if success else None,
            }
            if self.log_file:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(meta) + "\n")
            if success:
                logger.info("Saved debug transition screenshot: %s", dest_path)

        self._executor.submit(_worker)

    def capture_manual(self, label: str = "manual") -> None:
        """Capture an on-demand manual screenshot."""
        if not self.enabled:
            return

        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ts_str}_{label}.png"
        dest_path = self.output_dir / filename
        self._executor.submit(self._take_screenshot, dest_path)

    def shutdown(self) -> None:
        """Shut down background executor."""
        self._executor.shutdown(wait=False)
