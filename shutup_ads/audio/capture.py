"""Real-time and file audio capture pipelines via ffmpeg."""

from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """A single audio chunk yielded from capture."""
    samples: np.ndarray      # 1-D float32 audio samples normalized to [-1.0, 1.0]
    sample_rate: int         # 16000 Hz
    timestamp: float         # Wall clock / stream timestamp
    rms: float               # Root Mean Square energy


class PulseAudioCapture:
    """Streams live audio from PulseAudio / PipeWire monitor source."""

    def __init__(
        self,
        source: str = "default.monitor",
        sample_rate: int = 16000,
        chunk_seconds: float = 0.5,
    ):
        self.source = source
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.chunk_samples = int(sample_rate * chunk_seconds)
        self.chunk_bytes = self.chunk_samples * 4  # float32 = 4 bytes per sample
        self.process: subprocess.Popen | None = None

    def start(self) -> Generator[AudioChunk, None, None]:
        """Start ffmpeg capture process and yield AudioChunks continuously."""
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "pulse",
            "-i", self.source,
            "-ar", str(self.sample_rate),
            "-ac", "1",
            "-f", "f32le",
            "-",
        ]
        logger.info("Starting PulseAudio capture with cmd: %s", " ".join(cmd))
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self.chunk_bytes * 4,
        )

        try:
            while self.process.poll() is None:
                raw_bytes = self.process.stdout.read(self.chunk_bytes)
                if len(raw_bytes) < self.chunk_bytes:
                    break
                samples = np.frombuffer(raw_bytes, dtype=np.float32)
                rms = float(np.sqrt(np.mean(samples ** 2)))
                yield AudioChunk(
                    samples=samples,
                    sample_rate=self.sample_rate,
                    timestamp=time.time(),
                    rms=rms,
                )
        finally:
            self.stop()

    def stop(self) -> None:
        """Terminate the ffmpeg process."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except (subprocess.SubprocessError, OSError):
                self.process.kill()
            self.process = None


class FileAudioCapture:
    """Streams audio from a local media file (for offline testing & evaluation)."""

    def __init__(
        self,
        file_path: Path | str,
        sample_rate: int = 16000,
        chunk_seconds: float = 0.5,
        realtime: bool = False,
    ):
        self.file_path = Path(file_path)
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.chunk_samples = int(sample_rate * chunk_seconds)
        self.chunk_bytes = self.chunk_samples * 4
        self.realtime = realtime
        self.process: subprocess.Popen | None = None

    def start(self) -> Generator[AudioChunk, None, None]:
        """Stream chunks from audio/video file."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Media file not found: {self.file_path}")

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(self.file_path),
            "-ar", str(self.sample_rate),
            "-ac", "1",
            "-f", "f32le",
            "-",
        ]
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self.chunk_bytes * 4,
        )

        stream_time = 0.0
        try:
            while self.process.poll() is None:
                t_start = time.time()
                raw_bytes = self.process.stdout.read(self.chunk_bytes)
                if len(raw_bytes) < self.chunk_bytes:
                    break
                samples = np.frombuffer(raw_bytes, dtype=np.float32)
                rms = float(np.sqrt(np.mean(samples ** 2)))
                yield AudioChunk(
                    samples=samples,
                    sample_rate=self.sample_rate,
                    timestamp=stream_time,
                    rms=rms,
                )
                stream_time += self.chunk_seconds

                if self.realtime:
                    elapsed = time.time() - t_start
                    delay = self.chunk_seconds - elapsed
                    if delay > 0:
                        time.sleep(delay)
        finally:
            self.stop()

    def stop(self) -> None:
        """Terminate the ffmpeg process."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except (subprocess.SubprocessError, OSError):
                self.process.kill()
            self.process = None
