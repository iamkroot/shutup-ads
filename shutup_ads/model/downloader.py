"""Model assets downloader and cache manager for YAMNet."""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path
from rich.progress import Progress, BarColumn, DownloadColumn, TextColumn, TimeRemainingColumn

YAMNET_ONNX_URL = "https://huggingface.co/zeropointnine/yamnet-onnx/resolve/main/yamnet.onnx"
YAMNET_CLASS_MAP_URL = "https://huggingface.co/zeropointnine/yamnet-onnx/raw/main/yamnet_class_map.csv"

def get_cache_dir() -> Path:
    """Return cache directory for shutup-ads models and calibration files."""
    env_cache = os.environ.get("SHUTUP_ADS_CACHE_DIR")
    if env_cache:
        path = Path(env_cache)
    else:
        path = Path.home() / ".cache" / "shutup-ads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download_file(url: str, dest_path: Path, description: str) -> None:
    """Download a file with rich progress reporting."""
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TimeRemainingColumn(),
        transient=True,
    ) as progress:
        task_id = progress.add_task(description, total=None)

        def report_hook(block_num: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                progress.update(task_id, total=total_size, completed=block_num * block_size)

        try:
            urllib.request.urlretrieve(url, temp_path, reporthook=report_hook)
            temp_path.replace(dest_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise


def ensure_yamnet_assets(cache_dir: Path | None = None) -> tuple[Path, Path]:
    """Ensure yamnet.onnx and yamnet_class_map.csv exist locally, downloading if necessary.

    Returns:
        tuple[Path, Path]: (model_path, class_map_path)
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()

    model_path = cache_dir / "yamnet.onnx"
    class_map_path = cache_dir / "yamnet_class_map.csv"

    # Check existing files
    if not model_path.exists() or model_path.stat().st_size < 10_000_000:
        _download_file(YAMNET_ONNX_URL, model_path, "Downloading YAMNet ONNX model")

    if not class_map_path.exists() or class_map_path.stat().st_size < 1000:
        _download_file(YAMNET_CLASS_MAP_URL, class_map_path, "Downloading YAMNet Class Map")

    return model_path, class_map_path
