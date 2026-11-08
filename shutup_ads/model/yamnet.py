"""YAMNet ONNX model wrapper for real-time audio inference."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort

from shutup_ads.model.downloader import ensure_yamnet_assets


@dataclass
class YAMNetOutput:
    """Output tensors from YAMNet inference."""
    scores: np.ndarray        # Shape: [num_frames, 521]
    embeddings: np.ndarray    # Shape: [num_frames, 1024]
    spectrogram: np.ndarray   # Shape: [num_spec_frames, 64]


class YAMNetModel:
    """Wrapper around YAMNet ONNX inference session."""

    SAMPLE_RATE = 16000  # YAMNet requires 16 kHz audio

    def __init__(self, model_path: Path | None = None, class_map_path: Path | None = None):
        if model_path is None or class_map_path is None:
            default_model, default_map = ensure_yamnet_assets()
            model_path = model_path or default_model
            class_map_path = class_map_path or default_map

        self.model_path = Path(model_path)
        self.class_map_path = Path(class_map_path)

        # Initialize ONNX runtime session with CPU execution provider
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name

        # Load class mappings
        self.index_to_name: dict[int, str] = {}
        self.name_to_index: dict[str, int] = {}
        self._load_class_map()

    def _load_class_map(self) -> None:
        """Parse class map CSV."""
        with open(self.class_map_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = int(row["index"])
                name = row["display_name"]
                self.index_to_name[idx] = name
                self.name_to_index[name.lower()] = idx

    def infer(self, waveform: np.ndarray) -> YAMNetOutput:
        """Run inference on 1-D audio waveform sampled at 16000 Hz.

        Args:
            waveform: 1-D float32 array of audio samples in [-1.0, 1.0].

        Returns:
            YAMNetOutput containing scores, embeddings, and spectrogram.
        """
        if waveform.ndim != 1:
            waveform = waveform.flatten()

        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)

        # Avoid zero-length input
        if len(waveform) == 0:
            return YAMNetOutput(
                scores=np.zeros((0, len(self.index_to_name)), dtype=np.float32),
                embeddings=np.zeros((0, 1024), dtype=np.float32),
                spectrogram=np.zeros((0, 64), dtype=np.float32),
            )

        outputs = self.session.run(None, {self.input_name: waveform})
        return YAMNetOutput(
            scores=outputs[0],
            embeddings=outputs[1],
            spectrogram=outputs[2],
        )

    def class_name(self, index: int) -> str:
        """Get class display name for an index."""
        return self.index_to_name.get(index, f"Class {index}")

    def class_index(self, name: str) -> int | None:
        """Look up class index by case-insensitive name."""
        return self.name_to_index.get(name.lower())

    def get_top_k(self, scores: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """Get top k predicted classes aggregated over all frames in scores."""
        if len(scores) == 0:
            return []
        mean_scores = np.mean(scores, axis=0)
        top_indices = np.argsort(mean_scores)[::-1][:k]
        return [(self.class_name(idx), float(mean_scores[idx])) for idx in top_indices]

    def get_classes_score(
        self, scores: np.ndarray, indices: Sequence[int], agg: str = "max"
    ) -> float:
        """Aggregate score across multiple class indices across all frames."""
        if len(scores) == 0 or len(indices) == 0:
            return 0.0
        # Subset class columns: shape [frames, len(indices)]
        sub = scores[:, indices]
        if agg == "max":
            return float(np.max(sub))
        elif agg == "mean":
            return float(np.mean(sub))
        elif agg == "sum":
            return float(np.sum(np.mean(sub, axis=0)))
        else:
            raise ValueError(f"Unknown aggregation: {agg}")
