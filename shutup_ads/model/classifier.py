"""Calibrated transfer learning classifier head on top of YAMNet embeddings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def get_default_model_path() -> Path:
    config_dir = Path.home() / ".config" / "shutup-ads"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "classifier.joblib"


class CalibratedClassifier:
    """Logistic regression classifier trained on YAMNet 1024-D embeddings."""

    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path or get_default_model_path()
        self.pipeline: Pipeline | None = None
        self._load()

    def _load(self) -> None:
        """Attempt to load existing model from disk."""
        if self.model_path.exists():
            try:
                self.pipeline = joblib.load(self.model_path)
            except Exception:
                self.pipeline = None

    @property
    def is_fitted(self) -> bool:
        """Return True if a trained model is loaded and ready."""
        return self.pipeline is not None

    def fit(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray | Sequence[int],
        save: bool = True,
    ) -> None:
        """Fit classifier on embeddings (shape: [N, 1024]) and binary labels (0=sports, 1=ad).

        Args:
            embeddings: [N, 1024] float array
            labels: [N] binary array (0=sports, 1=ad)
            save: whether to persist model to self.model_path
        """
        X = np.asarray(embeddings, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int32)

        if len(np.unique(y)) < 2:
            raise ValueError("Training requires at least two classes (both sports=0 and ad=1 samples).")

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")),
        ])
        pipe.fit(X, y)
        self.pipeline = pipe

        if save:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipe, self.model_path)

    def predict_ad_probability(self, embeddings: np.ndarray) -> float | None:
        """Predict probability of commercial break given frame embeddings.

        Returns float in [0.0, 1.0] or None if model is not fitted.
        """
        if not self.is_fitted or self.pipeline is None or len(embeddings) == 0:
            return None

        # Predict across frames and take mean probability
        probas = self.pipeline.predict_proba(embeddings)
        # Class 1 is ad
        ad_probs = probas[:, 1]
        return float(np.mean(ad_probs))
