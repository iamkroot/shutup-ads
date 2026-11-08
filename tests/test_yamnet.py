"""Unit tests for YAMNet ONNX inference wrapper."""


import numpy as np
import pytest

from shutup_ads.model.yamnet import YAMNetModel, ensure_yamnet_assets


@pytest.fixture(scope="module")
def yamnet():
    model_path, class_map = ensure_yamnet_assets()
    return YAMNetModel(model_path, class_map)


def test_yamnet_initialization(yamnet):
    assert len(yamnet.index_to_name) == 521
    assert yamnet.class_name(0) == "Speech"
    assert yamnet.class_name(459) == "Basketball bounce"
    assert yamnet.class_name(355) == "Squeak"
    assert yamnet.class_name(396) == "Whistle"
    assert yamnet.class_name(132) == "Music"


def test_yamnet_inference_shapes(yamnet):
    # 1 second of 16kHz audio
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000, dtype=np.float32))
    out = yamnet.infer(audio)

    assert out.scores.ndim == 2
    assert out.scores.shape[1] == 521
    assert out.embeddings.ndim == 2
    assert out.embeddings.shape[1] == 1024
    assert out.spectrogram.ndim == 2
    assert out.spectrogram.shape[1] == 64


def test_yamnet_empty_input(yamnet):
    empty = np.zeros(0, dtype=np.float32)
    out = yamnet.infer(empty)
    assert len(out.scores) == 0
    assert len(out.embeddings) == 0
    assert len(out.spectrogram) == 0


def test_yamnet_top_k(yamnet):
    # 440 Hz tone typically triggers telephone / dial tone
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000, dtype=np.float32))
    out = yamnet.infer(audio)
    top_k = yamnet.get_top_k(out.scores, k=3)
    assert len(top_k) == 3
    assert all(isinstance(name, str) and isinstance(prob, float) for name, prob in top_k)
