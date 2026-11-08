"""Unit tests for basketball commercial detection heuristics."""

import numpy as np
import pytest

from shutup_ads.detector.classes import (
    BASKETBALL_BOUNCE,
    CHEERING,
    CROWD,
    MUSIC,
    POP_MUSIC,
    SPEECH,
    SQUEAK,
    WHISTLE,
)
from shutup_ads.detector.engine import BasketballDetector


@pytest.fixture
def detector():
    return BasketballDetector(ad_threshold=0.50, sports_threshold=0.30, ema_alpha=1.0)


def test_basketball_game_scores_high_sports(detector):
    """Simulate live basketball gameplay: court sounds, crowd cheering, commentator."""
    scores = np.zeros((2, 521), dtype=np.float32)
    scores[:, BASKETBALL_BOUNCE] = 0.65
    scores[:, SQUEAK] = 0.50
    scores[:, CROWD] = 0.70
    scores[:, CHEERING] = 0.60
    scores[:, SPEECH] = 0.40  # commentator over crowd
    scores[:, MUSIC] = 0.05

    embeddings = np.zeros((2, 1024), dtype=np.float32)
    metrics = detector.evaluate(scores, embeddings, rms=0.08)

    assert metrics.sports_score > 0.50
    assert metrics.crowd_score > 0.50
    assert metrics.court_score > 0.50
    assert metrics.ad_probability < 0.20


def test_commercial_break_scores_high_ad(detector):
    """Simulate commercial break: studio music, clean voiceover, zero crowd, zero bounce."""
    scores = np.zeros((2, 521), dtype=np.float32)
    scores[:, BASKETBALL_BOUNCE] = 0.0
    scores[:, SQUEAK] = 0.0
    scores[:, CROWD] = 0.01
    scores[:, CHEERING] = 0.0
    scores[:, MUSIC] = 0.85
    scores[:, POP_MUSIC] = 0.80
    scores[:, SPEECH] = 0.75  # clean narration

    embeddings = np.zeros((2, 1024), dtype=np.float32)
    metrics = detector.evaluate(scores, embeddings, rms=0.09)

    assert metrics.ad_probability > 0.60
    assert metrics.music_score > 0.70
    assert metrics.sports_score < 0.15


def test_arena_timeout_dj_not_flagged_as_commercial(detector):
    """Simulate short arena DJ music during live game with crowd singing and whistle."""
    scores = np.zeros((2, 521), dtype=np.float32)
    scores[:, MUSIC] = 0.55
    scores[:, CROWD] = 0.65
    scores[:, CHEERING] = 0.50
    scores[:, WHISTLE] = 0.40

    embeddings = np.zeros((2, 1024), dtype=np.float32)
    metrics = detector.evaluate(scores, embeddings, rms=0.08)

    # Sports penalty suppresses commercial false positive
    assert metrics.sports_score > 0.40
    assert metrics.ad_probability < 0.35
