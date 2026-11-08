"""Unit tests for the broadcast state machine."""

import pytest
from shutup_ads.detector.engine import DetectionMetrics
from shutup_ads.detector.state_machine import BroadcastState, BroadcastStateMachine


def make_dummy_metrics(ad_prob: float, sports_score: float) -> DetectionMetrics:
    return DetectionMetrics(
        sports_score=sports_score,
        commercial_score=ad_prob,
        ad_probability=ad_prob,
        crowd_score=sports_score,
        court_score=sports_score,
        music_score=ad_prob,
        speech_score=0.5,
        clean_speech_score=ad_prob,
        silence_score=0.0,
        rms_energy=0.05,
        is_cut_candidate=False,
        embedding_ad_prob=None,
    )


def test_transition_to_commercial_after_delay():
    sm = BroadcastStateMachine(
        ad_threshold=0.50,
        sports_threshold=0.30,
        ad_trigger_seconds=1.5,
        sports_trigger_seconds=1.5,
        min_ad_duration=10.0,
    )

    t = 100.0
    ad_metrics = make_dummy_metrics(ad_prob=0.85, sports_score=0.10)

    # 1. First ad chunk -> enters PENDING_COMMERCIAL
    ev = sm.update(ad_metrics, current_time=t)
    assert ev is None
    assert sm.current_state == BroadcastState.PENDING_COMMERCIAL

    # 2. Before delay expires (1.0s elapsed) -> remains PENDING
    ev = sm.update(ad_metrics, current_time=t + 1.0)
    assert ev is None
    assert sm.current_state == BroadcastState.PENDING_COMMERCIAL

    # 3. After delay (1.6s elapsed) -> transitions to COMMERCIAL_BREAK
    ev = sm.update(ad_metrics, current_time=t + 1.6)
    assert ev is not None
    assert ev.to_state == BroadcastState.COMMERCIAL_BREAK
    assert sm.current_state == BroadcastState.COMMERCIAL_BREAK
    assert sm.is_commercial is True


def test_transient_spike_does_not_trigger_commercial():
    sm = BroadcastStateMachine(ad_trigger_seconds=1.5)
    t = 100.0
    ad_metrics = make_dummy_metrics(ad_prob=0.85, sports_score=0.10)
    sport_metrics = make_dummy_metrics(ad_prob=0.10, sports_score=0.80)

    sm.update(ad_metrics, current_time=t)
    assert sm.current_state == BroadcastState.PENDING_COMMERCIAL

    # Ad signal drops before 1.5s
    sm.update(sport_metrics, current_time=t + 0.5)
    assert sm.current_state == BroadcastState.LIVE_SPORTS
    assert sm.is_commercial is False


def test_min_ad_duration_guard():
    sm = BroadcastStateMachine(
        ad_trigger_seconds=1.0,
        sports_trigger_seconds=1.0,
        min_ad_duration=10.0,
    )

    t = 100.0
    ad_metrics = make_dummy_metrics(ad_prob=0.85, sports_score=0.10)
    sport_metrics = make_dummy_metrics(ad_prob=0.10, sports_score=0.80)

    # Enter commercial break
    sm.update(ad_metrics, current_time=t)
    sm.update(ad_metrics, current_time=t + 1.2)
    assert sm.current_state == BroadcastState.COMMERCIAL_BREAK

    # Sports resumes early at 4.0s (less than min_ad_duration 10s)
    ev = sm.update(sport_metrics, current_time=t + 4.0)
    assert ev is None
    assert sm.current_state == BroadcastState.COMMERCIAL_BREAK

    # After min_ad_duration has passed (12.0s since t, which is 10.8s in commercial break)
    sm.update(sport_metrics, current_time=t + 12.0)
    assert sm.current_state == BroadcastState.PENDING_SPORTS

    # Sustained sports for sports_trigger_seconds (1.2s more -> 13.2s)
    ev = sm.update(sport_metrics, current_time=t + 13.2)
    assert ev is not None
    assert ev.to_state == BroadcastState.LIVE_SPORTS
    assert sm.current_state == BroadcastState.LIVE_SPORTS


def test_failsafe_timeout():
    sm = BroadcastStateMachine(
        ad_trigger_seconds=1.0,
        max_ad_duration=60.0,
    )
    t = 100.0
    ad_metrics = make_dummy_metrics(ad_prob=0.85, sports_score=0.10)

    sm.update(ad_metrics, current_time=t)
    sm.update(ad_metrics, current_time=t + 1.2)
    assert sm.current_state == BroadcastState.COMMERCIAL_BREAK

    # Timeout occurs at 65s
    ev = sm.update(ad_metrics, current_time=t + 65.0)
    assert ev is not None
    assert ev.to_state == BroadcastState.LIVE_SPORTS
    assert "Failsafe" in ev.reason
