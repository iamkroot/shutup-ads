"""Unit tests for screenshot capturer."""

from pathlib import Path

from shutup_ads.debug.capturer import ScreenshotCapturer
from shutup_ads.detector.engine import DetectionMetrics
from shutup_ads.detector.state_machine import BroadcastState, StateTransitionEvent


def test_screenshot_capturer_disabled(tmp_path):
    capturer = ScreenshotCapturer(output_dir=tmp_path, enabled=False)
    dummy_event = StateTransitionEvent(
        from_state=BroadcastState.LIVE_SPORTS,
        to_state=BroadcastState.COMMERCIAL_BREAK,
        timestamp=100.0,
        reason="test",
        duration_in_prev_state=5.0,
    )
    dummy_metrics = DetectionMetrics(
        sports_score=0.1,
        commercial_score=0.8,
        ad_probability=0.8,
        crowd_score=0.0,
        court_score=0.0,
        music_score=0.8,
        speech_score=0.2,
        clean_speech_score=0.0,
        silence_score=0.0,
        rms_energy=0.05,
        is_cut_candidate=False,
        embedding_ad_prob=None,
    )
    capturer.capture_transition(dummy_event, dummy_metrics)
    capturer.shutdown()
    assert len(list(tmp_path.iterdir())) == 0


def test_screenshot_capturer_enabled(tmp_path, monkeypatch):
    capturer = ScreenshotCapturer(output_dir=tmp_path, enabled=True)
    # Mock _take_screenshot to create a dummy image
    def mock_take(dest: Path) -> bool:
        dest.write_text("dummy image data")
        return True

    monkeypatch.setattr(capturer, "_take_screenshot", mock_take)

    dummy_event = StateTransitionEvent(
        from_state=BroadcastState.LIVE_SPORTS,
        to_state=BroadcastState.COMMERCIAL_BREAK,
        timestamp=100.0,
        reason="test",
        duration_in_prev_state=5.0,
    )
    dummy_metrics = DetectionMetrics(
        sports_score=0.1,
        commercial_score=0.8,
        ad_probability=0.8,
        crowd_score=0.0,
        court_score=0.0,
        music_score=0.8,
        speech_score=0.2,
        clean_speech_score=0.0,
        silence_score=0.0,
        rms_energy=0.05,
        is_cut_candidate=False,
        embedding_ad_prob=None,
    )
    capturer.capture_transition(dummy_event, dummy_metrics)
    capturer._executor.shutdown(wait=True)

    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) == 1
    assert "LIVE_SPORTS_to_COMMERCIAL_BREAK" in pngs[0].name

    jsonl = tmp_path / "transitions.jsonl"
    assert jsonl.exists()
    assert "LIVE_SPORTS" in jsonl.read_text()
