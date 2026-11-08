"""Unit tests for stream matching and pulse controller."""

import pytest
from shutup_ads.audio.controller import PulseController


def test_target_matching():
    ctrl = PulseController(target_apps=["firefox", "sunshine", "ff"])

    assert ctrl._matches_target("Firefox Developer Edition", "/usr/lib/firefox-developer-edition/firefox")
    assert ctrl._matches_target("Sunshine Game Stream", "sunshine")
    assert ctrl._matches_target("ff", "bash")
    assert not ctrl._matches_target("vesktop", "vesktop")
    assert not ctrl._matches_target("Spotify", "spotify")


def test_empty_target_matches_all():
    ctrl = PulseController(target_apps=[])
    assert ctrl._matches_target("anything", "anything")
