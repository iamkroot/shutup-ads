"""Basketball commercial break detection engine with asymmetric evidence accumulation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from shutup_ads.detector.classes import (
    ARENA_ATMOSPHERE_CLASSES,
    BASKETBALL_BOUNCE,
    BASKETBALL_COURT_CLASSES,
    BUZZER,
    COMMERCIAL_FOLEY_CLASSES,
    COMMERCIAL_MUSIC_CLASSES,
    NARRATION,
    SILENCE,
    SPEECH,
    SQUEAK,
    WHISTLE,
)
from shutup_ads.model.classifier import CalibratedClassifier

if TYPE_CHECKING:
    from shutup_ads.config import PresetConfig


@dataclass
class DetectionMetrics:
    """Detailed acoustic metrics for a single audio window."""
    sports_score: float             # Overall basketball action confidence [0.0, 1.0]
    commercial_score: float         # Raw commercial / advertising score [0.0, 1.0]
    ad_probability: float           # Final fused smoothed ad probability [0.0, 1.0]
    crowd_score: float              # Crowd / Cheering / Arena organ level
    court_score: float              # Basketball bounce / squeak / whistle level
    music_score: float              # Commercial music / jingle level
    speech_score: float             # Speech / dialogue level
    clean_speech_score: float       # In-ad narration / dialogue level
    silence_score: float            # Silence class score
    rms_energy: float               # Raw audio RMS level
    is_cut_candidate: bool          # Silence dip indicating commercial cut boundary
    embedding_ad_prob: float | None # Output of custom classifier if fitted


class BasketballDetector:
    """Evaluates YAMNet outputs against basketball and commercial acoustic profiles."""

    def __init__(
        self,
        classifier: CalibratedClassifier | None = None,
        ad_threshold: float = 0.40,
        sports_threshold: float = 0.18,
        alpha_attack: float = 0.35,   # Fast attack when ad evidence increases
        alpha_release: float = 0.10,  # Slow release to maintain confidence across dialogue gaps
        ema_alpha: float | None = None, # Backwards compatibility
        court_classes: Sequence[int] | None = None,
        crowd_classes: Sequence[int] | None = None,
        music_classes: Sequence[int] | None = None,
        foley_classes: Sequence[int] | None = None,
        preset: PresetConfig | None = None,
    ):
        if preset is not None:
            if ad_threshold == 0.40:
                ad_threshold = preset.ad_thresh
            if sports_threshold == 0.18:
                sports_threshold = preset.sports_thresh
            if alpha_attack == 0.35:
                alpha_attack = preset.alpha_attack
            if alpha_release == 0.10:
                alpha_release = preset.alpha_release
            if court_classes is None:
                court_classes = preset.court_classes
            if crowd_classes is None:
                crowd_classes = preset.crowd_classes
            if music_classes is None:
                music_classes = preset.music_classes
            if foley_classes is None:
                foley_classes = preset.foley_classes

        self.classifier = classifier
        self.ad_threshold = ad_threshold
        self.sports_threshold = sports_threshold
        self.alpha_attack = ema_alpha if ema_alpha is not None else alpha_attack
        self.alpha_release = ema_alpha if ema_alpha is not None else alpha_release

        self.court_classes = list(court_classes) if court_classes is not None else list(BASKETBALL_COURT_CLASSES)
        self.crowd_classes = list(crowd_classes) if crowd_classes is not None else list(ARENA_ATMOSPHERE_CLASSES)
        self.music_classes = list(music_classes) if music_classes is not None else list(COMMERCIAL_MUSIC_CLASSES)
        self.foley_classes = list(foley_classes) if foley_classes is not None else list(COMMERCIAL_FOLEY_CLASSES)

        self._smoothed_ad_prob: float = 0.0
        self._smoothed_sports_score: float | None = None

    def reset(self) -> None:
        """Reset internal smoothing states."""
        self._smoothed_ad_prob = 0.0
        self._smoothed_sports_score = None

    def evaluate(
        self,
        scores: np.ndarray,
        embeddings: np.ndarray,
        rms: float,
    ) -> DetectionMetrics:
        """Evaluate a YAMNet inference result (shape: [num_frames, 521])."""
        if len(scores) == 0:
            return DetectionMetrics(
                sports_score=0.0,
                commercial_score=0.0,
                ad_probability=self._smoothed_ad_prob,
                crowd_score=0.0,
                court_score=0.0,
                music_score=0.0,
                speech_score=0.0,
                clean_speech_score=0.0,
                silence_score=0.0,
                rms_energy=rms,
                is_cut_candidate=False,
                embedding_ad_prob=None,
            )

        # 1. Extract class probabilities (mean across window frames)
        frame_mean = np.mean(scores, axis=0)

        # Sports Indicators:
        # Court action: bounce, squeak, whistle, buzzer + generic court classes
        bounce = float(frame_mean[BASKETBALL_BOUNCE]) if BASKETBALL_BOUNCE in self.court_classes else 0.0
        squeak = float(frame_mean[SQUEAK]) if SQUEAK in self.court_classes else 0.0
        whistle = float(frame_mean[WHISTLE]) if WHISTLE in self.court_classes else 0.0
        buzzer = float(frame_mean[BUZZER]) if BUZZER in self.court_classes else 0.0
        specific_action = max(bounce * 1.5, squeak * 1.2, whistle * 1.8, buzzer * 1.5)
        generic_action = float(np.max(frame_mean[self.court_classes])) if self.court_classes else 0.0
        court_action = max(specific_action, generic_action)

        # Arena atmosphere (crowd, cheer, applause, chant, stadium organ)
        if self.crowd_classes:
            arena_sub = frame_mean[self.crowd_classes]
            crowd_cheer = float(np.max(arena_sub))
        else:
            crowd_cheer = 0.0

        # Sports score is strictly court action and arena crowd/organ sounds.
        # Human speech is excluded so voice timbre in commercials never triggers sports.
        sports_score = float(np.clip(
            max(crowd_cheer, court_action, 0.7 * crowd_cheer + 0.5 * court_action),
            0.0, 1.0
        ))

        # 2. Commercial Indicators:
        if self.music_classes:
            music_sub = frame_mean[self.music_classes]
            music_score = float(np.max(music_sub))
        else:
            music_score = 0.0

        speech = float(frame_mean[SPEECH])
        narration = float(frame_mean[NARRATION])
        speech_score = max(speech, narration)

        foley_sub = float(np.max(frame_mean[self.foley_classes])) if self.foley_classes else 0.0
        ad_speech = max(narration, 0.5 * speech, foley_sub)

        # Soft continuous music signal: in live gameplay music was never > 0.045
        music_norm = float(np.clip((music_score - 0.05) / 0.25, 0.0, 1.0))

        # Evidence accumulation: when commercial music is present, dialogue and foley augment confidence
        ad_signal = music_norm * 0.70 + (music_norm * ad_speech) * 0.30
        if music_score >= 0.25:
            ad_signal = max(ad_signal, music_score)

        # Live sports penalty: if actual basketball action/crowd is present, heavily suppress ad signal
        sports_penalty = 2.0 * sports_score
        heuristic_ad = float(np.clip(ad_signal - sports_penalty, 0.0, 1.0))

        # 3. Optional Trained Embedding Head
        embedding_ad_prob: float | None = None
        if self.classifier and self.classifier.is_fitted:
            embedding_ad_prob = self.classifier.predict_ad_probability(embeddings)

        # 4. Score Fusion
        if embedding_ad_prob is not None:
            raw_ad_prob = 0.50 * heuristic_ad + 0.50 * embedding_ad_prob
        else:
            raw_ad_prob = heuristic_ad

        # 5. Asymmetric Temporal EMA Smoothing
        if raw_ad_prob >= self._smoothed_ad_prob:
            alpha = self.alpha_attack
        else:
            alpha = self.alpha_release

        self._smoothed_ad_prob = float(
            alpha * raw_ad_prob + (1.0 - alpha) * self._smoothed_ad_prob
        )
        if self._smoothed_sports_score is None:
            self._smoothed_sports_score = float(sports_score)
        else:
            self._smoothed_sports_score = float(
                0.40 * sports_score + 0.60 * self._smoothed_sports_score
            )

        # 6. Silence / Cut detection
        silence_score = float(frame_mean[SILENCE])
        is_cut = (silence_score > 0.45) or (rms < 0.006)

        return DetectionMetrics(
            sports_score=self._smoothed_sports_score,
            commercial_score=heuristic_ad,
            ad_probability=self._smoothed_ad_prob,
            crowd_score=crowd_cheer,
            court_score=court_action,
            music_score=music_score,
            speech_score=speech_score,
            clean_speech_score=ad_speech,
            silence_score=silence_score,
            rms_energy=rms,
            is_cut_candidate=is_cut,
            embedding_ad_prob=embedding_ad_prob,
        )


SportsDetector = BasketballDetector

