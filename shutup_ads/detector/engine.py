"""Basketball commercial break detection engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from shutup_ads.detector.classes import (
    APPLAUSE,
    BASKETBALL_BOUNCE,
    BASKETBALL_SPORTS_CLASSES,
    BUZZER,
    CHEERING,
    COMMERCIAL_MUSIC_CLASSES,
    CROWD,
    JINGLE_MUSIC,
    MUSIC,
    NARRATION,
    SILENCE,
    SPEECH,
    SQUEAK,
    WHISTLE,
)
from shutup_ads.model.classifier import CalibratedClassifier


@dataclass
class DetectionMetrics:
    """Detailed acoustic metrics for a single audio window."""
    sports_score: float             # Overall basketball action confidence [0.0, 1.0]
    commercial_score: float         # Raw commercial / advertising score [0.0, 1.0]
    ad_probability: float           # Final fused ad probability [0.0, 1.0]
    crowd_score: float              # Crowd / Cheering level
    court_score: float              # Basketball bounce / squeak / whistle level
    music_score: float              # Commercial music / jingle level
    speech_score: float             # Announcer / narration speech level
    clean_speech_score: float       # Studio speech without crowd noise
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
        sports_threshold: float = 0.20,
        ema_alpha: float = 0.60,  # Responsive smoothing factor
    ):
        self.classifier = classifier
        self.ad_threshold = ad_threshold
        self.sports_threshold = sports_threshold
        self.ema_alpha = ema_alpha

        self._smoothed_ad_prob: float = 0.0
        self._smoothed_sports_score: float = 0.5

    def reset(self) -> None:
        """Reset internal smoothing states."""
        self._smoothed_ad_prob = 0.0
        self._smoothed_sports_score = 0.5

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
        # Court action: bounce, squeak, whistle, buzzer
        bounce = float(frame_mean[BASKETBALL_BOUNCE])
        squeak = float(frame_mean[SQUEAK])
        whistle = float(frame_mean[WHISTLE])
        buzzer = float(frame_mean[BUZZER])
        court_action = max(bounce * 1.5, squeak * 1.2, whistle * 1.8, buzzer * 1.5)

        # Crowd & arena atmosphere
        crowd = float(frame_mean[CROWD])
        cheer = float(frame_mean[CHEERING])
        applause = float(frame_mean[APPLAUSE])
        crowd_cheer = max(crowd, cheer, applause, (crowd + cheer + applause) / 2.0)

        # Sports score is strictly court action and arena crowd sounds.
        # Human speech is intentionally excluded so voice timbre in commercials never triggers sports!
        sports_score = float(np.clip(
            max(crowd_cheer, court_action, 0.7 * crowd_cheer + 0.5 * court_action),
            0.0, 1.0
        ))

        # 2. Commercial Indicators:
        # Commercial breaks require prominent, continuous studio music or jingles (> 0.18).
        music_sub = frame_mean[COMMERCIAL_MUSIC_CLASSES]
        music_score = float(np.max(music_sub))

        speech = float(frame_mean[SPEECH])
        narration = float(frame_mean[NARRATION])
        speech_score = max(speech, narration)

        if music_score < 0.18:
            heuristic_ad = 0.0
        else:
            # Voiceover in commercial strengthens ad confidence when paired with studio music
            ad_signal = max(music_score, 0.60 * music_score + 0.40 * speech_score)
            sports_penalty = 1.5 * sports_score
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

        # 5. Temporal Exponential Moving Average (EMA)
        self._smoothed_ad_prob = float(
            self.ema_alpha * raw_ad_prob + (1.0 - self.ema_alpha) * self._smoothed_ad_prob
        )
        self._smoothed_sports_score = float(
            self.ema_alpha * sports_score + (1.0 - self.ema_alpha) * self._smoothed_sports_score
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
            clean_speech_score=speech_score,
            silence_score=silence_score,
            rms_energy=rms,
            is_cut_candidate=is_cut,
            embedding_ad_prob=embedding_ad_prob,
        )
