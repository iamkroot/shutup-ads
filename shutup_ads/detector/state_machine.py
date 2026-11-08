"""State machine for broadcast state tracking with debounce, hysteresis, and ad guards."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass
from shutup_ads.detector.engine import DetectionMetrics


class BroadcastState(enum.Enum):
    LIVE_SPORTS = "LIVE_SPORTS"
    PENDING_COMMERCIAL = "PENDING_COMMERCIAL"
    COMMERCIAL_BREAK = "COMMERCIAL_BREAK"
    PENDING_SPORTS = "PENDING_SPORTS"


@dataclass
class StateTransitionEvent:
    from_state: BroadcastState
    to_state: BroadcastState
    timestamp: float
    reason: str
    duration_in_prev_state: float


class BroadcastStateMachine:
    """Manages state transitions between live sports and commercial breaks."""

    def __init__(
        self,
        ad_threshold: float = 0.40,
        sports_threshold: float = 0.18,
        ad_trigger_seconds: float = 3.5,     # 3.5s window rides through 2-3s network outro bumpers
        sports_trigger_seconds: float = 2.0, # Sustained sports required to unmute
        min_ad_duration: float = 20.0,       # Commercial breaks are at least 20-30s
        max_ad_duration: float = 270.0,      # 4.5m failsafe auto-unmute
    ):
        self.ad_threshold = ad_threshold
        self.sports_threshold = sports_threshold
        self.ad_trigger_seconds = ad_trigger_seconds
        self.sports_trigger_seconds = sports_trigger_seconds
        self.min_ad_duration = min_ad_duration
        self.max_ad_duration = max_ad_duration

        self.current_state = BroadcastState.LIVE_SPORTS
        self.state_enter_time: float | None = None
        self.session_start_time: float | None = None
        self._last_update_time: float | None = None

        # Accumulator for pending commercial trigger (leaky bucket)
        self._pending_accum: float = 0.0

        # Pending sports timer
        self._pending_sports_since: float | None = None

        # Statistics
        self.commercial_break_count = 0
        self.total_commercial_seconds = 0.0
        self.events: list[StateTransitionEvent] = []

    @property
    def is_commercial(self) -> bool:
        """True if currently in COMMERCIAL_BREAK state."""
        return self.current_state in (
            BroadcastState.COMMERCIAL_BREAK,
            BroadcastState.PENDING_SPORTS,
        )

    @property
    def time_in_current_state(self) -> float:
        """Seconds spent in current state."""
        if self.state_enter_time is None:
            return 0.0
        return time.time() - self.state_enter_time

    def _transition_to(self, new_state: BroadcastState, reason: str, now: float) -> StateTransitionEvent:
        prev_state = self.current_state
        prev_duration = (now - self.state_enter_time) if self.state_enter_time is not None else 0.0
        self.current_state = new_state
        self.state_enter_time = now
        self._pending_accum = 0.0
        self._pending_sports_since = None

        if prev_state == BroadcastState.COMMERCIAL_BREAK and new_state in (BroadcastState.LIVE_SPORTS, BroadcastState.PENDING_SPORTS):
            self.total_commercial_seconds += prev_duration

        if new_state == BroadcastState.COMMERCIAL_BREAK:
            self.commercial_break_count += 1

        event = StateTransitionEvent(
            from_state=prev_state,
            to_state=new_state,
            timestamp=now,
            reason=reason,
            duration_in_prev_state=prev_duration,
        )
        self.events.append(event)
        return event

    def update(self, metrics: DetectionMetrics, current_time: float | None = None) -> StateTransitionEvent | None:
        """Update state machine with new chunk metrics."""
        now = current_time if current_time is not None else time.time()

        if self.session_start_time is None:
            self.session_start_time = now
            self.state_enter_time = now
            self._last_update_time = now
            dt = 0.5
        else:
            dt = max(0.0, min(2.0, now - (self._last_update_time or now)))
            self._last_update_time = now
        duration = now - (self.state_enter_time or now)

        # Failsafe: max ad duration auto-restore
        if self.is_commercial and duration > self.max_ad_duration:
            return self._transition_to(
                BroadcastState.LIVE_SPORTS,
                f"Failsafe timeout ({self.max_ad_duration}s exceeded)",
                now,
            )

        if self.current_state == BroadcastState.LIVE_SPORTS:
            if metrics.ad_probability >= self.ad_threshold:
                self.current_state = BroadcastState.PENDING_COMMERCIAL
                self._pending_accum = 0.0
            return None

        elif self.current_state == BroadcastState.PENDING_COMMERCIAL:
            if metrics.ad_probability >= self.ad_threshold:
                self._pending_accum += dt
            else:
                # If ad probability drops significantly (e.g. below half threshold), cancel immediately
                if metrics.ad_probability < self.ad_threshold * 0.50:
                    self.current_state = BroadcastState.LIVE_SPORTS
                    self._pending_accum = 0.0
                    return None
                # Otherwise decay smoothly across dialogue pauses
                self._pending_accum = max(0.0, self._pending_accum - 0.7 * dt)
                if self._pending_accum <= 0.0:
                    self.current_state = BroadcastState.LIVE_SPORTS
                    return None

            if self._pending_accum >= self.ad_trigger_seconds:
                return self._transition_to(
                    BroadcastState.COMMERCIAL_BREAK,
                    f"Sustained ad probability ({metrics.ad_probability:.2f} >= {self.ad_threshold:.2f} for {self._pending_accum:.1f}s)",
                    now,
                )
            return None

        elif self.current_state == BroadcastState.COMMERCIAL_BREAK:
            # Check minimum ad duration guard (commercial breaks are at least 20s)
            if duration < self.min_ad_duration:
                return None

            # Return to sports ONLY on positive evidence of basketball court action or crowd sounds!
            # Dialogue gaps, music dips, or silence cuts in ads will NOT return to sports.
            if metrics.sports_score >= self.sports_threshold and metrics.music_score < 0.12:
                self.current_state = BroadcastState.PENDING_SPORTS
                self._pending_sports_since = now
            return None

        elif self.current_state == BroadcastState.PENDING_SPORTS:
            # Check if sports condition holds
            if metrics.sports_score < self.sports_threshold or metrics.music_score >= 0.12:
                # Revert back to commercial break
                self.current_state = BroadcastState.COMMERCIAL_BREAK
                self._pending_sports_since = None
                return None
            else:
                elapsed = now - (self._pending_sports_since or now)
                if elapsed >= self.sports_trigger_seconds:
                    return self._transition_to(
                        BroadcastState.LIVE_SPORTS,
                        f"Basketball action resumed (sports={metrics.sports_score:.2f} >= {self.sports_threshold:.2f} for {elapsed:.1f}s)",
                        now,
                    )
                return None

        return None

    def force_state(self, state: BroadcastState, reason: str = "Manual override") -> StateTransitionEvent:
        """Force state transition manually."""
        now = time.time()
        return self._transition_to(state, reason, now)
