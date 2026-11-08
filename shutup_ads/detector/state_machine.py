"""State machine for broadcast state tracking with debounce and hysteresis."""

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
        ad_threshold: float = 0.50,
        sports_threshold: float = 0.30,
        ad_trigger_seconds: float = 1.5,
        sports_trigger_seconds: float = 1.5,
        min_ad_duration: float = 10.0,
        max_ad_duration: float = 300.0,
    ):
        self.ad_threshold = ad_threshold
        self.sports_threshold = sports_threshold
        self.ad_trigger_seconds = ad_trigger_seconds
        self.sports_trigger_seconds = sports_trigger_seconds
        self.min_ad_duration = min_ad_duration
        self.max_ad_duration = max_ad_duration

        self.current_state = BroadcastState.LIVE_SPORTS
        self.state_enter_time = time.time()
        self.session_start_time = time.time()

        # Pending transition timer
        self._pending_since: float | None = None

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
        return time.time() - self.state_enter_time

    def _transition_to(self, new_state: BroadcastState, reason: str, now: float) -> StateTransitionEvent:
        prev_state = self.current_state
        prev_duration = now - self.state_enter_time
        self.current_state = new_state
        self.state_enter_time = now
        self._pending_since = None

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
        """Update state machine with new chunk metrics.

        Returns StateTransitionEvent if a transition occurred, else None.
        """
        now = current_time if current_time is not None else time.time()
        duration = now - self.state_enter_time

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
                self._pending_since = now
            return None

        elif self.current_state == BroadcastState.PENDING_COMMERCIAL:
            if metrics.ad_probability < self.ad_threshold:
                # Cancel pending
                self.current_state = BroadcastState.LIVE_SPORTS
                self._pending_since = None
                return None
            else:
                elapsed = now - (self._pending_since or now)
                if elapsed >= self.ad_trigger_seconds:
                    return self._transition_to(
                        BroadcastState.COMMERCIAL_BREAK,
                        f"Sustained ad probability ({metrics.ad_probability:.2f} >= {self.ad_threshold:.2f} for {elapsed:.1f}s)",
                        now,
                    )
                return None

        elif self.current_state == BroadcastState.COMMERCIAL_BREAK:
            # Check minimum ad duration guard
            if duration < self.min_ad_duration:
                return None

            # Condition to return to sports:
            # Either high basketball sports score OR low ad probability
            if (metrics.sports_score >= self.sports_threshold or
                metrics.ad_probability < (self.ad_threshold * 0.65)):
                self.current_state = BroadcastState.PENDING_SPORTS
                self._pending_since = now
            return None

        elif self.current_state == BroadcastState.PENDING_SPORTS:
            # Check if sports condition holds
            if (metrics.sports_score < self.sports_threshold and
                metrics.ad_probability >= (self.ad_threshold * 0.65)):
                # Cancel pending, revert to commercial
                self.current_state = BroadcastState.COMMERCIAL_BREAK
                self._pending_since = None
                return None
            else:
                elapsed = now - (self._pending_since or now)
                if elapsed >= self.sports_trigger_seconds:
                    return self._transition_to(
                        BroadcastState.LIVE_SPORTS,
                        f"Basketball action resumed (sports={metrics.sports_score:.2f}, ad_prob={metrics.ad_probability:.2f} for {elapsed:.1f}s)",
                        now,
                    )
                return None

        return None

    def force_state(self, state: BroadcastState, reason: str = "Manual override") -> StateTransitionEvent:
        """Force state transition manually."""
        return self._transition_to(state, reason, time.time())
