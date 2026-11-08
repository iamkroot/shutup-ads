"""Terminal UI dashboard built with Rich."""

import time

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from shutup_ads.audio.controller import StreamInfo
from shutup_ads.detector.engine import DetectionMetrics
from shutup_ads.detector.state_machine import BroadcastState, BroadcastStateMachine


def make_meter(val: float, width: int = 20, color: str = "cyan") -> str:
    """Render a text progress meter."""
    clamped = max(0.0, min(1.0, val))
    filled = round(clamped * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{color}]{bar}[/{color}] {clamped*100:5.1f}%"


class Dashboard:
    """Rich interactive dashboard for shutup-ads."""

    def __init__(
        self,
        action_mode: str = "attenuate",
        target_vol_pct: float = 0.15,
        preset_name: str = "Basketball",
    ):
        self.console = Console()
        self.action_mode = action_mode
        self.target_vol_pct = target_vol_pct
        self.preset_name = preset_name
        self.recent_events: list[str] = []

    def log_event(self, msg: str) -> None:
        """Add an event to the recent events log."""
        ts = time.strftime("%H:%M:%S")
        self.recent_events.append(f"[dim]{ts}[/dim] {msg}")
        if len(self.recent_events) > 6:
            self.recent_events.pop(0)

    def render(
        self,
        metrics: DetectionMetrics,
        state_machine: BroadcastStateMachine,
        streams: list[StreamInfo],
        top_classes: list[tuple[str, float]],
    ) -> Panel:
        """Render the full dashboard UI layout."""
        # 1. State Banner
        state = state_machine.current_state
        duration = state_machine.time_in_current_state

        preset_upper = self.preset_name.upper()
        icon = "🏀" if "basket" in self.preset_name.lower() else "🏟️"

        if state == BroadcastState.LIVE_SPORTS:
            state_text = Text(f" {icon}  LIVE {preset_upper}  [{duration:.1f}s] ", style="bold white on green", justify="center")
        elif state == BroadcastState.PENDING_COMMERCIAL:
            state_text = Text(f" ⏳  DETECTING COMMERCIAL BREAK...  [{duration:.1f}s] ", style="bold black on yellow", justify="center")
        elif state == BroadcastState.COMMERCIAL_BREAK:
            label = "MUTED" if self.action_mode == "mute" else f"ATTENUATED ({int(self.target_vol_pct*100)}%)"
            state_text = Text(f" 🔇  COMMERCIAL BREAK - {label}  [{duration:.1f}s] ", style="bold white on red", justify="center")
        elif state == BroadcastState.PENDING_SPORTS:
            state_text = Text(f" ⏳  RESUMING {preset_upper}...  [{duration:.1f}s] ", style="bold white on dark_green", justify="center")
        else:
            state_text = Text(f" {state.value} ", style="bold", justify="center")

        # 2. Main Metrics Table
        metrics_table = Table.grid(expand=True, padding=(0, 2))
        metrics_table.add_column(ratio=1)
        metrics_table.add_column(ratio=1)

        # Left Column: Confidence Scores
        left_table = Table(title="Broadcast Confidence", box=None, padding=(0, 1), expand=True)
        left_table.add_column("Signal", style="bold")
        left_table.add_column("Level / Probability", justify="right")

        ad_color = "red" if metrics.ad_probability > 0.5 else "green"
        sports_color = "green" if metrics.sports_score > 0.3 else "dim"

        left_table.add_row("Commercial Prob", make_meter(metrics.ad_probability, 18, ad_color))
        left_table.add_row("Basketball Action", make_meter(metrics.sports_score, 18, sports_color))
        left_table.add_row("Audio RMS Energy", make_meter(min(1.0, metrics.rms_energy * 5.0), 18, "yellow"))

        # Right Column: Basketball & Commercial Features
        right_table = Table(title=f"{self.preset_name} Acoustic Profile", box=None, padding=(0, 1), expand=True)
        right_table.add_column("Feature", style="bold")
        right_table.add_column("Level", justify="right")

        court_label = f"{icon} Court Action (Bounce/Squeak/Whistle)" if "basket" in self.preset_name.lower() else f"{icon} Game / Match Action"
        right_table.add_row(court_label, make_meter(metrics.court_score, 16, "cyan"))
        right_table.add_row("📣 Crowd & Cheering", make_meter(metrics.crowd_score, 16, "blue"))
        right_table.add_row("🎵 Studio Music / Jingles", make_meter(metrics.music_score, 16, "magenta"))
        right_table.add_row("🗣️ Clean Studio Voiceover", make_meter(metrics.clean_speech_score, 16, "bright_black"))

        metrics_table.add_row(left_table, right_table)

        # 3. Target Streams Table
        streams_table = Table(title="Target Audio Streams", box=None, padding=(0, 1), expand=True)
        streams_table.add_column("Index", style="dim", width=8)
        streams_table.add_column("Application", style="bold cyan")
        streams_table.add_column("Status", justify="center")
        streams_table.add_column("Volume", justify="right")

        if not streams:
            streams_table.add_row("-", "[yellow]Scanning for target apps (firefox, sunshine)...[/yellow]", "-", "-")
        else:
            for s in streams:
                if s.is_attenuated or s.is_muted:
                    status = "[bold red]ATTENUATED[/bold red]" if not s.is_muted else "[bold red]MUTED[/bold red]"
                else:
                    status = "[bold green]NORMAL[/bold green]"
                vol_str = f"{int(s.current_volume*100)}% (base: {int(s.baseline_volume*100)}%)"
                streams_table.add_row(f"#{s.index}", s.app_name, status, vol_str)

        # 4. Top AudioSet Classes
        classes_text = Text()
        classes_text.append("Top Classes: ", style="bold")
        for idx, (name, prob) in enumerate(top_classes[:4]):
            classes_text.append(f"{name} ({prob:.2f})", style="cyan" if "music" not in name.lower() else "magenta")
            if idx < 3:
                classes_text.append("  •  ", style="dim")

        # 5. Session Stats
        elapsed = time.time() - state_machine.session_start_time
        mins, secs = divmod(int(elapsed), 60)
        ad_mins, ad_secs = divmod(int(state_machine.total_commercial_seconds), 60)
        stats_line = (
            f"[bold]Session:[/bold] {mins:02d}:{secs:02d}  •  "
            f"[bold]Commercials Caught:[/bold] {state_machine.commercial_break_count}  •  "
            f"[bold]Total Ad Time Silenced:[/bold] {ad_mins:02d}:{ad_secs:02d}"
        )

        # 6. Event Log
        log_content = "\n".join(self.recent_events) if self.recent_events else "[dim]Waiting for first commercial transition...[/dim]"
        log_panel = Panel(log_content, title="Event History", height=5, border_style="dim")

        # 7. Hotkeys Footer
        footer = Text(
            "[c] Screenshot  │  [m] Toggle Attenuation  │  [s] Force Sports  │  [a] Force Commercial  │  [q] Quit",
            style="dim italic",
            justify="center",
        )

        content = Group(
            state_text,
            Text(""),
            metrics_table,
            Text(""),
            classes_text,
            Text(""),
            streams_table,
            Text(""),
            Text.from_markup(stats_line),
            log_panel,
            footer,
        )

        return Panel(
            content,
            title="[bold green]shutup-ads[/bold green] [dim]v0.1.0 (YAMNet Real-Time Basketball Detector)[/dim]",
            border_style="green",
            padding=(1, 2),
        )
