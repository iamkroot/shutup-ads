"""Command line interface for shutup-ads."""

from __future__ import annotations

import argparse
import logging
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Sequence

import numpy as np
from rich.console import Console
from rich.live import Live
from rich.table import Table

from shutup_ads.audio.capture import FileAudioCapture, PulseAudioCapture
from shutup_ads.audio.controller import DEFAULT_TARGET_APPS, PulseController
from shutup_ads.debug.capturer import ScreenshotCapturer
from shutup_ads.detector.engine import BasketballDetector
from shutup_ads.detector.state_machine import BroadcastState, BroadcastStateMachine
from shutup_ads.model.classifier import CalibratedClassifier
from shutup_ads.model.downloader import ensure_yamnet_assets
from shutup_ads.model.yamnet import YAMNetModel
from shutup_ads.ui.dashboard import Dashboard

console = Console()


class RawTerminalInput:
    """Context manager for non-blocking raw single-character terminal input."""

    def __init__(self) -> None:
        self.old_settings: list | None = None

    def __enter__(self) -> "RawTerminalInput":
        if sys.stdin.isatty():
            try:
                self.old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
            except Exception:
                self.old_settings = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except Exception:
                pass

    def get_key(self) -> str | None:
        """Read a single key non-blocking, or return None."""
        if not sys.stdin.isatty():
            return None
        rlist, _, _ = select.select([sys.stdin], [], [], 0)
        if rlist:
            try:
                return sys.stdin.read(1)
            except Exception:
                return None
        return None


def cmd_list_devices(args: argparse.Namespace) -> None:
    """List active PulseAudio / PipeWire streams and audio sinks."""
    controller = PulseController()
    streams = controller.list_all_streams()
    monitor = controller.get_monitor_source()

    console.print(f"\n[bold green]Active Audio Monitor:[/bold green] [cyan]{monitor}[/cyan]\n")

    table = Table(title="Active Application Audio Streams (Sink Inputs)")
    table.add_column("Index", style="dim", width=8)
    table.add_column("Application Name", style="bold")
    table.add_column("Binary", style="dim")
    table.add_column("Media Name")
    table.add_column("Volume", justify="right")
    table.add_column("Muted", justify="center")
    table.add_column("Auto-Targeted?", justify="center")

    for s in streams:
        target_str = "[bold green]YES (Watched)[/bold green]" if s["is_target"] else "[dim]No[/dim]"
        mute_str = "[red]Yes[/red]" if s["mute"] else "No"
        vol_str = f"{int(s['volume'] * 100)}%"
        table.add_row(
            f"#{s['index']}",
            s["app_name"],
            s["binary"],
            s["media_name"] or "-",
            vol_str,
            mute_str,
            target_str,
        )

    console.print(table)
    console.print()


def cmd_run(args: argparse.Namespace) -> None:
    """Run real-time sports audio monitoring and stream attenuation."""
    action_mode = "dry_run" if args.dry_run else args.mode
    target_vol = args.ad_volume / 100.0

    target_apps = [a.strip() for a in args.app.split(",") if a.strip()]

    console.print("[dim]Checking YAMNet model assets...[/dim]")
    model_path, class_map = ensure_yamnet_assets()

    console.print("[dim]Initializing inference engine...[/dim]")
    yamnet = YAMNetModel(model_path, class_map)

    classifier = CalibratedClassifier()
    detector = BasketballDetector(
        classifier=classifier,
        ad_threshold=args.ad_thresh,
        sports_threshold=args.sports_thresh,
    )
    state_machine = BroadcastStateMachine(
        ad_threshold=args.ad_thresh,
        sports_threshold=args.sports_thresh,
        ad_trigger_seconds=args.ad_delay,
        sports_trigger_seconds=args.sports_delay,
    )
    controller = PulseController(
        target_apps=target_apps,
        attenuate_volume=target_vol,
        action_mode=action_mode,
    )

    monitor_source = args.source or controller.get_monitor_source()
    capture = PulseAudioCapture(source=monitor_source, chunk_seconds=0.5)

    dashboard = Dashboard(action_mode=action_mode, target_vol_pct=target_vol)

    debug_enabled = getattr(args, "debug", False) or getattr(args, "debug_screenshots", False)
    debug_dir = getattr(args, "debug_dir", "debug_captures")
    capturer = ScreenshotCapturer(output_dir=debug_dir, enabled=debug_enabled)

    console.print(f"[bold green]Starting shutup-ads on monitor:[/bold green] [cyan]{monitor_source}[/cyan]")
    console.print(f"[bold]Target apps:[/bold] {', '.join(target_apps)} | [bold]Mode:[/bold] {action_mode}")
    if debug_enabled:
        console.print(f"[bold magenta]Debug screenshot mode ENABLED[/bold magenta] (captures -> [cyan]{debug_dir}[/cyan])")

    # Ring buffer for YAMNet (0.975s = 15600 samples)
    ring_buffer = np.zeros(16000, dtype=np.float32)

    headless = getattr(args, "headless", False) or not sys.stdout.isatty()

    try:
        with RawTerminalInput() as term_input:
            if headless:
                console.print("[bold green]Running in headless/background mode...[/bold green]")
                last_log_time = 0.0
                for chunk in capture.start():
                    key = term_input.get_key()
                    if key == "q":
                        break

                    num_new = len(chunk.samples)
                    ring_buffer = np.roll(ring_buffer, -num_new)
                    ring_buffer[-num_new:] = chunk.samples

                    yamnet_out = yamnet.infer(ring_buffer)
                    metrics = detector.evaluate(yamnet_out.scores, yamnet_out.embeddings, chunk.rms)
                    event = state_machine.update(metrics)

                    if event:
                        console.print(f"[bold yellow]State Transition:[/bold yellow] {event.to_state.value} ({event.reason})")
                        capturer.capture_transition(event, metrics)

                    active_streams = controller.apply_commercial_state(state_machine.is_commercial)

                    now = time.time()
                    if now - last_log_time >= 3.0:
                        last_log_time = now
                        stream_summary = ", ".join(f"Stream {s.index}: {s.app_name} ({int(s.current_volume*100)}%)" for s in active_streams)
                        console.print(
                            f"[dim]{time.strftime('%H:%M:%S')}[/dim] "
                            f"State: [bold]{state_machine.current_state.value}[/bold] | "
                            f"Ad Prob: {metrics.ad_probability*100:4.1f}% | "
                            f"Sports: {metrics.sports_score*100:4.1f}% | "
                            f"Streams: [{stream_summary or 'none'}]",
                            highlight=False,
                        )
            else:
                with Live(console=console, refresh_per_second=4, screen=False) as live:
                    for chunk in capture.start():
                        key = term_input.get_key()
                        if key == "q":
                            break
                        elif key == "c":
                            capturer.capture_manual("manual_check")
                            dashboard.log_event("Captured manual screenshot")
                        elif key == "m":
                            if state_machine.is_commercial:
                                ev = state_machine.force_state(BroadcastState.LIVE_SPORTS, "Manual un-attenuate key")
                            else:
                                ev = state_machine.force_state(BroadcastState.COMMERCIAL_BREAK, "Manual attenuate key")
                            dashboard.log_event(f"Manual override: {ev.to_state.value}")
                            controller.apply_commercial_state(state_machine.is_commercial)
                        elif key == "s":
                            ev = state_machine.force_state(BroadcastState.LIVE_SPORTS, "Force sports key")
                            dashboard.log_event("Forced state: LIVE_SPORTS")
                            controller.apply_commercial_state(False)
                        elif key == "a":
                            ev = state_machine.force_state(BroadcastState.COMMERCIAL_BREAK, "Force ad key")
                            dashboard.log_event("Forced state: COMMERCIAL_BREAK")
                            controller.apply_commercial_state(True)

                        num_new = len(chunk.samples)
                        ring_buffer = np.roll(ring_buffer, -num_new)
                        ring_buffer[-num_new:] = chunk.samples

                        yamnet_out = yamnet.infer(ring_buffer)
                        metrics = detector.evaluate(yamnet_out.scores, yamnet_out.embeddings, chunk.rms)

                        event = state_machine.update(metrics)
                        if event:
                            dashboard.log_event(f"State transition: {event.to_state.value} ({event.reason})")
                            capturer.capture_transition(event, metrics)

                        active_streams = controller.apply_commercial_state(state_machine.is_commercial)

                        top_k = yamnet.get_top_k(yamnet_out.scores, k=4)
                        live.update(dashboard.render(metrics, state_machine, active_streams, top_k))

    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        controller.cleanup()
        capturer.shutdown()
        console.print("\n[bold green]shutup-ads stopped cleanly. Audio streams restored.[/bold green]\n")


def cmd_test_file(args: argparse.Namespace) -> None:
    """Analyze a recorded audio or video file offline."""
    file_path = Path(args.file)
    if not file_path.exists():
        console.print(f"[bold red]Error: File not found: {file_path}[/bold red]")
        sys.exit(1)

    console.print(f"[bold green]Analyzing file:[/bold green] {file_path}")
    model_path, class_map = ensure_yamnet_assets()
    yamnet = YAMNetModel(model_path, class_map)
    detector = BasketballDetector(
        ad_threshold=args.ad_thresh,
        sports_threshold=args.sports_thresh,
    )
    state_machine = BroadcastStateMachine(
        ad_threshold=args.ad_thresh,
        sports_threshold=args.sports_thresh,
    )

    capture = FileAudioCapture(file_path=file_path, chunk_seconds=0.5, realtime=args.realtime)
    ring_buffer = np.zeros(16000, dtype=np.float32)

    table = Table(title="Commercial Breaks Detected in File")
    table.add_column("Break #", justify="center")
    table.add_column("Start Time", justify="right")
    table.add_column("End Time", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Trigger Reason")

    break_count = 0
    break_start_time = 0.0

    for chunk in capture.start():
        ring_buffer = np.roll(ring_buffer, -len(chunk.samples))
        ring_buffer[-len(chunk.samples):] = chunk.samples

        yamnet_out = yamnet.infer(ring_buffer)
        metrics = detector.evaluate(yamnet_out.scores, yamnet_out.embeddings, chunk.rms)
        event = state_machine.update(metrics, current_time=chunk.timestamp)

        if event:
            if event.to_state == BroadcastState.COMMERCIAL_BREAK:
                break_start_time = chunk.timestamp
            elif event.from_state == BroadcastState.COMMERCIAL_BREAK and event.to_state == BroadcastState.LIVE_SPORTS:
                break_count += 1
                dur = chunk.timestamp - break_start_time
                table.add_row(
                    str(break_count),
                    f"{int(break_start_time//60):02d}:{int(break_start_time%60):02d}",
                    f"{int(chunk.timestamp//60):02d}:{int(chunk.timestamp%60):02d}",
                    f"{dur:.1f}s",
                    event.reason,
                )

    console.print(table)
    console.print(f"\n[bold]Total Commercial Breaks:[/bold] {break_count}\n")


def cmd_calibrate(args: argparse.Namespace) -> None:
    """Interactive calibration wizard to train a custom broadcast embedding head."""
    console.print("\n[bold green]═══ shutup-ads Broadcast Calibration Wizard ═══[/bold green]\n")
    console.print(
        "This tool records audio embeddings from your specific broadcast stream to train a\n"
        "custom machine-learning classifier tailored to this game's commentators and commercials.\n"
    )

    controller = PulseController()
    monitor = controller.get_monitor_source()
    console.print(f"Monitoring source: [cyan]{monitor}[/cyan]\n")

    model_path, class_map = ensure_yamnet_assets()
    yamnet = YAMNetModel(model_path, class_map)

    def record_samples(label_name: str, duration_sec: int) -> list[np.ndarray]:
        console.input(f"▶ Press [bold green]ENTER[/bold green] to begin recording [bold]{duration_sec}s[/bold] of [bold cyan]{label_name}[/bold cyan]...")
        console.print(f"Recording {label_name} audio...")
        capture = PulseAudioCapture(source=monitor, chunk_seconds=0.5)
        chunks_needed = int(duration_sec / 0.5)
        embeddings_list = []
        ring_buffer = np.zeros(16000, dtype=np.float32)

        count = 0
        for chunk in capture.start():
            ring_buffer = np.roll(ring_buffer, -len(chunk.samples))
            ring_buffer[-len(chunk.samples):] = chunk.samples
            out = yamnet.infer(ring_buffer)
            if len(out.embeddings) > 0:
                embeddings_list.append(np.mean(out.embeddings, axis=0))
            count += 1
            if count >= chunks_needed:
                break
        capture.stop()
        console.print(f"✓ Collected {len(embeddings_list)} embedding frames for {label_name}.\n")
        return embeddings_list

    duration = args.duration
    sports_embeddings = record_samples("LIVE BASKETBALL", duration)
    ad_embeddings = record_samples("COMMERCIAL BREAK", duration)

    console.print("Training custom classifier head...")
    X = np.vstack([sports_embeddings, ad_embeddings])
    y = np.array([0] * len(sports_embeddings) + [1] * len(ad_embeddings))

    classifier = CalibratedClassifier()
    classifier.fit(X, y, save=True)

    console.print(f"[bold green]✓ Successfully trained and saved custom model to:[/bold green] {classifier.model_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="shutup-ads",
        description="Real-time basketball broadcast commercial detector and stream attenuator.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # run command
    run_parser = subparsers.add_parser("run", help="Run real-time commercial detector and stream attenuator.")
    run_parser.add_argument(
        "--app",
        type=str,
        default=",".join(DEFAULT_TARGET_APPS),
        help=f"Comma-separated target application names (default: {','.join(DEFAULT_TARGET_APPS)})",
    )
    run_parser.add_argument(
        "--mode",
        choices=["attenuate", "mute", "dry_run"],
        default="attenuate",
        help="Action on ad detection: attenuate (reduce vol), mute (0 vol), or dry_run (monitor only)",
    )
    run_parser.add_argument(
        "--ad-volume",
        type=float,
        default=15.0,
        help="Target volume percentage when attenuated (default: 15.0)",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Shortcut for --mode dry_run",
    )
    run_parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without interactive Rich TUI (clean line logging for daemons / background)",
    )
    run_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (captures screenshots on every state transition)",
    )
    run_parser.add_argument(
        "--debug-screenshots",
        action="store_true",
        help="Capture desktop screenshot whenever state transitions between sports and commercial",
    )
    run_parser.add_argument(
        "--debug-dir",
        type=str,
        default="debug_captures",
        help="Directory to save debug transition screenshots and metadata (default: debug_captures)",
    )
    run_parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="PulseAudio monitor source (defaults to auto-detected default sink monitor)",
    )
    run_parser.add_argument(
        "--ad-thresh",
        type=float,
        default=0.50,
        help="Ad probability threshold to trigger commercial state (default: 0.50)",
    )
    run_parser.add_argument(
        "--sports-thresh",
        type=float,
        default=0.25,
        help="Sports action score threshold to return to live sports (default: 0.25)",
    )
    run_parser.add_argument(
        "--ad-delay",
        type=float,
        default=2.0,
        help="Persistence seconds required before entering commercial state (default: 2.0)",
    )
    run_parser.add_argument(
        "--sports-delay",
        type=float,
        default=1.0,
        help="Persistence seconds required before returning to sports state (default: 1.0)",
    )

    # list-devices command
    subparsers.add_parser("list-devices", help="List audio sinks, monitors, and active stream inputs.")

    # test command
    test_parser = subparsers.add_parser("test", help="Evaluate commercial detection on a recorded media file.")
    test_parser.add_argument("--file", type=str, required=True, help="Path to audio/video file")
    test_parser.add_argument("--realtime", action="store_true", help="Simulate real-time streaming pace")
    test_parser.add_argument("--ad-thresh", type=float, default=0.50)
    test_parser.add_argument("--sports-thresh", type=float, default=0.30)

    # calibrate command
    cal_parser = subparsers.add_parser("calibrate", help="Record live stream samples to train a custom embedding head.")
    cal_parser.add_argument("--duration", type=int, default=20, help="Duration in seconds for each phase (default: 20)")

    args = parser.parse_args()

    if args.subcommand == "run":
        cmd_run(args)
    elif args.subcommand == "list-devices":
        cmd_list_devices(args)
    elif args.subcommand == "test":
        cmd_test_file(args)
    elif args.subcommand == "calibrate":
        cmd_calibrate(args)


if __name__ == "__main__":
    main()
