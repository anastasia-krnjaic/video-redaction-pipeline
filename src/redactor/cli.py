from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

from redactor import media


@click.group()
def main() -> None:
    """Inspect videos and create local development artifacts."""


@main.command()
def doctor() -> None:
    """Check Python, FFmpeg, and FFprobe availability."""
    click.echo(f"Python: {sys.version.split()[0]}")
    for tool in ("ffmpeg", "ffprobe"):
        try:
            click.echo(f"{tool}: {media.version_line(tool)}")
        except FileNotFoundError as exc:
            raise click.ClickException(str(exc)) from exc
        except subprocess.CalledProcessError as exc:
            raise click.ClickException(_subprocess_message(exc)) from exc


@main.command("inspect")
@click.argument(
    "input_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="JSON output path.",
)
def inspect_command(input_path: Path, output_path: Path) -> None:
    """Inspect a video file and write deterministic JSON metadata."""
    try:
        info = media.probe_media(input_path)
        media.write_media_info(info, output_path)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(_subprocess_message(exc)) from exc

    click.echo(f"Wrote metadata: {output_path}")


@main.command()
@click.argument(
    "input_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.argument("output_path", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--start", "start_seconds", required=True, type=float, help="Start time in seconds.")
@click.option(
    "--duration",
    "duration_seconds",
    required=True,
    type=float,
    help="Duration in seconds.",
)
@click.option("--overwrite", is_flag=True, help="Replace the output file if it already exists.")
def trim(
    input_path: Path,
    output_path: Path,
    start_seconds: float,
    duration_seconds: float,
    overwrite: bool,
) -> None:
    """Create a short MP4 development clip."""
    try:
        actual_duration = media.trim_video(
            input_path,
            output_path,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            overwrite=overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(_subprocess_message(exc)) from exc

    duration_text = "unknown" if actual_duration is None else f"{actual_duration:.3f}s"
    click.echo(
        "Trimmed "
        f"{input_path} -> {output_path} "
        f"(requested start={start_seconds:g}s duration={duration_seconds:g}s; "
        f"actual duration={duration_text})"
    )


@main.command("contact-sheet")
@click.argument(
    "input_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.argument("output_path", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--interval-seconds",
    required=True,
    type=float,
    help="Sample interval in seconds.",
)
@click.option("--columns", required=True, type=int, help="Number of sheet columns.")
@click.option("--max-frames", required=True, type=int, help="Maximum sampled frames.")
@click.option("--overwrite", is_flag=True, help="Replace the output file if it already exists.")
def contact_sheet(
    input_path: Path,
    output_path: Path,
    interval_seconds: float,
    columns: int,
    max_frames: int,
    overwrite: bool,
) -> None:
    """Create a JPEG contact sheet."""
    try:
        media.create_contact_sheet(
            input_path,
            output_path,
            interval_seconds=interval_seconds,
            columns=columns,
            max_frames=max_frames,
            overwrite=overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(_subprocess_message(exc)) from exc

    click.echo(
        f"Wrote contact sheet: {output_path} "
        f"(interval={interval_seconds:g}s columns={columns} max_frames={max_frames})"
    )


@main.command()
@click.argument(
    "input_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--category",
    required=True,
    type=click.Choice(["face", "license_plate"]),
    help="Detection category.",
)
@click.option(
    "--model",
    "model_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="ONNX model path.",
)
@click.option(
    "--threshold",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Detection score threshold.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for detections.jsonl, summary.json, and overlay.mp4.",
)
@click.option("--overwrite", is_flag=True, help="Replace outputs in a non-empty directory.")
@click.option(
    "--max-frames",
    type=click.IntRange(min=1),
    default=None,
    help="Maximum number of frames to process.",
)
def detect(
    input_path: Path,
    category: str,
    model_path: Path,
    threshold: float | None,
    output_dir: Path,
    overwrite: bool,
    max_frames: int | None,
) -> None:
    """Run frame-level diagnostic detection."""
    from redactor import detection

    resolved_threshold = threshold
    if resolved_threshold is None:
        resolved_threshold = detection.DEFAULT_THRESHOLDS[category]

    try:
        summary = detection.run_detection(
            input_path,
            category=category,
            model_path=model_path,
            threshold=resolved_threshold,
            output_dir=output_dir,
            overwrite=overwrite,
            max_frames=max_frames,
        )
    except (FileExistsError, FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(
        f"Processed {summary['frames_processed']} frames; "
        f"{summary['frames_containing_at_least_one_detection']} frames with detections; "
        f"{summary['total_detections']} total detections."
    )
    click.echo(f"Wrote detections: {output_dir / 'detections.jsonl'}")
    click.echo(f"Wrote summary: {output_dir / 'summary.json'}")
    click.echo(f"Wrote diagnostic overlay without audio: {output_dir / 'overlay.mp4'}")


@main.command()
@click.argument(
    "input_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--detections",
    "detections_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Face detections JSONL path.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for tracking diagnostics.",
)
@click.option("--strong-threshold", default=0.70, show_default=True, type=click.FloatRange(0, 1))
@click.option("--weak-threshold", default=0.45, show_default=True, type=click.FloatRange(0, 1))
@click.option("--min-observed-hits", default=3, show_default=True, type=click.IntRange(min=1))
@click.option("--min-strong-hits", default=2, show_default=True, type=click.IntRange(min=0))
@click.option("--confirmation-window", default=5, show_default=True, type=click.IntRange(min=1))
@click.option("--max-gap-seconds", default=0.30, show_default=True, type=float)
@click.option("--backward-seconds", default=1.00, show_default=True, type=float)
@click.option(
    "--min-auto-median-score", default=0.70, show_default=True, type=click.FloatRange(0, 1)
)
@click.option(
    "--max-auto-synthetic-fraction",
    default=0.40,
    show_default=True,
    type=click.FloatRange(0, 1),
)
@click.option("--min-auto-area-fraction", default=0.0005, show_default=True, type=float)
@click.option("--max-auto-area-fraction", default=0.08, show_default=True, type=float)
@click.option("--max-tail-prediction-seconds", default=0.10, show_default=True, type=float)
@click.option(
    "--association-iou-threshold", default=0.15, show_default=True, type=click.FloatRange(0, 1)
)
@click.option("--association-center-distance", default=0.75, show_default=True, type=float)
@click.option(
    "--manual-overrides",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional manual seed JSON file.",
)
@click.option("--overwrite", is_flag=True, help="Replace outputs in a non-empty directory.")
@click.option(
    "--show-rejected-overlay", is_flag=True, help="Include rejected proposals in overlay.mp4."
)
def track(
    input_path: Path,
    detections_path: Path,
    output_dir: Path,
    strong_threshold: float,
    weak_threshold: float,
    min_observed_hits: int,
    min_strong_hits: int,
    confirmation_window: int,
    max_gap_seconds: float,
    backward_seconds: float,
    min_auto_median_score: float,
    max_auto_synthetic_fraction: float,
    min_auto_area_fraction: float,
    max_auto_area_fraction: float,
    max_tail_prediction_seconds: float,
    association_iou_threshold: float,
    association_center_distance: float,
    manual_overrides: Path | None,
    overwrite: bool,
    show_rejected_overlay: bool,
) -> None:
    """Run offline face temporal-safety tracking diagnostics."""
    from redactor import tracking

    try:
        summary = tracking.run_tracking(
            input_path,
            detections_path=detections_path,
            output_dir=output_dir,
            strong_threshold=strong_threshold,
            weak_threshold=weak_threshold,
            min_observed_hits=min_observed_hits,
            min_strong_hits=min_strong_hits,
            confirmation_window=confirmation_window,
            max_gap_seconds=max_gap_seconds,
            backward_seconds=backward_seconds,
            min_auto_median_score=min_auto_median_score,
            max_auto_synthetic_fraction=max_auto_synthetic_fraction,
            min_auto_area_fraction=min_auto_area_fraction,
            max_auto_area_fraction=max_auto_area_fraction,
            max_tail_prediction_seconds=max_tail_prediction_seconds,
            association_iou_threshold=association_iou_threshold,
            association_center_distance=association_center_distance,
            manual_overrides=manual_overrides,
            overwrite=overwrite,
            show_rejected_overlay=show_rejected_overlay,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        ValueError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(
        f"Processed {summary['frames_processed']} frames; "
        f"{summary['auto_confirmed_tracks']} auto-confirmed tracks; "
        f"{summary['review_candidate_tracks']} review candidates; "
        f"{summary['rejected_tracks']} rejected proposals."
    )
    click.echo(f"Wrote tracks: {output_dir / 'tracks.jsonl'}")
    click.echo(f"Wrote review flags: {output_dir / 'review_flags.json'}")
    click.echo(f"Wrote review windows: {output_dir / 'review_windows.json'}")
    click.echo(f"Wrote track quality: {output_dir / 'track_quality.json'}")
    click.echo(f"Wrote summary: {output_dir / 'summary.json'}")
    click.echo(f"Wrote diagnostic overlay without audio: {output_dir / 'overlay.mp4'}")


def _subprocess_message(exc: subprocess.CalledProcessError) -> str:
    stderr = exc.stderr.strip() if exc.stderr else ""
    return stderr or str(exc)


if __name__ == "__main__":
    main()
