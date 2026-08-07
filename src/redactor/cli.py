from __future__ import annotations

import json
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
    required=False,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help=(
        "Model path. For face detection this is the ADAS XML path; for plates this is "
        "the OIM YOLOv9-S-608 ONNX path."
    ),
)
@click.option(
    "--detector-backend",
    type=click.Choice(
        ["openvino_face_adas_0001", "oim_yolov9_s_608"],
        case_sensitive=True,
    ),
    default=None,
    help=(
        "Advanced detector backend. Faces use openvino_face_adas_0001; plates use oim_yolov9_s_608."
    ),
)
@click.option(
    "--adas-xml",
    "adas_xml_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="OpenVINO face-detection-adas-0001 XML path. REDACTOR_ADAS_XML is also honored.",
)
@click.option(
    "--adas-bin",
    "adas_bin_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="OpenVINO face-detection-adas-0001 BIN path. REDACTOR_ADAS_BIN is also honored.",
)
@click.option(
    "--threshold",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Legacy detection score threshold. For ADAS, prefer --proposal-threshold.",
)
@click.option(
    "--proposal-threshold",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Proposal storage threshold. ADAS default: 0.20; OIM plate default: 0.30.",
)
@click.option(
    "--primary-threshold",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Primary observation threshold. ADAS default: 0.30; OIM plate default: 0.30.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for detections.jsonl, summary.json, and overlay.mp4.",
)
@click.option(
    "--detections-output",
    "detections_output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional detections JSONL output path. Defaults to OUTPUT_DIR/detections.jsonl.",
)
@click.option(
    "--clip-id",
    type=str,
    default=None,
    help="Optional canonical clip id to embed in detection records.",
)
@click.option("--no-overlay", is_flag=True, help="Skip diagnostic overlay.mp4 generation.")
@click.option("--no-summary", is_flag=True, help="Skip writing OUTPUT_DIR/summary.json.")
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
    model_path: Path | None,
    detector_backend: str | None,
    adas_xml_path: Path | None,
    adas_bin_path: Path | None,
    threshold: float | None,
    proposal_threshold: float | None,
    primary_threshold: float | None,
    output_dir: Path,
    detections_output_path: Path | None,
    clip_id: str | None,
    no_overlay: bool,
    no_summary: bool,
    overwrite: bool,
    max_frames: int | None,
) -> None:
    """Run frame-level diagnostic detection."""
    from redactor import detection

    resolved_backend = detection.resolve_detector_backend(category, detector_backend)
    resolved_model_path = resolve_detection_model_path(
        category=category,
        detector_backend=resolved_backend,
        model_path=model_path,
        adas_xml_path=adas_xml_path,
    )
    resolved_adas_bin_path = (
        adas_bin_path
        if resolved_backend == detection.DETECTOR_BACKEND_ADAS and adas_bin_path is not None
        else (
            detection.default_adas_bin_path(resolved_model_path)
            if resolved_backend == detection.DETECTOR_BACKEND_ADAS
            else None
        )
    )
    resolved_threshold = proposal_threshold if proposal_threshold is not None else threshold
    if resolved_threshold is None:
        resolved_threshold = detection.default_proposal_threshold(category, resolved_backend)

    try:
        summary = detection.run_detection(
            input_path,
            category=category,
            model_path=resolved_model_path,
            threshold=resolved_threshold,
            output_dir=output_dir,
            overwrite=overwrite,
            max_frames=max_frames,
            detector_backend=resolved_backend,
            adas_bin_path=resolved_adas_bin_path,
            primary_threshold=primary_threshold,
            clip_id=clip_id,
            detections_output_path=detections_output_path,
            write_overlay_video=not no_overlay,
            write_summary_file=not no_summary,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(
        f"Processed {summary['frames_processed']} frames; "
        f"{summary['frames_containing_at_least_one_detection']} frames with detections; "
        f"{summary['total_detections']} total detections."
    )
    click.echo(f"Detector inference FPS: {summary['detector_inference_fps']}")
    click.echo(f"Detection-pass FPS: {summary['processing_fps']}")
    click.echo(f"Wrote detections: {detections_output_path or output_dir / 'detections.jsonl'}")
    if not no_summary:
        click.echo(f"Wrote summary: {output_dir / 'summary.json'}")
    if not no_overlay:
        click.echo(f"Wrote diagnostic overlay without audio: {output_dir / 'overlay.mp4'}")


@main.command("detect-plates")
@click.argument(
    "input_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--model",
    "model_path",
    required=False,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help=("Open Image Models YOLOv9-S-608 ONNX path. REDACTOR_OIM_PLATE_MODEL is also honored."),
)
@click.option(
    "--threshold",
    type=click.FloatRange(0.0, 1.0),
    default=0.30,
    show_default=True,
    help="Global OIM primary/proposal score threshold.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for detections.jsonl, summary.json, and overlay.mp4.",
)
@click.option(
    "--detections-output",
    "detections_output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional detections JSONL output path. Defaults to OUTPUT_DIR/detections.jsonl.",
)
@click.option(
    "--clip-id",
    type=str,
    default=None,
    help="Optional canonical clip id to embed in detection records.",
)
@click.option("--no-overlay", is_flag=True, help="Skip diagnostic overlay.mp4 generation.")
@click.option("--no-summary", is_flag=True, help="Skip writing OUTPUT_DIR/summary.json.")
@click.option("--overwrite", is_flag=True, help="Replace outputs in a non-empty directory.")
@click.option(
    "--max-frames",
    type=click.IntRange(min=1),
    default=None,
    help="Maximum number of frames to process for a diagnostic run.",
)
def detect_plates(
    input_path: Path,
    model_path: Path | None,
    threshold: float,
    output_dir: Path,
    detections_output_path: Path | None,
    clip_id: str | None,
    no_overlay: bool,
    no_summary: bool,
    overwrite: bool,
    max_frames: int | None,
) -> None:
    """Run the supported OIM YOLOv9-S-608 license-plate detector."""
    from redactor import detection, plate_detection

    resolved_model_path = model_path or plate_detection.default_model_path()
    try:
        summary = detection.run_detection(
            input_path,
            category="license_plate",
            model_path=resolved_model_path,
            threshold=threshold,
            output_dir=output_dir,
            overwrite=overwrite,
            max_frames=max_frames,
            detector_backend=detection.DETECTOR_BACKEND_OIM_YOLOV9_S_608,
            primary_threshold=threshold,
            clip_id=clip_id,
            detections_output_path=detections_output_path,
            write_overlay_video=not no_overlay,
            write_summary_file=not no_summary,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("Detector: Open Image Models YOLOv9-S-608")
    click.echo(f"Inference mode: {plate_detection.INFERENCE_MODE}")
    click.echo(f"Configuration hash: {summary['configuration_hash']}")
    click.echo(
        f"Processed {summary['frames_processed']} frames; "
        f"{summary['frames_containing_at_least_one_detection']} frames with detections; "
        f"{summary['total_detections']} total detections."
    )
    click.echo(f"Detector inference FPS: {summary['detector_inference_fps']}")
    click.echo(f"Detection-pass FPS: {summary['processing_fps']}")
    click.echo(f"Wrote detections: {detections_output_path or output_dir / 'detections.jsonl'}")
    if not no_summary:
        click.echo(f"Wrote summary: {output_dir / 'summary.json'}")
    if not no_overlay:
        click.echo(f"Wrote diagnostic overlay without audio: {output_dir / 'overlay.mp4'}")


@main.command("track-plates")
@click.option(
    "--detections",
    "detections_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="OIM YOLOv9-S-608 license-plate detections JSONL path.",
)
@click.option(
    "--source-video",
    "source_video_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Source video used to create the detection JSONL.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help=(
        "Directory for tracks.jsonl, track_summary.json, gap_decisions.json, "
        "review_intervals.json, tracking_manifest.json, frame_span_report.json, and diagnostics."
    ),
)
@click.option("--no-overlay", is_flag=True, help="Skip diagnostic tracking_overlay.mp4 generation.")
@click.option(
    "--show-rejected-overlay",
    is_flag=True,
    help="Also draw rejected proposals on the diagnostic overlay.",
)
@click.option("--overwrite", is_flag=True, help="Replace outputs in a non-empty directory.")
def track_plates(
    detections_path: Path,
    source_video_path: Path,
    output_dir: Path,
    no_overlay: bool,
    show_rejected_overlay: bool,
    overwrite: bool,
) -> None:
    """Track OIM plates with shared accumulated evidence and bounded gap recovery."""
    from redactor import plate_tracking

    try:
        summary = plate_tracking.run_plate_tracking(
            source_video_path,
            detections_path=detections_path,
            output_dir=output_dir,
            overwrite=overwrite,
            write_overlay_video=not no_overlay,
            show_rejected_overlay=show_rejected_overlay,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        raise click.ClickException(
            _subprocess_message(exc) if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        ) from exc

    tracking_summary = summary["tracking_summary"]
    status_counts = tracking_summary["status_counts"]
    click.echo("Plate tracker: OIM YOLOv9-S-608 shared temporal evidence v1")
    click.echo(f"Detector configuration hash: {summary['detector_configuration_hash']}")
    click.echo(f"Tracking configuration hash: {summary['tracking_configuration_hash']}")
    click.echo(f"Evidence configuration hash: {tracking_summary['evidence_configuration_hash']}")
    click.echo(
        f"Processed {summary['frames_processed']} frames; "
        f"{status_counts['accepted_plate_track']} accepted tracks; "
        f"{status_counts['needs_review']} review tracks; "
        f"{status_counts['rejected_noise_track']} rejected tracks."
    )
    click.echo(f"Full source processed: {summary['full_source_processed']}")
    click.echo(f"Missing processed ranges: {len(summary['missing_processed_ranges'])}")
    click.echo(f"Review intervals: {tracking_summary['review_interval_count']}")
    click.echo(f"Processing FPS: {summary['processing_fps']}")
    click.echo(f"Wrote tracks: {output_dir / 'tracks.jsonl'}")
    click.echo(f"Wrote track summary: {output_dir / 'track_summary.json'}")
    click.echo(f"Wrote gap decisions: {output_dir / 'gap_decisions.json'}")
    click.echo(f"Wrote review intervals: {output_dir / 'review_intervals.json'}")
    click.echo(f"Wrote manifest: {output_dir / 'tracking_manifest.json'}")
    click.echo(f"Wrote frame-span report: {output_dir / 'frame_span_report.json'}")
    if not no_overlay:
        click.echo(f"Wrote diagnostic overlay without audio: {output_dir / 'tracking_overlay.mp4'}")


@main.command("render-plates")
@click.option(
    "--source-video",
    "source_video_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Source video used by the selected OIM plate detector and plate tracker.",
)
@click.option(
    "--tracks",
    "tracks_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Plate tracks JSONL from track-plates.",
)
@click.option(
    "--review-intervals",
    "review_intervals_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Plate review_intervals.json from track-plates.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for automatic_preview.mp4, render manifest, validation, and log files.",
)
@click.option(
    "--mode",
    type=click.Choice(["preview", "reviewed_final"]),
    default="preview",
    show_default=True,
    help="Preview renders accepted automatic plate regions; reviewed_final refuses blockers.",
)
@click.option(
    "--output-video",
    "output_video_path",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Optional output MP4 path. Defaults to automatic_preview.mp4 in output-dir.",
)
@click.option(
    "--allow-partial-preview",
    is_flag=True,
    help=(
        "Allow an explicitly watermarked non-final preview when plate tracks do not cover the "
        "full source video. Reviewed-final renders remain blocked."
    ),
)
@click.option("--overwrite", is_flag=True, help="Replace outputs in a non-empty directory.")
def render_plates(
    source_video_path: Path,
    tracks_path: Path,
    review_intervals_path: Path,
    output_dir: Path,
    mode: str,
    output_video_path: Path | None,
    allow_partial_preview: bool,
    overwrite: bool,
) -> None:
    """Render non-final pixelated plate previews from accepted OIM tracks only."""
    from redactor import plate_rendering

    try:
        summary = plate_rendering.run_plate_render(
            source_video_path,
            tracks_path=tracks_path,
            review_intervals_path=review_intervals_path,
            output_dir=output_dir,
            mode=mode,
            output_video_path=output_video_path,
            allow_partial_preview=allow_partial_preview,
            overwrite=overwrite,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        raise click.ClickException(
            _subprocess_message(exc) if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        ) from exc

    click.echo("Plate renderer: accepted OIM plate tracks only")
    click.echo(f"Render configuration hash: {summary['render_configuration_hash']}")
    click.echo(f"Mode: {summary['mode']}")
    click.echo(f"Full source processed: {summary['full_source_processed']}")
    click.echo(f"Partial processing: {summary['partial_processing']}")
    click.echo(f"Rendered frames: {summary['rendered_frame_count']}")
    click.echo(f"Rendered regions: {summary['rendered_region_count']}")
    click.echo(f"Final approval: {summary['final_approval_status']}")
    click.echo(f"Media validation passed: {summary['media_validation_passed']}")
    click.echo(f"Rendered-region validation passed: {summary['rendered_region_validation_passed']}")
    click.echo(f"Wrote manifest: {output_dir / 'render_manifest.json'}")
    click.echo(f"Wrote validation: {output_dir / 'validation.json'}")


@main.command("build-plate-review-bundle")
@click.option(
    "--source-video",
    "source_video_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Source video for original review clips.",
)
@click.option(
    "--tracks",
    "tracks_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Plate tracks JSONL from track-plates.",
)
@click.option(
    "--review-intervals",
    "review_intervals_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Plate review_intervals.json from track-plates.",
)
@click.option(
    "--automatic-preview",
    "automatic_preview_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Automatic plate preview MP4 generated by render-plates.",
)
@click.option(
    "--diagnostic-overlay",
    "diagnostic_overlay_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Optional diagnostic overlay MP4 from track-plates.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Review bundle directory.",
)
@click.option("--spot-check-count", default=3, show_default=True, type=click.IntRange(0, 20))
@click.option("--context-frames", default=15, show_default=True, type=click.IntRange(0, 300))
@click.option("--overwrite", is_flag=True, help="Replace outputs in a non-empty directory.")
def build_plate_review_bundle(
    source_video_path: Path,
    tracks_path: Path,
    review_intervals_path: Path,
    automatic_preview_path: Path,
    diagnostic_overlay_path: Path | None,
    output_dir: Path,
    spot_check_count: int,
    context_frames: int,
    overwrite: bool,
) -> None:
    """Build a local review bundle for plate tracking and preview blockers."""
    from redactor import plate_rendering

    try:
        summary = plate_rendering.build_plate_review_bundle(
            source_video_path,
            tracks_path=tracks_path,
            review_intervals_path=review_intervals_path,
            automatic_preview_path=automatic_preview_path,
            diagnostic_overlay_path=diagnostic_overlay_path,
            output_dir=output_dir,
            spot_check_count=spot_check_count,
            context_frames=context_frames,
            overwrite=overwrite,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        raise click.ClickException(
            _subprocess_message(exc) if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        ) from exc

    click.echo("Built plate review bundle")
    click.echo(f"Required review items: {summary['required_review_items']}")
    click.echo(f"Optional spot checks: {summary['optional_spot_checks']}")
    click.echo(f"Final export blocked: {summary['final_export_blocked']}")
    click.echo(f"Open: {output_dir / 'index.html'}")


@main.command("validate-plate-review-decisions")
@click.option(
    "--review-tasks",
    "tasks_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="review_tasks.json from build-plate-review-bundle.",
)
@click.option(
    "--decisions",
    "decisions_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Exported plate analyst decisions JSON.",
)
def validate_plate_review_decisions(tasks_path: Path, decisions_path: Path) -> None:
    """Validate exported plate review decisions without altering privacy regions."""
    from redactor import plate_rendering

    try:
        result = plate_rendering.validate_plate_review_decisions(tasks_path, decisions_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc
    if not result["passed"]:
        raise click.ClickException("; ".join(result["errors"]))
    click.echo("Plate review decisions are valid.")
    click.echo(f"Unresolved blocking items: {result['unresolved_blocking_count']}")
    click.echo(f"Correction files required: {result['correction_files_required']}")


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
@click.option(
    "--primary-threshold",
    "--strong-threshold",
    "strong_threshold",
    default=None,
    type=click.FloatRange(0, 1),
    help="Primary/strong observation threshold. ADAS default: 0.30.",
)
@click.option(
    "--proposal-threshold",
    "--weak-threshold",
    "weak_threshold",
    default=None,
    type=click.FloatRange(0, 1),
    help="Proposal/weak observation threshold. ADAS default: 0.20.",
)
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

    resolved_strong_threshold = (
        tracking.DEFAULT_STRONG_THRESHOLD if strong_threshold is None else strong_threshold
    )
    resolved_weak_threshold = (
        tracking.DEFAULT_WEAK_THRESHOLD if weak_threshold is None else weak_threshold
    )
    try:
        summary = tracking.run_tracking(
            input_path,
            detections_path=detections_path,
            output_dir=output_dir,
            strong_threshold=resolved_strong_threshold,
            weak_threshold=resolved_weak_threshold,
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


@main.command("recover-face-gaps")
@click.argument(
    "input_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--tracks",
    "tracks_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Frame-indexed face tracks JSONL path.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for privacy gap-recovery outputs.",
)
@click.option(
    "--target-track-id",
    "target_track_ids",
    multiple=True,
    required=True,
    type=int,
    help="Track id belonging to the privacy target. Repeat for fragmented identities.",
)
@click.option(
    "--keep-track-id",
    "keep_track_ids",
    multiple=True,
    type=int,
    help="Track id for a keep identity that generated privacy regions must avoid.",
)
@click.option(
    "--competing-track-id",
    "competing_track_ids",
    multiple=True,
    type=int,
    help="Track id for another identity used as a proximity/switch safety check.",
)
@click.option("--frame-start", type=int, default=None, help="Inclusive recovery span start.")
@click.option("--frame-end", type=int, default=None, help="Inclusive recovery span end.")
@click.option(
    "--preserved-regions",
    "preserved_regions_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional accepted privacy-region JSONL to preserve where it already covers a gap.",
)
@click.option("--short-interpolation-gap-frames", default=9, show_default=True, type=int)
@click.option("--max-bridge-gap-frames", default=180, show_default=True, type=int)
@click.option("--max-one-sided-gap-frames", default=45, show_default=True, type=int)
@click.option("--max-raw-disagreement-norm", default=1.50, show_default=True, type=float)
@click.option("--max-refined-disagreement-norm", default=0.75, show_default=True, type=float)
@click.option("--max-center-jump-norm", default=0.85, show_default=True, type=float)
@click.option("--max-scale-change", default=0.65, show_default=True, type=float)
@click.option("--max-keep-overlap-fraction", default=0.01, show_default=True, type=float)
@click.option("--competing-identity-distance-norm", default=0.35, show_default=True, type=float)
@click.option("--scene-cut-mean-delta", default=45.0, show_default=True, type=float)
@click.option(
    "--visual-refinement",
    type=click.Choice(["none", "bottom_component"], case_sensitive=True),
    default="none",
    show_default=True,
    help="Optional visual refinement strategy. The bottom-component strategy is opt-in.",
)
@click.option(
    "--conservative-fallback",
    "conservative_fallback_enabled",
    is_flag=True,
    help="Emit conservative fallback review records instead of hard rejection where applicable.",
)
@click.option(
    "--no-preserve-existing-regions",
    "preserve_existing_regions",
    flag_value=False,
    default=True,
    help="Regenerate gaps even when preserved regions cover them.",
)
@click.option("--no-overlay", is_flag=True, help="Skip diagnostic overlay.mp4 generation.")
@click.option("--overwrite", is_flag=True, help="Replace outputs in a non-empty directory.")
def recover_face_gaps(
    input_path: Path,
    tracks_path: Path,
    output_dir: Path,
    target_track_ids: tuple[int, ...],
    keep_track_ids: tuple[int, ...],
    competing_track_ids: tuple[int, ...],
    frame_start: int | None,
    frame_end: int | None,
    preserved_regions_path: Path | None,
    short_interpolation_gap_frames: int,
    max_bridge_gap_frames: int,
    max_one_sided_gap_frames: int,
    max_raw_disagreement_norm: float,
    max_refined_disagreement_norm: float,
    max_center_jump_norm: float,
    max_scale_change: float,
    max_keep_overlap_fraction: float,
    competing_identity_distance_norm: float,
    scene_cut_mean_delta: float,
    visual_refinement: str,
    conservative_fallback_enabled: bool,
    preserve_existing_regions: bool,
    no_overlay: bool,
    overwrite: bool,
) -> None:
    """Recover privacy regions across face-track gaps after ordinary tracking."""
    from redactor import privacy_gap_recovery

    settings = privacy_gap_recovery.PrivacyGapRecoverySettings(
        short_interpolation_gap_frames=short_interpolation_gap_frames,
        max_bridge_gap_frames=max_bridge_gap_frames,
        max_one_sided_gap_frames=max_one_sided_gap_frames,
        max_raw_disagreement_norm=max_raw_disagreement_norm,
        max_refined_disagreement_norm=max_refined_disagreement_norm,
        max_center_jump_norm=max_center_jump_norm,
        max_scale_change=max_scale_change,
        max_keep_overlap_fraction=max_keep_overlap_fraction,
        competing_identity_distance_norm=competing_identity_distance_norm,
        scene_cut_mean_delta=scene_cut_mean_delta,
        enable_visual_refinement=visual_refinement != "none",
        visual_refinement=visual_refinement,
        conservative_fallback_enabled=conservative_fallback_enabled,
        render_overlay=not no_overlay,
        preserve_existing_regions=preserve_existing_regions,
    )
    try:
        summary = privacy_gap_recovery.run_privacy_gap_recovery(
            input_path,
            tracks_path=tracks_path,
            output_dir=output_dir,
            target_track_ids=target_track_ids,
            keep_track_ids=keep_track_ids,
            competing_track_ids=competing_track_ids,
            frame_start=frame_start,
            frame_end=frame_end,
            preserved_regions_path=preserved_regions_path,
            settings=settings,
            overwrite=overwrite,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc

    metrics = summary["metrics"]
    click.echo(
        f"Considered {metrics['gaps_considered']} gaps; "
        f"{metrics['accepted_intervals']} accepted; "
        f"{metrics['needs_review_intervals']} need review; "
        f"{metrics['rejected_intervals']} rejected."
    )
    click.echo(
        f"Recovered {metrics['recovered_frames']} frames; "
        f"{metrics['unresolved_frames']} unresolved."
    )
    click.echo(f"Wrote privacy regions: {output_dir / 'privacy_regions.jsonl'}")
    click.echo(f"Wrote interval decisions: {output_dir / 'interval_decisions.json'}")
    click.echo(f"Wrote metrics: {output_dir / 'metrics.json'}")
    if not no_overlay:
        click.echo(f"Wrote diagnostic overlay without audio: {output_dir / 'overlay.mp4'}")


@main.command("render-faces")
@click.argument(
    "input_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--privacy-regions",
    "privacy_regions_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Privacy regions JSONL from face gap recovery or preserved reviewed corrections.",
)
@click.option(
    "--interval-decisions",
    "interval_decisions_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional interval decision JSON for final-approval checks.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for redacted video and render manifests.",
)
@click.option(
    "--output-video",
    "output_video_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional MP4 output path. Defaults to OUTPUT_DIR/redacted_MODE.mp4.",
)
@click.option(
    "--mode",
    type=click.Choice(["preview", "final"], case_sensitive=True),
    default="preview",
    show_default=True,
    help="Preview is visibly non-final; final refuses unsafe or unresolved intervals.",
)
@click.option(
    "--method",
    type=click.Choice(["gaussian_blur", "pixelate"], case_sensitive=True),
    default="gaussian_blur",
    show_default=True,
    help="Face redaction method.",
)
@click.option("--margin-pixels", default=0, show_default=True, type=click.IntRange(min=0))
@click.option("--margin-fraction", default=0.0, show_default=True, type=float)
@click.option("--blur-kernel", default=51, show_default=True, type=click.IntRange(min=3))
@click.option("--pixel-size", default=12, show_default=True, type=click.IntRange(min=2))
@click.option(
    "--comparison-frames",
    default=0,
    show_default=True,
    type=click.IntRange(min=0),
    help="Optional number of side-by-side diagnostic frames to write.",
)
@click.option("--overwrite", is_flag=True, help="Replace outputs in a non-empty directory.")
def render_faces(
    input_path: Path,
    privacy_regions_path: Path,
    interval_decisions_path: Path | None,
    output_dir: Path,
    output_video_path: Path | None,
    mode: str,
    method: str,
    margin_pixels: int,
    margin_fraction: float,
    blur_kernel: int,
    pixel_size: int,
    comparison_frames: int,
    overwrite: bool,
) -> None:
    """Render accepted face privacy regions into a redacted MP4."""
    from redactor import rendering

    settings = rendering.RenderSettings(
        mode=mode,
        method=method,
        margin_pixels=margin_pixels,
        margin_fraction=margin_fraction,
        blur_kernel=blur_kernel,
        pixel_size=pixel_size,
    )
    try:
        summary = rendering.run_face_render(
            input_path,
            privacy_regions_path=privacy_regions_path,
            interval_decisions_path=interval_decisions_path,
            output_dir=output_dir,
            output_video_path=output_video_path,
            settings=settings,
            comparison_frames=comparison_frames,
            overwrite=overwrite,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(
        f"Rendered {summary['processed_frame_count']} frames; "
        f"{summary['rendered_region_count']} privacy regions."
    )
    if summary["preview_non_final"]:
        click.echo("Output is a non-final preview.")
    else:
        click.echo(f"Final approval status: {summary['final_approval_status']}")
    click.echo(f"Wrote video: {output_dir / str(summary['output_video'])}")
    click.echo(f"Wrote manifest: {output_dir / 'render_manifest.json'}")
    click.echo(f"Wrote validation: {output_dir / 'validation.json'}")


@main.command("validate-ground-truth")
@click.argument(
    "ground_truth_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--source-video",
    "source_video_values",
    multiple=True,
    help="Optional clip_id=video_path source hash validation. Repeat per clip.",
)
def validate_ground_truth(
    ground_truth_path: Path,
    source_video_values: tuple[str, ...],
) -> None:
    """Validate machine-readable redaction ground truth."""
    from redactor import evaluation

    source_video_paths = _parse_source_video_values(source_video_values)
    try:
        result = evaluation.validate_ground_truth_file(
            ground_truth_path,
            source_video_paths=source_video_paths,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(
        f"Ground truth {result['ground_truth_id']} schema {result['schema_version']}: "
        f"{result['status']} ({result['error_count']} errors)"
    )
    for error in result["errors"]:
        click.echo(f"- {error}")
    if result["status"] != "valid":
        raise click.ClickException("ground-truth validation failed")


@main.command("evaluate-faces")
@click.option(
    "--ground-truth",
    "ground_truth_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Machine-readable redaction ground-truth JSON.",
)
@click.option("--clip-id", required=True, help="Clip id inside the ground-truth file.")
@click.option(
    "--mode",
    required=True,
    type=click.Choice(["detector", "automatic", "reviewed_final"], case_sensitive=True),
    help="Evaluation mode.",
)
@click.option(
    "--source-video",
    "source_video_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Source video for hash validation and visual evidence.",
)
@click.option(
    "--detections",
    "detections_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="ADAS detection JSONL.",
)
@click.option(
    "--adjudication",
    "adjudication_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Corrected ADAS adjudication JSON for detector identity anchors.",
)
@click.option(
    "--tracks",
    "tracks_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Ordinary face-tracking JSONL.",
)
@click.option(
    "--privacy-regions",
    "privacy_regions_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Privacy-region JSONL from gap recovery or preserved reviewed corrections.",
)
@click.option(
    "--interval-decisions",
    "interval_decisions_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Privacy gap interval-decision JSON.",
)
@click.option(
    "--preserved-regions",
    "preserved_regions_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional preserved reviewed regions used by reviewed_final runs.",
)
@click.option(
    "--rendered-video",
    "rendered_video_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional rendered output video.",
)
@click.option(
    "--render-manifest",
    "render_manifest_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional render_manifest.json.",
)
@click.option(
    "--render-validation",
    "render_validation_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional renderer validation.json.",
)
@click.option(
    "--detector-threshold",
    default=0.30,
    show_default=True,
    type=click.FloatRange(0, 1),
    help="Primary detector threshold used for detector recall.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for evaluation reports.",
)
@click.option("--overwrite", is_flag=True, help="Replace outputs in a non-empty directory.")
def evaluate_faces(
    ground_truth_path: Path,
    clip_id: str,
    mode: str,
    source_video_path: Path,
    detections_path: Path,
    adjudication_path: Path | None,
    tracks_path: Path | None,
    privacy_regions_path: Path | None,
    interval_decisions_path: Path | None,
    preserved_regions_path: Path | None,
    rendered_video_path: Path | None,
    render_manifest_path: Path | None,
    render_validation_path: Path | None,
    detector_threshold: float,
    output_dir: Path,
    overwrite: bool,
) -> None:
    """Evaluate the face pipeline against machine-readable ground truth."""
    from redactor import evaluation

    try:
        metrics = evaluation.run_face_evaluation(
            ground_truth_path=ground_truth_path,
            clip_id=clip_id,
            mode=mode,
            source_video_path=source_video_path,
            detections_path=detections_path,
            output_dir=output_dir,
            adjudication_path=adjudication_path,
            tracks_path=tracks_path,
            privacy_regions_path=privacy_regions_path,
            interval_decisions_path=interval_decisions_path,
            preserved_regions_path=preserved_regions_path,
            rendered_video_path=rendered_video_path,
            render_manifest_path=render_manifest_path,
            render_validation_path=render_validation_path,
            detector_threshold=detector_threshold,
            overwrite=overwrite,
        )
    except (FileExistsError, FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(
        f"Evaluation status: {metrics['acceptance']['final_status']} "
        f"for {metrics['clip_id']} ({metrics['evaluation_mode']})"
    )
    click.echo(
        f"Detector recall: {metrics['detector']['frames_with_valid_detection']}/"
        f"{metrics['detector']['detector_scored_object_frames']} "
        f"({metrics['detector']['object_frame_recall']})"
    )
    click.echo(
        f"Privacy coverage: {metrics['privacy']['covered_privacy_frames']}/"
        f"{metrics['privacy']['privacy_required_frames']} "
        f"({metrics['privacy']['privacy_coverage']})"
    )
    click.echo(f"Wrote evaluation report: {output_dir / 'report.md'}")


@main.command("build-review-bundle")
@click.option(
    "--source-video",
    "source_video_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Source video used by the pipeline.",
)
@click.option("--case-id", default=None, help="Stable case or clip id for the bundle.")
@click.option(
    "--detections",
    "detections_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional detection JSONL.",
)
@click.option(
    "--tracks",
    "tracks_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional tracking JSONL.",
)
@click.option(
    "--track-quality",
    "track_quality_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional track_quality.json.",
)
@click.option(
    "--track-review-flags",
    "track_review_flags_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional review_flags.json from tracking.",
)
@click.option(
    "--track-review-windows",
    "track_review_windows_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional review_windows.json from tracking.",
)
@click.option(
    "--privacy-regions",
    "privacy_regions_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional privacy_regions.jsonl.",
)
@click.option(
    "--interval-decisions",
    "interval_decisions_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional interval_decisions.json.",
)
@click.option(
    "--render-manifest",
    "render_manifest_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional render_manifest.json.",
)
@click.option(
    "--render-validation",
    "render_validation_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional renderer validation.json.",
)
@click.option(
    "--automatic-preview-video",
    "automatic_preview_video_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional historical automatic preview video for input traceability.",
)
@click.option(
    "--reviewed-final-video",
    "reviewed_final_video_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional historical reviewed-final video for input traceability.",
)
@click.option(
    "--evaluation-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Optional evaluator output directory.",
)
@click.option(
    "--corrected-privacy-regions",
    "corrected_privacy_regions_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional reviewed/corrected privacy_regions.jsonl used to render Corrected final clips.",
)
@click.option(
    "--corrected-interval-decisions",
    "corrected_interval_decisions_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional reviewed/corrected interval_decisions.json for presentation context.",
)
@click.option(
    "--corrected-render-manifest",
    "corrected_render_manifest_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional reviewed-final render_manifest.json for presentation context.",
)
@click.option(
    "--corrected-render-validation",
    "corrected_render_validation_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional reviewed-final render validation.json for presentation context.",
)
@click.option(
    "--corrected-evaluation-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Optional reviewed-final evaluator output directory for presentation context.",
)
@click.option(
    "--ground-truth",
    "ground_truth_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional machine-readable ground truth for manifesting benchmark context.",
)
@click.option(
    "--reviewer-findings",
    "reviewer_findings_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional reviewer-discovered findings JSON to promote audit misses into tasks.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Review bundle output directory.",
)
@click.option("--random-sample-count", default=3, show_default=True, type=click.IntRange(min=0))
@click.option(
    "--random-sample-frames",
    default=24,
    show_default=True,
    type=click.IntRange(min=1),
)
@click.option("--random-seed", default=20260803, show_default=True, type=int)
@click.option("--merge-gap-frames", default=12, show_default=True, type=click.IntRange(min=0))
@click.option("--context-frames", default=18, show_default=True, type=click.IntRange(min=0))
@click.option(
    "--one-frame-miss-context-frames",
    default=12,
    show_default=True,
    type=click.IntRange(min=0),
    help="Context frames on each side for one-frame detector-miss continuity tasks.",
)
@click.option(
    "--review-render-method",
    type=click.Choice(["gaussian_blur", "pixelate"], case_sensitive=True),
    default="pixelate",
    show_default=True,
    help="Redaction method for aligned review-comparison clips.",
)
@click.option("--review-margin-pixels", default=0, show_default=True, type=click.IntRange(min=0))
@click.option("--review-margin-fraction", default=0.15, show_default=True, type=float)
@click.option("--review-blur-kernel", default=51, show_default=True, type=click.IntRange(min=3))
@click.option("--review-pixel-size", default=16, show_default=True, type=click.IntRange(min=2))
@click.option("--no-visuals", is_flag=True, help="Skip clip/contact-sheet generation.")
@click.option("--overwrite", is_flag=True, help="Replace outputs in a non-empty directory.")
def build_review_bundle(
    source_video_path: Path,
    case_id: str | None,
    detections_path: Path | None,
    tracks_path: Path | None,
    track_quality_path: Path | None,
    track_review_flags_path: Path | None,
    track_review_windows_path: Path | None,
    privacy_regions_path: Path | None,
    interval_decisions_path: Path | None,
    render_manifest_path: Path | None,
    render_validation_path: Path | None,
    automatic_preview_video_path: Path | None,
    reviewed_final_video_path: Path | None,
    evaluation_dir: Path | None,
    corrected_privacy_regions_path: Path | None,
    corrected_interval_decisions_path: Path | None,
    corrected_render_manifest_path: Path | None,
    corrected_render_validation_path: Path | None,
    corrected_evaluation_dir: Path | None,
    ground_truth_path: Path | None,
    reviewer_findings_path: Path | None,
    output_dir: Path,
    random_sample_count: int,
    random_sample_frames: int,
    random_seed: int,
    merge_gap_frames: int,
    context_frames: int,
    one_frame_miss_context_frames: int,
    review_render_method: str,
    review_margin_pixels: int,
    review_margin_fraction: float,
    review_blur_kernel: int,
    review_pixel_size: int,
    no_visuals: bool,
    overwrite: bool,
) -> None:
    """Build a local HTML review bundle from face pipeline evidence."""
    from redactor import review_bundle

    config = review_bundle.ReviewBundleConfig(
        case_id=case_id,
        random_sample_count=random_sample_count,
        random_sample_frame_count=random_sample_frames,
        random_seed=random_seed,
        merge_gap_frames=merge_gap_frames,
        context_frames=context_frames,
        generate_visuals=not no_visuals,
        one_frame_miss_context_frames=one_frame_miss_context_frames,
        review_render=review_bundle.ReviewRenderConfig(
            method=review_render_method,
            margin_pixels=review_margin_pixels,
            margin_fraction=review_margin_fraction,
            blur_kernel=review_blur_kernel,
            pixel_size=review_pixel_size,
        ),
    )
    try:
        summary = review_bundle.build_review_bundle(
            source_video_path=source_video_path,
            output_dir=output_dir,
            config=config,
            detections_path=detections_path,
            tracks_path=tracks_path,
            track_quality_path=track_quality_path,
            track_review_flags_path=track_review_flags_path,
            track_review_windows_path=track_review_windows_path,
            privacy_regions_path=privacy_regions_path,
            interval_decisions_path=interval_decisions_path,
            render_manifest_path=render_manifest_path,
            render_validation_path=render_validation_path,
            automatic_preview_video_path=automatic_preview_video_path,
            reviewed_final_video_path=reviewed_final_video_path,
            evaluation_dir=evaluation_dir,
            corrected_privacy_regions_path=corrected_privacy_regions_path,
            corrected_interval_decisions_path=corrected_interval_decisions_path,
            corrected_render_manifest_path=corrected_render_manifest_path,
            corrected_render_validation_path=corrected_render_validation_path,
            corrected_evaluation_dir=corrected_evaluation_dir,
            ground_truth_path=ground_truth_path,
            reviewer_findings_path=reviewer_findings_path,
            overwrite=overwrite,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            raise click.ClickException(_subprocess_message(exc)) from exc
        raise click.ClickException(str(exc)) from exc

    click.echo(
        f"Built review bundle: {summary.get('reviewer_task_count', summary['total_items'])} "
        f"reviewer tasks, {summary.get('technical_only_items', 0)} technical-only signals."
    )
    click.echo(f"Open: {output_dir / 'index.html'}")


@main.command("serve-review-bundle")
@click.argument(
    "bundle_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option("--port", default=0, show_default=True, type=click.IntRange(min=0, max=65535))
@click.option(
    "--open", "open_browser", is_flag=True, help="Open the bundle in the default browser."
)
def serve_review_bundle(bundle_dir: Path, port: int, open_browser: bool) -> None:
    """Serve a local review bundle from 127.0.0.1 for reliable embedded video playback."""
    from redactor import review_bundle

    try:
        server, url = review_bundle.create_review_bundle_server(bundle_dir, port=port)
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Serving review bundle on {url}")
    click.echo("Press Ctrl+C to stop.")
    if open_browser:
        review_bundle.webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("Stopping review bundle server.")
    finally:
        server.server_close()


@main.command("validate-review-decisions")
@click.option(
    "--queue",
    "queue_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="review_queue.json from a review bundle.",
)
@click.option(
    "--decisions",
    "decisions_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Exported review decisions JSON.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional validation JSON output path.",
)
def validate_review_decisions(
    queue_path: Path,
    decisions_path: Path,
    output_path: Path | None,
) -> None:
    """Validate exported local review decisions against a review queue."""
    from redactor import review_bundle

    try:
        result = review_bundle.validate_review_decisions(queue_path, decisions_path)
        if output_path is not None:
            review_bundle.write_json(output_path, result)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(
        f"Review decisions {result['status']}: {result['decision_count']} decisions, "
        f"{result['unresolved_blocking_item_count']} unresolved blocking items."
    )
    if result["errors"]:
        for error in result["errors"]:
            click.echo(f"- {error}")
        raise click.ClickException("review decision validation failed")


@main.command("validate-final-audit-decisions")
@click.option(
    "--tasks",
    "tasks_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="audit_tasks.json from the final pipeline audit package.",
)
@click.option(
    "--decisions",
    "decisions_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Exported final audit decisions JSON.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional validation JSON output path.",
)
def validate_final_audit_decisions(
    tasks_path: Path,
    decisions_path: Path,
    output_path: Path | None,
) -> None:
    """Validate exported final full-video audit decisions."""
    from redactor import final_package, review_bundle

    try:
        result = final_package.validate_audit_decisions(
            final_package.load_json(decisions_path),
            final_package.load_json(tasks_path),
        )
        if output_path is not None:
            review_bundle.write_json(output_path, result)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(
        f"Final audit decisions: {result['decision_count']} decisions, "
        f"{len(result['unresolved_blocking_items'])} unresolved blocking items."
    )
    if result["errors"]:
        for error in result["errors"]:
            click.echo(f"- {error}")
        raise click.ClickException("final audit decision validation failed")


@main.command("summarize-final-audit")
@click.option(
    "--tasks",
    "tasks_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="audit_tasks.json from the final pipeline audit package.",
)
@click.option(
    "--decisions",
    "decisions_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional exported final audit decisions JSON.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional summary JSON output path.",
)
def summarize_final_audit(
    tasks_path: Path,
    decisions_path: Path | None,
    output_path: Path | None,
) -> None:
    """Summarize pending/completed final full-video audit work."""
    from redactor import final_package, review_bundle

    try:
        decisions = final_package.load_json(decisions_path) if decisions_path else None
        summary = final_package.summarize_audit(final_package.load_json(tasks_path), decisions)
        if output_path is not None:
            review_bundle.write_json(output_path, summary)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(
        f"Final audit summary: {summary['total_tasks']} tasks, "
        f"{summary['pending_review_count']} pending, "
        f"{summary['unresolved_blocking_count']} unresolved blockers."
    )


@main.command("build-final-review-workspace")
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Final review workspace package directory.",
)
@click.option(
    "--v1-package",
    "v1_package_root",
    default=Path("artifacts/final_pipeline_package_v1"),
    show_default=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Existing final_pipeline_package_v1 root used as frozen input evidence.",
)
@click.option(
    "--adas-xml",
    "adas_xml_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Frozen OpenVINO face-detection-adas-0001 XML path. REDACTOR_ADAS_XML is honored.",
)
@click.option(
    "--adas-bin",
    "adas_bin_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Frozen OpenVINO face-detection-adas-0001 BIN path. REDACTOR_ADAS_BIN is honored.",
)
@click.option(
    "--plate-model",
    "plate_model_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Frozen OIM YOLOv9-S-608 ONNX path. REDACTOR_OIM_PLATE_MODEL is honored.",
)
@click.option(
    "--frontend",
    type=click.Choice(["react", "legacy"]),
    default="react",
    show_default=True,
    help="Workspace frontend to place at the package entry point.",
)
@click.option(
    "--skip-frontend-build",
    is_flag=True,
    help="Copy an existing frontend/dist build instead of running npm ci and npm run build.",
)
@click.option(
    "--react-reference-package",
    "react_reference_package",
    default=Path("artifacts/final_pipeline_package_v2_3_4"),
    show_default=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Frozen package to use as the backend base for React frontend packages.",
)
@click.option(
    "--regenerate-backend",
    is_flag=True,
    help="Regenerate backend package artifacts instead of copying the frozen v2.3.4 backend base.",
)
@click.option("--overwrite", is_flag=True, help="Replace an existing v2 package directory.")
def build_final_review_workspace(
    output_dir: Path,
    v1_package_root: Path,
    adas_xml_path: Path | None,
    adas_bin_path: Path | None,
    plate_model_path: Path | None,
    frontend: str,
    skip_frontend_build: bool,
    react_reference_package: Path,
    regenerate_backend: bool,
    overwrite: bool,
) -> None:
    """Build the local case-based final review workspace package."""
    from redactor import review_workspace

    try:
        if frontend == "react" and not regenerate_backend:
            summary = review_workspace.build_react_workspace_package_from_reference(
                package_root=output_dir,
                reference_root=react_reference_package,
                build_frontend=not skip_frontend_build,
                overwrite=overwrite,
            )
        else:
            summary = review_workspace.build_workspace_package(
                package_root=output_dir,
                v1_package_root=v1_package_root,
                adas_xml_path=adas_xml_path,
                adas_bin_path=adas_bin_path,
                plate_model_path=plate_model_path,
                frontend=frontend,
                build_frontend=not skip_frontend_build,
                overwrite=overwrite,
            )
    except (
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        raise click.ClickException(
            _subprocess_message(exc) if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        ) from exc
    click.echo(f"Built final review workspace: {output_dir}")
    if "raw_signal_count" in summary:
        click.echo(f"Raw audit signals: {summary['raw_signal_count']}")
    if "reviewer_task_count" in summary:
        click.echo(f"Reviewer tasks: {summary['reviewer_task_count']}")
    if "day" in summary:
        click.echo(f"Day lineage proven: {summary['day']['lineage_proven']}")
    click.echo(f"Frontend: {summary['frontend']}")
    if "day" in summary and summary["day"]["blockers"]:
        click.echo("Day case blockers:")
        for blocker in summary["day"]["blockers"]:
            click.echo(f"- {blocker}")
    click.echo(f"Open with: redactor serve-final-review-workspace {output_dir} --open")


@main.command("generate-final-review-api-schema")
@click.option(
    "--output",
    "output_path",
    default=Path("docs/api/final_review_workspace_v1.schema.json"),
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
def generate_final_review_api_schema(output_path: Path) -> None:
    """Write the machine-readable Final Review Workspace API v1 contract."""
    from redactor import review_api_contract

    review_api_contract.write_schema(output_path)
    click.echo(f"Wrote API schema: {output_path}")


@main.command("audit-final-review-backend-parity")
@click.option(
    "--reference",
    "reference_root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Frozen reference final review package root.",
)
@click.option(
    "--candidate",
    "candidate_root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Candidate final review package root.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional JSON audit output path.",
)
def audit_final_review_backend_parity(
    reference_root: Path,
    candidate_root: Path,
    output_path: Path | None,
) -> None:
    """Compare backend/privacy artifacts between two final review packages."""
    from redactor import review_workspace

    audit = review_workspace.backend_parity_audit(reference_root, candidate_root)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        click.echo(f"Wrote backend parity audit: {output_path}")
    click.echo(
        f"Compared {audit['compared_count']} backend artifacts; errors: {audit['error_count']}"
    )
    if not audit["passed"]:
        raise click.ClickException("; ".join(audit["errors"][:10]))


@main.command("serve-final-review-workspace")
@click.argument(
    "package_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--port", default=0, show_default=True, type=click.IntRange(min=0, max=65535))
@click.option("--open", "open_browser", is_flag=True, help="Open the workspace in a browser.")
def serve_final_review_workspace(package_root: Path, port: int, open_browser: bool) -> None:
    """Serve a final review workspace on 127.0.0.1 with local write endpoints."""
    import time

    from redactor import review_workspace

    server = review_workspace.serve_workspace(
        package_root,
        port=port,
        open_browser=open_browser,
    )
    host, resolved_port = server.server_address
    click.echo(f"Serving {package_root} at http://{host}:{resolved_port}/index.html")
    click.echo("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.shutdown()
        server.server_close()
        click.echo("Stopped final review workspace server.")


@main.command("validate-review-session")
@click.option(
    "--session-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Review session directory containing events.jsonl.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional validation JSON output path.",
)
def validate_review_session(session_dir: Path, output_path: Path | None) -> None:
    """Validate a final review workspace session event hash chain."""
    from redactor import review_sessions

    try:
        result = review_sessions.validate_event_hash_chain(session_dir)
        if output_path is not None:
            review_sessions.write_json(output_path, result)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"Review session event chain: {result['event_count']} events, "
        f"{result['error_count']} errors."
    )
    if not result["passed"]:
        raise click.ClickException("review session validation failed")


@main.command("summarize-review-session")
@click.option(
    "--session-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Review session directory.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional summary JSON output path.",
)
def summarize_review_session(session_dir: Path, output_path: Path | None) -> None:
    """Summarize a final review workspace session."""
    from redactor import review_sessions

    try:
        summary = review_sessions.refresh_session_summary(session_dir)
        if output_path is not None:
            review_sessions.write_json(output_path, summary)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"Review session summary: {summary['event_count']} events, "
        f"{summary['decision_count']} current decisions, "
        f"{summary['correction_count']} corrections."
    )


@main.command("finalize-reviewed-case")
@click.option(
    "--package-root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Final review workspace package root.",
)
@click.option("--case-id", required=True, type=str, help="Case id to finalize.")
@click.option(
    "--session-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Validated review session directory.",
)
@click.option("--overwrite", is_flag=True, help="Replace an existing final reviewed MP4.")
def finalize_reviewed_case(
    package_root: Path,
    case_id: str,
    session_dir: Path,
    overwrite: bool,
) -> None:
    """Finalize one case after all review blockers have been resolved."""
    from redactor import review_finalization

    try:
        result = review_finalization.finalize_reviewed_case(
            package_root=package_root,
            case_id=case_id,
            session_dir=session_dir,
            overwrite=overwrite,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        raise click.ClickException(
            _subprocess_message(exc) if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        ) from exc
    if result["finalized"]:
        click.echo(f"Finalized {case_id}: {result['final_video']}")
        click.echo(f"Final SHA-256: {result['final_video_sha256']}")
    else:
        click.echo(f"Finalization blocked for {case_id}:")
        for blocker in result["blockers"]:
            click.echo(f"- {blocker}")
        raise click.ClickException("finalization blockers remain")


@main.command("validate-face-policy")
@click.argument(
    "policy_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--source-video",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional source video used to validate policy hash and frame count.",
)
def validate_face_policy(policy_path: Path, source_video: Path | None) -> None:
    """Validate a machine-readable face privacy policy."""
    from redactor import face_policy

    try:
        result = face_policy.validate_policy_file(policy_path, source_video=source_video)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"Face privacy policy {result['status']}: "
        f"{result['track_assignment_count']} track assignments; mode={result['mode']}."
    )
    for error in result["errors"]:
        click.echo(f"- {error}")
    if result["status"] != "valid":
        raise click.ClickException("face policy validation failed")


@main.command("redact-video")
@click.argument(
    "input_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option("--faces/--no-faces", default=True, show_default=True, help="Run face redaction.")
@click.option(
    "--policy",
    "policy_mode",
    type=click.Choice(
        ["redact_all", "redact_all_except_keep", "redact_selected", "review_only"],
        case_sensitive=True,
    ),
    default="redact_all",
    show_default=True,
    help="Case-level face privacy policy.",
)
@click.option(
    "--privacy-policy",
    "policy_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional machine-readable face privacy policy JSON.",
)
@click.option(
    "--keep-track-id",
    "keep_track_ids",
    multiple=True,
    type=int,
    help="Explicit case-policy keep track. Not required for default redact_all.",
)
@click.option(
    "--selected-track-id",
    "selected_track_ids",
    multiple=True,
    type=int,
    help="Explicit redact track for redact_selected policy.",
)
@click.option(
    "--detections",
    "detections_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Advanced: reuse an existing ADAS detections JSONL instead of running detection.",
)
@click.option(
    "--model",
    "model_path",
    required=False,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Face model path. For ADAS this is the XML path.",
)
@click.option(
    "--adas-xml",
    "adas_xml_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="OpenVINO face-detection-adas-0001 XML path. REDACTOR_ADAS_XML is also honored.",
)
@click.option(
    "--adas-bin",
    "adas_bin_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="OpenVINO face-detection-adas-0001 BIN path. REDACTOR_ADAS_BIN is also honored.",
)
@click.option("--proposal-threshold", default=0.20, show_default=True, type=click.FloatRange(0, 1))
@click.option("--primary-threshold", default=0.30, show_default=True, type=click.FloatRange(0, 1))
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Root directory for the full-video face privacy case output.",
)
@click.option("--no-detection-overlay", is_flag=True, help="Skip diagnostic detection overlay.")
@click.option("--overwrite", is_flag=True, help="Replace outputs in non-empty output directories.")
def redact_video(
    input_path: Path,
    faces: bool,
    policy_mode: str,
    policy_path: Path | None,
    keep_track_ids: tuple[int, ...],
    selected_track_ids: tuple[int, ...],
    detections_path: Path | None,
    model_path: Path | None,
    adas_xml_path: Path | None,
    adas_bin_path: Path | None,
    proposal_threshold: float,
    primary_threshold: float,
    output_dir: Path,
    no_detection_overlay: bool,
    overwrite: bool,
) -> None:
    """Run the production-oriented full-video face privacy workflow."""
    if not faces:
        raise click.ClickException("redact-video currently supports --faces only")
    from redactor import detection, face_policy

    resolved_detections_path = detections_path
    try:
        if resolved_detections_path is None:
            detection_dir = output_dir / "detection"
            resolved_backend = detection.resolve_detector_backend("face", None)
            resolved_model_path = resolve_detection_model_path(
                category="face",
                detector_backend=resolved_backend,
                model_path=model_path,
                adas_xml_path=adas_xml_path,
            )
            resolved_adas_bin = (
                adas_bin_path
                if adas_bin_path is not None
                else detection.default_adas_bin_path(resolved_model_path)
            )
            detection.run_detection(
                input_path,
                category="face",
                model_path=resolved_model_path,
                threshold=proposal_threshold,
                output_dir=detection_dir,
                overwrite=overwrite,
                detector_backend=resolved_backend,
                adas_bin_path=resolved_adas_bin,
                primary_threshold=primary_threshold,
                write_overlay_video=not no_detection_overlay,
            )
            resolved_detections_path = detection_dir / "detections.jsonl"
        audit = face_policy.run_full_video_policy_flow(
            input_path,
            output_dir=output_dir,
            detections_path=resolved_detections_path,
            mode=policy_mode,
            policy_path=policy_path,
            keep_track_ids=keep_track_ids,
            selected_track_ids=selected_track_ids,
            overwrite=overwrite,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        "Full-video face privacy preview complete: "
        f"{audit['privacy_summary']['redact_track_count']} redact tracks; "
        f"{audit['privacy_summary']['review_required_interval_count']} review-required intervals."
    )
    click.echo(f"Final export status: {audit['final_export_status']}")
    click.echo(f"Wrote audit manifest: {output_dir / 'audit_manifest.json'}")


@main.command("face-pipeline")
@click.argument(
    "input_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Root directory for detection, tracking, and privacy gap-recovery outputs.",
)
@click.option(
    "--model",
    "model_path",
    required=False,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Face model path. For ADAS this is the XML path.",
)
@click.option(
    "--adas-xml",
    "adas_xml_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="OpenVINO face-detection-adas-0001 XML path. REDACTOR_ADAS_XML is also honored.",
)
@click.option(
    "--adas-bin",
    "adas_bin_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="OpenVINO face-detection-adas-0001 BIN path. REDACTOR_ADAS_BIN is also honored.",
)
@click.option("--proposal-threshold", default=0.20, show_default=True, type=click.FloatRange(0, 1))
@click.option("--primary-threshold", default=0.30, show_default=True, type=click.FloatRange(0, 1))
@click.option("--target-track-id", "target_track_ids", multiple=True, type=int)
@click.option("--keep-track-id", "keep_track_ids", multiple=True, type=int)
@click.option("--competing-track-id", "competing_track_ids", multiple=True, type=int)
@click.option(
    "--frame-start", type=int, default=None, help="Inclusive privacy recovery span start."
)
@click.option("--frame-end", type=int, default=None, help="Inclusive privacy recovery span end.")
@click.option("--max-frames", type=click.IntRange(min=1), default=None)
@click.option("--no-detection-overlay", is_flag=True)
@click.option("--no-recovery-overlay", is_flag=True)
@click.option("--overwrite", is_flag=True, help="Replace outputs in non-empty output directories.")
def face_pipeline(
    input_path: Path,
    output_dir: Path,
    model_path: Path | None,
    adas_xml_path: Path | None,
    adas_bin_path: Path | None,
    proposal_threshold: float,
    primary_threshold: float,
    target_track_ids: tuple[int, ...],
    keep_track_ids: tuple[int, ...],
    competing_track_ids: tuple[int, ...],
    frame_start: int | None,
    frame_end: int | None,
    max_frames: int | None,
    no_detection_overlay: bool,
    no_recovery_overlay: bool,
    overwrite: bool,
) -> None:
    """Diagnostic detection/tracking path. For full-video redaction use redact-video."""
    from redactor import detection, privacy_gap_recovery, tracking

    resolved_backend = detection.resolve_detector_backend("face", None)
    resolved_model_path = resolve_detection_model_path(
        category="face",
        detector_backend=resolved_backend,
        model_path=model_path,
        adas_xml_path=adas_xml_path,
    )
    resolved_adas_bin = (
        adas_bin_path
        if resolved_backend == detection.DETECTOR_BACKEND_ADAS and adas_bin_path is not None
        else (
            detection.default_adas_bin_path(resolved_model_path)
            if resolved_backend == detection.DETECTOR_BACKEND_ADAS
            else None
        )
    )
    detection_dir = output_dir / "detection"
    tracking_dir = output_dir / "tracking"
    recovery_dir = output_dir / "privacy_gap_recovery"
    try:
        detection_summary = detection.run_detection(
            input_path,
            category="face",
            model_path=resolved_model_path,
            threshold=proposal_threshold,
            output_dir=detection_dir,
            overwrite=overwrite,
            max_frames=max_frames,
            detector_backend=resolved_backend,
            adas_bin_path=resolved_adas_bin,
            primary_threshold=primary_threshold,
            write_overlay_video=not no_detection_overlay,
        )
        tracking_summary = tracking.run_tracking(
            input_path,
            detections_path=detection_dir / "detections.jsonl",
            output_dir=tracking_dir,
            strong_threshold=primary_threshold,
            weak_threshold=proposal_threshold,
            overwrite=overwrite,
        )
        click.echo(
            f"Detection processed {detection_summary['frames_processed']} frames; "
            f"tracking produced {tracking_summary['auto_confirmed_tracks']} auto-confirmed tracks."
        )
        if target_track_ids:
            recovery_summary = privacy_gap_recovery.run_privacy_gap_recovery(
                input_path,
                tracks_path=tracking_dir / "tracks.jsonl",
                output_dir=recovery_dir,
                target_track_ids=target_track_ids,
                keep_track_ids=keep_track_ids,
                competing_track_ids=competing_track_ids,
                frame_start=frame_start,
                frame_end=frame_end,
                settings=privacy_gap_recovery.PrivacyGapRecoverySettings(
                    render_overlay=not no_recovery_overlay,
                ),
                overwrite=overwrite,
            )
            metrics = recovery_summary["metrics"]
            click.echo(
                f"Privacy gap recovery considered {metrics['gaps_considered']} gaps; "
                f"recovered {metrics['recovered_frames']} frames."
            )
        else:
            click.echo("Skipped privacy gap recovery because no --target-track-id was provided.")
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc


def _subprocess_message(exc: subprocess.CalledProcessError) -> str:
    stderr = exc.stderr.strip() if exc.stderr else ""
    return stderr or str(exc)


def _parse_source_video_values(values: tuple[str, ...]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise click.ClickException("--source-video must use clip_id=path")
        clip_id, raw_path = value.split("=", 1)
        if not clip_id or not raw_path:
            raise click.ClickException("--source-video must use clip_id=path")
        parsed[clip_id] = Path(raw_path)
    return parsed


def resolve_detection_model_path(
    *,
    category: str,
    detector_backend: str,
    model_path: Path | None,
    adas_xml_path: Path | None,
) -> Path:
    from redactor import detection

    if detector_backend == detection.DETECTOR_BACKEND_ADAS:
        return adas_xml_path or model_path or detection.default_adas_xml_path()
    if detector_backend == detection.DETECTOR_BACKEND_OIM_YOLOV9_S_608:
        from redactor import plate_detection

        return model_path or plate_detection.default_model_path()
    if model_path is not None:
        return model_path
    if category == "face":
        raise click.ClickException("face detection requires an ADAS XML model path")
    if category == "license_plate":
        raise click.ClickException("license-plate detection requires --model with an ONNX path")
    raise click.ClickException(f"unsupported category: {category}")


if __name__ == "__main__":
    main()
