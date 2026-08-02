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
    required=False,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Model path. For face detection this is the ADAS XML path.",
)
@click.option(
    "--detector-backend",
    type=click.Choice(
        ["openvino_face_adas_0001", "license_plate_yunet"],
        case_sensitive=True,
    ),
    default=None,
    help=("Advanced detector backend. Face detection supports only openvino_face_adas_0001."),
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
    help="Proposal storage threshold. ADAS default: 0.20.",
)
@click.option(
    "--primary-threshold",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Primary observation threshold. ADAS default: 0.30.",
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
    """Run face detection, ordinary tracking, and optional privacy gap recovery."""
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
    if model_path is not None:
        return model_path
    if category == "face":
        raise click.ClickException("face detection requires an ADAS XML model path")
    if category == "license_plate":
        raise click.ClickException("license-plate detection requires --model with an ONNX path")
    raise click.ClickException(f"unsupported category: {category}")


if __name__ == "__main__":
    main()
