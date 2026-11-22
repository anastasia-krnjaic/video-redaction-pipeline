"""Generate a model-free, public-safe demonstration using the real review pipeline.

Geometry and observations are authored fixtures, not predictions from a detector.
Only the temporal association, bounded interpolation, renderer, and workspace are
production components. No pretrained model or external media is loaded.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

from redactor import case_pipeline, media, plate_tracking, tracking
from redactor.privacy_records import APPROVAL_ACCEPTED, PrivacyRegionRecord

WIDTH, HEIGHT, FPS, FRAME_COUNT = 640, 360, 20, 160
GIF_WIDTH, GIF_HEIGHT, GIF_FPS = 840, 416, 10
GAPS = {"face": (52, 53, 54), "license_plate": (92, 93, 94)}
MANUAL_START = 120
MANUAL_BOX = (526, 110, 582, 166)
SYNTHETIC = "synthetic_fixture_no_model_inference"


def target_box(category: str, frame: int) -> tuple[int, int, int, int]:
    if not 0 <= frame < FRAME_COUNT:
        raise ValueError(f"frame must be between 0 and {FRAME_COUNT - 1}")
    if category == "face":
        x, y, width, height = 64 + round(frame * 1.5), 95, 64, 64
    elif category == "license_plate":
        x, y, width, height = 400 - frame * 2, 249, 88, 32
    else:
        raise ValueError(f"unsupported synthetic category: {category}")
    return x, y, x + width, y + height


def source_frame(frame: int) -> np.ndarray:
    """Return entirely generated pixels; plate-like targets contain only shapes."""
    image = np.full((HEIGHT, WIDTH, 3), (36, 28, 24), dtype=np.uint8)
    cv.putText(
        image,
        "SYNTHETIC / INJECTED OBSERVATIONS",
        (22, 28),
        cv.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 225, 235),
        1,
        cv.LINE_AA,
    )
    cv.putText(
        image,
        "No model inference. No real people or plates.",
        (22, 50),
        cv.FONT_HERSHEY_SIMPLEX,
        0.43,
        (160, 170, 180),
        1,
        cv.LINE_AA,
    )
    x1, y1, x2, y2 = target_box("face", frame)
    cv.circle(image, (x1 + 32, y1 + 32), 30, (100, 185, 240), -1)
    for x in (x1 + 20, x1 + 44):
        cv.circle(image, (x, y1 + 23), 5, (35, 35, 35), -1)
    cv.ellipse(image, (x1 + 32, y1 + 36), (15, 12), 0, 0, 180, (35, 35, 35), 3)
    x1, y1, x2, y2 = target_box("license_plate", frame)
    cv.rectangle(image, (x1 - 18, y1 - 40), (x2 + 18, y2 + 15), (100, 85, 65), -1)
    cv.rectangle(image, (x1, y1), (x2, y2), (230, 225, 210), -1)
    for offset in range(8, 80, 12):
        cv.rectangle(image, (x1 + offset, y1 + 8), (x1 + offset + 5, y1 + 23), (45, 45, 45), -1)
    if frame >= MANUAL_START:
        x1, y1, x2, y2 = MANUAL_BOX
        for y in range(y1, y2, 7):
            for x in range(x1, x2, 7):
                color = (220, 90, 170) if ((x - x1 + y - y1) // 7) % 2 else (35, 220, 230)
                cv.rectangle(image, (x, y), (x + 6, y + 6), color, -1)
        cv.putText(
            image,
            "Manual demo",
            (510, 187),
            cv.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 225, 235),
            1,
            cv.LINE_AA,
        )
    gap = next((name for name, frames in GAPS.items() if frame in frames), None)
    label = f"Frame {frame:03d} / {FRAME_COUNT - 1}"
    if gap:
        label += f"   {gap} observation omitted"
    cv.putText(
        image, label, (22, 338), cv.FONT_HERSHEY_SIMPLEX, 0.48, (200, 225, 235), 1, cv.LINE_AA
    )
    return image


def write_source(path: Path) -> None:
    """Encode generated frames directly as a browser-compatible H.264 video."""
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-n",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{WIDTH}x{HEIGHT}",
        "-framerate",
        str(FPS),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "1",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    subprocess.run(
        command, input=b"".join(source_frame(i).tobytes() for i in range(FRAME_COUNT)), check=True
    )


def temporal_fixture(category: str) -> tuple[list[PrivacyRegionRecord], dict[str, Any]]:
    """Exercise existing tracking with deliberate gaps and unchanged safety gates."""
    frames = [i for i in range(FRAME_COUNT) if i not in GAPS[category]]
    records: list[PrivacyRegionRecord] = []
    if category == "face":
        result = tracking.build_tracks(
            {
                i: [
                    tracking.TrackObservation(
                        i, target_box(category, i), 0.92, SYNTHETIC, detector_id=SYNTHETIC
                    )
                ]
                for i in frames
            },
            manual_seeds={},
            frame_count=FRAME_COUNT,
            strong_threshold=0.30,
            max_gap_frames=3,
            max_tail_prediction_frames=0,
            fps=FPS,
            width=WIDTH,
            height=HEIGHT,
        )
        for frame, boxes in result.frame_boxes.items():
            for box in boxes:
                if box.render_eligible:
                    records.append(
                        PrivacyRegionRecord(
                            category=category,
                            source_frame=frame,
                            bbox_xyxy=box.bbox_xyxy,
                            object_id=f"face_track_{box.track_id}",
                            provenance=f"synthetic_fixture_{box.provenance}",
                            approval_state=APPROVAL_ACCEPTED,
                            render_eligible=True,
                            source_record={
                                "input_kind": SYNTHETIC,
                                "temporal_record": box.to_dict(),
                            },
                        )
                    )
        details = {
            "interpolated_frames": result.interpolated_frames,
            "confirmed_tracks": sorted(result.auto_confirmed_track_ids),
        }
    else:
        result = plate_tracking.track_plate_observations(
            {
                i: [
                    plate_tracking.PlateObservation(
                        frame_index=i,
                        bbox_xyxy=target_box(category, i),
                        score=0.92,
                        detector_id=SYNTHETIC,
                        candidate_id=f"synthetic_plate_{i}",
                        source_detection_index=0,
                        metadata={"input_kind": SYNTHETIC},
                    )
                ]
                for i in frames
            },
            frame_count=FRAME_COUNT,
            fps=FPS,
            width=WIDTH,
            height=HEIGHT,
            config=plate_tracking.default_config(),
        )
        for frame, boxes in result.frame_tracks.items():
            for box in boxes:
                if box.status == plate_tracking.ACCEPTED_TRACK:
                    records.append(
                        PrivacyRegionRecord(
                            category=category,
                            source_frame=frame,
                            bbox_xyxy=box.bbox_xyxy,
                            object_id=plate_tracking.format_track_uid(box.track_id),
                            provenance=f"synthetic_fixture_{box.provenance}",
                            approval_state=APPROVAL_ACCEPTED,
                            render_eligible=True,
                            source_record={
                                "input_kind": SYNTHETIC,
                                "temporal_record": box.to_dict(),
                            },
                        )
                    )
        details = {
            "tracks": [track.to_summary_dict() for track in result.tracks],
            "gap_decisions": [gap.to_dict() for gap in result.gap_decisions],
        }
    return records, {
        "input_kind": SYNTHETIC,
        "category": category,
        "observed_frames": frames,
        "omitted_observation_frames": list(GAPS[category]),
        "observations": [
            {"frame": i, "bbox_xyxy": list(target_box(category, i)), "score": 0.92} for i in frames
        ],
        **details,
    }


class SyntheticAdapter:
    """Use the existing case-pipeline extension point without invoking detectors."""

    def __init__(self, category: str) -> None:
        self.category = category

    def run(
        self, context: case_pipeline.CaseContext, options: case_pipeline.CasePipelineOptions
    ) -> case_pipeline.CategoryRunResult:
        del options
        records, evidence = temporal_fixture(self.category)
        directory = context.output_dir / self.category
        evidence_path = directory / "synthetic_observations_and_tracking.json"
        case_pipeline.write_json(evidence_path, evidence)
        task_path = context.output_dir if self.category == "face" else directory
        case_pipeline.write_json(task_path / "review" / "review_tasks.json", {"tasks": []})
        artifact = case_pipeline.artifact_reference(
            context.output_dir,
            evidence_path,
            role="synthetic_fixture_evidence",
            category=self.category,
        )
        stages = (
            case_pipeline.StageStatus(
                stage=case_pipeline.STAGE_DETECT,
                status=case_pipeline.STATUS_SKIPPED,
                category=self.category,
                reason=SYNTHETIC,
            ),
            case_pipeline.StageStatus(
                stage=case_pipeline.STAGE_TEMPORAL_ASSOCIATION,
                status=case_pipeline.STATUS_SUCCEEDED,
                category=self.category,
                artifacts=(artifact,),
                summary={"input_kind": SYNTHETIC},
            ),
        )
        return case_pipeline.CategoryRunResult(
            category=self.category,
            stages=stages,
            privacy_regions=tuple(records),
            artifacts=(artifact,),
            summary={"input_kind": SYNTHETIC},
        )


def render_without_workspace(source: Path, output: Path) -> dict[str, Any]:
    """Use the same production plan/renderer without the frontend packaging step."""
    info = media.probe_media(source)
    context = case_pipeline.CaseContext(
        case_id="case",
        output_dir=output / "case",
        selected_categories=tuple(GAPS),
        source_identity=case_pipeline.SourceIdentity.from_video(source, info=info),
        source_info=info,
        source_video=source,
    )
    results = [
        SyntheticAdapter(category).run(context, case_pipeline.CasePipelineOptions())
        for category in GAPS
    ]
    plan = case_pipeline.write_render_plan(context, results)
    if plan["blockers"]:
        raise RuntimeError(f"synthetic render plan blocked: {plan['blockers']}")
    case_pipeline.render_combined_automatic_preview(
        context,
        effective_regions_path=context.output_dir / "render_plan/effective_privacy_regions.jsonl",
        overwrite=False,
    )
    return {"render_plan": plan}


def comparison_frame(original: np.ndarray, automatic: np.ndarray, frame: int) -> np.ndarray:
    """Present matching decoded source/output frames; do not simulate redaction."""
    if not 0 <= frame < FRAME_COUNT:
        raise ValueError("comparison frame outside the synthetic source span")
    if any(image.shape != (HEIGHT, WIDTH, 3) for image in (original, automatic)):
        raise ValueError("comparison inputs must match the synthetic source dimensions")
    canvas = np.full((GIF_HEIGHT, GIF_WIDTH, 3), (27, 22, 18), dtype=np.uint8)
    text_color, muted, accent = (245, 241, 235), (185, 178, 165), (180, 210, 95)

    def label(text: str, xy: tuple[int, int], size: float, color: tuple[int, ...]) -> None:
        cv.putText(canvas, text, xy, cv.FONT_HERSHEY_SIMPLEX, size, color, 1, cv.LINE_AA)

    label("VIDEO PRIVACY / SYNTHETIC DEMO", (24, 33), 0.78, text_color)
    label(
        "Injected observations. Real tracking + interpolation + rendering.", (24, 59), 0.48, muted
    )
    label("INPUT / generated shapes", (24, 94), 0.52, text_color)
    label("OUTPUT / automatic pixelation", (432, 94), 0.52, accent)
    for left, pixels in ((24, original), (432, automatic)):
        canvas[110:326, left : left + 384] = cv.resize(
            pixels, (384, 216), interpolation=cv.INTER_AREA
        )
    label(
        "Missing observations: face 52-54 / plate 92-94. Both gaps interpolated.",
        (24, 352),
        0.46,
        muted,
    )
    cv.line(canvas, (24, 378), (816, 378), (80, 70, 60), 3)
    for frames in GAPS.values():
        start, end = (
            24 + round(value / (FRAME_COUNT - 1) * 792) for value in (frames[0], frames[-1])
        )
        cv.line(canvas, (start, 378), (end, 378), (105, 180, 250), 7)
    cursor = 24 + round(frame / (FRAME_COUNT - 1) * 792)
    cv.circle(canvas, (cursor, 378), 5, accent, -1)
    label(f"Source frame {frame:03d}/{FRAME_COUNT - 1}  |  8 seconds", (24, 407), 0.43, muted)
    label("No model inference / no real media", (490, 407), 0.43, muted)
    return canvas


def write_comparison_gif(source: Path, automatic: Path, destination: Path) -> dict[str, Any]:
    """Encode every second matched decoded frame at 10 fps: 8 seconds, no slowdown."""
    frames = []
    captures = [cv.VideoCapture(str(path)) for path in (source, automatic)]
    try:
        for frame in range(FRAME_COUNT):
            decoded = [capture.read() for capture in captures]
            if not all(ok for ok, _ in decoded):
                raise ValueError("comparison video ended before the expected source frame")
            if frame % (FPS // GIF_FPS) == 0:
                frames.append(comparison_frame(decoded[0][1], decoded[1][1], frame))
        if any(capture.read()[0] for capture in captures):
            raise ValueError("comparison video exceeds the expected source frame count")
    finally:
        for capture in captures:
            capture.release()
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{GIF_WIDTH}x{GIF_HEIGHT}",
            "-framerate",
            str(GIF_FPS),
            "-i",
            "pipe:0",
            "-filter_complex_threads",
            "1",
            "-filter_complex",
            "split[a][b];[a]palettegen=max_colors=192:stats_mode=diff[p];"
            "[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
            "-loop",
            "0",
            str(destination),
        ],
        input=b"".join(frame.tobytes() for frame in frames),
        check=True,
    )
    return {
        "sha256": media.sha256_file(destination),
        "size_bytes": destination.stat().st_size,
        "width": GIF_WIDTH,
        "height": GIF_HEIGHT,
        "frame_count": len(frames),
        "fps": GIF_FPS,
        "duration_seconds": len(frames) / GIF_FPS,
        "source_frame_stride": FPS // GIF_FPS,
        "presentation": "matching decoded input and automatic output; no playback slowdown",
    }


def generate_demo(
    output: Path,
    *,
    gif_path: Path | None = None,
    skip_workspace: bool = False,
    overwrite_gif: bool = False,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new or empty output directory: {output}")
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"{executable} must already be installed")
    frontend = Path(case_pipeline.__file__).resolve().parents[2] / "frontend" / "dist"
    if not skip_workspace and not (frontend / "index.html").is_file():
        raise RuntimeError("Build the frontend first: npm --prefix frontend run build")
    if gif_path is not None:
        if gif_path.suffix.lower() != ".gif":
            raise ValueError("--gif requires a .gif destination")
        if gif_path.exists() and not overwrite_gif:
            raise FileExistsError(
                f"GIF already exists; pass --overwrite-gif to replace: {gif_path}"
            )
    output.mkdir(parents=True, exist_ok=True)
    source = output / "synthetic_source.mp4"
    write_source(source)
    if skip_workspace:
        result = render_without_workspace(source, output)
    else:
        result = case_pipeline.run_case_pipeline(
            source,
            output_dir=output / "case",
            categories=("face", "license_plate"),
            adapters={category: SyntheticAdapter(category) for category in GAPS},
        )
    manifest = {
        "schema_version": "synthetic_demo.v1",
        "input_kind": SYNTHETIC,
        "source_sha256": media.sha256_file(source),
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "source": "synthetic_source.mp4",
        "workspace": None if skip_workspace else "case/review_workspace",
        "automatic": "case/combined_preview/automatic_preview.mp4",
        "model_inference_performed": False,
        "external_media_used": False,
        "omitted_observation_frames": {key: list(value) for key, value in GAPS.items()},
        "manual_exercise": {
            "category": "other_sensitive",
            "frame_start": MANUAL_START,
            "frame_end": FRAME_COUNT - 1,
            "bbox_xyxy": list(MANUAL_BOX),
            "instruction": "Add the checkerboard region in Manual redaction; "
            "Preview, Save, Build reviewed video, then verify Latest.",
        },
        "limitations": [
            "Not detector accuracy or real-video privacy evidence.",
            "Uses short bounded tracker interpolation, not optical-flow recovery.",
            "Profile IDs name pipeline configuration, not executed model inference.",
            "Timestamps and encoded bytes can differ across environments.",
        ],
        "rendered_region_count": result["render_plan"]["effective_region_count"],
    }
    if gif_path is not None:
        manifest["comparison_gif"] = write_comparison_gif(
            source, output / manifest["automatic"], gif_path
        )
    case_pipeline.write_json(output / "synthetic_demo.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/synthetic_demo"),
        help="new or empty directory (default: artifacts/synthetic_demo)",
    )
    parser.add_argument("--gif", type=Path, help="explicit destination for the INPUT/OUTPUT GIF")
    parser.add_argument("--overwrite-gif", action="store_true", help="replace the requested GIF")
    parser.add_argument(
        "--skip-workspace",
        action="store_true",
        help="run tracking/shared rendering without building a frontend review workspace",
    )
    args = parser.parse_args()
    manifest = generate_demo(
        args.output.resolve(),
        gif_path=args.gif.resolve() if args.gif is not None else None,
        skip_workspace=args.skip_workspace,
        overwrite_gif=args.overwrite_gif,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
