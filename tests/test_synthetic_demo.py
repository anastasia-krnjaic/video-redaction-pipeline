from __future__ import annotations

import json
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
import pytest

from redactor import review_corrections, review_working_state, review_workspace
from scripts import generate_synthetic_demo as demo


@pytest.mark.parametrize("category", ["face", "license_plate"])
def test_synthetic_fixture_recovers_missing_observations_without_models(category: str) -> None:
    records, evidence = demo.temporal_fixture(category)
    repeated, _ = demo.temporal_fixture(category)
    assert [record.to_dict() for record in records] == [record.to_dict() for record in repeated]
    assert {record.source_frame for record in records} == set(range(demo.FRAME_COUNT))
    assert set(evidence["observed_frames"]).isdisjoint(demo.GAPS[category])
    recovered = [record for record in records if record.source_frame in demo.GAPS[category]]
    assert all("interpolat" in record.provenance for record in recovered)
    assert all(record.source_record["input_kind"] == demo.SYNTHETIC for record in records)


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def read_frame(path: Path, index: int) -> np.ndarray:
    capture = cv.VideoCapture(str(path))
    try:
        for _ in range(index + 1):
            ok, frame = capture.read()
            assert ok
        return frame
    finally:
        capture.release()


def test_generated_pixels_are_deterministic_bounded_and_not_manual_observations() -> None:
    for frame in range(demo.FRAME_COUNT):
        first, second = demo.source_frame(frame), demo.source_frame(frame)
        assert np.array_equal(first, second)
        assert first.shape == (demo.HEIGHT, demo.WIDTH, 3)
        for category in demo.GAPS:
            x1, y1, x2, y2 = demo.target_box(category, frame)
            assert 0 <= x1 < x2 <= demo.WIDTH
            assert 0 <= y1 < y2 <= demo.HEIGHT
            if frame >= demo.MANUAL_START:
                mx1, my1, mx2, my2 = demo.MANUAL_BOX
                assert x2 <= mx1 or x1 >= mx2 or y2 <= my1 or y1 >= my2
    for frame in (-1, demo.FRAME_COUNT):
        with pytest.raises(ValueError, match="frame must be"):
            demo.source_frame(frame)
    x1, y1, x2, y2 = demo.MANUAL_BOX
    before = demo.source_frame(demo.MANUAL_START - 1)[y1:y2, x1:x2]
    after = demo.source_frame(demo.MANUAL_START)[y1:y2, x1:x2]
    assert np.unique(before.reshape(-1, 3), axis=0).shape[0] == 1
    assert np.unique(after.reshape(-1, 3), axis=0).shape[0] == 2


def test_gif_uses_actual_rendered_frames_and_reproducible_encoding(tmp_path: Path) -> None:
    if not all(shutil.which(name) for name in ("ffmpeg", "ffprobe")):
        pytest.skip("synthetic GIF integration requires ffmpeg and ffprobe")
    output, gif = tmp_path / "demo", tmp_path / "comparison.gif"
    manifest = demo.generate_demo(output, gif_path=gif, skip_workspace=True)
    assert manifest["workspace"] is None
    assert not (output / "case/review_workspace").exists()
    assert manifest["model_inference_performed"] is False
    assert manifest["external_media_used"] is False
    assert manifest["rendered_region_count"] == 2 * demo.FRAME_COUNT
    assert manifest["comparison_gif"]["duration_seconds"] == 8.0
    assert manifest["comparison_gif"]["size_bytes"] < 5_000_000
    metadata = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=width,height,nb_frames:format=duration",
                "-of",
                "json",
                str(gif),
            ]
        )
    )
    stream = metadata["streams"][0]
    assert (stream["width"], stream["height"]) == (840, 416)
    assert int(stream["nb_frames"]) == 80
    assert float(metadata["format"]["duration"]) == 8.0

    for category, gap in demo.GAPS.items():
        frame = gap[0]
        decoded = read_frame(gif, frame // 2).astype(float)
        x1, y1, x2, y2 = demo.target_box(category, frame)
        x1, y1, x2, y2 = (round(value * 0.6) for value in (x1, y1, x2, y2))
        before = decoded[110 + y1 : 110 + y2, 24 + x1 : 24 + x2]
        after = decoded[110 + y1 : 110 + y2, 432 + x1 : 432 + x2]
        assert np.abs(before - after).mean() > 5
    repeated = demo.write_comparison_gif(
        output / manifest["source"], output / manifest["automatic"], tmp_path / "repeat.gif"
    )
    assert repeated["sha256"] == manifest["comparison_gif"]["sha256"]
    with pytest.raises(FileExistsError, match="GIF already exists"):
        demo.generate_demo(tmp_path / "refused", gif_path=gif, skip_workspace=True)
    assert not (tmp_path / "refused").exists()
    assert demo.media.sha256_file(gif) == repeated["sha256"]
    with pytest.raises(ValueError, match=".gif destination"):
        demo.generate_demo(tmp_path / "invalid", gif_path=tmp_path / "bad.mp4", skip_workspace=True)
    assert not (tmp_path / "invalid").exists()


def test_generated_workspace_renders_gaps_and_accepts_manual_correction(tmp_path: Path) -> None:
    if not all(shutil.which(name) for name in ("ffmpeg", "ffprobe")):
        pytest.skip("synthetic demo integration requires ffmpeg and ffprobe")
    if not (Path(__file__).resolve().parents[1] / "frontend/dist/index.html").is_file():
        pytest.skip("synthetic demo integration requires an already built frontend")
    output = tmp_path / "demo"
    manifest = demo.generate_demo(output)
    assert manifest["model_inference_performed"] is False
    assert manifest["rendered_region_count"] == 2 * demo.FRAME_COUNT
    automatic = output / manifest["automatic"]
    source = output / manifest["source"]
    for category, gap in demo.GAPS.items():
        frame = gap[1]
        x1, y1, x2, y2 = demo.target_box(category, frame)
        before = read_frame(source, frame)[y1:y2, x1:x2].astype(float)
        after = read_frame(automatic, frame)[y1:y2, x1:x2].astype(float)
        assert np.abs(before - after).mean() > 5
    with pytest.raises(FileExistsError):
        demo.generate_demo(output)

    package = output / manifest["workspace"]
    server = review_workspace.serve_workspace(package, port=0)
    server.materialization_jobs.debounce_seconds = 60
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    actor = {"reviewer_id": "synthetic-demo"}
    try:
        session = post_json(
            f"{base_url}/api/v1/sessions",
            {
                "reviewer": {"display_name": "Synthetic demo", **actor},
                "source_revision": "synthetic_demo.v1",
            },
        )["session"]
        correction = {
            "case_id": "case",
            "category": "other_sensitive",
            "operation": review_corrections.OP_ADD,
            "configuration_freeze_hash": session["configuration_freeze_hash"],
            "interpolation_method": "none",
            "keyframe_boxes": [
                {"bbox_xyxy": list(demo.MANUAL_BOX), "source_frame": demo.MANUAL_START}
            ],
            "reason": "Synthetic manual correction exercise",
            "reviewer_identity": actor,
            "source_frame_start": demo.MANUAL_START,
            "source_frame_end": demo.MANUAL_START,
            "source_video_hash": manifest["source_sha256"],
            "task_id": "manual_demo",
        }
        payload = {
            "actor": actor,
            "correction": correction,
            "session_id": session["session_id"],
            "reason": correction["reason"],
        }
        preview = post_json(f"{base_url}/api/v1/corrections/preview", payload)
        saved = post_json(
            f"{base_url}/api/v1/corrections",
            {
                **payload,
                "preview_token": preview["preview"]["preview_token"],
            },
        )
        assert saved["ok"] is True
        session_dir = package / "sessions" / session["session_id"]
        effective, _ = review_working_state.compute_effective_privacy_regions(
            package,
            session_dir,
            case_id="case",
        )
        manual = [record for record in effective if record.category == "other_sensitive"]
        assert [(record.source_frame, record.bbox_xyxy) for record in manual] == [
            (demo.MANUAL_START, demo.MANUAL_BOX),
        ]
    finally:
        server.shutdown()
        server.server_close()
