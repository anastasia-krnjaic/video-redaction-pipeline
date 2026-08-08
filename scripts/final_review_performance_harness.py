#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tempfile
import time
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Final Review Workspace mutations.")
    parser.add_argument("--source-root", type=Path, default=Path("src"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--label", default="optimized")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))

    import cv2 as cv
    import numpy as np

    from redactor import case_manifests, media, privacy_entities, review_sessions, review_workspace
    from redactor.privacy_records import APPROVAL_ACCEPTED, CATEGORY_PLATE, PrivacyRegionRecord

    def write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def post_json(url: str, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        encoded = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body, (time.perf_counter() - started) * 1000.0

    def write_video(path: Path, *, frame_count: int = 8) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 64))
        try:
            for index in range(frame_count):
                frame = np.zeros((64, 96, 3), dtype=np.uint8)
                frame[:, :] = (50 + index * 6, 80, 130)
                cv.rectangle(frame, (20, 20), (54, 34), (240, 240, 240), -1)
                writer.write(frame)
        finally:
            writer.release()

    def setup_package(root: Path) -> tuple[Any, dict[str, Any], str]:
        source = root / "media" / "synthetic" / "original_reference.mp4"
        write_video(source)
        manifest = case_manifests.case_manifest_from_media(
            case_id="synthetic",
            display_name="Synthetic",
            source_path="media/synthetic/original_reference.mp4",
            source_video=source,
            video_roles=[
                case_manifests.VideoRole(
                    label="Original",
                    path="media/synthetic/original_reference.mp4",
                    role="original",
                    source_frame_end=7,
                    source_frame_start=0,
                )
            ],
        )
        automatic = root / "media" / "synthetic" / "automatic_baseline.mp4"
        shutil.copy2(source, automatic)
        manifest = replace(
            manifest,
            video_roles=(
                manifest.video_roles[0],
                case_manifests.VideoRole(
                    label="Automatic",
                    path=automatic.relative_to(root).as_posix(),
                    role="automatic",
                    sha256=media.sha256_file(automatic),
                    source_frame_end=manifest.source_frame_end,
                    source_frame_start=manifest.source_frame_start,
                ),
            ),
        )
        case_manifests.write_case_manifest(root / "cases" / "synthetic.json", manifest)
        record = PrivacyRegionRecord(
            approval_state=APPROVAL_ACCEPTED,
            bbox_xyxy=(20, 20, 54, 34),
            category=CATEGORY_PLATE,
            object_id="plate_1",
            provenance="plate_detector_observation",
            render_eligible=True,
            source_frame=0,
            source_record={"region_id": "region_1", "track_uid": "plate_1"},
        )
        from redactor import review_finalization

        review_finalization.write_privacy_region_records(
            root / "regions" / "synthetic_current_regions.jsonl",
            [record],
        )
        entity = {
            "automatic_source_track_ids": ["plate_1"],
            "case_id": "synthetic",
            "category": CATEGORY_PLATE,
            "current_policy": privacy_entities.POLICY_REDACT,
            "default_privacy_policy": privacy_entities.POLICY_REDACT,
            "entity_id": "entity_plate_1",
            "first_visible_source_frame": 0,
            "label": "Plate 01",
            "last_visible_source_frame": 0,
            "member_detector_tracker_ids": ["plate_1"],
            "privacy_action": privacy_entities.POLICY_REDACT,
        }
        write_json(root / "review_entities.json", {"cases": {"synthetic": [entity]}})
        write_json(root / "reviewer_tasks.json", {"tasks": []})
        write_json(root / "workspace_manifest.json", {"case_count": 1})
        freeze = {"schema_version": "benchmark.freeze.v1"}
        write_json(root / "configuration_freeze.json", freeze)
        session = review_sessions.create_review_session(
            package_root=root,
            reviewer=review_sessions.ReviewerIdentity(None, None),
            source_revision="benchmark",
            configuration_freeze_hash=review_workspace.stable_hash(freeze),
            case_manifest_hashes={"synthetic": manifest.manifest_hash()},
            source_hashes={"synthetic": manifest.source_video_hash or ""},
        )
        return manifest, entity, session.session_id

    def run_operation(name: str, operation: str) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix=f"frw_{name}_") as tmp:
            package = Path(tmp) / "package"
            manifest, entity, session_id = setup_package(package)
            server = review_workspace.serve_workspace(package, port=0)
            server.materialization_jobs.debounce_seconds = 60
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                if operation.startswith("decision:"):
                    action = operation.split(":", 1)[1]
                    response, latency = post_json(
                        f"{base_url}/api/v1/decisions",
                        {
                            "action": action,
                            "actor": {"reviewer_id": "benchmark"},
                            "case_id": manifest.case_id,
                            "note": "benchmark",
                            "session_id": session_id,
                            "task_id": f"task_{name}",
                        },
                    )
                elif operation == "policy_noop":
                    response, latency = post_json(
                        f"{base_url}/api/v1/object-policy",
                        {
                            "actor": {"reviewer_id": "benchmark"},
                            "case_id": manifest.case_id,
                            "entity_id": entity["entity_id"],
                            "policy": privacy_entities.POLICY_REDACT,
                            "session_id": session_id,
                        },
                    )
                elif operation == "policy_keep":
                    response, latency = post_json(
                        f"{base_url}/api/v1/object-policy",
                        {
                            "actor": {"reviewer_id": "benchmark"},
                            "case_id": manifest.case_id,
                            "entity_id": entity["entity_id"],
                            "policy": privacy_entities.POLICY_KEEP_VISIBLE,
                            "session_id": session_id,
                        },
                    )
                elif operation == "policy_repeat_10":
                    latencies = []
                    response = {}
                    for index in range(10):
                        response, latency = post_json(
                            f"{base_url}/api/v1/object-policy",
                            {
                                "actor": {"reviewer_id": "benchmark"},
                                "case_id": manifest.case_id,
                                "entity_id": entity["entity_id"],
                                "policy": privacy_entities.POLICY_KEEP_VISIBLE,
                                "session_id": session_id,
                            },
                        )
                        latencies.append(latency)
                    latency = sum(latencies)
                else:
                    raise ValueError(operation)
                jobs = dict(server.materialization_jobs.jobs)
                plan = dict(response.get("materialization_plan") or {})
                return {
                    "api_ack_ms": round(latency, 3),
                    "changed_source_frame_count": plan.get("changed_source_frame_count"),
                    "changed_source_frame_intervals": plan.get("changed_source_frame_intervals"),
                    "job_count": len(jobs),
                    "jobs_started": sum(
                        1 for job in jobs.values() if job.get("status") == "running"
                    ),
                    "materialization_required": bool(response.get("materialization_required")),
                    "materialization_strategy": plan.get("materialization_strategy"),
                    "operation": name,
                    "planned_frames_to_encode": plan.get("planned_frames_to_encode"),
                    "planned_frames_to_render": plan.get("planned_frames_to_render"),
                    "queued_jobs": sum(1 for job in jobs.values() if job.get("status") == "queued"),
                    "skipped_jobs": sum(
                        1
                        for job in jobs.values()
                        if job.get("status") == "skipped_no_effective_region_change"
                    ),
                    "success": bool(response.get("ok")),
                }
            finally:
                server.shutdown()
                server.server_close()

    operations = {
        "decision_correct": "decision:looks_correct",
        "decision_needs_correction": "decision:privacy_leak",
        "decision_unsure": "decision:uncertain_escalate",
        "policy_noop_redact": "policy_noop",
        "policy_keep_visible": "policy_keep",
        "policy_repeat_same_entity_10": "policy_repeat_10",
    }
    samples: list[dict[str, Any]] = []
    for name, operation in operations.items():
        for _ in range(max(1, args.iterations)):
            samples.append(run_operation(name, operation))
    summary: dict[str, Any] = {}
    for name in operations:
        rows = [row for row in samples if row["operation"] == name]
        latencies = sorted(float(row["api_ack_ms"]) for row in rows)
        p95_index = min(len(latencies) - 1, int(round((len(latencies) - 1) * 0.95)))
        summary[name] = {
            "api_ack_ms_p50": round(statistics.median(latencies), 3),
            "api_ack_ms_p95": round(latencies[p95_index], 3),
            "iterations": len(rows),
            "materialization_required_count": sum(
                1 for row in rows if row["materialization_required"]
            ),
            "changed_source_frame_counts": sorted(
                int(row["changed_source_frame_count"])
                for row in rows
                if row.get("changed_source_frame_count") is not None
            ),
            "queued_job_count": sum(int(row["queued_jobs"]) for row in rows),
            "skipped_job_count": sum(int(row["skipped_jobs"]) for row in rows),
        }
    payload = {
        "label": args.label,
        "operations": summary,
        "samples": samples,
        "schema_version": "final_review_performance_metrics.v1",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [f"# Final Review Performance Metrics ({args.label})", ""]
    lines.append("| Operation | p50 API ack ms | p95 API ack ms | queued jobs | skipped jobs |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for name, row in summary.items():
        lines.append(
            f"| {name} | {row['api_ack_ms_p50']} | {row['api_ack_ms_p95']} | "
            f"{row['queued_job_count']} | {row['skipped_job_count']} |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
