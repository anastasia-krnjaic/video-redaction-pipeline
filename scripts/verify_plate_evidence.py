"""Read-only checks for the compact plate detector evidence; standard library only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def compact_case(report: dict) -> dict:
    """Select counts and scope only; omit boxes, frame lists, and raw proposals."""
    scope = report["certified_scope"]
    target = report["per_target_metrics"][scope["target_id"]]
    require(report["mode"] == "detector", "source report is not detector evaluation")
    require(report["publishable_metrics"] is True, "source report is not publishable")
    require(report["validation"]["status"] == "ready", "source validation is not ready")
    require(not report["validation"]["blockers"], "source report has validation blockers")
    require(len(report["per_target_metrics"]) == 1, "expected one target per benchmark")
    return {
        "benchmark_id": scope["benchmark_id"],
        "target_id": scope["target_id"],
        "canonical_frame_namespace": scope["canonical_frame_namespace"],
        "dense_frame_interval": scope["dense_frame_interval"],
        "parent_evaluation_frame_interval": scope["parent_evaluation_frame_interval"],
        "evaluation_video_sha256": scope["evaluation_video"]["sha256"],
        "detector_scored_frames": target["detector_scored_frames"],
        "hit_frame_count": target["hit_frame_count"],
        "miss_frame_count": target["miss_frame_count"],
        "target_frame_recall": target["target_frame_recall"],
        "maximum_consecutive_miss_run": target["maximum_consecutive_miss_run"],
        "formal_precision": target["formal_precision"],
    }


def verify(local_source_run: Path | None = None) -> None:
    repo = Path(__file__).resolve().parents[1]
    evidence_dir = repo / "data/evaluations/plate_detector_oim_yolov9_s_608_v1"
    manifest = read_json(evidence_dir / "input_manifest.json")
    metrics = read_json(evidence_dir / "metrics.json")
    require(metrics["schema_version"] == "compact_plate_detector_evidence.v1", "wrong schema")
    require(metrics["mode"] == "detector", "wrong evaluation mode")
    require(sha256(evidence_dir / "metrics.json") == manifest["metrics_sha256"], "metrics hash")
    for name, record in manifest["tracked_inputs"].items():
        require(sha256(repo / record["path"]) == record["sha256"], f"input hash: {name}")
    gt = read_json(repo / manifest["tracked_inputs"]["ground_truth"]["path"])
    policy = read_json(repo / manifest["tracked_inputs"]["evaluation_policy"]["path"])
    config = read_json(repo / manifest["tracked_inputs"]["detector_configuration"]["path"])
    require(gt["content_sha256"] == metrics["ground_truth_content_sha256"], "GT identity")
    require(policy["content_sha256"] == metrics["evaluation_policy_sha256"], "policy identity")
    require(config["configuration_hash"] == metrics["configuration_hash"], "config identity")
    require(config["detector_id"] == metrics["detector_id"], "detector identity")
    require(config["model_sha256"] == metrics["model_sha256"], "model identity")
    for key in ("model_file", "model_revision", "source_repository", "inference_mode"):
        require(config[key] == metrics[key], f"model configuration: {key}")
    require(config["primary_threshold"] == metrics["primary_threshold"], "threshold identity")
    benchmarks = {b["benchmark_id"]: b for b in gt["benchmarks"]}
    hits = denominator = 0
    for case in metrics["cases"]:
        benchmark = benchmarks[case["benchmark_id"]]
        records = [r for r in benchmark["spatial_records"] if r["detector_scored"]]
        require({r["target_id"] for r in records} == {case["target_id"]}, "GT targets")
        require(len(records) == case["detector_scored_frames"], "GT denominator")
        for key in ("dense_frame_interval", "parent_evaluation_frame_interval"):
            require(case[key] == benchmark["frame_mapping"][key], f"GT mapping: {key}")
        require(
            case["evaluation_video_sha256"] == benchmark["evaluation_video"]["sha256"],
            "source video identity",
        )
        count = case["detector_scored_frames"]
        require(case["hit_frame_count"] + case["miss_frame_count"] == count, "count sum")
        require(round(case["hit_frame_count"] / count, 6) == case["target_frame_recall"], "rate")
        precision = {key: policy["formal_precision"][key] for key in ("value", "reason")}
        require(case["formal_precision"] == precision, "precision policy")
        hits += case["hit_frame_count"]
        denominator += count
    require(
        metrics["aggregate"]
        == {
            "hit_frame_count": hits,
            "detector_scored_frames": denominator,
            "miss_frame_count": denominator - hits,
            "micro_recall": round(hits / denominator, 6),
        },
        "aggregate arithmetic",
    )
    if local_source_run is not None:
        for record in manifest["local_source_reports"]:
            path = local_source_run / record["path_within_run"]
            require(sha256(path) == record["sha256"], f"source report hash: {path.name}")
            report = read_json(path)
            case = next(c for c in metrics["cases"] if c["benchmark_id"] == record["benchmark_id"])
            require(compact_case(report) == case, "compact record differs from source report")
            require(report["input_artifact_hashes"] == record["input_artifact_hashes"], "inputs")
            require(report["ground_truth_content_sha256"] == gt["content_sha256"], "source GT")
            require(report["evaluation_policy_sha256"] == policy["content_sha256"], "source policy")
        integrity = manifest["local_source_integrity"]
        path = local_source_run / integrity["path_within_run"]
        require(sha256(path) == integrity["sha256"], "source integrity report hash")
        require(read_json(path)["passed"] is True, "source integrity report failed")
    print("PASS: compact counts, tracked input hashes, benchmark scope, and arithmetic")
    if local_source_run is not None:
        print("PASS: exact original report hashes and compact derivation")
    else:
        print("Original local reports, detector inference, and media were not revalidated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-source-run", type=Path)
    args = parser.parse_args()
    try:
        verify(args.local_source_run)
    except (KeyError, OSError, ValueError, StopIteration) as error:
        parser.exit(1, f"FAIL: {error}\n")
