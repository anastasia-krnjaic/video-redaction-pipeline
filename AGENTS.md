# Repository Guidance

## System Overview Maintenance

`docs/SYSTEM_OVERVIEW.md` is the canonical human-readable description of the current pipeline. Any task that materially changes the pipeline must update `docs/SYSTEM_OVERVIEW.md` as part of the same change. Before changing the canonical overview, preserve the previous canonical document as the next sequential immutable file under `docs/system_overview_versions/`, then update `docs/system_overview_versions/CHANGELOG.md`.

Every archived version must preserve the implementation sources and configuration sources on which it was based. Archived versions must never be edited retroactively. Do not create a new archived version for spelling, grammar, or formatting corrections that do not change the documented system behavior.

A material system change includes any change to detector model or model version, model selection status, confidence thresholds, shared detection logic, track association, track confirmation criteria, interpolation or prediction rules, analyst decisions or manual override workflow, identity grouping, rendering eligibility, redaction rendering, audit records or manifests, output artifacts, evaluation methodology, benchmark clips, or production workflow.

## Ground-Truth Benchmark Rules

Default selected face benchmark: `data/ground_truth/face_benchmark_v1.1.json`.

1. Versioned ground-truth files under `data/ground_truth/` are frozen sources of truth once created.
2. Never edit a frozen ground-truth version in place.
3. Every evaluation must validate ground-truth, video, and detection hashes.
4. Zero-leak claims apply only to GOLD_DENSE intervals.
5. Detector-ignore frames are excluded from raw detector recall.
6. Tracking, prediction, and interpolation do not count as raw detector hits.
7. Final redaction coverage may use reviewed tracking and interpolation.
8. Reference ROIs may be used for candidate search but not as automatic scoring boxes.
9. New ground-truth semantics require a new version.
10. Future detector, tracker, renderer, and evaluation work must use the newest explicitly selected ground-truth version.
