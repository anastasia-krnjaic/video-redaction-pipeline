# Repository Guidance

## System Overview Maintenance

`docs/SYSTEM_OVERVIEW.md` is the canonical human-readable description of the current pipeline. Any task that materially changes the pipeline must update `docs/SYSTEM_OVERVIEW.md` as part of the same change. Before changing the canonical overview, preserve the previous canonical document as the next sequential immutable file under `docs/system_overview_versions/`, then update `docs/system_overview_versions/CHANGELOG.md`.

Every archived version must preserve the implementation sources and configuration sources on which it was based. Archived versions must never be edited retroactively. Do not create a new archived version for spelling, grammar, or formatting corrections that do not change the documented system behavior.

A material system change includes any change to detector model or model version, model selection status, confidence thresholds, shared detection logic, track association, track confirmation criteria, interpolation or prediction rules, analyst decisions or manual override workflow, identity grouping, rendering eligibility, redaction rendering, audit records or manifests, output artifacts, evaluation methodology, benchmark clips, or production workflow.

## Ground-Truth Benchmark Rules

Default selected face benchmark: `data/ground_truth/face_benchmark_v1.1.json`.

Default selected face detector evidence record: `data/evaluations/face_detector_adas_v1.0/`.

Selected face detector policy:

- Primary detector: `openvino_face_adas_0001`.
- Proposal/storage threshold: `0.20`.
- Primary detector threshold: `0.30`.
- Weak proposals from `0.20` through below `0.30` may support temporal tracking but must not independently confirm a track.
- ADAS is the only active face detector backend. Historical YuNet and OpenVINO 0205 comparison material must stay in the archive/evidence record and must not be reintroduced into the active runtime path without a new explicit model-selection task.

1. Versioned ground-truth files under `data/ground_truth/` are frozen sources of truth once created.
2. Never edit a frozen ground-truth version in place.
3. Every evaluation must validate ground-truth, video, and detection hashes.
4. Zero-leak claims apply only to GOLD_DENSE intervals.
5. Detector-ignore frames are excluded from raw detector recall.
6. Tracking, prediction, and interpolation do not count as raw detector hits.
7. Final redaction coverage may use reviewed tracking and interpolation.
8. Reference ROIs may be used for candidate search but not as automatic scoring boxes.
9. New ground-truth semantics require a new version.
10. Raw ADAS detector misses still require temporal recovery and reviewed rendering evidence; final zero-leak claims have not yet been established.
11. OpenCV YuNet and OpenVINO face-detection-0205 are historical comparison models, not active production face detector backends.
12. Future detector, tracker, renderer, and evaluation work must use the newest explicitly selected ground-truth version and selected face-detector evidence record.
13. Face privacy gap recovery is a downstream continuity layer, not detector recall. It must preserve provenance, separate detector/ordinary-tracker evidence from synthetic privacy regions, and emit `needs_review` or `rejected` intervals when bidirectional, one-sided, scene-cut, keep-overlap, identity-competition, jump, scale, or validity checks fail.
14. The optional bottom-connected visual-component refinement is strategy-specific and disabled by default; do not encode clip-specific frame numbers, person IDs, or coordinates in production recovery logic.
