# Ground-Truth Benchmarks

Versioned `GroundTruth_*.md` files are immutable semantic sources of truth for frozen benchmark versions. Do not edit a frozen ground-truth version in place.

| Version | Status | Meaning |
|---|---|---|
| v1.0 | Frozen, superseded for face detector evaluation | Initial reference registered before complete face-visibility verification. It remains immutable historical evidence. |
| v1.1 | Current selected face benchmark | Corrected face benchmark after complete human verification of 660 unique face frames and 1,020 target decisions. |

`face_benchmark_v1.1.json` is the current machine-readable face-evaluation mirror of `GroundTruth_v1.1.md`. The Markdown remains authoritative if a human-readable statement and the JSON appear to disagree.

`redaction_ground_truth_v1.0.json` is the selected machine-readable redaction-ground-truth file for Face Pipeline v1 automated evaluation. It derives from GroundTruth v1.1 and `face_benchmark_v1.1.json` without adding new boxes or changing semantics. Its companion schema description is `redaction_ground_truth_schema_v1.0.json`.

`plate_ground_truth_v1.0.json` is the selected machine-readable license-plate spatial Ground Truth
for the certified `day_plate` and `night_selective` windows. It contains 600 proposal-assisted
human-reviewed plate-panel boxes: 240 day records and 360 night records. Its companion schema is
`plate_ground_truth_schema_v1.0.json`, and its provenance binds back to the completed annotation
review export from `artifacts/plate_ground_truth_annotation_v2_assisted/run_20260811T185326Z`.

The plate Ground Truth covers only `artifacts/eval_day_plate.mp4` parent frames `240-479` and
`artifacts/eval_night_selective.mp4` parent frames `420-779`. The close plate clip
`artifacts/eval_day_plate_close.mp4` is excluded from certified plate scoring annotations.

A new semantic version is required if mappings, targets, actions, scored ranges, ignore ranges, dense intervals, or acceptance rules change.

Detector recall and final redaction coverage are separate metrics. Detector recall is scored only on detector-scored frames and requires visual target correspondence. Final redaction coverage may use reviewed tracking and interpolation.

Automated evaluation must keep detector, ordinary tracking, synthetic privacy recovery, reviewed/preserved corrections, and rendering evidence separate.

Reference ROIs are search aids, not tight scoring boxes. Exact detector-hit adjudication requires visual target correspondence.

The current day-plate gold interval describes the far-plate clip, not `eval_day_plate_close.mp4`.

Plate metrics remain future work. The current `evaluate-plates` path is still capability-only and
must not publish recall or privacy-coverage numbers until a separate plate evaluator phase is
implemented and audited.

License-plate text must never be stored.
