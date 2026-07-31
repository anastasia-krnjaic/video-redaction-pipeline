# Ground-Truth Benchmarks

Versioned `GroundTruth_*.md` files are immutable semantic sources of truth for frozen benchmark versions. Do not edit a frozen ground-truth version in place.

| Version | Status | Meaning |
|---|---|---|
| v1.0 | Frozen, superseded for face detector evaluation | Initial reference registered before complete face-visibility verification. It remains immutable historical evidence. |
| v1.1 | Current selected face benchmark | Corrected face benchmark after complete human verification of 660 unique face frames and 1,020 target decisions. |

`face_benchmark_v1.1.json` is the current machine-readable face-evaluation mirror of `GroundTruth_v1.1.md`. The Markdown remains authoritative if a human-readable statement and the JSON appear to disagree.

A new semantic version is required if mappings, targets, actions, scored ranges, ignore ranges, dense intervals, or acceptance rules change.

Detector recall and final redaction coverage are separate metrics. Detector recall is scored only on detector-scored frames and requires visual target correspondence. Final redaction coverage may use reviewed tracking and interpolation.

Reference ROIs are search aids, not tight scoring boxes. Exact detector-hit adjudication requires visual target correspondence.

The current day-plate gold interval describes the far-plate clip, not `eval_day_plate_close.mp4`.

License-plate text must never be stored.
