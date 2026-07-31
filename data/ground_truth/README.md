# Ground-Truth Benchmarks

`GroundTruth_v1.0.md` is the immutable semantic source of truth for the frozen benchmark version. Do not edit a frozen ground-truth version in place.

`face_benchmark_v1.0.json` is the machine-readable face-evaluation mirror of the frozen Markdown. The Markdown remains authoritative if a human-readable statement and the JSON appear to disagree.

A new semantic version is required if mappings, targets, actions, scored ranges, ignore ranges, dense intervals, or acceptance rules change.

Detector recall and final redaction coverage are separate metrics. Detector recall is scored only on detector-scored frames and requires visual target correspondence. Final redaction coverage may use reviewed tracking and interpolation.

Reference ROIs are search aids, not tight scoring boxes. Exact detector-hit adjudication requires visual target correspondence.

The current day-plate gold interval describes the far-plate clip, not `eval_day_plate_close.mp4`.

License-plate text must never be stored.
