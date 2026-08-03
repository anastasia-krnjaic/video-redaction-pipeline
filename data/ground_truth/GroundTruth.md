# GroundTruth

The current human-readable face benchmark source is frozen in:

- `data/ground_truth/GroundTruth_v1.1.md`

The current selected machine-readable redaction-ground-truth file for Face Pipeline v1 automated evaluation is:

- `data/ground_truth/redaction_ground_truth_v1.0.json`

The JSON derives from GroundTruth v1.1 and `face_benchmark_v1.1.json` without adding new annotations, tight per-frame boxes, or semantic changes. If human-readable wording and derived JSON ever appear to conflict, treat the frozen versioned GroundTruth document as authoritative and create a new version instead of editing frozen files in place.
