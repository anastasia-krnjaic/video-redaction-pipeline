# Face Detector Bake-Off Archive

This archive preserves the history of the face-detector model comparison that led to the current Face Pipeline v1 detector decision.

## What Was Evaluated

Three face detector candidates were compared on the day and night face clips:

- OpenVINO `face-detection-adas-0001`
- OpenCV YuNet
- OpenVINO `face-detection-0205`

The comparison started as a visual bake-off and was followed by a focused ADAS-only GroundTruth v1.1 validation after ADAS was selected as the provisional primary detector.

## Current Decision

The active face detector is OpenVINO `face-detection-adas-0001`, recorded in runtime outputs as `openvino_face_adas_0001`.

OpenCV YuNet and OpenVINO `face-detection-0205` are no longer active face detector backends. They are retained only as historical comparison context and must not be treated as supported production options.

## Evidence Locations

Tracked evidence:

- `data/evaluations/face_detector_adas_v1.0/`

Ignored historical artifacts, when present locally:

- `artifacts/face_detector_bakeoff/`
- `artifacts/face_detector_bakeoff_v2/`
- `artifacts/baseline/`
- `artifacts/face_ground_truth_eval_v1_1/`

These ignored directories may contain generated detections, comparison summaries, review packages, overlays, and model copies used during the evaluation process. They are intentionally not moved into tracked Git history.

## Supported Workflow Boundary

Archived commands and scripts from the bake-off are not part of the supported Face Pipeline v1 workflow. The current supported entry point is `redactor face-pipeline`, which uses ADAS only for face detection.

This archive is for reproducibility and historical context, not for active detector selection.
