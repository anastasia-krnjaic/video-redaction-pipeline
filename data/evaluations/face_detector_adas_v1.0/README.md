# ADAS Face Detector Evaluation v1.0

This directory is the tracked, reproducible evidence record for selecting `openvino_face_adas_0001` as the provisional primary face detector for the evaluated face clips. It registers the completed ADAS-only identity repair and certified raw-detector recall results derived from GroundTruth v1.1.

No inference was rerun to create this record. The completed adjudication and correction log were copied byte-for-byte from the completed ignored ADAS evaluation package, then validated against GroundTruth v1.1 and canonical ADAS detections.

## Contents

| File | Purpose |
| --- | --- |
| `input_manifest.json` | Relative input paths, hashes, copied evidence hashes, and provenance. |
| `metrics.json` | Certified ADAS raw-detector metrics with corrected FPS provenance fields. |
| `selected_thresholds.json` | Selected detector policy and threshold rationale. |
| `selected_detector_decision.md` | Human-readable detector decision record. |
| `adjudication.completed.corrected.json` | Byte-for-byte copy of the corrected ADAS adjudication export. |
| `correction_log.json` | Byte-for-byte copy of the ADAS correction log. |
| `validation.json` | Validation summary for this tracked evidence package. |

This record intentionally excludes video, frame images, model binaries, browser state, generated overlays, and absolute local paths.

## Decision

Selected primary detector: `openvino_face_adas_0001` (OpenVINO face-detection-adas-0001).

Selected detector policy:

- Proposal/storage threshold: `0.20`.
- Primary detector threshold: `0.30`.
- Weak proposals from `0.20` through below `0.30` may support temporal tracking but must not independently confirm a track.

YuNet and OpenVINO face-detection-0205 remain comparison baselines. They were not fully re-adjudicated after ADAS selection. This is an engineering selection for the evaluated operating domain, not a universal detector ranking.

## Certified Primary-Threshold Results

At threshold `0.30`:

| Target | Hits | Denominator | Recall |
| --- | ---: | ---: | ---: |
| `person_day_01` | 299 | 300 | 0.996667 |
| `person_night_01` | 359 | 360 | 0.997222 |
| `person_night_02` | 175 | 192 | 0.911458 |
| Micro | 833 | 852 | 0.977700 |

Worst-target recall is `0.911458`. Maximum consecutive misses are day `1`, standing night `1`, seated night `9`.

At threshold `0.20`, ADAS has `834 / 852` micro hits: `299 / 300`, `359 / 360`, and `176 / 192`. The maximum seated-night miss run remains `9`.

## FPS Provenance Audit

The historical ignored final ADAS report records `128.294961` FPS. Inspection shows that value is the arithmetic mean of ADAS tracking validation processing rates, not detector inference throughput.

Explicit throughput fields in this tracked record are:

| Field | Value | Source |
| --- | ---: | --- |
| `detector_inference_fps_day` | 61.512487 | `artifacts/face_detector_bakeoff_v2/adas_day/summary.json` |
| `detector_inference_fps_night` | 66.489572 | `artifacts/face_detector_bakeoff_v2/adas_night/summary.json` |
| `tracking_or_postprocessing_fps_day` | 118.293567 | `artifacts/adas_tracking_validation_v2/day_face/summary.json` |
| `tracking_or_postprocessing_fps_night` | 138.296354 | `artifacts/adas_tracking_validation_v2/night_face/summary.json` |

The ignored historical report was not altered. This tracked record separates the throughput families so tracking/postprocessing throughput is not labeled as detector inference throughput.

## Limitations

These metrics measure raw detector recall on GroundTruth v1.1 detector-scored frames. They do not measure tracking, interpolation, prediction, identity persistence, final rendering, final redaction coverage, or zero-leak performance. Raw detector misses still require temporal recovery and reviewed rendering evidence in later phases.
