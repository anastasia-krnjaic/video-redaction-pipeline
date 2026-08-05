# License Plate Detector Decision: OIM YOLOv9-S-608 v1

Status: `selected_for_prototype_plate_pipeline`.

Selected detector: Open Image Models YOLOv9-S-608 end-to-end license-plate detector.

This decision supersedes the earlier v2.1 metric-only RT-DETR selection after manual
full-video visual adjudication. RT-DETR had higher measured recall, but it also
produced a destructive false-redaction burden. OIM YOLOv9-S-608 was selected because
it provides cleaner, plate-focused localization with a practical review burden.

## Selected Configuration

| Field | Value |
| --- | --- |
| Detector ID | `oim_yolov9_s_608` |
| Source | `ankandrew/open-image-models` |
| Source revision | `f22000e02b30642f317cdba7755c0631638b109e` |
| Model file | `yolo-v9-s-608-license-plates-end2end.onnx` |
| Model SHA-256 | `2b878b38d9aa07b6ddc3ea75c4ffcb39869bc5c218e0a14002f60ab2f7b0be9a` |
| License finding | MIT repository license; no separate weight-specific notice found |
| Inference mode | `uniform_small_object_tiles` |
| Primary threshold | `0.30` |
| Proposal threshold | none selected |
| Tile size / overlap | `1280`, `0.25` |
| Cross-tile NMS IoU | `0.45` |
| Configuration hash | `7d914438255905e2a249e5bbd30fd6e70eace1e91b57e205bd572d004ffb5a55` |

## Benchmark Clips

| Clip | Source | SHA-256 | Evaluated range | Role |
| --- | --- | --- | --- | --- |
| `day_plate_primary` | `artifacts/eval_day_plate_close.mp4` | `4f19396fa1393c2861bfd26302b3147fc8ae29acfd55bf10f6e7be46a30231c9` | `0-239` | Primary daytime benchmark |
| `night_plate_primary` | `artifacts/eval_night_selective.mp4` | `0b55aa3a497941711ac8634a049c557637e20928e8d3bd8b63e5b030b6b272e0` | `420-779` | Night benchmark |
| `day_plate_close_stress` | `artifacts/eval_day_plate.mp4` | `63d41905e2b6c65a3cdc72aa73ee31668fe1c1b4fdfefa2c5688cd38c90d1097` | `240-479` | Close/stress visual clip |

## Finalist Metrics

| Metric | RT-DETR v2 | OIM YOLOv9-S-608 |
| --- | ---: | ---: |
| All-visible recall | `0.898936` | `0.861702` |
| Precision | `0.307552` | `0.958580` |
| True non-plate detections | `553` | `13` |
| Duplicate detections | `208` | `1` |
| Night target recall | `0.850000` | `0.730556` |
| Longest night target gap | `14` | `41` |

## Calibration Result

The selected OIM threshold remains `0.30`. Narrow global calibration showed that
lowering the threshold to `0.20` or `0.25` increased false detections without
shortening the certified night target gap. Raising the threshold reduced false
detections only by losing recall. No separate lower proposal threshold is selected
for this detector-only milestone.

## Rejected Alternatives

RT-DETR v2 is rejected from the active path because its false non-plate detection
burden and duplicate burden would cause too much destructive redaction and review
work despite higher recall.

RF-DETR Medium is rejected because visual review showed an excessive non-plate
detection burden.

Intel YOLOv8n and OpenVINO 0106/0123 remain historical negative evidence from the
earlier bake-offs. OpenCV LPD-YuNet remains only a historical baseline.

## Limitations

This is a prototype detector selection, not a production-ready plate subsystem.
OIM still has nighttime gaps, including the certified longest target gap of 41
frames. Plate tracking, bounded interpolation, review handling, and final plate
rendering remain the next stages.

## Reproducibility

Calibration evidence is in `artifacts/oim_plate_detector_calibration_v1/`.
Historical comparison evidence is retained under
`docs/archive/license_plate_detector_bakeoff/` and
`experiments/archive/license_plate_detector_bakeoff/`.
