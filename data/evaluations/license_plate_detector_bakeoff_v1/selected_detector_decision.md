# License Plate Detector Bake-off v1.1

Status: no production plate detector selected.

This record freezes the bounded v1/v1.1 plate-detector bake-off evidence. It did
not change the Face Pipeline v1 and did not implement plate tracking, plate gap
recovery, plate rendering, or plate review UI behavior.

## Ground Truth Available

The plate benchmark evidence is range-level. It does not contain tight
per-frame plate boxes, so IoU and precise localization metrics are not
certified.

| Clip | Target | Scored frames | Available localization aid |
|---|---|---:|---|
| `day_plate_far_gold` | `plate_day_01` | 240 | Generous reference ROI `[650, 380, 1060, 570]` |
| `night_selective_plate_gold` | `plate_night_01` | 360 | Generous reference ROI `[60, 560, 500, 900]` |
| `day_plate_close_stress` | `plate_day_01` | not certified | No certified close-clip ROI or per-frame boxes |

The close-plate clip remains a stress/visual case only. `data/ground_truth/README.md`
states that the current day-close clip is not the frozen day-plate gold mapping.

## v1.1 Correction Scope

v1.1 audited the Intel YOLOv8n OpenVINO adapter and made one final general
small-object attempt with OpenVINO Barrier 0106.

Adapter audit result:

- the downloaded Intel IR input is `[1, 3, 640, 640]`;
- the output is `[1, 5, 8400]`;
- bundled metadata says `task: detect`, `nms: false`;
- IR runtime metadata contains `model_type=yolo_v8`, `scale_values=255`,
  `reverse_input_channels=YES`, and no built-in NMS;
- no angle channel is present, so the local IR is not an OBB tensor despite
  the public model-card wording;
- v1 did not fundamentally misdecode OBB output, but it failed to preserve
  pre-NMS diagnostics and did not document the OBB/axis-aligned mismatch.

The corrected adapter records the raw axis-aligned model geometry and the
derived conservative redaction bbox. Oriented geometry remains `null` for this
specific IR.

Exact DLStreamer `gvadetect` parity could not be run locally because the
`gvadetect` GStreamer plugin is not installed. The adapter was run on Intel's
official `ParkingVideo.mp4` sample, and visual/reference evidence was generated
against the published expected GIF.

## Candidates

| Candidate | Runtime | Source/license finding | Result |
|---|---|---|---|
| `openvino_vehicle_license_plate_barrier_0106_fp32` | OpenVINO CPU | Open Model Zoo / Apache-2.0 project license | v1 rejected: missed the frozen day plate entirely. |
| `openvino_vehicle_license_plate_barrier_0106_fp16_int8` | OpenVINO CPU | Open Model Zoo / Apache-2.0 project license | v1 rejected: missed the frozen day plate entirely. |
| `openvino_vehicle_license_plate_barrier_0106_fp32_tiled` | OpenVINO CPU | Open Model Zoo / Apache-2.0 project license | v1.1 rejected: general tiling improved day recall but remained too low. |
| `intel_yolov8n_lpr_openvino` | OpenVINO CPU | Hugging Face card has MIT tag; bundled detector metadata/notice state AGPL-3.0 for YOLOv8n | v1.1 rejected: high recall, but proposal noise remains excessive after audited postprocessing. |
| `opencv_lpd_yunet_2023mar` | OpenCV DNN | OpenCV Zoo / Apache-2.0 | Historical baseline only; existing evidence shows zero important daytime detections and sparse nighttime detections. |
| `openvino_vehicle_license_plate_barrier_0123` | TensorFlow/OpenVINO conversion candidate | Apache-2.0 original model manifest | Not run: installed OpenVINO 2026 converter could not convert the graph without the older Model Optimizer path. |

## Corrected Metrics at Threshold 0.20

The table uses target-ROI center hits for the two frozen benchmark intervals.
That is a selection proxy, not certified tight-box localization.

| Candidate | Day recall | Day longest miss | Night recall | Night longest miss | FP frames | Duplicate frames before/after NMS | Mean proposals before/after NMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| `intel_yolov8n_lpr_openvino` | 224 / 240 | 5 | 360 / 360 | 0 | day 240, night 360 | day 204/183, night 360/360 | day 87.600/43.642, night 189.592/86.136 |
| `openvino_vehicle_license_plate_barrier_0106_fp32_tiled` | 71 / 240 | 94 | 265 / 360 | 42 | day 96, night 27 | day 34/9, night 258/0 | day 1.371/0.904, night 2.292/0.811 |

The 0106 tiled experiment used one configuration across all clips:

- whole-frame inference plus uniform `960 px` tiles;
- `25%` tile overlap;
- cross-tile NMS IoU `0.45`;
- no target-specific crops, ROIs, frame numbers, or per-video thresholds.

## Decision

No detector is selected for the future active plate pipeline.

Intel YOLOv8n remains the closest recall candidate, but after corrected
postprocessing it still produces off-target proposals in every certified frame
and many duplicates after NMS. OpenVINO Barrier 0106 remains cleaner, but it
does not recover enough of the day plate target even with the one general
multi-scale/tiled mode.

No weak/primary plate threshold policy is selected.

## Rejected Alternatives

OpenVINO Barrier 0106 is rejected because it fails the most important day plate
case. The tiled variant improves from `0/240` to `71/240`, but that is still not
a usable detector foundation.

Intel YOLOv8n LPR is rejected because its recall is strong only with an
unmanageable proposal burden and an unresolved license ambiguity for production
selection. The package-level Hugging Face model card is MIT, while the bundled
detector metadata and third-party notices identify YOLOv8n as AGPL-3.0.

LPD-YuNet remains historical baseline evidence only. It should not be used as
the future active plate detector.

## Artifacts

Ignored bake-off packages:

- `artifacts/license_plate_detector_bakeoff_v1/`
- `artifacts/license_plate_detector_bakeoff_v1_1/`

Key v1.1 files:

- `intel_adapter_audit.json`
- `official_sample_parity.json`
- `metrics_summary.json`
- `threshold_comparison.csv`
- `selection_decision.json`
- `report.md`
- per-candidate detection JSONL, overlays, contact sheets, and comparison sheets

Model weights and official sample media remain ignored and are not tracked.

## Bounded Fallback Plan

The next direction is to fine-tune one permissively licensed detector
architecture, such as RT-DETR or YOLOX, on a clearly licensed plate dataset plus
project-specific annotations. Training is out of scope for the bake-off.

Minimum proposed preparation:

- collect at least `300-500` tight project plate boxes spanning day, night,
  small, oblique, motion-blurred, and occluded cases;
- split validation by source clip/scene, holding out at least `20%`;
- verify dataset, pre-trained checkpoint, model architecture, and training-code
  licenses before training;
- produce frozen validation detections, a threshold grid, missed/false-positive
  visual evidence, and an updated detector-selection record before plate
  tracking begins.
