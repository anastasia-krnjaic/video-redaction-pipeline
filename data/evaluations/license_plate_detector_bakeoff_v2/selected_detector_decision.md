# License Plate Detector Bake-off v2

Status: `no_detector_selected`.

This record freezes the bounded v2 plate-detector bake-off. It did not change
the active Face Pipeline, did not reuse the failed Intel YOLO/OpenVINO 0106
models as active candidates, and did not implement plate tracking, rendering,
gap recovery, or review UI behavior.

## Clip Mapping

| Clip | Role | Source | SHA-256 | Frames Scored | Quantitative Note |
|---|---|---|---|---:|---|
| `day_plate_primary` | Primary daytime benchmark | `artifacts/eval_day_plate_close.mp4` | `4f19396fa1393c2861bfd26302b3147fc8ae29acfd55bf10f6e7be46a30231c9` | 240 | Officer-near-vehicle scene. No validated per-frame plate boxes or valid reference ROI exist for this derivative clip, so target recall is not certified. |
| `night_plate_primary` | Nighttime benchmark | `artifacts/eval_night_selective.mp4` | `0b55aa3a497941711ac8634a049c557637e20928e8d3bd8b63e5b030b6b272e0` | 360 | Uses the existing generous `plate_night_01` search ROI. This is not tight-box localization truth. |
| `day_plate_close_stress` | Driving-behind stress clip | `artifacts/eval_day_plate.mp4` | `63d41905e2b6c65a3cdc72aa73ee31668fe1c1b4fdfefa2c5688cd38c90d1097` | 240 | Qualitative stress run only. |

The previous v1/v1.1 visualization used the driving-behind clip as the
quantitative day benchmark. Bake-off v2 corrects that mapping and treats the
driving-behind clip as stress evidence only.

## Candidates

| Candidate | Source Revision | License Finding | Runtime |
|---|---|---|---|
| `open_image_models_yolov9_s_608` | `ankandrew/open-image-models` @ `f22000e02b30642f317cdba7755c0631638b109e` | MIT repository license; no separate local weight notice found. | OpenVINO CPU on ONNX. |
| `rtdetr_v2_license_plate` | `justjuu/rtdetr-v2-license-plate-detection` @ `0ca7598f58649fe4a27db22ad72dca7be70314b1` | Apache-2.0 model card; no separate local LICENSE file in clone. | Hugging Face Transformers / PyTorch, MPS when available. |
| `rfdetr_medium_alpr` | `autolane/rfdetr-alpr` @ `851dc6b2db275bca2dec61a16f6d9be8c026fdc9` | Apache-2.0 model card; README says same as RF-DETR. | OpenVINO CPU on `rfdetr_alpr.onnx`. |

## Repository Sample Inference

Open Image Models shipped a local repository test asset
`test/assets/car_image.webp`; the v2 adapter loaded it and produced a plausible
plate box at score `0.919922`. No official expected-output file was present, so
this is a smoke/parsing check rather than numeric parity.

The RT-DETR v2 and RF-DETR Medium ALPR local repository clones did not include
sample media, so their sample status is recorded as `not_available`.

## Shared Inference Policy

Every candidate was evaluated with the same threshold grid:

`0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70`

Two standardized modes were evaluated for each candidate:

- native whole-frame inference at the model's documented input size;
- whole-frame plus uniform `1280 px` overlapping tiles with `25%` overlap,
  merged back into source coordinates by NMS.

No clip-specific thresholds, coordinates, ROIs, or frame rules were introduced.

## Metrics At Each Candidate's Best Consistent Mode

Day proposal proxy is not certified target recall; it only reports whether at
least one final plate proposal existed in a visible-frame range without
validated target boxes.

| Candidate | Selected Mode For Analysis | Threshold | Day Proposal Proxy | Night ROI Recall | Night Longest Miss | Night False/Off-Target Frames | Night Duplicate Frames | Processing FPS Range |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `open_image_models_yolov9_s_608` | `uniform_small_object_tiles` | 0.30 | 93 / 240 | 263 / 360 | 41 | 81 | 1 | 2.09-2.19 |
| `rtdetr_v2_license_plate` | `native_whole_frame` | 0.05 | 240 / 240 | 306 / 360 | 14 | 188 | 145 | 1.64-1.99 |
| `rfdetr_medium_alpr` | `uniform_small_object_tiles` | 0.05 | 240 / 240 | 359 / 360 | 1 | 360 | 349 | 0.85-1.64 |

## Visual Findings

Open Image Models YOLOv9-s was comparatively clean but missed too many
target-visible intervals, especially at night and through much of the corrected
primary day clip.

RT-DETR v2 produced proposals through the day clip but missed too many night ROI
frames and retained a large off-target/duplicate burden.

RF-DETR Medium ALPR had the strongest night ROI recall, but the overlays show a
systematic flood of off-target proposals on the seated person, road markings,
signage, and other background structures. That false burden is not a practical
starting point for plate tracking or review.

## Decision

No detector is selected for the future active plate pipeline.

No candidate satisfies all v2 acceptance criteria:

- strong object-frame evidence on the corrected primary daytime benchmark;
- strong night object-frame recall;
- manageable longest missed runs;
- low enough false/off-target burden for tracking and review;
- no systematic duplicate flood;
- usable conservative localization;
- practical local runtime;
- acceptable documented license path.

## Limitations

The plate evidence remains range-level for the current day and stress clips. IoU,
center-distance error, width/height error, and certified daytime target recall
cannot be reported without reviewed localization boxes.

Model-card metrics were used only as source context. They were not treated as
selection evidence on the body-camera clips.

## Next Step

Do not start plate tracking from these detectors. The bounded next step is to
create a small, reviewed plate-localization annotation set for the project clips
and fine-tune one permissively licensed detector architecture, such as RT-DETR
or YOLOX, after dataset and checkpoint licenses are verified.

Minimum preparation:

- collect at least `300-500` tight project plate boxes spanning day, night,
  small, oblique, motion-blurred, and occluded cases;
- split validation by source clip/scene, holding out at least `20%`;
- preserve model/data/license audit records;
- rerun a frozen threshold grid with missed/false-positive visual evidence
  before any plate tracking work begins.

Ignored artifact package:

`artifacts/license_plate_detector_bakeoff_v2/`
