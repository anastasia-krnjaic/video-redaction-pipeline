# License Plate Detector Bake-off Archive

This archive preserves the license-plate detector research that preceded the
current selected detector. The archived commands and modules are historical
reproducibility material, not supported production workflow.

## Why This Exists

License-plate localization was evaluated separately from the frozen Face Pipeline
because early plate detectors either missed important plates or produced too much
false redaction. The goal was to find one practical local detector before adding
plate tracking, gap recovery, review, or rendering.

## Models Evaluated

- OpenCV LPD-YuNet: historical baseline. It missed the important daytime plate and
  produced sparse or false nighttime detections.
- OpenVINO vehicle-license-plate-detection-barrier-0106 and 0123: inadequate on
  the primary daytime plate benchmark, including the tiled 0106 follow-up.
- Intel YOLOv8n plate model: high raw recall but extreme false/off-target and
  duplicate proposal burden. Temporal stabilization did not make it suitable.
- RF-DETR Medium ALPR: rejected after visual review because of excessive
  non-plate detections.
- RT-DETR v2 license-plate detector: strong recall, but the final all-visible
  adjudication showed too many true non-plate detections and duplicates for a
  safe prototype redaction path.
- Open Image Models YOLOv9-S-608: selected after manual finalist review because
  it provided the cleanest plate-localization behavior and a much lower false
  redaction burden.

## Current Decision

Open Image Models YOLOv9-S-608 is the only active plate detector. The selected
configuration is tracked in
`data/evaluations/license_plate_detector_oim_yolov9_s_608_v1/`.

This does not claim that the plate subsystem is production ready. OIM has known
nighttime gaps that require the next plate-tracking and review milestone.

## Archive Boundaries

The archived experiment modules are retained for reproducibility and historical
context only. They may reference rejected models and old candidate identifiers.
Production code must not import from this archive, and model weights remain
local-only ignored files.
