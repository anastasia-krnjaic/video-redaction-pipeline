# Archived License Plate Detector Experiments

This directory contains historical bake-off, finalist-adjudication, and proposal
stabilization scripts. They are not part of the supported runtime.

The active detector is now Open Image Models YOLOv9-S-608, implemented in
`src/redactor/plate_detection.py`. The old scripts are preserved so previous
metrics and negative findings can be reproduced without leaving rejected
detectors selectable from the production CLI.

Model weights, source videos, overlays, and large generated artifacts are not
tracked by Git.
