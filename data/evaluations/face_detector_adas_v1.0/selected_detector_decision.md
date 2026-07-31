# Selected Detector Decision

Evaluation record: `face_detector_adas_v1.0`

Selected primary detector: `openvino_face_adas_0001` (OpenVINO face-detection-adas-0001).

Selected detector policy:

- Proposal/storage threshold: `0.20`.
- Primary detector threshold: `0.30`.
- Weak proposals from `0.20` through below `0.30` may support temporal tracking but must not independently confirm a track.

At the primary threshold `0.30`, ADAS achieved:

- `person_day_01`: `299 / 300` recall frames.
- `person_night_01`: `359 / 360` recall frames.
- `person_night_02`: `175 / 192` recall frames.
- Micro: `833 / 852`.
- Micro recall: `0.977700`.
- Worst-target recall: `0.911458`.
- Maximum consecutive misses: day `1`, standing night `1`, seated night `9`.

At threshold `0.20`, ADAS achieved:

- `person_day_01`: `299 / 300` recall frames.
- `person_night_01`: `359 / 360` recall frames.
- `person_night_02`: `176 / 192` recall frames.
- Micro: `834 / 852`.
- Maximum seated miss run remains `9`.

YuNet and OpenVINO face-detection-0205 remain comparison baselines. They were not fully re-adjudicated after ADAS selection, and exact certified statistics for those unselected models were intentionally not pursued further.

This is an engineering selection for the evaluated operating domain, not a universal detector ranking. The metrics measure raw detector recall, not tracking, interpolation, prediction, identity persistence, rendering, or final redaction quality.

FPS provenance audit: the historical ignored ADAS final report value `128.294961` is tracking/postprocessing throughput, not detector inference throughput. Detector inference throughput was `61.512487` FPS on the day clip and `66.489572` FPS on the night clip.
