# System Overview Changelog

## v004 — 2026-07-31

Registered the completed ADAS selected-detector evaluation as tracked evidence:

- documented `openvino_face_adas_0001` as the selected primary face detector;
- recorded selected thresholds `0.30` primary and `0.20` proposal/storage;
- summarized certified ADAS raw-detector recall on GroundTruth v1.1 denominators `300`, `360`, and `192`;
- documented that raw detector misses still require temporal recovery and reviewed rendering;
- clarified that YuNet remains a comparison baseline and 0205 remains an optional targeted fallback;
- documented that final zero-leak claims have not yet been established;
- separated detector inference FPS from tracking/postprocessing FPS provenance.

## v003 — 2026-07-31

Selected corrected ground-truth benchmark v1.1 after complete human face-visibility verification:

- documented 660 unique reviewed face frames and 1,020 reviewed face-visibility decisions;
- recorded zero uncertain decisions;
- updated face target-frame denominators to `300`, `360`, and `192`;
- updated the face micro denominator from `912` to `852`;
- documented `person_night_02` as 192 detector-scored frames, 168 detector-ignore frames, and 360 final-redaction frames;
- preserved v1.0 as immutable historical evidence.

## v002 — 2026-07-30

Registered frozen ground-truth benchmark v1.0 as the evaluation source of truth:

- added GOLD_DENSE versus CONTEXT_ONLY methodology;
- added face target-frame denominators;
- separated raw detector recall from final redaction coverage;
- documented visual adjudication requirements and reference ROI limits;
- documented the current day-close-plate limitation.

## v001 — 2026-07-30

Initial documented architecture covering:

- media ingestion;
- face detection;
- face tracking v2;
- analyst-review workflow;
- current plate baseline;
- current limitations and planned milestones.
