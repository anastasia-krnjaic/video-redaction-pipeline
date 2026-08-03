# System Overview Changelog

## v009 — 2026-08-02

Added machine-readable face ground truth and automated Face Pipeline v1 evaluation:

- documented `data/ground_truth/redaction_ground_truth_v1.0.json` as the selected machine-readable redaction ground truth for Face Pipeline v1;
- added `validate-ground-truth` and `evaluate-faces` workflow notes;
- documented detector, automatic, and reviewed-final evaluation modes;
- clarified that renderer approval is not by itself a ground-truth privacy-quality result;
- clarified that tracking, synthetic privacy recovery, and reviewed/preserved corrections do not count as raw detector hits;
- recorded evaluator outputs and current limits of range-level truth.

## v008 — 2026-08-01

Added Face Redaction Renderer v1 to the documented Face Pipeline:

- documented `render-faces` as the explicit rendering step after privacy gap recovery;
- added preview and final rendering semantics, including final export blocking for review-required, rejected, unresolved, or invalid privacy regions;
- documented Gaussian blur, pixelation, privacy margins, provenance-preserving manifests, audio remuxing, and FFprobe media validation;
- clarified that rendering applies accepted privacy regions without converting them into detector evidence;
- recorded processed-frame accounting for `--max-frames` so source-video frame counts are distinct from processed-frame counts.

## v007 — 2026-08-01

Consolidated Face Pipeline v1 around ADAS-only face detection:

- removed OpenCV YuNet from the active face detector runtime path and face-pipeline CLI;
- documented OpenVINO `face-detection-adas-0001` as the only active face detector backend;
- moved the three-model detector bake-off context into `docs/archive/face_detector_bakeoff/`;
- rewrote the canonical overview for a broader reader and moved technical details into `docs/FACE_PIPELINE_TECHNICAL_NOTES.md`;
- clarified that historical YuNet and OpenVINO 0205 comparison material is not supported production behavior;
- preserved existing tracking and privacy gap-recovery behavior and evidence boundaries.

## v006 — 2026-08-01

Integrated reusable face privacy gap recovery after ordinary ADAS tracking:

- added a tracked post-tracking component for privacy-region continuity across face-track gaps;
- documented interpolation, bidirectional tracking, one-sided tracking, preserved accepted regions, and conservative review/rejection paths;
- recorded scale-normalized safety checks for center motion, forward/backward disagreement, scale changes, keep-person overlap, competing identities, scene cuts, gap duration, invalid boxes, and jumps;
- clarified that optional bottom-connected visual refinement is strategy-specific and disabled by default;
- documented that privacy coverage remains separate from raw detector recall and does not establish final zero-leak rendering claims;
- added a non-authoritative ignored night regression artifact preserving accepted `adas_privacy_bridge_v2` regions.

## v005 — 2026-07-31

Integrated ADAS as the primary face detector backend:

- documented `openvino_face_adas_0001` as the default integrated source backend;
- kept YuNet selectable as a comparison baseline and 0205 as an optional targeted fallback candidate;
- recorded the selected ADAS thresholds `0.20` proposal/storage and `0.30` primary observation;
- documented that weak ADAS observations may support tracking but do not independently confirm tracks;
- recorded zero-mismatch parity against certified ADAS canonical detections under ignored integration artifacts;
- separated integrated detector-call FPS from detection-pass, tracking, and postprocessing throughput;
- clarified that ADAS tracking validation and reviewed rendering remain next phases and that final zero-leak claims have not yet been established.

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
