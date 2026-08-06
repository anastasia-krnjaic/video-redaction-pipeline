# System Overview Changelog

## v024 — 2026-08-05

Added Final Pipeline Freeze, Audit Package, Repository Cleanup, and Showcase Draft v1:

- documented the generated `artifacts/final_pipeline_package_v1/` behavior-freeze package;
- recorded that the package freezes current face and plate reference behavior without detector, tracking, evidence, gap-recovery, or rendering changes;
- clarified that day face and day plate media are source-incompatible and therefore not combined;
- documented that night face and plate media are source-compatible and have a non-final combined demo preview;
- separated machine-derived metrics from pending human-reviewed full-video privacy metrics.

## v023 — 2026-08-05

Added Shared Temporal Evidence Accumulation and Hysteresis Engine v1 for plates:

- documented the shared accumulated-evidence engine and category-profile boundary;
- recorded that the engine is active for plate track confirmation but not for production face tracking;
- added audited evidence components, hysteresis, retrospective promotion, and bounded stitch-decision behavior;
- clarified that interpolation can support plate continuity but cannot satisfy the detector-observation count required for automatic confirmation;
- recorded the selected plate evidence configuration hash and before/after plate-rendering result.

## v022 — 2026-08-05

Added Shared Post-Detection Privacy Pipeline Foundation v1:

- documented common privacy observation, temporal-region, gap-decision, privacy-region, processing-span, renderer, and review-task contracts;
- added category profiles for faces and plates while preserving separate detector thresholds and category safety behavior;
- recorded that plate rendering now delegates its full-preview rendering and rendered-region verification to the shared renderer core through the existing wrapper;
- clarified that the frozen face tracking/recovery engine remains category-specific with adapters and parity tests rather than a broad rewrite;
- preserved existing supported CLI commands and postponed final unified face/plate rendering.

## v021 — 2026-08-05

Added Full-Video Plate Processing and Partial-Span Safety v1:

- documented that `detect-plates` processes complete source videos by default and `--max-frames` is diagnostic only;
- added plate frame-span reports for detector/tracker inputs and renderer manifests;
- blocked normal plate preview/final rendering from partial tracking inputs unless an explicitly watermarked `--allow-partial-preview` diagnostic is requested;
- recorded that benchmark frame windows remain evaluation-only and must not control production plate processing;
- preserved the selected OIM detector, Plate Tracking v1 configuration, and plate rendering style.

## v020 — 2026-08-05

Added Plate Rendering and Review v1 after OIM plate tracking:

- documented `redactor render-plates` for non-final automatic plate previews from accepted tracks only;
- documented `redactor build-plate-review-bundle` and `redactor validate-plate-review-decisions`;
- recorded the global plate render style: pixelation, 15% margin, pixel size 16, H.264/yuv420p;
- clarified that rejected tracks, `needs_review` tracks, unresolved gaps, and speculative prediction are excluded from clean previews;
- added decoded rendered-region validation for expected plate boxes;
- clarified that reviewed-final plate completion and unified face/plate rendering remain future work.

## v019 — 2026-08-05

Added Plate Tracking and Gap Recovery v1 after the selected OIM plate detector:

- documented `redactor track-plates` as the supported prototype plate tracking command;
- recorded deterministic OIM plate association, track classification, bounded two-anchor interpolation, review intervals, and rejected-noise diagnostics;
- clarified that interpolated plate boxes are temporal support, not detector hits;
- recorded the global tracking configuration hash `27fc564737bf7bf3d2bc5389c4394fe991b5b9e90f9e76789ed0d8231e3dff9d`;
- clarified that plate review integration and final plate rendering remain future work.

## v018 — 2026-08-05

Selected Open Image Models YOLOv9-S-608 as the active detector-only plate backend:

- documented `oim_yolov9_s_608` with uniform small-object tiles and threshold `0.30`;
- recorded that manual visual adjudication superseded the earlier metric-only RT-DETR selection;
- added active detector-only CLI guidance for `redactor detect-plates`;
- archived prior plate bake-off and stabilization experiments as historical material;
- clarified that plate tracking, plate gap recovery, review integration, and final plate rendering are not implemented yet.

## v017 — 2026-08-04

Added Full-Video Automatic Face Privacy Policy v1:

- documented `redactor redact-video` as the supported production-oriented face workflow;
- added the full-video track census and machine-readable case privacy-policy boundary;
- recorded `redact_all` as the safest default policy and `redact_all_except_keep` as an explicit case-policy choice;
- clarified that Ground Truth, benchmark windows, target track IDs, and preserved benchmark corrections are not operational privacy policy inputs;
- documented that rejected or unsupported candidates remain review blockers rather than automatic privacy regions;
- preserved lower-level targeted track/frame commands as diagnostic tools.

## v016 — 2026-08-03

Repaired Review Bundle v1.2.3.1 grouped media controls:

- documented that grouped playback controls are scoped to the currently visible Guided-review task;
- disabled grouped playback in View-all mode so hidden task media cannot start in the background;
- recorded dynamic `Play together` / `Pause together` labels, restart-to-zero behavior, and on-page playback errors;
- preserved detector, tracking, privacy recovery, rendering, evaluator, Ground Truth, and privacy-region behavior.

## v015 — 2026-08-03

Resolved Review Bundle v1.2.3 reviewer-discovered daytime privacy-miss workflow:

- documented that deterministic random audits can reveal whole-video misses outside selected benchmark windows;
- added reviewer-discovered finding handling that promotes optional audit misses into blocking corrected-result tasks while preserving raw audit evidence;
- recorded that reviewer findings remain distinct from frozen Ground Truth unless a new Ground Truth version is explicitly selected;
- added loopback `redactor serve-review-bundle` playback guidance and browser-local decision auto-advance/export behavior;
- clarified that benchmark evaluator range-level success and whole-video review findings are separate claims.

## v014 — 2026-08-03

Added Review Bundle v1.2.2 frame-mapping and rendered-region verification:

- documented that privacy-region JSONL uses source-video frame indices while short review clips use local frame indices beginning at zero;
- recorded the source-to-local mapping rule used by task-level review media;
- added bbox-localized decoded-pixel validation so JSON privacy coverage must be confirmed by visible rendered changes in the intended frame;
- documented declared JSON coverage separately from visibly verified rendered coverage;
- preserved detector, tracking, privacy recovery, evaluation, rendering thresholds, and reviewed privacy-region coordinates.

## v013 — 2026-08-03

Repaired Review Bundle v1.2 media generation into v1.2.1:

- documented that task-level review clips are encoded as browser-compatible H.264/yuv420p MP4s;
- recorded that FFprobe metadata validation is not enough to prove a review clip contains visible frames;
- added decoded-frame visual-content validation for first, middle, and final sampled frames;
- clarified that bundle generation fails when generated clips turn black or repeat frames while matching source frames contain visible structure;
- preserved the v1.2 aligned media semantics, privacy-region provenance, and pixelated review comparison style.

## v012 — 2026-08-03

Refined benchmark review media into Review Bundle v1.2:

- documented that Original, Automatic, and Corrected final clips are generated from one shared source-frame interval;
- recorded the explicit review-render style used for comparison clips: pixelation with 15% margin and block size 16;
- clarified that Automatic comes from automatic privacy-region JSONL and Corrected final comes from reviewed/corrected privacy-region JSONL;
- added media audit expectations for source-frame range, output frame count, renderer configuration hash, privacy-region input hash, and corrected-region traceability;
- clarified that comparison styling is separate from the global renderer defaults and that renderer approval alone is not Ground-Truth acceptance.

## v011 — 2026-08-03

Refined Automated Review Bundle v1 into task-oriented Review Bundle v1.1:

- documented the raw-signal versus reviewer-task presentation model;
- added guided review semantics with required tasks, corrected-result confirmations, optional spot checks, and Technical audit;
- clarified that technical-only signals do not inflate the default reviewer task count;
- documented explicit media roles for Original, Automatic, Corrected final, and Diagnostic evidence;
- recorded duplicate-media-role validation and local decision export fields for reviewer tasks;
- reiterated that review decisions do not rewrite privacy regions or override renderer/evaluator acceptance.

## v010 — 2026-08-03

Added Automated Review Bundle v1 to the documented Face Pipeline:

- documented `build-review-bundle` and `validate-review-decisions`;
- described pipeline-derived risks, Ground-Truth-only benchmark evidence, and deterministic random audit samples as separate review item sources;
- clarified that review bundles do not prove absence of silent misses and do not override renderer or evaluator approval;
- documented exported local review decisions and blocker preservation for intervals that still need corrected privacy regions;
- recorded the local-only bundle structure with relative assets, contact sheets, short clips, queue JSON, summary JSON, and checklist.

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
