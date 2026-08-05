# Repository Guidance

## System Overview Maintenance

`docs/SYSTEM_OVERVIEW.md` is the canonical human-readable description of the current pipeline. Any task that materially changes the pipeline must update `docs/SYSTEM_OVERVIEW.md` as part of the same change. Before changing the canonical overview, preserve the previous canonical document as the next sequential immutable file under `docs/system_overview_versions/`, then update `docs/system_overview_versions/CHANGELOG.md`.

Every archived version must preserve the implementation sources and configuration sources on which it was based. Archived versions must never be edited retroactively. Do not create a new archived version for spelling, grammar, or formatting corrections that do not change the documented system behavior.

A material system change includes any change to detector model or model version, model selection status, confidence thresholds, shared detection logic, track association, track confirmation criteria, interpolation or prediction rules, analyst decisions or manual override workflow, identity grouping, rendering eligibility, redaction rendering, audit records or manifests, output artifacts, evaluation methodology, benchmark clips, or production workflow.

## Ground-Truth Benchmark Rules

Default selected face benchmark: `data/ground_truth/face_benchmark_v1.1.json`.

Default selected machine-readable redaction ground truth: `data/ground_truth/redaction_ground_truth_v1.0.json`.

Default selected face detector evidence record: `data/evaluations/face_detector_adas_v1.0/`.

Default selected license-plate detector evidence record:
`data/evaluations/license_plate_detector_oim_yolov9_s_608_v1/`.

Selected face detector policy:

- Primary detector: `openvino_face_adas_0001`.
- Proposal/storage threshold: `0.20`.
- Primary detector threshold: `0.30`.
- Weak proposals from `0.20` through below `0.30` may support temporal tracking but must not independently confirm a track.
- ADAS is the only active face detector backend. Historical YuNet and OpenVINO 0205 comparison material must stay in the archive/evidence record and must not be reintroduced into the active runtime path without a new explicit model-selection task.

Selected license-plate detector policy:

- Primary detector: `oim_yolov9_s_608` (Open Image Models YOLOv9-S-608).
- Inference mode: `uniform_small_object_tiles`.
- Primary threshold: `0.30`.
- No separate lower proposal threshold is selected for the active plate detector.
- OIM YOLOv9-S-608 is the only active plate detector backend. Historical LPD-YuNet, OpenVINO 0106/0123, Intel YOLOv8n, RF-DETR, and RT-DETR comparison material must stay in the archive/evidence record and must not be reintroduced into the active runtime path without a new explicit model-selection task.
- Plate tracking and bounded plate gap recovery are implemented as a prototype stage after OIM detection. They use one global configuration, preserve detector observations separately from interpolation, and leave long or unsafe gaps as review intervals.
- Plate review integration and final plate rendering are not implemented yet. Do not claim plate redaction is complete from detector or tracking evidence.

1. Versioned ground-truth files under `data/ground_truth/` are frozen sources of truth once created.
2. Never edit a frozen ground-truth version in place.
3. Every evaluation must validate ground-truth, video, and detection hashes.
4. Zero-leak claims apply only to GOLD_DENSE intervals.
5. Detector-ignore frames are excluded from raw detector recall.
6. Tracking, prediction, and interpolation do not count as raw detector hits.
7. Final redaction coverage may use reviewed tracking and interpolation.
8. Reference ROIs may be used for candidate search but not as automatic scoring boxes.
9. New ground-truth semantics require a new version.
10. Raw ADAS detector misses still require temporal recovery and reviewed rendering evidence; final zero-leak claims have not yet been established.
11. OpenCV YuNet and OpenVINO face-detection-0205 are historical comparison models, not active production face detector backends.
12. Future detector, tracker, renderer, and evaluation work must use the newest explicitly selected ground-truth version and selected face-detector evidence record.
13. Face privacy gap recovery is a downstream continuity layer, not detector recall. It must preserve provenance, separate detector/ordinary-tracker evidence from synthetic privacy regions, and emit `needs_review` or `rejected` intervals when bidirectional, one-sided, scene-cut, keep-overlap, identity-competition, jump, scale, or validity checks fail.
14. The optional bottom-connected visual-component refinement is strategy-specific and disabled by default; do not encode clip-specific frame numbers, person IDs, or coordinates in production recovery logic.
15. Face rendering must consume privacy regions without reclassifying them as detector evidence. Preview renders must remain visibly or machine-readably non-final. Final renders must block when relevant intervals need review, are rejected, contain unresolved privacy-required frames, or contain invalid boxes.
16. When `--max-frames` is used, summaries must distinguish total source-video frames from the frames actually processed by detection, tracking, recovery, or rendering.
17. Face evaluation must use `redactor evaluate-faces` modes explicitly: `detector`, `automatic`, or `reviewed_final`. Never count tracking, synthetic privacy bridges, preserved regions, or reviewed corrections as raw detector hits.
18. Machine-readable redaction ground truth may represent range-level truth, detector-scored ranges, detector-ignore ranges, privacy-required ranges, and optional boxes only where boxes actually exist. Do not invent per-frame boxes for evaluation.
19. Renderer approval is not by itself ground-truth privacy-quality proof. The evaluator must separately attribute detector, ordinary-tracker, automatic privacy recovery, preserved/reviewed corrections, and rendering evidence.
20. Automated review bundles must keep pipeline-derived risks, Ground-Truth-derived benchmark failures, and deterministic random audit samples distinct. A review decision of `accepted_current` must not clear a final blocker when the underlying privacy-required interval still lacks valid reviewed regions.
21. Review bundles are local artifacts only. They must use relative paths and local assets, avoid network calls, and remind reviewers to export decisions before closing the browser.
22. Review Bundle v1.1 must preserve raw review signals while presenting concise reviewer tasks. Technical-only signals such as handled detector misses or provenance transitions should stay in Technical audit rather than inflating the default task count.
23. Review Bundle v1.2 comparison media must render Original, Automatic, and Corrected final clips from the same source-frame interval. Automatic and Corrected final must use the same explicit review-render configuration, and Corrected final must be traceable to the reviewed/corrected privacy-region JSONL rather than only to an opaque rendered video.
24. Review Bundle v1.2.1 task clips must be browser-compatible H.264/yuv420p MP4s and must pass decoded-frame visual-content validation. FFprobe metadata alone is not sufficient proof that a generated review clip contains visible source content.
25. Review Bundle v1.2.2 task media must record source-frame to local-clip-frame mapping and verify rendered-region application inside expected privacy bboxes. JSON privacy coverage is not sufficient unless decoded output confirms that each expected region changed on the intended source frame.
26. Review Bundle v1.2.3 must treat reviewer-discovered random-audit privacy misses as blocking human-review evidence, distinct from frozen Ground Truth. The original random-audit item must remain auditable, but the default workflow should present the corrected-result follow-up rather than duplicate the same interval as optional. Use `redactor serve-review-bundle` for reliable local embedded playback; it must bind only to loopback and serve only files under the selected bundle directory.
27. The supported production-oriented face workflow is `redactor redact-video` with a case privacy policy. The default policy is `redact_all`: redact every qualifying full-video face track unless the case policy explicitly marks a track as `keep`.
28. Ground Truth, benchmark object IDs, benchmark frame windows, source-video names, manually selected target tracks, and preserved benchmark corrections must not act as the production privacy policy.
29. Rejected or unsupported face-track candidates must remain visible in the track census and review queue, but must not be silently rendered as defensible automatic privacy regions.
30. Explicit keep-person handling is a case-policy decision. Ambiguous keep fragments or unsupported weak sequences must remain review blockers; do not infer keep identity from benchmark labels, broad ROIs, clothing, or role.
31. Plate detector calibration must use one global threshold policy across clips. Do not introduce per-video plate thresholds, ROIs, coordinates, frame ranges, or source-specific production logic.
32. Plate tracking must use one global time-normalized configuration across clips. Interpolated plate boxes are temporal support only, not raw detector hits. Long gaps, scene cuts, unstable motion, scale jumps, ambiguous tracks, and possible static-overlay false positives must stay auditable instead of being silently accepted.
