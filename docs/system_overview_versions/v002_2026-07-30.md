# Video Redaction Pipeline - System Overview

- **Document version:** v002
- **Last updated:** 2026-07-30
- **System status:** Face pipeline under analyst review; ground-truth benchmark v1.0 registered; plate detector selection in progress
- **Implementation reference:** current working tree; base commit `7226499588c9477e45f50fd137ac54c9181a8282`
- **Canonical document:** `docs/SYSTEM_OVERVIEW.md`

# 1. Purpose

The video redaction pipeline is intended to propose and apply selective redactions for sensitive visual regions while preserving analyst control and traceability. In the current working tree, it proposes and reviews candidate regions; final rendering and audit packaging are still planned. The current prototype focuses on faces, license plates, temporal consistency, analyst review, and reproducible local outputs.

# 2. System at a Glance

```text
Input video
-> frame-level detections
-> temporal tracklets
-> quality classification
-> analyst decisions
-> approved redaction regions
-> rendered video
-> audit outputs
```

Input videos are local clips prepared for development and evaluation. Frame-level detection runs a selected model on each decoded frame and writes one record per frame. Temporal tracking links compatible detections into short-lived tracklets so analysts can reason about intervals rather than isolated boxes. Quality classification separates strong automatic candidates from items that need review or rejection. Analyst decisions are intended to approve, reject, group, and assign keep or redact actions. Approved redaction regions will feed the renderer. The final rendered video and audit outputs are planned, but the current working tree is still centered on detection, face tracking diagnostics, and review preparation.

# 3. Current Implementation Status

| Area | Status | Notes |
| --- | --- | --- |
| Video ingestion | Implemented | Local media inspection, hashing, trimming, and contact sheets are implemented with FFprobe and FFmpeg helpers. |
| Face detection | Implemented | OpenCV YuNet face detection is available through the detection command and local model file. |
| Face tracking | Implemented and under review | Face-only tracking v2 creates tracklets, quality states, review flags, review windows, overlays, and contact sheets. |
| Analyst face review | Implemented and under review | Review packages and an analyst-decision template exist; final decisions are not yet fully applied to rendering. |
| Manual face correction | Implemented in tracking diagnostics | Manual seed input can be loaded and connected to later automatic observations; the end-to-end analyst correction workflow is not complete. |
| Face rendering | Planned | No final face-only redacted output is produced by the current implementation. |
| License-plate detection | Evaluated but not selected | OpenCV LPD-YuNet has been measured as a baseline and is not considered production quality for the current clips. |
| License-plate tracking | Planned | Plate tracking is intentionally separated until a plate detector is selected. |
| Unified rendering | Planned | Combined face and plate rendering is not implemented. |
| Audit manifest | Planned | Diagnostic summaries exist; a production manifest and final verifier are not complete. |

# 4. Shared Detection Concepts

## 4.1 Frame-level proposals

A detector processes video frames and produces candidate regions with a model confidence score. The score is useful for ranking and thresholding model outputs, but it is not a calibrated legal, statistical, or safety probability. A high score can still be wrong, and a low score can still point to a real sensitive region.

Each detection record stores the frame index, category, confidence score, and a clipped rectangular region. Detection overlays are diagnostic only and are written without audio.

## 4.2 Confidence bands

The current shared face tracking pass uses two score bands. These values apply to the face proposal and tracking logic in the current implementation; they should not be assumed for future plate detectors.

| Score range | Meaning | System behavior |
| --- | --- | --- |
| Below `0.45` | Below weak threshold | Does not enter the primary face tracking pass. |
| `0.45` to below `0.70` | Weak proposal | Retained only as supporting evidence and review material. |
| `0.70` and above | Strong detection | Counts as stronger observed evidence, but one strong frame is not enough for automatic approval. |

Low-score detections do not enter the primary tracking pass. Weak proposals are kept so the tracker and analyst can see plausible low-confidence regions, especially in difficult lighting or profile views. Strong detections are better observations, but automatic approval still requires repeated evidence, timing consistency, size checks, and spatial continuity.

## 4.3 Observed versus synthetic regions

The pipeline records where each track box came from:

| Provenance | Meaning |
| --- | --- |
| `observed_strong` | A model detection at or above the strong threshold. |
| `observed_weak` | A model detection between the weak and strong thresholds. |
| `manual` | A box supplied by an analyst as a manual seed or correction. |
| `interpolated` | A box filled between two observed endpoints across a short safe gap. |
| `predicted` | A short tail box after the last observation, used only under conservative conditions. |

Only real observations and manual inputs count as evidence that a tracklet exists. Interpolated and predicted boxes are synthetic continuity aids; they must not make a tracklet appear more reliable than the observed evidence supports.

# 5. Shared Temporal Logic

Frame-level detections are single-frame proposals. Tracklets are chains of compatible detections over time. Real-world identities are analyst-level groupings across one or more tracklets. The current tracker creates tracklets; it does not perform identity recognition.

Association is based on temporal proximity, spatial overlap, center movement, and box scale consistency. Each detection can be assigned to at most one tracklet. A candidate association may pass when there is enough rectangular overlap or when the normalized center movement is small enough. Large scale jumps are rejected even if positions are near each other.

| Parameter | Current value | Meaning |
| --- | ---: | --- |
| Weak face threshold | `0.45` | Minimum score retained for face tracking proposals. |
| Strong face threshold | `0.70` | Score needed for a face observation to count as strong. |
| Confirmation window | `5` frames | Evidence must be close together, not scattered across the clip. |
| Minimum observed hits | `3` | Minimum real or manual observations for automatic confirmation. |
| Minimum strong hits | `2` | Minimum strong model observations for automatic confirmation. |
| Minimum automatic median score | `0.70` | Median scored observation must meet the strong threshold. |
| Median area fraction | `0.0005` to `0.08` | Tracklet size must be plausible relative to the frame. |
| Maximum synthetic fraction | `0.40` | Interpolated and predicted boxes cannot dominate an automatic tracklet. |
| Association overlap threshold | `0.15` | Minimum intersection-over-union overlap when center movement is not enough. |
| Association center distance | `0.75` | Maximum normalized center movement when overlap is low. |
| Allowed area-scale ratio | `0.50` to `2.00` | Consecutive boxes cannot shrink or grow by an unsafe factor. |
| Maximum internal gap | `0.30` seconds | Short bracketed gaps may be interpolated when motion is safe. |
| Maximum unbracketed tail prediction | `0.10` seconds | Short unmatched tails may be predicted only for trusted tracklets. |
| Association search horizon | `1.00` second | Sparse compatible observations can still connect before gap handling is decided. |
| Review-window merge gap | `15` frames | Nearby review flags are merged into a single analyst review window. |

# 6. Track Quality Classification

Lifecycle and quality are separate concepts. Lifecycle describes where a tracklet is in construction: `tentative`, `active`, or `ended`. Quality describes how the ended tracklet should be treated: `auto_confirmed`, `review_candidate`, or `rejected`.

The important product distinction is render eligibility. In the current tracker, only `auto_confirmed` tracklets are automatically render-eligible. Review candidates are preserved for analyst review. Rejected proposals remain available in diagnostics but are excluded from automatic redaction.

## 6.1 auto_confirmed

A tracklet becomes automatically render-eligible only when all required gates pass. The current criteria are:

- at least `3` observed or manual hits;
- at least `2` strong model hits;
- at least `3` observations within a `5`-frame window;
- at least `2` strong observations within a `5`-frame window;
- median observed model score of at least `0.70`;
- median area fraction from `0.0005` through `0.08`;
- synthetic fraction no greater than `0.40`;
- acceptable spatial and scale continuity.

One high-score frame is insufficient. Interpolation and prediction do not count toward confirmation. Analyst review can still override an automatic decision in the intended workflow, but final application of those decisions to rendering is not complete.

## 6.2 review_candidate

A review candidate is a potentially real and temporally consistent region that fails one or more automatic safety gates. Examples include a face under poor lighting with repeated scores between `0.45` and `0.70`, a small but persistent face, a track with good temporal support but too much interpolation, a weak-only tracklet, or a large region that may be a close face but cannot be trusted automatically.

Review candidates are preserved. They are not render-eligible until analyst approval. This is the primary safe-degradation path: when the system has enough signal to suspect a sensitive region but not enough signal to act automatically, it asks a person to decide.

## 6.3 rejected

Rejected proposals lack sufficient or consistent evidence. Examples include an isolated weak box, an isolated strong box with no temporal support, spatially inconsistent detections, or a tracklet that never gathers enough evidence. Rejected items remain available in diagnostics, including optional overlays, but are excluded from automatic redaction.

| Quality status | Evidence pattern | Render eligibility | Analyst role |
| --- | --- | --- | --- |
| `auto_confirmed` | Repeated, strong, temporally close, size-plausible, spatially continuous evidence | Automatically eligible | May override or group if needed. |
| `review_candidate` | Potentially real evidence that fails at least one automatic gate | Not eligible until approved | Primary review workload. |
| `rejected` | Isolated, inconsistent, or insufficient evidence | Not eligible | Diagnostic reference only unless manually revisited. |

# 7. Gap Handling

## 7.1 Internal interpolation

Internal interpolation fills a short missing span only when there are observed endpoints before and after the gap and the spatial transition is safe. The current maximum internal gap is `0.30` seconds, which is about nine frames at the current `30000/1001` frame-rate clips. Interpolated boxes are marked as synthetic and are review-flagged.

## 7.2 Tail prediction

Tail prediction is more conservative because there is no future endpoint. The current maximum tail duration is `0.10` seconds, about three frames at the current evaluation frame rate. Tail prediction is allowed only for sufficiently trusted `auto_confirmed` tracklets and only while staying within the synthetic-fraction budget.

## 7.3 Missing prefix before the first detection

Tracking cannot reconstruct a person before the first detection without additional evidence. If a face is visible before the first automatic detection, the current automatic tracker leaves that prefix unresolved.

Manual seeds are the intended correction path. An analyst can supply a box on an earlier frame, and the tracker can connect that seed to a later automatic tracklet when the motion and scale are compatible. Safe intervals may then be interpolated. Any unresolved motion remains review-required. This seed mechanism exists in tracking diagnostics, but the full analyst correction and final rendering workflow is not complete end to end.

# 8. Analyst Review

The intended analyst workflow is:

1. Review auto-confirmed tracklets.
2. Approve or reject review candidates.
3. Group approved tracklets into logical identities when needed.
4. Assign an action: keep or redact.
5. Add manual seeds where automatic detection missed the beginning or an entire interval.
6. Review the final rendered output.

The current system reduces many raw diagnostic events into a smaller number of review windows. Review flags are sorted by frame and merged when the next flag starts no more than `15` frames after the current window ends. Mandatory review spans are also emitted for the first and last two seconds of a clip, and tracklets that begin or end away from a scene boundary are flagged for review.

The current review package includes auto-confirmed track contact sheets, review-candidate contact sheets, review-window contact sheets, track-quality tables, and an analyst-decision template. The template contains fields for track decisions, identity grouping, keep or redact action, notes, and manual seeds. It is preparation for analyst control, not a completed rendering policy engine.

# 9. Face Detection

The selected face detector is OpenCV YuNet:

| Property | Current value |
| --- | --- |
| Model | OpenCV YuNet face detector |
| Filename | `models/face_detection_yunet_2023mar.onnx` |
| Model version | `2023mar` |
| SHA-256 | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| Inference backend | OpenCV `FaceDetectorYN` with default backend and target; no GPU backend is configured. |

YuNet was selected for the current face branch because it runs locally, is lightweight, has acceptable speed for development clips, and provides useful performance on both profile and frontal faces. The observed behavior is strongest in daylight. Nighttime performance is workable, but difficult low-light, occluded, or highly angled faces may produce weak or missing detections. Local review also showed occasional false positives on clothing, hair, reflections, or background regions.

The table below uses the face proposal threshold used by tracking v2. The detection command also has a face default threshold of `0.60`, but tracking v2 intentionally uses `0.45` so weak evidence remains available for review.

| Clip | Frames | Detector threshold | Frames with at least one detection | Total detections | Processing rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Daylight face clip | `1,079` | `0.45` | `1,029` | `1,496` | `21.342279` frames/sec |
| Nighttime selective-redaction clip | `1,589` | `0.45` | `1,583` | `2,181` | `23.565718` frames/sec |

"Frames with at least one detection" is not per-person recall. A frequently detected person can hide misses of another person in the same frame. No formal recall is claimed without independent ground-truth labels.

# 10. Face Tracking v2

Face tracking v2 is the current face temporal-safety diagnostic layer.

| Clip | Total proposals | Auto-confirmed | Review candidates | Rejected | Review windows | Interpolated frames | Predicted frames |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Daylight face clip | `50` | `1` | `22` | `27` | `2` | `243` | `0` |
| Nighttime selective-redaction clip | `41` | `4` | `20` | `17` | `8` | `149` | `9` |

The first tracker confirmed too many tracklets automatically and created a noisy review surface. Its local audit reported `13` confirmed day tracks and `20` confirmed night tracks, with duplicate candidates, fragmentation candidates, many suspicious tracks to inspect, and raw review intervals that were not meaningfully reduced. It also allowed isolated detections to gain synthetic predictions too easily.

The second version separates automatic approval from analyst review, does not count synthetic boxes as evidence, restricts prediction, introduces size and quality gates, and aggregates review events into fewer review windows. Track IDs remain local diagnostic identifiers and should not be treated as recognized identities.

# 11. License-Plate Detection

**Evaluated baseline - not selected as the final plate detector**

The currently evaluated plate model is OpenCV LPD-YuNet:

| Property | Current value |
| --- | --- |
| Model | OpenCV LPD-YuNet license-plate detector |
| Filename | `models/license_plate_detection_lpd_yunet_2023mar.onnx` |
| Model version | `2023mar` |
| SHA-256 | `6d4978a7b6d25514d5e24811b82bfb511d166bdd8ca3b03aa63c1623d4d039c7` |
| Inference backend | OpenCV neural-network module with local adapter code. |

| Clip | Frames | Threshold | Frames with detections | Total detections | Processing rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Close daytime plate clip | `540` | `0.40` | `0` | `0` | `3.560561` frames/sec |
| Nighttime selective-redaction clip | `1,589` | `0.35` | `23` | `38` | `2.645604` frames/sec |

The reference check found two issues. First, the local adapter did not exactly match the official OpenCV Zoo reference behavior on selected frames: the local adapter produced no detections on checked frames where the reference script produced detections. Second, the official reference also produced large false-positive regions on those same frames, so both an adapter mismatch and model/domain limitations were observed.

Current decision: LPD-YuNet remains a documented baseline. It is not considered a production-quality plate detector for this pipeline. Plate work is temporarily separated from the face branch, and multiple alternative plate detectors will be evaluated later on the same benchmark clips before plate tracking is added.

# 12. Inputs and Evaluation Clips

The current evaluation set contains four local scenarios:

| Scenario | Purpose |
| --- | --- |
| Daylight face clip | Tests profile face detection, head rotation, partial occlusion, and moving camera behavior. |
| Nighttime selective-redaction clip | Tests low light, glare, multiple people, selective keep/redact policy, and a visible plate target. |
| Close daytime plate clip | Tests close plate localization under changing scale, perspective, camera motion, and partial occlusion. |
| Optional distant-plate stress case | Qualitative stress test for a small visually ambiguous plate; not required for final success. |

The workflow uses automated proposals plus targeted human review rather than a complete manual ground-truth dataset. Formal recall, precision, or overlap metrics must not be claimed outside the explicitly registered benchmark semantics.

# 13. Ground Truth and Evaluation

The frozen benchmark source is `data/ground_truth/GroundTruth_v1.0.md`, with machine-readable face-evaluation fields mirrored in `data/ground_truth/face_benchmark_v1.0.json`. The Markdown document remains the semantic source of truth and must not be edited in place.

The benchmark distinguishes two review levels:

| Level | Meaning | Allowed claims |
| --- | --- | --- |
| `GOLD_DENSE` | Every frame in the interval was visually reviewed at native frame rate. | Zero-leak claims may be made only for these intervals after the corresponding output is reviewed. |
| `CONTEXT_ONLY` | Surrounding clip context was reviewed to confirm scene and target selection. | Useful for debugging and scene selection, but not for per-frame zero-leak claims. |

The current face benchmark uses `30000/1001` fps, zero-based frame indexes, half-open time intervals, and 1920x1080 `xyxy` pixel coordinates. The registered face target-frame denominators are:

| Target | Action | Detector-scored face frames |
| --- | --- | ---: |
| `person_day_01` | redact | `300` |
| `person_night_01` | keep | `360` |
| `person_night_02` | redact | `252` |
| **Micro total** |  | `912` |

Raw detector recall and final redaction coverage are separate metrics. Raw detector recall is scored only on detector-scored frames, excludes detector-ignore frames, and does not count tracker-only, predicted, or interpolated boxes as detector hits. Final redaction coverage may use reviewed tracking and interpolation, but a single visible uncovered target frame in a certified redact interval remains a failure.

The benchmark contains temporal labels and reference search ROIs, but it does not contain tight per-frame scoring boxes. Detector-hit adjudication therefore requires visual target correspondence. Reference ROIs may guide candidate search and review triage, but they are not automatic scoring boxes.

The current day-close-plate clip is not the registered day-plate gold interval. The frozen day-plate gold interval describes the earlier far-plate clip; `eval_day_plate_close.mp4` requires a future versioned mapping before it can be treated as plate ground truth.

# 14. Current Outputs

Detection-stage outputs:

- `detections.jsonl`: one frame-indexed record per processed frame.
- `overlay.mp4`: diagnostic overlay video without audio.
- `summary.json`: model, source, threshold, processing, and count summary.

Tracking-stage outputs:

- `tracks.jsonl`: frame-indexed track boxes with provenance, quality, and render eligibility.
- `overlay.mp4`: diagnostic face-tracking overlay without audio.
- `summary.json`: track counts, thresholds, review counts, and processing summary.
- `track_quality.json`: per-track quality summary.
- `review_flags.json`: raw review prompts.
- `review_windows.json`: aggregated review intervals.
- `track_contact_sheets/`: sampled crops for auto-confirmed and review-candidate tracklets.

Analyst-stage outputs:

- `artifacts/tracking_v2_review/analyst_decisions.template.json`: current template for track decisions, identity grouping, keep/redact actions, notes, and manual seeds.
- Review contact sheets and track-quality tables are available as local diagnostic review aids.
- Final analyst decisions, final rendered redaction output, unified face/plate rendering, production audit manifests, and final verifier output are planned but not yet implemented.

# 15. Current Limitations

- Difficult faces may still be missed before their first detection.
- Manual review remains necessary.
- The tracker creates tracklets rather than recognized identities.
- Current analyst decisions are not yet fully applied to rendering.
- LPD-YuNet is not an acceptable final plate detector.
- Ground truth is currently certified only for the frozen dense benchmark windows, not for full source videos or full evaluation clips.
- The production manifest and final verifier are not complete.

# 16. Next Planned Milestones

1. Complete face tracklet review.
2. Group approved face tracklets into logical identities.
3. Add manual seeds for unresolved face prefixes.
4. Produce and verify face-only selective redaction.
5. Freeze the face branch.
6. Evaluate several alternative plate detectors separately.
7. Select a plate detector and add temporal plate tracking.
8. Combine face and plate regions in one renderer.
9. Finalize audit and production documentation.

# 17. Source Traceability

This document reflects the current working tree on top of commit `7226499588c9477e45f50fd137ac54c9181a8282`. At the time this document was written, the working tree had uncommitted ground-truth benchmark registration changes.

Current local model files and hashes:

| Model file | SHA-256 |
| --- | --- |
| `models/face_detection_yunet_2023mar.onnx` | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| `models/license_plate_detection_lpd_yunet_2023mar.onnx` | `6d4978a7b6d25514d5e24811b82bfb511d166bdd8ca3b03aa63c1623d4d039c7` |

| Documented concept | Repository source |
| --- | --- |
| CLI commands and defaults | `src/redactor/cli.py` |
| Media ingestion, hashing, trimming, contact sheets | `src/redactor/media.py` |
| Detector configuration and detection summaries | `src/redactor/detection.py`; `src/redactor/models.py` |
| Face tracking and quality gates | `src/redactor/tracking.py`; `tests/test_tracking.py` |
| Detection and media behavior tests | `tests/test_detection.py`; `tests/test_media.py`; `tests/test_models.py` |
| Evaluation clips and policies | `data/evaluation_plan.yaml`; `data/source.yaml` |
| Frozen ground-truth source | `data/ground_truth/GroundTruth_v1.0.md` |
| Machine-readable face benchmark | `data/ground_truth/face_benchmark_v1.0.json` |
| Evaluation clip metadata | `artifacts/eval_day_face.metadata.json`; `artifacts/eval_night_selective.metadata.json`; `artifacts/eval_day_plate_close.metadata.json`; `artifacts/eval_day_plate.metadata.json` - local diagnostic outputs ignored by Git |
| Face proposal baseline metrics | `artifacts/baseline/day_face_proposals_t045/summary.json`; `artifacts/baseline/night_face/summary.json` - local diagnostic outputs ignored by Git |
| Tracking-v2 metrics | `artifacts/tracking_v2/day_face/summary.json`; `artifacts/tracking_v2/night_face/summary.json`; `artifacts/tracking_v2/day_face/track_quality.json`; `artifacts/tracking_v2/night_face/track_quality.json` - local diagnostic outputs ignored by Git |
| Analyst review package | `artifacts/tracking_v2_review/report.md`; `artifacts/tracking_v2_review/analyst_decisions.template.json` - local diagnostic outputs ignored by Git |
| First tracker audit | `artifacts/tracking_audit/report.md` - local diagnostic output ignored by Git |
| Plate baseline metrics | `artifacts/baseline/day_plate/summary.json`; `artifacts/baseline/night_plate/summary.json` - local diagnostic outputs ignored by Git |
| Official-reference comparison | `artifacts/reference_check/lpd_reference_results.json`; `artifacts/reference_check/lpd_yunet_reference.py` - local diagnostic outputs ignored by Git |
| Dependency and package context | `pyproject.toml`; `README.md` |
