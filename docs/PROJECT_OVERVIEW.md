# Video Redaction Pipeline Project Overview

## What The System Does

This project is a local video privacy-redaction prototype. It detects sensitive objects in video, follows them across time, renders redaction over approved regions, and creates review material for uncertain intervals.

The current system covers two privacy-object categories:

- Faces
- License plates

The system is designed for local processing. It does not upload videos, model output, or review data to a cloud service.

## Why Video Privacy Redaction Is Difficult

Redacting one still image is usually straightforward: find the sensitive object, cover it, and save the result. Video is harder because privacy targets move, disappear, reappear, blur, shrink, cross behind other objects, and sometimes produce weak or missing detections.

If each frame is handled independently, redaction can flicker. A face or plate may be covered in one frame and visible in the next. A detector may also briefly latch onto the wrong object, or a moving target may split into several short fragments.

The project therefore treats detection as only the first step. After detection, the pipeline asks a different question: is there enough evidence over time to trust this object track, recover short safe gaps, and send uncertain spans to review?

## High-Level Workflow

The workflow has six broad stages:

1. Detect sensitive objects in each frame.
2. Normalize detections into a shared observation format.
3. Connect observations over time.
4. Accumulate evidence before trusting a track.
5. Render approved regions and keep uncertain intervals reviewable.
6. Validate output media and preserve audit records.

The system keeps faces and plates as separate detector categories, because the best detector for one category is not the best detector for the other. After detection, both categories use shared ideas: normalized observations, temporal evidence, provenance, rendering, and review tasks.

## Models Selected

Several face detector candidates were evaluated. The active face detector is OpenVINO `face-detection-adas-0001`. It is the only face detector in the supported active face pipeline.

Several license-plate detector candidates were also evaluated. The active plate detector is Open Image Models YOLOv9-S-608. It was selected because manual visual adjudication found it produced much cleaner plate localization and far fewer destructive false redactions than the higher-recall finalist. The remaining plate gaps are expected to be handled through tracking, bounded interpolation, and review rather than by accepting a noisier detector.

Rejected detector experiments are preserved as historical evidence. They are not part of the active runtime path.

## Temporal Consistency And Uncertainty

The face pipeline uses its frozen face-specific tracking and privacy-gap recovery behavior. Existing reviewed day and night face outputs are the accepted face references for this freeze.

The plate pipeline uses the selected OIM detector, then applies a shared temporal evidence engine. Plate observations start as tentative. A track becomes trusted only when there is enough evidence across time: persistence, detector support, density, confidence, box stability, motion consistency, and absence of strong competition.

When a track becomes trusted, earlier valid observations from the same track can be promoted retrospectively. This helps avoid late-start redaction when the system has enough later evidence that the earlier observations belonged to the same plate.

The engine also uses hysteresis: one weak frame or short miss does not immediately break a confirmed track, but long ambiguous gaps stay unresolved. Persistent screen-fixed false positives are not accepted just because they last a long time.

## Day And Night Results

The current frozen package separates machine-derived results from human-reviewed results.

| Area | Machine-derived status | Human-reviewed status |
| --- | --- | --- |
| Day face | Reviewed reference output available | Pending full-video audit |
| Night face | Reviewed reference output available | Pending full-video audit |
| Day plate | Automatic preview available; review still required | Pending full-video audit |
| Night plate | Automatic preview available; review still required | Pending full-video audit |

The latest plate evidence engine kept day and close/stress behavior stable and increased full-night rendered plate regions substantially. It also preserved review blockers for long ambiguous gaps. This is useful progress, but it is not a final privacy guarantee.

Day face and day plate reference media originally came from different extracted clips. The current review workspace traces both back to a common parent source and extracts one canonical Day original interval, but it does not fabricate a combined result from the old encoded clips. The frozen pipelines must be rerun on that canonical parent interval before a combined Day current result can be reviewed.

Night face and night plate references share the same source video, so the workspace includes a combined Night current result for review.

## Human Review And Safety Behavior

The system does not claim that a clean automatic pass proves every privacy target was found. Some misses are silent: if a face or plate produces no detector, tracker, or review evidence, software alone may not know it was present.

The final package therefore includes a local case-based review workspace. It shows Original and Current result side by side where available, keeps raw machine signals in a technical audit, and presents concise reviewer tasks for unresolved intervals, uncertain tracks, provenance changes, and deterministic spot checks.

Reviewer decisions and corrections are stored as local append-only events with hash-chain validation. Reviewer identity is declared locally; it is not enterprise authentication.

Human-reviewed leak metrics remain pending until a reviewer exports valid decisions. Pending metrics are represented as `null`, not zero.

## Current Limitations

The prototype is not production-ready and does not claim zero leaks. Known limitations include:

- Plate previews are still review-required.
- Some nighttime plate gaps remain unresolved.
- Some false plate redactions are visible and must be adjudicated.
- The canonical Day source interval is identified, but the combined Day result still needs a frozen-pipeline rerun with local model files.
- Human full-video audit has not yet been completed.
- Face production tracking is intentionally frozen and has not migrated to the shared evidence engine.

## Next Possible Improvements

The next practical step is manual full-video audit of the frozen package. That review should mark concrete leaks, false redactions, wrong-target events, and cosmetic-only issues.

After that, the next engineering step is reviewed plate completion: add only evidence-supported plate corrections for the blocking intervals, then produce a reviewed plate candidate. A later milestone can combine reviewed face and reviewed plate regions into one final output when source compatibility allows.
