# Plate Evaluation

This document describes the frozen plate evaluation implementation for
`plate_ground_truth_v1.0`. It documents scoring behavior only; it does not
publish the current detector recall, automatic coverage, or reviewed-final
coverage results. Current frozen release results are recorded in
[EVALUATION.md](EVALUATION.md).

## Ground Truth Scope

The plate Ground Truth is `data/ground_truth/plate_ground_truth_v1.0.json`.
Its content SHA-256 is bound by the evaluation policy and must remain
unchanged for reports using `plate_evaluation_policy_v1.0`.

Certified plate windows are:

- `day_plate`: dense frames `0-239`, canonical parent frames `240-479`.
- `night_selective`: dense frames `0-359`, canonical parent frames `420-779`.

All scoring inputs are normalized to evaluation-video parent frame identity.
The scoring path rejects ambiguous mapping and does not infer mapping from
duration, approximate time, or directory names.

## Claim Layers

The implementation separates three claim layers:

- `detector`: direct selected-detector proposal matching against GT.
- `automatic`: automatic rendered-region GT coverage with automatic provenance
  only.
- `reviewed_final`: reviewed/final GT coverage gated by lineage and human
  final-review evidence.

These layers are reported separately. There is no blended detector,
automatic, and reviewed-final score.

## Detector Matching Policy

Detector mode uses `plate_detector_matching_v1.0` from
`data/evaluations/plate_evaluation_policy_v1.0.json`.
The policy is both internally content-hashed and evaluator-pinned to the
supported `plate_evaluation_policy_v1.0` semantic SHA-256. Re-stamping changed
scoring semantics under the same policy ID fails closed.

Eligible detector predictions must be direct selected OIM YOLOv9-S-608
outputs, `license_plate`, inside the 1920x1080 coordinate space, and scored at
or above `0.30`. Tracker, interpolation, automatic privacy, manual, reviewed,
and preserved-region provenance cannot receive detector credit.

Pair geometry uses xyxy half-open area semantics. A prediction matches a GT
box only when every policy threshold passes: positive intersection, IoU,
GT coverage, prediction coverage, area ratio, width ratio, height ratio, and
center-inside-expanded-GT. One-to-one assignment is performed globally across
all GT plate instances and eligible detector predictions on each canonical
frame, then per-target metrics are derived from that assignment. Greedy
assignment is deterministic using the policy tie-break order, with stable GT
identity only as a final tie-break.

## Precision Limitation

Formal plate detector precision is intentionally unavailable. The Ground Truth
does not include a spatial negative/ignore map sufficient to certify every
unmatched detector proposal as a false positive.

Reports use:

- `formal_precision.value = null`
- `formal_precision.reason = insufficient_spatial_negative_ignore_map`

Unmatched and duplicate proposals may appear as diagnostics, not certified
precision.

## Automatic Coverage Policy

Automatic mode uses `plate_automatic_coverage_v1.0`. For each required GT
target frame, the scoring path gathers eligible automatic plate regions, expands
them with the runtime automatic render margin, clips them to the GT box, and
computes deterministic rectangle-union area.

A GT frame is covered when union GT coverage is at least `0.95`. Overlapping
regions are not double-counted.

Eligible automatic provenance is limited to direct detector evidence,
accepted automatic tracks, accepted persistence/gap recovery, accepted bounded
interpolation, and effective automatic privacy regions. `needs_review`,
manual correction, reviewer redraw, reviewed replacement, preserved reviewed
regions, Latest reviewed-only geometry, and Final-only geometry are excluded.
Manual or reviewed contamination fails closed.

## Over-Redaction Diagnostics

Automatic reports include diagnostics such as automatic redacted area per
frame, redacted area divided by frame area, duplicate region counts, unmatched
automatic plate-region counts, and association-unverified coverage frames.
These are not a universal scene precision score and do not label conservative
margin as an error.

## Reviewed-Final Requirements

Reviewed-final mode uses the same `0.95` geometric GT coverage threshold on
authoritative reviewed regions, but geometric coverage alone is not enough to
publish metrics.

Required trust gates include:

- reviewed region set SHA-256.
- Latest reviewed revision identity and current pointer.
- Final revision identity and current pointer.
- supplied reviewed-region artifact raw SHA-256 matching the immutable Latest
  reviewed revision.
- supplied reviewed-region semantic records SHA-256 matching the authoritative
  reviewed-region identity.
- source lineage.
- category-run lineage.
- media validation.
- region validation.
- human sweep acknowledgement.
- current finalization.

If any gate fails, `publishable_metrics` is false and exact blockers are
reported. Current release dispositions are recorded in [EVALUATION.md](EVALUATION.md)
after applying these gates.

Allowed wording when all gates pass is limited to the certified windows: no
uncovered GT plate frames found in those windows, with human final-review
evidence present. The scoring path does not claim automated pixel-level zero
leak.

## Fail-Closed Behavior

Reports fail closed for invalid GT, GT hash mismatch, unsupported policy,
policy hash mismatch, source hash mismatch, wrong benchmark or frame mapping,
incomplete source span, wrong detector/config/threshold, contaminated
detector or automatic provenance, missing artifacts, stale Latest reviewed or
Final pointers, invalid finalization, missing sweep evidence, failed media or
region validation, and unsupported report schemas.

`--validate-only` validates lineage and inputs and writes non-publishable
reports without calculating scoring metrics.

## Implementation Separation

The implementation freezes policy and scoring code separately from release
reporting. Synthetic tests may calculate metrics to verify policy behavior.
Real detector, automatic, and reviewed-final artifacts are reported through the
release evaluation disposition rather than by rewriting this policy reference.
