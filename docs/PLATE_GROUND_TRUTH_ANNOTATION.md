# Plate Ground Truth Annotation

This document describes the local Phase A annotation package for license-plate spatial review. The
package prepares source-backed review frames and draft files only. It is not yet Ground Truth.

## Certified Targets

- `day_plate`: `plate_day_01`, `artifacts/eval_day_plate.mp4`, parent frames `240-479`,
  dense frames `0-239`, source interval `bodycam_sample.mp4` `00:01:41.000-00:01:49.000`.
- `night_selective`: `plate_night_01`, `artifacts/eval_night_selective.mp4`, parent frames
  `420-779`, dense frames `0-359`, source interval `night_candidate.mp4`
  `00:04:56.000-00:05:08.000`.
- `artifacts/eval_day_plate_close.mp4` is excluded from this package.

## Annotation Policy

Annotate the smallest defensible rectangle around the physical plate panel. Do not draw the whole
vehicle, bumper, broad search ROI, text overlay, or reflection as the target box. Characters do not
need to be readable. For glare or overlay cases, annotate the defensible physical panel extent and
record a note when uncertainty remains.

The broad ROI is search-only metadata. It guides navigation and draft filtering, but it is never a
Ground Truth box.

## Draft Policy

Historical model-derived proposals may seed the package only when source hash, frame range, target
identity, and clip mapping align with the Phase A target. Proposals are hidden by default, do not
count as reviewed, and are stored separately from human boxes. Showing a proposal records
`draft_shown=true`.

## Secondary Plate Handling

Secondary records support `additional_localizable_plate`, `ignore_small`, and
`ambiguous_review_required`. Additional localizable plates require a box and receive deterministic
provisional IDs. Ignore-small records require a reason and remain non-negative examples. Ambiguous
records block final export until resolved.

## Temporal Ground Truth Conflicts

The frozen temporal Ground Truth remains authoritative: the mandatory day and night plate targets are
visible/localizable throughout the certified Phase A intervals. If a reviewer disagrees on a frame,
they must save the frame as `temporal_gt_conflict` with a non-empty note. Whitespace-only notes are
invalid. A conflict does not count as `reviewed_box`, does not complete the mandatory annotation, and
blocks final export until explicitly resolved.

A temporal conflict is resolved only when the reviewer saves a valid tight human box and the status
becomes `reviewed_box`. Showing a draft never resolves a conflict or creates a human box. Conflict
creation, notes, resolution, and the final human box remain in the append-only change log.

## Progress And Export Gates

Progress is counted by benchmark and total mandatory frames. The initial real package should show
`0 / 240` day reviewed boxes, `0 / 360` night reviewed boxes, and `0 / 600` total reviewed boxes.
Draft presence never changes those counts.

Final review export is locked until all 600 mandatory frames have valid human boxes and no unresolved
temporal or ambiguous records remain. Completed export revalidates the full package identity before
writing: package schema, certified target IDs, benchmark IDs, object type, action, parent/dense frame
mapping, expected counts, manifest identity hash, and actual evaluation video SHA-256 values must all
match the certified Phase A configuration.

## Local Save Contract

The annotation server is a local loopback, single-reviewer workspace. One active human reviewer is
expected. Multiple browser tabs are possible, but each save includes a `state_revision`; stale tabs
whose expected revision no longer matches the current revision are rejected instead of silently
overwriting newer work. The tool does not provide multi-user merge semantics.

## Commands

```bash
redactor build-plate-annotation-package
redactor validate-plate-annotation-package artifacts/plate_ground_truth_annotation_v1/<run_id>
redactor summarize-plate-annotation-package artifacts/plate_ground_truth_annotation_v1/<run_id>
redactor serve-plate-annotation-package artifacts/plate_ground_truth_annotation_v1/<run_id>
```
