# Ground-Truth Changelog

## plate_ground_truth_v1.0 - 2026-08-11

Created the versioned machine-readable license-plate spatial Ground Truth.

- Derived from the completed audited annotation package
  `artifacts/plate_ground_truth_annotation_v2_assisted/run_20260811T185326Z`.
- Added `plate_ground_truth_schema_v1.0.json` and `plate_ground_truth_v1.0.json`.
- Published 600 human-reviewed plate-panel boxes: 240 for `day_plate` and 360 for
  `night_selective`.
- Preserved proposal-assisted review provenance with separate counts for migrated manual review,
  proposal accepted unchanged, proposal human-adjusted, and manual/no-proposal frames.
- Preserved GroundTruth v1.1 temporal policy for the certified plate windows.
- Excluded `artifacts/eval_day_plate_close.mp4` from certified plate Ground Truth.
- Did not calculate or publish plate detector recall, automatic coverage, or reviewed-final
  metrics.

## redaction_ground_truth_v1.0 - 2026-08-02

Created the selected machine-readable redaction-ground-truth schema and Face Pipeline v1 JSON file.

- Derived from frozen `GroundTruth_v1.1.md` and `face_benchmark_v1.1.json`.
- Preserved existing face semantics, dense windows, detector-scored ranges, detector-ignore ranges, privacy-required ranges, and keep-clear ranges.
- Added no new per-frame boxes or keyframes.
- Added schema support for future object classes while evaluating only face objects in this milestone.

## v1.1 - 2026-07-31

Created after complete human face-visibility verification of `GroundTruth_v1.0.md`.

- Reviewed 660 unique face frames.
- Reviewed 1,020 target decisions.
- Recorded zero uncertain decisions.
- Changed 60 `person_night_02` labels from `visible_or_partial_face` to `detector_ignore_cap_or_edge_only`.
- No `person_day_01` or `person_night_01` labels changed.

Changed contiguous parent-frame ranges for `person_night_02`:

| Parent frames | Old label | New label |
|---|---|---|
| `420-425` | `visible_or_partial_face` | `detector_ignore_cap_or_edge_only` |
| `442-446` | `visible_or_partial_face` | `detector_ignore_cap_or_edge_only` |
| `480-483` | `visible_or_partial_face` | `detector_ignore_cap_or_edge_only` |
| `502-535` | `visible_or_partial_face` | `detector_ignore_cap_or_edge_only` |
| `693-703` | `visible_or_partial_face` | `detector_ignore_cap_or_edge_only` |

Counts changed:

| Target | v1.0 detector-scored | v1.0 detector-ignore | v1.1 detector-scored | v1.1 detector-ignore |
|---|---:|---:|---:|---:|
| `person_night_02` | 252 | 108 | 192 | 168 |

Face detector denominators changed from:

- v1.0 micro total: 912
- v1.1 micro total: 852

Verification evidence:

- File: `data/ground_truth/verifications/face_visibility_verification_v1.1.completed.json`
- SHA-256: `f4401d88fffdd9dd5a90ea0c04f94bca77d5dd6b12b8081343dbbab1fb0ce429`

`GroundTruth_v1.0.md` remains immutable historical evidence. It is not deleted or invalidated.
