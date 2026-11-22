# OIM plate detector: compact evaluation evidence v1

This record preserves **raw detector recall on two certified dense intervals**
from the existing `run_20260812T004154Z` evaluation. It does not rerun inference,
alter Ground Truth, or score current reviewed/final media.

| Benchmark | Detector hits | Scored target frames | Recall | Longest miss run |
| --- | ---: | ---: | ---: | ---: |
| `day_plate` | 236 | 240 | 0.983333 | 2 frames |
| `night_selective` | 321 | 360 | 0.891667 | 7 frames |
| Micro aggregate | 557 | 600 | 0.928333 | — |

The selected detector is `oim_yolov9_s_608`, with uniform small-object tiles and
threshold `0.30`. The [selected configuration](../license_plate_detector_oim_yolov9_s_608_v1/selected_configuration.json)
pins the model SHA-256 and detector configuration. The
[evaluation policy](../plate_evaluation_policy_v1.0.json) freezes matching semantics.

## Files and derivation

- `metrics.json`: scalar counts, rates, certified frame intervals, source hashes,
  detector identity, and limitations. Its schema is
  `compact_plate_detector_evidence.v1`, not the full evaluator report schema.
- `input_manifest.json`: exact SHA-256 values of the compact metrics, tracked
  Ground Truth/policy/configuration, original local reports, and report-integrity
  record. Raw file hashes and semantic `content_sha256` values are distinct.
- [`scripts/verify_plate_evidence.py`](../../../scripts/verify_plate_evidence.py): read-only, standard-library verification. Its
  `compact_case` function specifies the extraction from the original reports.

The original reports were produced at commit
`1cf539af07350879cd6e9be6b2c4421387aa1ac9` (`frozen-plate-evaluator-v1`).
They report detector mode, publishable metrics, ready validation and no blockers.
Their nested validation objects retain `validation_only: true`, the validation
helper's metadata; the top-level scored reports contain populated target metrics.
This compact record preserves counts from those scored reports, not from the
separate `reviewed_final_*_validation_only` reports.

The aggregate is the sum of hits divided by the sum of scored target frames,
rounded to six decimal places. No per-frame boxes, images, proposal lists, plate
characters, session exports, or model files are included here.

## Verify without models or media

Run from the repository root:

```bash
python3 -B scripts/verify_plate_evidence.py
```

This checks tracked input hashes, GT denominators and frame mappings, detector
identity, and arithmetic. It does **not** independently establish detector hits:
the source images and original detections are excluded from the public repository.

On the original evaluation machine, also compare the exact local source reports:

```bash
python3 -B scripts/verify_plate_evidence.py \
  --local-source-run artifacts/plate_evaluation_v1/run_20260812T004154Z
```

The second command checks source-report hashes and reproduces the compact
extraction in memory. Both commands are read-only and write no reports.
Rerunning detection/scoring requires authorized copies of the source videos and
detector inputs identified by the manifest, plus `redactor evaluate-plates
--mode detector`; the private asset paths are not a public quickstart.

## Scope and limits

The [frozen GT](../../ground_truth/plate_ground_truth_v1.0.json) contains 600
proposal-assisted, human-reviewed target-frame boxes: one target in each interval.
Day dense frames `0–239` map to evaluation-video frames `240–479`; Night dense
frames `0–359` map to `420–779`. These are source-bound intervals, not full videos
or an independent held-out dataset.

Formal precision is unavailable because GT lacks a complete spatial negative/ignore
map. Tracking, interpolation, manual corrections and rendering do not receive
detector credit. Recall is not final privacy coverage or a zero-leak guarantee.

The older [detector-selection decision](../license_plate_detector_oim_yolov9_s_608_v1/selected_detector_decision.md)
uses a different historical comparison contract. Its precision, recall and gap
figures must not be combined with this record. Current and historical reviewed-final
evidence is described separately in [Evaluation](../../../docs/EVALUATION.md).
