# Evaluation

Detector recall, automatic coverage, reviewed-final geometry and human review are
separate evidence layers. Metrics apply to their recorded source, configuration,
Ground Truth and revision; they do not automatically transfer to newer Final media.

## Public detector results

The public source repository includes compact, non-media evidence for these results:

| Detector / target | Hits | Scored target frames | Recall |
| --- | ---: | ---: | ---: |
| ADAS / Day face | 299 | 300 | 0.996667 |
| ADAS / standing Night face | 359 | 360 | 0.997222 |
| ADAS / seated Night face | 175 | 192 | 0.911458 |
| **ADAS micro aggregate** | **833** | **852** | **0.977700** |
| OIM / Day plate | 236 | 240 | 0.983333 |
| OIM / Night plate | 321 | 360 | 0.891667 |
| **OIM micro aggregate** | **557** | **600** | **0.928333** |

Both tables use primary threshold `0.30`. These are small, source-bound benchmark
results, not universal detector rankings or full-video privacy scores.

- [Face evidence](../data/evaluations/face_detector_adas_v1.0/README.md) contains
  metrics, corrected adjudication, validation and input hashes for GroundTruth v1.1.
  Longest miss runs are 1, 1 and 9 frames. The `0.20` proposal threshold is temporal
  support, not the primary operating point in this table.
- [Plate evidence](../data/evaluations/plate_detector_oim_yolov9_s_608_v1/README.md)
  contains compact metrics derived from the existing validated local run, input
  hashes and a read-only verifier. Longest miss runs are 2 and 7 frames. GT contains
  600 proposal-assisted, human-reviewed boxes across two certified dense intervals.
  Formal precision is unavailable without a complete spatial negative/ignore map.

The records support checking identities, denominators and arithmetic in a clean
checkout. Independent inference/scoring reproduction requires excluded local media
and detector artifacts. Neither record measures tracking, manual corrections,
rendered privacy quality or zero leaks. See [Plate Evaluation](PLATE_EVALUATION.md)
for the matching and coverage contract.

## Historical automatic plate coverage

The local run `artifacts/plate_evaluation_v1/run_20260812T004154Z/` records automatic
eligible coverage of Day `240/240`, Night `0/360`, aggregate `240/600 = 0.4`.
These are historical automatic-layer diagnostics; the full machine reports are
ignored local artifacts, not part of the compact public detector record.

Night coverage is zero under that run's frozen automatic eligibility rules, which
exclude reviewed/manual corrections. This is not a measure of the current Final
video. Automatic coverage must not be relabeled as detector recall.

## Historical reviewed-final plate evidence

The August 16 reconciliation reported Night strict geometric coverage of
`284/360 = 0.788889` for **this exact historical revision**:

| Identity | Value |
| --- | --- |
| Latest reviewed revision | `latest_reviewed_133bd8a1e52cfa36` |
| Reviewed-region set SHA-256 | `133bd8a1e52cfa36163768630d487557b226e2b2e4f4931986013063741633e9` |
| Rendered video SHA-256 | `5a010c26f5f3367f14a1fc25327ec93e7f244bf8c22b760946140dd928968e60` |

The source report is
`artifacts/reviewed_final_evaluation_reconciliation_v1/run_20260816T192539Z/reviewed_final_results/night/metrics.json`.
Its associated `night_uncovered_audit/visual_classification.json` records review of
the 76 geometrically uncovered frames: 74 visually protected below the 0.95 geometry
threshold, two occluded/not localizable, and zero classified true visible leaks or
actionable undercoverage. These are historical local records, not a fresh audit.
The report also contains inconsistent top-level/per-target publishability flags;
it is retained as historical evidence, not a current public certification.

The current Night Final instead uses `latest_reviewed_41fcfdd5b7e2349d` and media
SHA-256 `cc8982b317c439d7965dd151c562f0c01004b3252cfd5f63266fa95b0a21e88a`.
**The historical 284/360 metric and 76-frame visual disposition do not certify this
newer revision.** A new compatible evaluation, or verified evidence of equivalence,
would be required to attach numerical results to it.

Day reviewed-final scoring remains `DIFFERENT_SOURCE_NOT_COMPARABLE`: the frozen
`day_plate` benchmark and the full-case Day Final use different source identities.
Unavailable is not zero and is not a failed score.

## Current reviewed-final and human evidence

Current Final identities are indexed in [Release Evidence](RELEASE_EVIDENCE.md).
Local manifests record full-case human sweep acknowledgements bound to the current
Latest reviewed revisions, source/configuration checks, media/region validation,
and immutable Final publication. This is recorded human approval, not independently
repeated visual verification or an automated numerical privacy benchmark.

Face reviewed-final numerical scoring is
`NOT_SUPPORTED_BY_FROZEN_FACE_EVALUATION_CONTRACT`. Source lineage and human approval
do not supply a missing face coverage metric.

The local Night submission sample covers case source frames `484–686`, 203 frames
(about 6.77 seconds). Its separate acceptance record reports no known visible leak
observed in that sample. This bounded human observation is neither a full-video
zero-leak certification nor permission to redistribute the media.

## Claim boundaries

- Preserve frozen GT and scoring policy; do not tune them to fit Final media.
- Attribute every evaluated result to its exact source and revision.
- Keep raw detector, automatic, reviewed-final and human evidence separate.
- Restrict dense-interval claims to their certified windows; full-case human review
  does not extend a numerical benchmark to the full original source video.
- Do not equate geometric coverage, renderer approval or visual acceptance with
  automated pixel-level or global zero-leak proof.
- Do not publish formal plate precision or treat historical selection precision as
  a result under the current frozen plate evaluation contract.
