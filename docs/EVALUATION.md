# Evaluation

This document records the frozen release evaluation disposition. It does not create new benchmarks, reinterpret Ground Truth, tune thresholds, or combine incompatible evidence layers.

## Evidence Layers

The release keeps these claim layers separate:

- Detector recall: selected-detector proposals matched against frozen Ground Truth.
- Automatic eligible coverage: machine-generated privacy regions matched against frozen Ground Truth.
- Reviewed-final geometric coverage: authoritative reviewed regions in current Final media matched against frozen Ground Truth windows.
- Full-video human review: reviewer watched and approved the exact current Latest reviewed revision before Final publication.

Human full-video approval is review evidence, not a numerical benchmark. Geometric coverage is not a pixel-level zero-leak proof.

## Plate Detector Recall

Frozen detector recall uses the selected plate detector and the frozen plate evaluation policy.

| Case | Covered | Total | Recall |
| --- | ---: | ---: | ---: |
| Day | 236 | 240 | 0.983333 |
| Night | 321 | 360 | 0.891667 |
| Aggregate | 557 | 600 | 0.928333 |

Formal detector precision is not published because the frozen plate Ground Truth does not provide a complete spatial negative/ignore map.

## Automatic Eligible Coverage

Automatic eligible coverage measures machine-generated privacy regions only.

| Case | Covered | Total | Coverage |
| --- | ---: | ---: | ---: |
| Day | 240 | 240 | 1.0 |
| Night | 0 | 360 | 0.0 |
| Aggregate | 240 | 600 | 0.4 |

Night automatic coverage is zero for this benchmark layer because the frozen automatic eligibility rules do not credit the final reviewed/manual/human-corrected state.

## Reviewed-Final Plate Coverage

Day reviewed-final plate scoring is unavailable:

```text
DIFFERENT_SOURCE_NOT_COMPARABLE
```

The frozen `day_plate` benchmark is source-bound to `artifacts/eval_day_plate.mp4`, while the product Day Final uses a different full-case source. The correct release interpretation is unavailable, not zero and not failed.

Night reviewed-final strict geometric coverage is:

| Case | Covered | Total | Coverage |
| --- | ---: | ---: | ---: |
| Night | 284 | 360 | 0.788889 |

The 76 strict geometric uncovered frames were exhaustively visually audited:

| Classification | Count |
| --- | ---: |
| TRUE_VISIBLE_PRIVACY_LEAK | 0 |
| PLATE_PRESENT_BUT_UNDER_COVERED | 0 |
| VISUALLY_PROTECTED_BELOW_0_95_GEOMETRIC_THRESHOLD | 74 |
| OCCLUDED_OR_NOT_VISUALLY_LOCALIZABLE | 2 |

The frozen disposition is:

```text
NIGHT_FINAL_PRIVACY_SAFE_UNDER_VISUAL_AUDIT
```

The 0.788889 result is strict geometric Ground Truth coverage, not a visual privacy leak rate.

## Face Reviewed-Final Status

Face reviewed-final numerical scoring is:

```text
NOT_SUPPORTED_BY_FROZEN_FACE_EVALUATION_CONTRACT
```

The release does not invent a face metric from visual sweep approval. Face evidence remains documented through source lineage, reviewed media validation, region validation, and full-video human approval.

## Evaluation Guardrails

Do not:

- change Ground Truth after the release;
- adapt benchmark source identity post hoc;
- tune thresholds to match Final media;
- combine detector, automatic, reviewed-final, and human sweep evidence into one undefined score;
- present visual sweep approval as an automated numerical metric;
- describe strict geometric coverage as pixel-level zero-leak proof.
