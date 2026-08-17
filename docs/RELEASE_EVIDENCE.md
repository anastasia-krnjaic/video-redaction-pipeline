# Release Evidence

Release evidence is stored under ignored `artifacts/` roots. These artifacts are not required for a clean source checkout, but they preserve local certification, evaluation, delivery, performance, cleanup, and reconciliation proof for the frozen release.

## Evidence Roots

| Evidence | Relative artifact root |
| --- | --- |
| Canonical review workspace | `artifacts/reviewed_final_combined_v2_1/run_20260813T210250Z/review_workspace/` |
| Final certification | `artifacts/reviewed_final_certification_v1/run_20260816T181759Z/` |
| Evaluation reconciliation | `artifacts/reviewed_final_evaluation_reconciliation_v1/run_20260816T192539Z/` |
| Delivery packages | `artifacts/reviewed_final_delivery_v1/run_20260816T181759Z/` |
| Production-readiness evidence | `artifacts/performance_production_readiness_v1/run_20260816T195729Z/` |
| Cleanup audit | `artifacts/cleanup_audit_v1/run_20260816T224650Z/` |
| Session reconciliation | `artifacts/review_session_reconciliation_v1/run_20260815T201114Z/` |

## Certified Media

| Case | Final revision | Final media SHA-256 |
| --- | --- | --- |
| Day | `finalization_aa68ef16b4396b6d` | `f2c16de58e145a7c5073e6fdbf9af9b6c39a9033ce1034a59099901d6333d8a0` |
| Night | `finalization_80444a7c087a8a59` | `5a010c26f5f3367f14a1fc25327ec93e7f244bf8c22b760946140dd928968e60` |

## Delivery Packages

| Package | Relative path | SHA-256 |
| --- | --- | --- |
| Day | `artifacts/reviewed_final_delivery_v1/run_20260816T181759Z/day/day_final_delivery.zip` | `dd20170dd1674899051b42ef36806af0c6160acf265afa73fa4f84af0f006eb0` |
| Night | `artifacts/reviewed_final_delivery_v1/run_20260816T181759Z/night/night_final_delivery.zip` | `a17a4f7f3bc5bad4eb99ddc61f89c05c44a0c608ca8498ed92ef88d484c98286` |
| Combined | `artifacts/reviewed_final_delivery_v1/run_20260816T181759Z/combined_day_night_final_delivery.zip` | `dfc71335b72d7e9201fed1407840e2b03aa5f87e9a95cf8cac94e87469437f69` |

## How To Use This Index

Use this document to locate evidence when the ignored artifact roots are present on the release machine. Do not treat these relative paths as required files in a fresh source checkout.

Portable delivery package identity is carried by package manifests and checksums, not by absolute local paths.
