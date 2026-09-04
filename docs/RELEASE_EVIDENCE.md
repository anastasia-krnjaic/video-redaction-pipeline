# Release Evidence

Release evidence is stored under ignored repository-local artifact roots. These artifacts are not required for a clean source checkout, but they preserve local approval, evaluation, delivery, performance, cleanup, and reconciliation proof for the frozen release.

This document is the canonical index for current Final IDs and package checksums. Paths below are repository-relative evidence locations for local audit only; package identity is carried by manifests and SHA-256 values, not by absolute local paths.

## Evidence Roots

| Evidence | Relative artifact root |
| --- | --- |
| Current submission release package | `artifacts/submission_release_v1_0_3/` |
| Current full final package zips | `artifacts/submission_release_v1_0_3/full_packages/` |
| Current submission sample | `artifacts/submission_release_v1_0_3/submission/Closure_video_redaction_submission/sample/` |

## Current Approved Final Media

| Case | Latest reviewed revision | Finalization ID | Final media SHA-256 |
| --- | --- | --- | --- |
| Day | `latest_reviewed_4808aa899c5da1a8` | `finalization_a9cc6da4ac152e74` | `4f0c57731b4d99ed2b4397472c8d991d3debb9962de1b4419cb941cff296d837` |
| Night | `latest_reviewed_41fcfdd5b7e2349d` | `finalization_2e7257daf1187805` | `cc8982b317c439d7965dd151c562f0c01004b3252cfd5f63266fa95b0a21e88a` |

For both current cases, the reviewer watched the exact Latest reviewed video in full and recorded explicit full-video approval for that revision. The immutable Final media was then produced from the approved Latest reviewed revision.

## Current Full Final Packages

| Package | Relative path | SHA-256 |
| --- | --- | --- |
| Day | `artifacts/submission_release_v1_0_3/full_packages/day_final_review_package.zip` | `675a9a9362350824e10bbd941cebf20f041b5237c74f68300156482d1da5791a` |
| Night | `artifacts/submission_release_v1_0_3/full_packages/night_final_review_package.zip` | `396a51bdbf4988be7e128be35c61de3190896abb0376b5ccc7b4fea593ef023d` |

## Submission Sample

| Field | Value |
| --- | --- |
| Case | Night |
| Source frames | 484-686 inclusive |
| Frame count | 203 |
| Duration | About 6.77 seconds |
| Parent Latest reviewed | `latest_reviewed_41fcfdd5b7e2349d` |
| Parent Finalization ID | `finalization_2e7257daf1187805` |
| Parent Final media SHA-256 | `cc8982b317c439d7965dd151c562f0c01004b3252cfd5f63266fa95b0a21e88a` |

The sample evidence root contains machine validation, checksums, a manifest, reviewed-region records, and a human visual acceptance record. The sample was separately watched after generation and accepted with no known visible sensitive-data leak observed. This sample claim is a bounded human visual acceptance record, not a mathematical zero-leak certification for the complete product videos.

## Historical Evidence

Earlier release-candidate Final media and delivery packages are historical only. They are superseded by the current approved Final media and package checksums above and must not be used as current submission evidence.

## How To Use This Index

Use this document to locate evidence when the ignored artifact roots are present on the release machine. Do not treat these relative paths as required files in a fresh source checkout.

Portable delivery package identity is carried by package manifests and checksums, not by absolute local paths. The current Final evidence here should be read together with [EVALUATION.md](EVALUATION.md), which keeps detector metrics, automatic coverage, reviewed-final geometric evidence, and human full-video review separate.
