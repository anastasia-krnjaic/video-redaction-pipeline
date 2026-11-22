# Release Evidence

Public detector evidence is tracked under
[`data/evaluations/face_detector_adas_v1.0/`](../data/evaluations/face_detector_adas_v1.0/README.md)
and [`data/evaluations/plate_detector_oim_yolov9_s_608_v1/`](../data/evaluations/plate_detector_oim_yolov9_s_608_v1/README.md).
The plate record includes a read-only verifier, exact source-report hashes and
compact counts; it excludes media and per-frame proposal dumps.

Review and delivery evidence below is stored under ignored repository-local artifact
roots. It is a private local audit index, not a public demo download list. These
artifacts are not required for a clean source checkout.

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

This describes the recorded human approval. Day covers case source frames `0–3866`
and Night `0–1588`; it is not a claim about every frame of the original upstream
video. No current reviewed-final numerical face score or compatible Day plate score
is published. The historical Night `284/360` result belongs to an older revision,
as detailed below.

## Current Full Final Packages

| Package | Relative path | SHA-256 |
| --- | --- | --- |
| Day | `artifacts/submission_release_v1_0_3/full_packages/day_final_review_package.zip` | `675a9a9362350824e10bbd941cebf20f041b5237c74f68300156482d1da5791a` |
| Night | `artifacts/submission_release_v1_0_3/full_packages/night_final_review_package.zip` | `396a51bdbf4988be7e128be35c61de3190896abb0376b5ccc7b4fea593ef023d` |

These packages contain `Original.mp4` and `Automatic.mp4` as well as reviewed/Final
media. They must remain local; their names do not imply that all included media is
redacted or cleared for redistribution. See [Third-party notices](../THIRD_PARTY_NOTICES.md).

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

In particular, the Night geometric result `284/360 = 0.788889` and associated
76-frame visual audit were recorded for `latest_reviewed_133bd8a1e52cfa36`, with
reviewed-region set SHA-256
`133bd8a1e52cfa36163768630d487557b226e2b2e4f4931986013063741633e9`
and media SHA-256
`5a010c26f5f3367f14a1fc25327ec93e7f244bf8c22b760946140dd928968e60`.
They do not describe the current `latest_reviewed_41fcfdd5b7e2349d` Final above.
The original evidence remains at
`artifacts/reviewed_final_evaluation_reconciliation_v1/run_20260816T192539Z/`.

## How To Use This Index

Use this document to locate evidence when the ignored artifact roots are present on the release machine. Do not treat these relative paths as required files in a fresh source checkout.

Portable delivery package identity is carried by package manifests and checksums, not by absolute local paths. The current Final evidence here should be read together with [EVALUATION.md](EVALUATION.md), which keeps detector metrics, automatic coverage, reviewed-final geometric evidence, and human full-video review separate.

Source bodycam footage, derivatives, screenshots containing it, model weights and
generated review packages are excluded from the public repository. Recorded privacy
review does not establish copyright, consent or redistribution rights.

## Portfolio Release Preparation v1 validation — 2026-11-21

These checks apply to the uncommitted release-preparation working tree based on
`99587d0dc5acdd4842007234dd0554a1ff4b9c1f`, not to a new Git tag. No changes were
staged, committed, pushed, or published. The owner has intentionally chosen
portfolio publication without granting an open-source license for project source;
no project `LICENSE` file is added.

The clean-source validation copies contained only tracked files plus the proposed
new public source/fixture/documentation files. Ignored media, model weights,
private annotation packages, sessions, `.venv`, `node_modules`, and frontend
builds were excluded. A fresh temporary Python environment installed `.[dev]`
without OpenVINO. Python caches and pytest outputs were directed outside the
repository; pytest's cache provider was disabled.

| Check | Result | Wall time |
| --- | --- | ---: |
| Public default Python suite | **799 passed, 4 skipped, 0 failed/errors** | 89.78 s |
| Optional private plate lineage | **1 passed, 25 deselected** | 4.43 s |
| Real Firefox synthetic-workspace smoke | **2 passed, 0 skipped** | 10.39 s |
| Synthetic demo temporal/media/API tests, frontend built | **3 passed** | 3.35 s |
| README synthetic tracking smoke | **1 passed** | 11.10 s |
| Ruff lint | Passed | 0.12 s |
| Ruff format check | Passed; 161 files already formatted | 0.09 s |
| Git diff whitespace check | Passed | 0.14 s |
| Fresh `npm ci --no-audit --no-fund` | Passed; 227 packages installed in temporary copy | 3.02 s |
| Frontend typecheck | Passed | 2.93 s |
| Frontend lint | Passed | 3.16 s |
| Frontend unit tests | **72 passed across 13 files** | 2.12 s |
| Frontend production build | Passed | 1.45 s |
| Model hash/load/blank-frame inference smoke | Both active models passed using existing local weights | 2.41 s |
| Compact plate evidence verification | Public checks and exact local report derivation passed | <0.1 s each |

The four default skips are two opt-in Firefox tests, one explicitly local-assets
test, and one demo integration test requiring a real frontend build. Its two
model-free temporal tests run in the default suite. The optional tests were run
separately with their prerequisites and are not reported as clean-clone tests.
The clean public suite did not create `frontend/dist` or `frontend/node_modules`.

The earlier clean-copy run exposed one additional archived test that imported
OpenVINO merely to check a missing model path. That test now supplies an empty
runtime module and still asserts the missing-file failure. Backend packaging
checks use tiny temporary frontend fixtures; an autouse test guard rejects
implicit frontend installation/build commands. No production behavior was changed.

Validation commands (from the appropriate repository/source-copy root, with the
selected Python environment active):

```bash
export PYTHONDONTWRITEBYTECODE=1
python -m pytest -rs -p no:cacheprovider --basetemp=/tmp/redactor-public-tests
REDACTOR_RUN_LOCAL_ASSET_TESTS=1 python -m pytest -m local_assets -rs \
  -p no:cacheprovider --basetemp=/tmp/redactor-local-tests tests/test_plate_ground_truth.py
REDACTOR_RUN_FIREFOX_E2E=1 python -m pytest -rs -p no:cacheprovider \
  --basetemp=/tmp/redactor-firefox-tests tests/test_final_review_workspace_firefox.py
python -m pytest -q -rs -p no:cacheprovider --basetemp=/tmp/redactor-demo-tests \
  tests/test_synthetic_demo.py
python -m ruff check --no-cache .
python -m ruff format --check --output-format concise .
npm --prefix frontend ci --no-audit --no-fund
npm --prefix frontend run typecheck
npm --prefix frontend run lint
npm --prefix frontend test -- --cache=false
npm --prefix frontend run build
python -B scripts/verify_plate_evidence.py
git diff --check
```

`.[dev]` and `.[dev,inference]` were installed in separate fresh temporary venvs.
The inference installation pins OpenVINO 2026.2.1 and compiled/inferred with both
existing, hash-verified local models; no weights were downloaded. See
[Model setup](MODEL_SETUP.md) for observed versions and upstream model identities.
The real Firefox run used the already installed Playwright 1.62.0/browser pair;
no browser binary was downloaded.

With explicit owner approval, Ruff formatted the 34 previously flagged files
(22 production modules and 12 test files). Independent before/after checks found
identical Python ASTs, string values, and comment sequences for every formatted
file. All 40 production Python files also retain identical ASTs relative to the
audited HEAD. No formatter exclusions were added. The public Python suite and
real Firefox smoke were rerun after formatting; the table records those final
runs. Frontend validation above remains applicable because frontend source and
configuration did not change.

Existing historical snapshots, frozen Ground Truth, selected model/configuration
files, and frontend source are unchanged. A new v037 overview snapshot preserves
the pre-correction document and its source/configuration provenance. Formatting
does not change detector, tracking, rendering, review, or finalization behavior.


## Final portfolio-publication validation — 2026-11-22

The final working tree remains based on `99587d0dc5acdd4842007234dd0554a1ff4b9c1f`.
The no-license portfolio status is intentional, not an unresolved release choice.
No project source `LICENSE` or license metadata was added. Third-party terms remain
separate, as recorded in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

A new temporary source-only copy and fresh `.[dev]` environment were used for the
public suite. The environment contained no OpenVINO; the copy contained no model
weights, private videos, ignored annotation packages, frontend build or installed
frontend dependencies. Separate temporary frontend/browser validation used a fresh
`npm ci` and production build. All Python runs set `PYTHONDONTWRITEBYTECODE=1`,
disabled pytest caching, and used basetemp/cache/output directories outside the
repository. No browser or model download was performed.

| Final check | Result | Wall time |
| --- | --- | ---: |
| Fresh Python `.[dev]` installation | Passed | 11.44 s |
| Public asset-free Python suite | **801 passed, 4 skipped, 0 failures/errors** | 92.46 s |
| Synthetic demo tests with built frontend | **5 passed** | 5.97 s |
| README synthetic tracking smoke, fresh developer environment | **1 passed** | 0.88 s |
| Existing real Firefox smoke | **2 passed, 0 skipped** | 12.65 s |
| Ruff check | Passed | 0.47 s |
| Ruff format check | Passed; 161 files already formatted | 0.47 s |
| Fresh `npm ci --no-audit --no-fund` | Passed; 227 packages | 4.31 s |
| Frontend typecheck | Passed | 0.25 s |
| Frontend lint | Passed | 3.07 s |
| Frontend unit tests | **72 passed across 13 files** | 1.99 s |
| Frontend production build | Passed | 4.29 s |
| Clean-copy synthetic demo and GIF generation | Passed; 160 source frames, 320 rendered regions | 1.85 s |
| Public plate evidence verifier | Passed | 0.57 s |
| Clean-copy documentation links | 98 local targets checked; none broken | 0.13 s |
| Git diff whitespace check | Passed | 0.04 s |

The four public-suite skips are the two explicitly opt-in Firefox tests, one
private plate-lineage test and the workspace integration test needing a built
frontend. Firefox and the workspace integration were exercised separately above.
The private lineage check was not counted as a clean-clone test; its earlier
result remains in the preparation record. The final local plate report derivation
was independently checked against the original report as well.

The additional demo tests check deterministic source geometry, absent synthetic
observations during the stated gaps, actual rendered pixels, manual correction
Preview/Save through the real API, GIF frame count/duration/size, repeatable encoding
and explicit overwrite protection. No test was weakened or production gate bypassed.

### Public synthetic assets

| Candidate asset | Dimensions | Size |
| --- | --- | ---: |
| [`video_redaction_demo.gif`](assets/video_redaction_demo.gif) | 840 × 416; 80 frames, 10 fps, 8 seconds | 562,546 bytes |
| [`review_workspace_synthetic.png`](assets/review_workspace_synthetic.png) | 1400 × 881 | 226,005 bytes |

The GIF uses injected observations with the real trackers, bounded tracker
interpolation and shared renderer; it does not run either pretrained detector or
the separate face privacy-gap-recovery stage. The clean-copy run reproduced the
candidate GIF SHA-256 exactly:
`6bcfbe1715fcd816b47236257266f104c3aaa38737c2237cfddeca939a6ff699`.

The screenshot was captured from the existing React interface and Python server,
using only the generated synthetic case. It shows an unsaved manual-correction
draft at source frame 159, not a verified or finalized correction. No frontend
code or displayed UI was altered. Decoded GIF frames and the actual screenshot
were visually inspected: no real people, readable plates, third-party branding,
source media, personal paths or session identifiers appear.

The generator remains the visual source of truth. To reproduce without a frontend
build, from a source-only copy with developer dependencies and FFmpeg/FFprobe:

```bash
python scripts/generate_synthetic_demo.py \
  --output /tmp/redactor-demo-final --skip-workspace \
  --gif /tmp/redactor-demo-final.gif
```

Use a new output path; no models, private inputs or network access are required by
the generator. See the [demo guide](DEMO_GUIDE.md) for the complete workspace path
and explicit regeneration of the README GIF.

### Final diff and public hygiene

The complete candidate tree contains only source, documentation, sanitized evidence
and the two inspected synthetic visual assets. No real media, face/plate crops,
readable plate transcriptions, model binaries, review sessions, delivery archives,
dependency trees, browser binaries, personal absolute paths or credential-pattern
matches were found. Existing Git author metadata is retained; no history was rewritten.

The 61 changed/new files are classified once each: 28 formatting-only, 10 portfolio
documentation, 11 public-test portability/fixture files, 2 setup/metadata files,
4 evaluation-evidence files, 3 synthetic-demo tooling/documentation files,
2 synthetic visual assets and 1 third-party notice file. All 40 production Python
modules retain identical ASTs relative to the audited HEAD; production comments,
frozen ground truth, detector configuration, model hashes, frontend source and
existing overview snapshots are unchanged. **No production algorithm or runtime
logic changed.** No files were staged, and no commit, tag, push, release or visibility
change was performed.
