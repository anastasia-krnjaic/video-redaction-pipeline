# Third-party notices and excluded assets

This repository is published as a portfolio project. No open-source license is
granted for the project source code. Third-party software and model components
remain subject to their respective licenses.

This inventory identifies dependencies, external model sources and local development
media. It is not a replacement for their licenses or a grant of rights to this
project's source. The absence of a project-wide `LICENSE` file is intentional.

## Python dependencies and tools

Dependency requirements are defined in [pyproject.toml](pyproject.toml). The following
license labels were read from installed distribution metadata during portfolio
preparation; installed versions are observations, not a Python dependency lock.

| Component | Inspected version | Metadata license / expression |
| --- | --- | --- |
| Click | 8.4.2 | BSD-3-Clause |
| NumPy | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| opencv-python | 4.14.0.94 | Apache 2.0 |
| OpenVINO runtime | 2026.2.1 | Apache-2.0 |
| setuptools | 83.0.0 | MIT |
| pytest | 8.4.2 | MIT |
| Ruff | 0.16.0 | MIT |
| Playwright Python | 1.62.0 | Apache-2.0 |

Distribution license files can contain additional bundled-component notices.
Consult the actual installed wheel before redistributing dependencies. Python
environments, wheels and browser binaries are not included in this repository.
FFmpeg and ffprobe are external command-line prerequisites; their terms depend on
the selected build and enabled components. This repository does not bundle them.

## Frontend dependencies and tools

The versions and labels below come from the tracked
[frontend lockfile](frontend/package-lock.json), not from a license inferred for all
transitive dependencies or generated bundles.

| Component | Locked version | Lockfile license |
| --- | --- | --- |
| React / React DOM | 18.3.1 | MIT |
| lucide-react | 0.468.0 | ISC |
| TypeScript | 5.7.2 | Apache-2.0 |
| Vite | 6.0.5 | MIT |
| Vitest | 2.1.8 | MIT |
| ESLint | 9.17.0 | MIT |
| typescript-eslint | 8.18.1 | MIT |
| @vitejs/plugin-react | 4.3.4 | MIT |
| @playwright/test | 1.49.1 | Apache-2.0 |
| @types/node | 22.10.2 | MIT |
| @types/react | 18.3.12 | MIT |
| @types/react-dom | 18.3.1 | MIT |

`node_modules` and generated frontend bundles are excluded. Before distributing a
compiled frontend, preserve the notices required by its included dependencies and
assets. The frontend package's `private: true` field is an npm publication guard;
it does not determine the GitHub repository's visibility or source-code license.

## Active external models

### OpenVINO face-detection-adas-0001

Source: Intel/OpenVINO Open Model Zoo. The official
[model manifest](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/face-detection-adas-0001/model.yml)
explicitly references the project's
[Apache-2.0 license](https://github.com/openvinotoolkit/open_model_zoo/blob/master/LICENSE).
The runtime uses separately supplied XML/BIN files; expected local hashes are
pinned in `src/redactor/review_workspace.py`.

Model files are not included here. Anyone redistributing an exact model should
retain its applicable license/notices and verify its source/version; an upstream
model name alone does not establish the provenance of arbitrary local files.

### Open Image Models YOLOv9-S-608 plate detector

Source: [ankandrew/open-image-models](https://github.com/ankandrew/open-image-models),
recorded revision `f22000e02b30642f317cdba7755c0631638b109e`. The
[license at that revision](https://github.com/ankandrew/open-image-models/blob/f22000e02b30642f317cdba7755c0631638b109e/LICENSE)
is MIT for the upstream repository software. The
[selected configuration](data/evaluations/license_plate_detector_oim_yolov9_s_608_v1/selected_configuration.json)
records the exact ONNX filename, model SHA-256 and this project's license finding:
no separate weight-specific notice was found in the audited artifacts.

**Exact checkpoint and training-data redistribution rights remain unconfirmed.**
The MIT repository label is not presented as a blanket clearance for weights or
training data. Supply weights locally from their upstream source; do not include
them in this project's public source or demo package without resolving those terms.

### Historical candidates

Archived comparisons discuss YuNet/LPD-YuNet, other OpenVINO models, Intel YOLOv8n,
RF-DETR and RT-DETR. They are not active detector dependencies. Their weights,
upstream source checkouts, training/evaluation media and reference animations are
not included. Some local archived Intel assets carry AGPL-3.0 and additional
dataset-use notices; no blanket redistribution claim is made for the archive.

## Bodycam and demonstration media

[data/source.yaml](data/source.yaml) identifies local development footage from
Law&Crime Network's [YouTube video](https://www.youtube.com/watch?v=Bw-IJxxtCRs)
and explicitly requires verifying usage rights before sharing. No redistribution
grant for that footage was established during portfolio preparation.

Raw videos, broadcast screenshots/contact sheets, detector overlays, redacted
derivatives, generated sessions and review packages remain local and excluded.
Some review ZIPs also contain unredacted `Original.mp4`; a filename containing
“Final” or an approval record does not make the whole package safe to publish.
Privacy redaction does not establish copyright, consent or redistribution rights.

The README's [synthetic demo](docs/assets/video_redaction_demo.gif) is generated by
this repository's [demo script](scripts/generate_synthetic_demo.py). It uses drawn
shapes and injected observations, with no real people, plate characters, copied
footage, or third-party branding. Its tracking and rendering demonstration is not
evidence of pretrained detector accuracy or privacy quality on real video.

Other public visual examples should use original synthetic material, appropriately
consented self-recordings, or assets with documented licenses covering the intended
use. Do not reuse upstream demo images or animations solely because they appear
beside permissively licensed source code.

## Evaluation records

Public evaluation files contain derived counts, anonymous target identifiers,
geometry/interval annotations and provenance hashes, not source-frame imagery or
license-plate transcriptions. Their inclusion does not grant access to or rights
over the excluded source videos. See [Evaluation](docs/EVALUATION.md) for the
distinction between detector results, historical reviewed-final records and current
human approval.
