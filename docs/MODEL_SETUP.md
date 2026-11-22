# Local Model Setup

The public developer suite and synthetic demo do not need model weights or
OpenVINO. Real inference uses exactly two pretrained detectors. Model binaries
are intentionally not stored in Git; obtain them separately and verify their
bytes before processing your own video. No training of these models is claimed.

## Python and tools

From the repository root, with Python 3.12 available:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,inference]"
redactor doctor
python -c "import openvino; print(openvino.__version__)"
```

The `inference` extra pins **OpenVINO 2026.2.1**, the version observed in the
working environment. Both the ADAS IR adapter and the OIM ONNX adapter use
OpenVINO's CPU runtime; ONNX Runtime and the upstream OIM Python package are not
required by the active implementation. Core dependencies are Click, NumPy and
OpenCV. The `dev` extra supplies pytest, Ruff and Python Playwright.

Release-preparation environment: Python **3.12.2**, OpenVINO **2026.2.1**,
OpenCV Python **4.14.0.94**, FFmpeg and FFprobe **9.0.1**, Node **26.0.0**,
npm **11.16.0**, on macOS. Fresh temporary environments successfully installed
both `.[dev]` and `.[dev,inference]`. The developer-only environment resolved
NumPy **2.5.5**; the inference environment resolved NumPy **2.4.6** to satisfy
OpenVINO's dependency constraint. Both resolved Click **8.5.1**, pytest **8.4.2**,
Ruff **0.17.6** and Playwright **1.65.2**. Existing Firefox automation was checked
with the already installed Playwright **1.62.0** and its browser, without
downloading another browser. These observations are not a full dependency lock.
Python metadata allows 3.11 and 3.12;
3.11 and other operating systems are not claimed as tested by this preparation.
Frontend package versions are recorded in `frontend/package-lock.json`.

Install FFmpeg/FFprobe and Node/npm separately for your platform. `redactor
doctor` checks Python and the media tools; it does not validate OpenVINO or
model hashes. Use the checks below as well. See [release evidence](RELEASE_EVIDENCE.md)
for validation outcomes and any installation limitations.

## Frozen face model

- Runtime ID: `openvino_face_adas_0001`.
- Model: Open Model Zoo **face-detection-adas-0001**, **2022.3, FP32**.
- Official [2022.3 model manifest](https://github.com/openvinotoolkit/open_model_zoo/blob/2022.3.0/models/intel/face-detection-adas-0001/model.yml).
- Official [XML](https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/face-detection-adas-0001/FP32/face-detection-adas-0001.xml)
  and [BIN](https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/face-detection-adas-0001/FP32/face-detection-adas-0001.bin).

Use this exact FP32 release, not a newer export or FP16 variant. The local XML's
embedded conversion metadata identifies OpenVINO 2022.3; that model export
version is distinct from the installed inference-runtime version above.

## Frozen plate model

- Runtime ID: `oim_yolov9_s_608`.
- Model: Open Image Models **YOLOv9-S-608**, end-to-end ONNX.
- Upstream repository revision: `f22000e02b30642f317cdba7755c0631638b109e`.
- [Pinned upstream model hub](https://github.com/ankandrew/open-image-models/blob/f22000e02b30642f317cdba7755c0631638b109e/open_image_models/detection/core/hub.py).
- [Upstream ONNX asset](https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-s-608-license-plates-end2end.onnx).

The upstream asset URL is not a content-addressed release: the SHA-256 below is
the authority for the selected checkpoint. Stop on a mismatch; do not substitute
a new checkpoint or update the frozen hash to make it pass. Repository software
licensing and exact weight redistribution rights are separate; see
[third-party notices](../THIRD_PARTY_NOTICES.md).

## Local layout and hashes

Create these directories and place the separately obtained files at these paths:

```text
models/
├── face-detection-adas-0001/
│   ├── face-detection-adas-0001.xml
│   └── face-detection-adas-0001.bin
└── open-image-models/
    └── yolo-v9-s-608-license-plates-end2end.onnx
```

| File | Expected SHA-256 |
| --- | --- |
| `face-detection-adas-0001.xml` | `ef8fb491060e64da9d31893045f36dbb24f59455308e961274dd234190db997b` |
| `face-detection-adas-0001.bin` | `54883903421c8ee5289334ca4da779976a8be66daf7d57e45d005d0bc2b8c637` |
| `yolo-v9-s-608-license-plates-end2end.onnx` | `2b878b38d9aa07b6ddc3ea75c4ffcb39869bc5c218e0a14002f60ab2f7b0be9a` |

Run the existing frozen-model validator from the repository root. It reads the
files and fails on missing files or unexpected hashes; it does not download or
modify them:

```bash
python - <<'PY'
from redactor.review_workspace import resolve_frozen_model_paths, validate_frozen_models

result = validate_frozen_models(resolve_frozen_model_paths())
for name, record in result["files"].items():
    print(name, record["sha256"])
print("All frozen model hashes match.")
PY
```

For an independent byte check, use `shasum -a 256` on macOS or `sha256sum` on
Linux with the three file paths, then compare every value with the table.

## Paths and runtime configuration

| Model file | CLI option | Environment override |
| --- | --- | --- |
| ADAS XML | `--adas-xml` | `REDACTOR_ADAS_XML` |
| ADAS BIN | `--adas-bin` | `REDACTOR_ADAS_BIN` |
| OIM ONNX | `--plate-model` | `REDACTOR_OIM_PLATE_MODEL` |

Explicit CLI paths take precedence. Otherwise the adapters use the environment
overrides and the default layout above. With a custom XML and no BIN override,
the face adapter uses the adjacent `.bin` file. The verification command honors
the environment overrides too. For a custom directory, set both `REDACTOR_ADAS_XML`
and `REDACTOR_ADAS_BIN`: the frozen-model validator resolves those overrides
independently and does not infer a custom BIN path from the XML override.

The selected runtime uses face proposals at `0.20` and primary face evidence at
`0.30`; plate detection uses `0.30` with uniform small-object tiling. Setup does
not retune those policies. No archived comparison model is needed.

## Process a local video

Build the frontend once, then process a video you have permission to use:

```bash
npm --prefix frontend ci
npm --prefix frontend run build
redactor redact-video input.mp4 \
  --categories face,license_plate \
  --adas-xml models/face-detection-adas-0001/face-detection-adas-0001.xml \
  --adas-bin models/face-detection-adas-0001/face-detection-adas-0001.bin \
  --plate-model models/open-image-models/yolo-v9-s-608-license-plates-end2end.onnx \
  --output-dir artifacts/my_case
redactor serve-final-review-workspace artifacts/my_case/review_workspace --open
```

Use a new output directory for each run. Automatic is a preview; inspect it,
save any corrections, build Latest reviewed, and review the full current result
before Finalization. Generated workspaces and delivery packages can contain
Original media. Keep them local; they are not public demonstration bundles.
