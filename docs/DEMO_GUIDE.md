# Demo Guide

This guide is for a 5 to 10 minute local demonstration of the frozen review workflow. It assumes the frontend has already been built and that either a generated review workspace or the local release evidence workspace is available.

## Start

From the repository root:

```bash
source .venv/bin/activate
npm --prefix frontend run build
redactor serve-final-review-workspace path/to/release_review_workspace --open
```

If the release artifacts are not present, generate a fresh local workspace first:

```bash
redactor redact-video path/to/input.mp4 \
  --categories face,license_plate \
  --adas-xml models/face-detection-adas-0001/face-detection-adas-0001.xml \
  --adas-bin models/face-detection-adas-0001/face-detection-adas-0001.bin \
  --plate-model models/open-image-models/yolo-v9-s-608-license-plates-end2end.onnx \
  --output-dir path/to/demo_automatic_preview \
  --overwrite

redactor serve-final-review-workspace path/to/demo_automatic_preview/review_workspace --open
```

For correction demonstrations, use a disposable copy instead of the frozen release workspace:

```bash
DEMO_WORKSPACE="$(mktemp -d)/review_workspace"
cp -R path/to/release_review_workspace "$DEMO_WORKSPACE"
redactor serve-final-review-workspace "$DEMO_WORKSPACE" --open
```

## Flow

1. Open the Day or Night case.
2. Show the media role selector: Original, Automatic, Latest reviewed, and Final when available.
3. Switch between Original and Automatic to show what the machine pass produced.
4. Open Guidance and explain that it is advisory and auditable, not a mandatory per-card approval gate.
5. Open Entities and show source-backed Person and Plate policy actions, preview media, and human-confirmed candidate merge behavior.
6. In a disposable workspace, use Manual redaction to draw one correction, Preview it, and show real rendered privacy pixels inside the region.
7. Save the correction, remove it if demonstrating retraction, and point out the saved/applying state and updated reviewed media lifecycle.
8. Show Latest reviewed as the revision that the reviewer watches.
9. Explain that Approve full-video review is bound to the exact current Latest reviewed revision.
10. Show Final and the download/package options.
11. Open Audit Trail or Technical details to show event provenance and hashes.

## What Should Be Visible

- No Restart review button.
- Guidance terminology rather than blocking Open-query language.
- Guidance remaining is optional.
- Needs correction is explicitly blocking.
- Optional QA is separate.
- The player is visually larger than the guidance panel on desktop.
- Manual playback continues while frame state updates unless the reviewer performs a deliberate static-frame action.
- Preview shows pixelation or blur from generated privacy-rendered media, not an opaque placeholder.
- Current full-video approval becomes stale after a later media-affecting change.
- Final appears only after current Latest reviewed and current sweep gates pass.

## What Not To Modify In A Frozen Workspace

During a release demo of approved Final media:

- do not create new decisions;
- do not create entity actions;
- do not save corrections;
- do not reset a case;
- do not create another sweep;
- do not create another Final.

Use a disposable copy for interactive mutation demonstrations.

## Commands To Mention

```bash
redactor validate-review-session --session-dir SESSION_DIR
redactor summarize-review-session --session-dir SESSION_DIR
redactor serve-final-review-workspace WORKSPACE_ROOT --open
```

The demo should describe detector, automatic coverage, reviewed-final scoring, and human full-video review as separate evidence layers.
