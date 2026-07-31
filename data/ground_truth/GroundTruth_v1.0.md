# Ground Truth — Face and License-Plate Redaction Benchmark

**Version:** 1.0  
**Created:** 2026-07-30  
**Status:** Frozen gold reference for the selected evaluation scenes  
**Classes:** `face`, `license_plate`  
**Actions:** `redact`, `keep`, `ignore`

---

## 1. Scope and certification level

This document is the golden decision and temporal reference for the three benchmark scenes selected from the two uploaded source videos:

1. daytime license plate;
2. daytime face;
3. nighttime selective redaction.

The complete source videos contain **21,907 frames** in total. This file does **not** claim that every frame of both complete source videos has been manually boxed.

Two review levels are distinguished:

- **`GOLD_DENSE`** — every frame in the interval was visually reviewed at the native frame rate. Zero-leak claims are allowed only for these intervals.
- **`CONTEXT_ONLY`** — the surrounding evaluation clip was reviewed at approximately 5 fps to confirm the scene, targets, and transitions. These ranges are useful for clip selection and debugging, but must not be used to claim zero leakage on every frame.

The certified dense set contains **900 manually reviewed frames**:

| Benchmark | Dense frames |
|---|---:|
| Daytime plate | 240 |
| Daytime face | 300 |
| Nighttime selective redaction | 360 |
| **Total** | **900** |

This Markdown file freezes the target identities, actions, visibility decisions, frame ranges, and acceptance rules. It deliberately does not store license-plate characters or identify any person.

---

## 2. Source integrity

All time mappings below apply only to files matching these hashes.

| Source file | SHA-256 | Codec | Resolution | Exact FPS | Duration | Frames | Bytes |
|---|---|---|---:|---:|---:|---:|---:|
| `bodycam_sample.mp4` | `ab9b5f007c1ab1cb4acff90366ce4ae1c4b447ba036c00c4daab57e7fd801a57` | AV1 | 1920×1080 | 30000/1001 | 300.033067 s | 8,992 | 62,410,564 |
| `night_candidate.mp4` | `6748c67baa06ee97858af3b7bb44780bfce142fdc5b26e90c086192d4dba2f48` | AV1 | 1920×1080 | 30000/1001 | 430.004000 s | 12,915 | 78,923,861 |

### Time and frame conventions

- Frame indexes are **zero-based**.
- Time ranges are **half-open**: `[start, end)`.
- Exact frame time is `frame_index × 1001 / 30000`.
- Source timecodes are authoritative.
- Parent evaluation-clip frame numbers below are nominal mappings at 30000/1001 fps. A re-encoded or differently trimmed derivative must carry its own hash and trim manifest.
- When pixel coordinates are used, the coordinate system is the original **1920×1080** frame and the box format is `[x1, y1, x2, y2]`.

---

## 3. Annotation policy

### 3.1 What is a positive instance

A `face` is positive when a facial region can be localized by a human, including:

- frontal and profile views;
- head rotation;
- partial occlusion;
- edge truncation;
- visible upper face when the lower face is covered.

A frame is `detector-ignore` when only a cap or other non-facial head covering remains visible, or when the face is too truncated to define a defensible face target.

A `license_plate` is positive when the physical plate panel can be localized. The characters do **not** need to be readable. A plate remains positive when partly crossed by a television graphic or glare, provided its location is still unambiguous.

### 3.2 Actions

- **`redact`** — the final output must conceal the target.
- **`keep`** — the target is a real instance of the class, but selective-redaction logic must leave it unredacted.
- **`ignore`** — excluded from detector recall and precision scoring.

### 3.3 General rules

- Annotation scope is `all_visible_localizable_instances` inside each certified dense interval.
- Tiny distant vehicle details that cannot be reliably localized at source resolution are `ignore_small`, not negatives.
- Reflections are not separate face instances unless they are independently recognizable and localizable.
- A prediction from tracking or interpolation may satisfy **final redaction coverage**, but does not count as a raw detector hit.
- A tracker may bridge an occlusion after a confirmed target, but must not invent an unresolved prefix before the first confirmed observation. Such a prefix requires a manual seed or review.
- For a `redact` track, short detector-ignore gaps may retain a continuous conservative redaction zone to prevent flicker.
- For a `keep` track, the face must remain clear; it must not be blurred merely because the face detector finds it.
- License-plate text is out of scope and must not be transcribed into evaluation artifacts.

---

## 4. Benchmark clip map

| Benchmark ID | Source | Source interval | Approx. clip duration | Certified interval in evaluation clip | Certified source interval | Dense frames |
|---|---|---:|---:|---:|---:|---:|
| `day_plate` | `bodycam_sample.mp4` | `00:01:33.000–00:01:52.000` | 19.019 s | `00:08.000–00:16.000` | `00:01:41.000–00:01:49.000` | 240 |
| `day_face` | `bodycam_sample.mp4` | `00:03:27.000–00:04:03.000` | 36.003 s | `00:10.000–00:20.000` | `00:03:37.000–00:03:47.000` | 300 |
| `night_selective` | `night_candidate.mp4` | `00:04:42.000–00:05:35.000` | 53.020 s | `00:14.000–00:26.000` | `00:04:56.000–00:05:08.000` | 360 |

### Derivative-clip warning

Do not infer source alignment from duration alone. In particular, if `eval_day_plate_close.mp4` or another shortened derivative is used, it must be accompanied by:

- SHA-256;
- exact source start and end time;
- exact FPS and frame count;
- whether timestamps were reset;
- mapping from derivative frame 0 to source time.

Until that manifest exists, the source intervals in the table above are the authoritative reference.

---

## 5. Target registry

| Track ID | Benchmark | Class | Action | Description | Main challenges |
|---|---|---|---|---|---|
| `plate_day_01` | `day_plate` | `license_plate` | `redact` | Rear plate of the black sedan/convertible traveling on the shoulder ahead of the camera | Small target, changing scale and distance, ego motion, road vibration |
| `person_day_01` | `day_face` | `face` | `redact` | Face of the stopped man in the blue-and-white patterned shirt beside the white police vehicle | Strong profile, head rotation, partial self-occlusion, moving body camera |
| `person_night_01` | `night_selective` | `face` | `keep` | Standing person centered near the white pickup; lower face covered by a gaiter/mask | Low light, upper-face-only visibility, selective-redaction exception |
| `person_night_02` | `night_selective` | `face` | `redact` | Seated/crouched person wearing a black cap near the lower-right portion of the frame | Edge truncation, cap occlusion, head-down pose, large scale changes |
| `plate_night_01` | `night_selective` | `license_plate` | `redact` | Rear plate of the white Toyota Tacoma | Glare, low light, television overlay crossing the plate in some frames |

---

## 6. Context review of the selected clips

These observations are `CONTEXT_ONLY` unless the time falls inside a certified dense interval.

### 6.1 `day_plate`

- `plate_day_01` is present and localizable in every 5-fps context sample across the selected clip.
- No scored face is present.
- Other vehicles are visible, but their possible plate regions are too small or ambiguous to be reliable positives and are `ignore_small`.
- The certified dense interval was chosen because the plate changes scale and position while remaining continuously visible.

### 6.2 `day_face`

- During approximately the first 0–1.8 seconds, the back of the subject's head is shown and there is no defensible face target.
- The face becomes localizable at approximately clip time 2.0 seconds and remains present through the end of the selected clip in the context review.
- The vehicle-window reflection is not an independent face instance.
- No scored license plate is present in the certified dense interval.

### 6.3 `night_selective`

- `person_night_01` is the standing `keep` target.
- `person_night_02` is the seated/crouched `redact` target.
- `plate_night_01` is the rear plate of the white pickup and is a `redact` target.
- The scene intentionally tests selective redaction: the seated face and plate must be hidden while the standing face remains clear.
- The seated face is intermittently hidden by the cap and lower frame edge; those states are separated from genuinely observable face frames below.

---

## 7. Certified frame-level decisions

All frame ranges in this section refer first to the **dense subclip**, where dense frame 0 is the first frame of the certified interval.

### 7.1 Daytime plate — `GOLD_DENSE`

**Dense interval:** evaluation clip `00:08.000–00:16.000`  
**Dense-subclip frames:** `0–239`  
**Nominal parent evaluation-clip frames:** `240–479`

| Track | Detector-scored frames | Detector-ignore frames | Final redaction required |
|---|---|---|---|
| `plate_day_01` | `0–239` | none | `0–239` |

**Human-reviewed result**

- The plate is visible and localizable in **240/240 frames**.
- There is no visibility gap.
- The target remains small, but the plate panel is consistently distinguishable from the bumper.
- Any missed frame in this interval is a privacy failure.

### 7.2 Daytime face — `GOLD_DENSE`

**Dense interval:** evaluation clip `00:10.000–00:20.000`  
**Dense-subclip frames:** `0–299`  
**Nominal parent evaluation-clip frames:** `300–599`

| Track | Detector-scored frames | Detector-ignore frames | Final redaction required |
|---|---|---|---|
| `person_day_01` | `0–299` | none | `0–299` |

**Human-reviewed result**

- The facial region is present in **300/300 frames**.
- The most difficult rotation occurs at approximately dense time `00:06.000–00:07.100`, nominal dense frames `180–212`.
- During that interval, the face becomes an extreme profile/head-away view, but enough facial region remains localizable for privacy redaction.
- The track must remain continuous; this rotation is not a negative gap.

### 7.3 Nighttime selective redaction — `GOLD_DENSE`

**Dense interval:** evaluation clip `00:14.000–00:26.000`  
**Dense-subclip frames:** `0–359`  
**Nominal parent evaluation-clip frames:** `420–779`

#### `person_night_01` — standing `keep` face

| Decision | Dense-subclip frames | Nominal parent frames |
|---|---|---|
| Detector-scored | `0–359` | `420–779` |
| Keep clear | `0–359` | `420–779` |
| Detector-ignore | none | none |

- The upper face/eyes are localizable in **360/360 frames**.
- The lower face is covered, but this does not remove the instance from the face class.
- Any redaction of the identifiable face area is a selective-redaction false positive.

#### `person_night_02` — seated `redact` face

| Decision | Dense-subclip frames | Nominal parent frames | Dense-relative time |
|---|---|---|---|
| Detector-scored: visible/partial face | `0–26` | `420–446` | `[00:00.000, 00:00.901)` |
| Detector-ignore: cap/edge only | `27–58` | `447–478` | `[00:00.901, 00:01.969)` |
| Detector-scored: visible/partial face | `59–283` | `479–703` | `[00:01.969, 00:09.476)` |
| Detector-ignore: cap/edge only | `284–359` | `704–779` | `[00:09.476, 00:12.000)` |
| Continuous final redaction zone | `0–359` | `420–779` | entire dense interval |

Counts:

- Detector-scored face frames: **252**
- Detector-ignore frames: **108**
- Continuous conservative redaction-track frames: **360**

The ignore ranges must not reduce detector recall. They exist because the cap/head remains in view while no defensible facial region is available. The final privacy layer should nevertheless keep the redact track continuous through these short truncation/occlusion states so that the mask does not flicker and is already present when the face becomes visible again.

#### `plate_night_01` — white-pickup rear plate

| Decision | Dense-subclip frames | Nominal parent frames |
|---|---|---|
| Detector-scored | `0–359` | `420–779` |
| Final redaction required | `0–359` | `420–779` |
| Detector-ignore | none | none |

- The plate is visible and localizable in **360/360 frames**.
- A television graphic crosses the plate in portions of the interval, but the plate panel remains unambiguous.
- The graphic is an occluder, not a reason to remove the plate from ground truth.

---

## 8. Reference search regions

These are deliberately generous **QA/search ROIs**, not tight scoring boxes. They help verify that a detector or tracker is following the intended object. Coordinates use the original 1920×1080 source frame.

| Track | Reference ROI `[x1, y1, x2, y2]` |
|---|---|
| `plate_day_01` | `[650, 380, 1060, 570]` |
| `person_day_01` | `[960, 80, 1360, 520]` |
| `person_night_01` | `[380, 120, 800, 500]` |
| `person_night_02` | `[1160, 620, 1760, 1080]` |
| `plate_night_01` | `[60, 560, 500, 900]` |

A predicted target outside its reference ROI is not automatically wrong because the camera moves. It is a review trigger.

---

## 9. Acceptance criteria

### 9.1 Raw detector evaluation

Compute detector metrics only on `detector-scored` frames.

- A detector hit must correspond to the correct class and target.
- Tracker-only or interpolated boxes do not count as detector hits.
- `detector-ignore` frames are excluded from both numerator and denominator.
- An output overlapping an `ignore_small` object is neither rewarded nor penalized unless it creates a visible false redaction.

Recommended primary metric:

```text
object_frame_recall =
    number of detector-scored target frames with a valid detector hit
    / total detector-scored target frames
```

### 9.2 Final redaction evaluation

For every `redact` target in a certified interval:

- **Maximum consecutive uncovered visible frames:** `0`
- **Allowed privacy leaks:** `0`
- **Allowed flicker events:** `0`
- **Unresolved prefix before first confirmed target:** not allowed
- A single visible frame with an identifiable, uncovered face or plate is a failure.

Automated mask/box coverage can be used as a pre-check, but visual zero-leak review is authoritative. A numerical overlap score must not overrule visible exposed facial or plate pixels.

### 9.3 Selective-redaction evaluation

For `person_night_01`:

- **False-redaction frames on the identifiable keep face:** `0`
- Redaction of the person's clothing or background is not by itself a keep-face failure, but any intentional blur/mask covering the identifiable upper face is a failure.
- The system must simultaneously redact `person_night_02` and `plate_night_01`.

### 9.4 Required report fields

Every model/run report should include:

- source and derivative SHA-256;
- model name and version;
- detector threshold;
- NMS threshold;
- preprocessing resolution;
- tracker/interpolation settings;
- object-frame recall by target;
- final redaction leaks by target;
- maximum consecutive missed frames;
- flicker-event count;
- false-redaction frames on `person_night_01`;
- list of manually reviewed frames or intervals.

---

## 10. Frozen expected outcomes

| Target | Dense scored frames | Expected final action | Gold requirement |
|---|---:|---|---|
| `plate_day_01` | 240 | redact | Covered in every one of 240 frames |
| `person_day_01` | 300 | redact | Covered in every one of 300 frames |
| `person_night_01` | 360 | keep | Identifiable face clear in every one of 360 frames |
| `person_night_02` | 252 detector-scored; 360 continuous track | redact | No visible-face leak; continuous conservative track |
| `plate_night_01` | 360 | redact | Covered in every one of 360 frames |

This version contains no unresolved target identity or action ambiguity in the certified intervals.

---

## 11. Machine-readable manifest

```yaml
schema_version: "1.0"
created: "2026-07-30"
certification: "GOLD_DENSE_WINDOWS_ONLY"
fps: "30000/1001"
frame_indexing: "zero_based"
time_intervals: "half_open"
coordinate_system:
  width: 1920
  height: 1080
  bbox_format: "xyxy"

sources:
  bodycam_sample:
    filename: "bodycam_sample.mp4"
    sha256: "ab9b5f007c1ab1cb4acff90366ce4ae1c4b447ba036c00c4daab57e7fd801a57"
    duration_seconds: 300.033067
    frames: 8992
  night_candidate:
    filename: "night_candidate.mp4"
    sha256: "6748c67baa06ee97858af3b7bb44780bfce142fdc5b26e90c086192d4dba2f48"
    duration_seconds: 430.004000
    frames: 12915

benchmarks:
  day_plate:
    source: "bodycam_sample"
    source_interval_seconds: [93.0, 112.0]
    dense:
      evaluation_clip_interval_seconds: [8.0, 16.0]
      source_interval_seconds: [101.0, 109.0]
      dense_subclip_frames: [0, 239]
      nominal_parent_frames: [240, 479]
      targets:
        plate_day_01:
          class: "license_plate"
          action: "redact"
          detector_scored_ranges: [[0, 239]]
          detector_ignore_ranges: []
          final_redaction_ranges: [[0, 239]]

  day_face:
    source: "bodycam_sample"
    source_interval_seconds: [207.0, 243.0]
    dense:
      evaluation_clip_interval_seconds: [10.0, 20.0]
      source_interval_seconds: [217.0, 227.0]
      dense_subclip_frames: [0, 299]
      nominal_parent_frames: [300, 599]
      targets:
        person_day_01:
          class: "face"
          action: "redact"
          detector_scored_ranges: [[0, 299]]
          detector_ignore_ranges: []
          final_redaction_ranges: [[0, 299]]
          difficulty_ranges:
            extreme_profile_nominal: [[180, 212]]

  night_selective:
    source: "night_candidate"
    source_interval_seconds: [282.0, 335.0]
    dense:
      evaluation_clip_interval_seconds: [14.0, 26.0]
      source_interval_seconds: [296.0, 308.0]
      dense_subclip_frames: [0, 359]
      nominal_parent_frames: [420, 779]
      targets:
        person_night_01:
          class: "face"
          action: "keep"
          detector_scored_ranges: [[0, 359]]
          detector_ignore_ranges: []
          keep_clear_ranges: [[0, 359]]

        person_night_02:
          class: "face"
          action: "redact"
          detector_scored_ranges: [[0, 26], [59, 283]]
          detector_ignore_ranges: [[27, 58], [284, 359]]
          final_redaction_ranges: [[0, 359]]

        plate_night_01:
          class: "license_plate"
          action: "redact"
          detector_scored_ranges: [[0, 359]]
          detector_ignore_ranges: []
          final_redaction_ranges: [[0, 359]]

acceptance:
  privacy_leaks_allowed: 0
  max_consecutive_uncovered_visible_frames: 0
  flicker_events_allowed: 0
  keep_face_false_redaction_frames_allowed: 0
  unresolved_prefix_allowed: false
```

---

## 12. Versioning rule

This file is frozen as `v1.0`.

Change the version only when at least one of the following changes:

- source or derivative hash;
- source-to-clip mapping;
- target identity;
- action (`redact`, `keep`, `ignore`);
- detector-scored or detector-ignore frame range;
- zero-leak acceptance rule.

Cosmetic wording changes that do not alter evaluation semantics may use a patch version.
