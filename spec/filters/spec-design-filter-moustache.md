---
title: 2D Moustache Overlay Filter
version: 1.0
date_created: 2026-03-06
last_updated: 2026-03-06
tags: [design, filter, face-tracking, 2d-overlay, cpu]
---

# 2D Moustache Overlay Filter

The 2D Moustache Overlay Filter composites a pre-drawn moustache sprite
onto the live camera feed each frame. The moustache is anchored to the
detected face in the correct anatomical position (between the nose tip
and the upper lip), scales with face distance, and rotates to match head
roll. The user can select from six distinct moustache styles loaded from
a shared sprite sheet.

---

## 1. What the User Sees

When the filter is active the camera feed shows a moustache overlaid on
any detected face:

- The moustache sits horizontally between the nose tip and the upper lip,
  centred on the face.
- The moustache width tracks the face width (mouth corner distance) and
  scales proportionally with face distance from the camera.
- The moustache rotates with the head roll angle so it stays flat
  relative to the face even when the user tilts their head.
- When no face is detected the camera feed is shown unmodified (no
  moustache drawn).
- The widget bar exposes a **Moustache** selector control allowing the
  user to pick one of six moustache styles (indices 0–5) without
  restarting the filter.

---

## 2. Scope

| In scope | Out of scope |
|----------|--------------|
| `MoustacheFilter` class in `filters/moustache.py` | Changes to `BaseFilter` or `FilterPass` interfaces |
| Loading and splitting `2dmodels/moustaches2.jpg` into six RGBA sprites | Training or tracking logic inside `tracking/` |
| Alpha-masking the white sprite background to transparency | Changing the sprite sheet format or adding new art assets |
| Anchoring, scaling, and rotating the sprite to the face each frame | Any 3D projection — this is a flat 2D overlay |
| `moustache_index` selection parameter | Per-moustache colour tinting or independent scale adjust (future) |
| Unit tests in `tests/test_moustache_filter.py` | Visual regression tests requiring a GPU |
| Integration in `app/application.py` | Changes to `OverlayPass` or `GamePass` |

---

## 3. Definitions

| Term | Meaning |
|------|---------|
| **Sprite sheet** | The file `2dmodels/moustaches2.jpg`, a JPEG containing the six moustache images in a 3-column × 2-row grid |
| **Sprite** | One extracted cell from the sprite sheet; converted to RGBA before use |
| **Alpha mask** | A per-pixel transparency channel derived from the luminance of the source JPEG; pixels brighter than a threshold are made transparent (white background removal) |
| **Anchor point** | The pixel position on the frame where the centre of the moustache is placed; computed from face landmarks each frame |
| **Mouth width** | Pixel distance between the left mouth corner (landmark 61) and the right mouth corner (landmark 291) |
| **Scale factor** | The ratio that maps the sprite's natural width to the desired render width on frame |
| **Head roll** | Rotation of the face around the depth axis (tilting head left/right), expressed in degrees; provided by `HeadPose.roll` |
| **Landmark** | A normalised 3-D facial point from MediaPipe; `x, y ∈ [0, 1]` with `(0, 0)` at top-left |
| **FaceTrackResult** | Dataclass from `FaceTracker.process()` containing 478 landmarks, a `face_detected` flag, and `head_pose` |
| **Overlay texture** | A CPU-drawn RGBA texture uploaded to the GPU each frame for alpha-blended compositing |
| **Blit pass** | A full-screen GPU draw that copies one texture to another unchanged |
| **BGRA** | Four-channel byte order used internally by OpenCV (`blue, green, red, alpha`) |

---

## 4. Requirements

### 4.1 Sprite Sheet Loading

- **REQ-MS-001** At `setup()` the filter loads `2dmodels/moustaches2.jpg`
  relative to the project root and splits it into exactly six RGBA
  sprites arranged in a 3-column × 2-row grid.
  - Index 0 = top-left, 1 = top-centre, 2 = top-right
  - Index 3 = bottom-left, 4 = bottom-centre, 5 = bottom-right
- **REQ-MS-002** Each sprite is alpha-masked by converting the white
  background to full transparency: pixels with luminance ≥ 220 (0–255
  scale) are set to `alpha = 0`; all other pixels retain full opacity
  (`alpha = 255`).
- **REQ-MS-003** The six pre-processed sprites are cached in memory for
  the full lifetime of the filter to avoid per-frame file I/O.

### 4.2 Moustache Selection

- **REQ-MS-004** The filter exposes a single user-adjustable parameter:
  `moustache_index` (integer, default 0). Valid values are 0–5
  inclusive.
- **REQ-MS-005** Setting `moustache_index` to a value outside `[0, 5]`
  is silently clamped to the nearest valid bound (0 or 5).
- **REQ-MS-006** Changing `moustache_index` takes effect on the very
  next rendered frame without any pipeline teardown or reload.

### 4.3 Face Anchoring

- **REQ-MS-007** When a face is detected the selected moustache sprite is
  composited onto the frame at an anchor position computed from the
  478-point landmark set:
  - **Anchor centre X** (pixels):
    `(landmark[61].x + landmark[291].x) / 2 × frame_width`
  - **Anchor centre Y** (pixels):
    `(landmark[1].y + landmark[13].y) / 2 × frame_height`
  - **Render width** (pixels):
    `|landmark[291].x − landmark[61].x| × frame_width × 1.4`
  - **Rotation**: `head_pose.roll` degrees, applied around the sprite
    centre.
- **REQ-MS-008** The sprite aspect ratio is preserved when the sprite
  is scaled to the computed render width.
- **REQ-MS-009** When no face is detected the filter passes the camera
  frame through unmodified — no moustache is drawn.

### 4.4 Integration

- **REQ-MS-010** The filter receives face tracking data each frame via
  `update_face_result(result: Optional[FaceTrackResult])`. It does not
  hold a reference to `AppState` or `FaceTracker`.
- **REQ-MS-011** The filter name property returns `"Moustache"`.
- **REQ-MS-012** The filter is registered in `app/application.py`
  with `enabled = False` (off by default, user-activated via widget).

---

## 5. Constraints

- **CON-MS-001** The filter must implement the `BaseFilter` interface:
  `setup`, `_build_pipeline`, `apply`, `teardown`.
- **CON-MS-002** `apply` must not allocate new GPU buffers or textures
  on the hot path; all GPU resources are pre-allocated in
  `_build_pipeline` or lazy-initialised once on the first `apply` call.
- **CON-MS-003** No GPU readback (CPU ← GPU copy) may occur inside
  `apply`; pixel data flows CPU → GPU only (via
  `queue.write_texture`).
- **CON-MS-004** The moustache compositing is CPU-side (OpenCV): the
  sprite is drawn into a NumPy BGRA canvas which is then uploaded as
  an overlay texture and alpha-blended by a GPU pass — identical to the
  overlay strategy used in `FaceLandmarkFilter`.

---

## 6. Interfaces & Data Contracts

### 6.1 Class Interface

File: `filters/moustache.py`

```python
class MoustacheFilter(BaseFilter):
    """2D moustache sprite overlaid on the detected face."""

    @property
    def name(self) -> str:
        """Returns 'Moustache'."""

    def update_face_result(
        self, result: Optional[FaceTrackResult]
    ) -> None:
        """Inject the latest face tracking result before apply()."""

    def setup(
        self,
        device: wgpu.GPUDevice,
        texture_format: wgpu.TextureFormat,
    ) -> None:
        """Load sprites, build GPU pipelines, cache resources."""

    def _build_pipeline(
        self,
        device: wgpu.GPUDevice,
        texture_format: wgpu.TextureFormat,
    ) -> None:
        """Compile WGSL shaders and allocate GPU pipeline objects."""

    def apply(
        self,
        encoder: wgpu.GPUCommandEncoder,
        input_texture: wgpu.GPUTexture,
        output_texture: wgpu.GPUTexture,
    ) -> None:
        """
        Record GPU commands for one frame:
        1. Blit pass — copy input_texture to output_texture.
        2. Overlay pass — composite moustache sprite (when face detected).
        """

    def teardown(self) -> None:
        """Release all GPU objects and sprite caches."""
```

### 6.2 `params` Dictionary

| Key | Type | Default | Range | Description |
|-----|------|---------|-------|-------------|
| `moustache_index` | `int` | `0` | `[0, 5]` | Index of the selected moustache in the sprite sheet |

### 6.3 Landmark Indices Used

| Index | Anatomical point |
|-------|-----------------|
| 1 | Nose tip |
| 13 | Upper lip centre |
| 61 | Left mouth corner |
| 291 | Right mouth corner |

### 6.4 Sprite Sheet Layout

```
┌─────────┬─────────┬─────────┐
│  idx 0  │  idx 1  │  idx 2  │  ← row 0  (top)
├─────────┼─────────┼─────────┤
│  idx 3  │  idx 4  │  idx 5  │  ← row 1  (bottom)
└─────────┴─────────┴─────────┘
  col 0     col 1     col 2
```

Each cell has dimensions:
- `cell_width  = sheet_width  // 3`
- `cell_height = sheet_height // 2`

### 6.5 Anchor Computation

```python
def _compute_anchor(
    landmarks: List[Landmark],
    frame_width: int,
    frame_height: int,
) -> Tuple[int, int, int, float]:
    """
    Returns (centre_x, centre_y, render_width, roll_degrees).
    """
    cx = (landmarks[61].x + landmarks[291].x) / 2 * frame_width
    cy = (landmarks[1].y  + landmarks[13].y)  / 2 * frame_height
    rw = abs(landmarks[291].x - landmarks[61].x) * frame_width * 1.4
    return int(cx), int(cy), int(rw), head_pose.roll
```

---

## 7. Acceptance Criteria

- **AC-MS-001** Given the app is running and no face is detected, when
  the Moustache filter is active, then the output frame is identical to
  the input frame (no pixel modification).

- **AC-MS-002** Given a face is detected with `face_detected = True`,
  when `apply()` is called, then a non-transparent region appears
  between the nose tip and upper lip area of the output frame.

- **AC-MS-003** Given `moustache_index = 0` is set, when the user
  changes it to `3`, then the next frame renders moustache sprite at
  index 3 without restarting or reloading the filter.

- **AC-MS-004** Given `moustache_index = -1` is set (out of range),
  when the value is read back, then it returns `0` (clamped to lower
  bound).

- **AC-MS-005** Given `moustache_index = 99` is set (out of range),
  when the value is read back, then it returns `5` (clamped to upper
  bound).

- **AC-MS-006** Given the face has a roll angle of `R` degrees, when
  `apply()` is called, then the rendered moustache is rotated by `R`
  degrees relative to horizontal.

- **AC-MS-007** Given 478 landmarks are supplied, when
  `_compute_anchor()` is called, then the returned `centre_x` equals
  `int((lm[61].x + lm[291].x) / 2 × frame_width)` and `centre_y`
  equals `int((lm[1].y + lm[13].y) / 2 × frame_height)`.

- **AC-MS-008** Given the filter is torn down via `teardown()`, when
  any GPU resource attribute is accessed, then it returns `None` (all
  references released).

- **AC-MS-009** Given the sprite sheet loads successfully, when
  `setup()` completes, then exactly six RGBA sprites are cached and each
  sprite's alpha channel is `0` wherever the source pixel luminance
  is ≥ 220.

- **AC-MS-010** Given `update_face_result(None)` is called, when
  `apply()` runs, then no moustache is drawn and no exception is raised.

---

## 8. Validation Criteria

1. `pytest tests/test_moustache_filter.py -v` — all tests pass with no GPU.
2. `pytest tests/` — no regressions in existing tests.
3. Launching the application with the Moustache filter active shows the
   moustache anchored between the nose and upper lip on the detected face.
4. Selecting each of the six moustache indices shows a visually distinct
   moustache.
5. Tilting the head produces a corresponding rotation of the moustache
   sprite.

---

## 9. Related Specifications

- [spec/filters/spec-design-filter-face-landmarks.md](spec-design-filter-face-landmarks.md) — reference implementation for the overlay pattern
- [spec/capabilities/spec-capability-face-tracking.md](../capabilities/spec-capability-face-tracking.md) — landmark coordinate system and `FaceTrackResult` contract
- [spec/main.md](../main.md) — overall pipeline order and filter chain architecture
