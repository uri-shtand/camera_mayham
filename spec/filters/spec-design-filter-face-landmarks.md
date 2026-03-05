---
title: Face Landmark Visualisation Filter
version: 2.0
date_created: 2026-03-05
last_updated: 2026-03-06
tags: [design, filter, face-tracking, gpu, wgpu]
---

# Face Landmark Visualisation Filter

The Face Landmark Visualisation Filter overlays all 478 MediaPipe face landmarks on the live camera feed as white dots, adds head-pose arrows to show facial orientation, and displays a clear face-detected indicator. Rendering is done on the GPU via WebGPU (wgpu) so there is no CPU overhead in the hot render loop. The filter plugs into the existing filter chain without changing any shared interfaces.

---

## 1. What the User Sees

When the filter is active the camera feed shows three layers on top of the video:

1. **Landmark dots** — 478 small white circles, one at each MediaPipe facial keypoint (eyes, nose, mouth, jawline, etc.).
2. **Head-pose arrows** — three arrows drawn from a fixed origin in the **top-left corner** of the frame, pointing in the directions of the face's pitch (up/down), yaw (left/right), and roll (tilt) axes. The arrow lengths scale with the magnitude of each angle.
3. **Face-detected indicator** — a small badge in the top-right corner of the frame:
   - Green circle + check when a face is detected.
   - Red circle + cross when no face is detected.

When no face is present only the badge is shown; the dots and arrows disappear.

---

## 2. Scope

| In scope | Out of scope |
|----------|--------------|
| `FaceLandmarkFilter` class in `filters/face_landmarks.py` | Changes to `BaseFilter` or `FilterPass` interfaces |
| Dot, arrow, and badge rendering on the GPU | Exposing dot colour or dot size as user-adjustable settings |
| Pipeline integration in `rendering/pipeline.py` | Training or tracking logic inside `tracking/` |
| Unit tests in `tests/test_face_landmark_filter.py` | Any UI controls beyond the filter on/off toggle |

---

## 3. Definitions

| Term | Meaning |
|------|---------|
| **Landmark** | A normalised 3-D facial point from MediaPipe; `x, y ∈ [0, 1]` with `(0, 0)` at top-left |
| **NDC** | Normalised Device Coordinates — WebGPU clip space where `x, y ∈ [-1, 1]` with `(0, 0)` at screen centre |
| **Head pose** | The orientation of the face in 3-D space, expressed as pitch, yaw, and roll angles derived from landmark geometry |
| **Instanced draw** | A single GPU draw call that renders the same quad geometry once per landmark using per-instance position data |
| **SDF circle** | A fragment shader technique that computes a smooth circle by measuring distance from a point centre |
| **Blit pass** | A full-screen draw that copies one GPU texture to another unchanged |
| **FaceTrackResult** | Dataclass from `FaceTracker.process()` containing the landmark list, a `face_detected` flag, and head-pose angles |

---

## 4. Requirements

### 4.1 Landmark Dots

- **REQ-LM-001** The filter renders a white dot at each of the 478 detected landmark positions.
- **REQ-LM-002** Dots are rendered as smooth SDF circles with a fixed radius of 3 px and colour `rgba(1, 1, 1, 1)` (opaque white). These values are hard-coded and not exposed to the user.
- **REQ-LM-003** When no face is detected the dots are hidden and the frame is passed through unmodified.

### 4.2 Head-Pose Arrows

- **REQ-LM-004** Three arrows are drawn from a fixed origin in the top-left corner of the frame along the pitch, yaw, and roll axes of the detected face.
- **REQ-LM-005** Arrow colour and direction follow the axis convention:
  - **Yaw** (left/right rotation) — blue arrow.
  - **Pitch** (up/down rotation) — green arrow.
  - **Roll** (tilt) — red arrow.
- **REQ-LM-006** Arrow length is proportional to the magnitude of the corresponding angle, capped at a sensible maximum so they remain inside the frame.
- **REQ-LM-007** When no face is detected the arrows are hidden.

### 4.3 Face-Detected Indicator

- **REQ-LM-008** A badge is always visible in the top-right corner of the frame regardless of whether a face is detected.
- **REQ-LM-009** When a face is detected the badge shows a **green filled circle** with a white check mark (✓).
- **REQ-LM-010** When no face is detected the badge shows a **red filled circle** with a white cross (✗).

### 4.4 Integration

- **REQ-LM-011** The filter receives face tracking data each frame via `update_face_result(result)`. It does not hold a reference to `AppState` or `FaceTracker`.
- **REQ-LM-012** The filter name property returns `"Face Landmarks"`.

---

## 5. Constraints

- **CON-LM-001** The filter must implement the `BaseFilter` interface: `setup`, `_build_pipeline`, `apply`, `teardown`.
- **CON-LM-002** `apply` must not allocate new GPU buffers or textures; all buffers are pre-allocated in `_build_pipeline`.
- **CON-LM-003** The vertex buffer supports at most 478 landmarks (`MAX_LANDMARKS`).
- **CON-LM-004** `apply` uses a two-pass strategy: a blit pass that copies the input to the output, then an overlay pass that draws dots, arrows, and the badge on top.

---

## 6. Data Contracts

### 6.1 Class Interface

File: `filters/face_landmarks.py`

```python
class FaceLandmarkFilter(BaseFilter):
    name: str  # "Face Landmarks"

    def update_face_result(
        self, result: Optional[FaceTrackResult]
    ) -> None: ...

    def _build_pipeline(
        self, device: wgpu.GPUDevice,
        texture_format: wgpu.TextureFormat,
    ) -> None: ...

    def apply(
        self,
        encoder: wgpu.GPUCommandEncoder,
        input_texture: wgpu.GPUTexture,
        output_texture: wgpu.GPUTexture,
    ) -> None: ...

    def teardown(self) -> None: ...
```

`FaceLandmarkFilter` has no user-facing `params` dictionary. All visual properties are fixed.

### 6.2 GPU Resources

| Resource | Type | Size | Purpose |
|----------|------|------|---------|
| `_blit_bgl` | BindGroupLayout | — | Bindings 0 (texture) + 1 (sampler) for the blit pass |
| `_landmark_bgl` | BindGroupLayout | — | Binding 0 (uniform) for the overlay pass |
| `_landmark_param_buffer` | GPUBuffer | 32 bytes | `UNIFORM \| COPY_DST` — dot radius, colour, NDC scale |
| `_landmark_vertex_buffer` | GPUBuffer | 478 × 8 bytes | `VERTEX \| COPY_DST` — per-instance NDC positions |
| `_blit_pipeline` | GPURenderPipeline | — | Full-screen blit |
| `_landmark_pipeline` | GPURenderPipeline | — | Instanced SDF circles with alpha blend |

### 6.3 WGSL Uniform: `LandmarkParams` (32 bytes)

```wgsl
struct LandmarkParams {
    dot_r    : f32,   // offset  0  — fixed: 1.0
    dot_g    : f32,   // offset  4  — fixed: 1.0
    dot_b    : f32,   // offset  8  — fixed: 1.0
    dot_a    : f32,   // offset 12  — fixed: 1.0
    radius_x : f32,   // offset 16  — NDC half-width  (aspect-ratio corrected)
    radius_y : f32,   // offset 20  — NDC half-height (aspect-ratio corrected)
    _pad0    : f32,   // offset 24
    _pad1    : f32,   // offset 28
}
```

`radius_x` and `radius_y` are derived each frame from the 3 px dot radius and the current texture dimensions: `radius_x = 2 * 3.0 / width`, `radius_y = 2 * 3.0 / height`.

### 6.4 Coordinate Conversion

MediaPipe normalised coordinates map to NDC as follows:

```
ndc_x =  x * 2 - 1
ndc_y =  1 - y * 2      # flip Y: MediaPipe y grows downward, NDC y grows upward
```

Boundary check: `(0, 0)` → `(-1, 1)` top-left; `(1, 1)` → `(1, -1)` bottom-right; `(0.5, 0.5)` → `(0, 0)` centre.

### 6.5 Pipeline Integration

`rendering/pipeline.py` must push the latest face result into every filter that supports it before recording the filter pass:

```python
for flt in state.enabled_filters():
    if hasattr(flt, "update_face_result"):
        flt.update_face_result(state.face_result)
```

No changes to `BaseFilter` or `FilterPass` are required.

---

## 7. Acceptance Criteria

| ID | Scenario | Expected result |
|----|----------|-----------------|
| AC-LM-001 | `apply` called with a valid 478-landmark result | 478 instanced quads drawn in the overlay pass |
| AC-LM-002 | `apply` called with `face_result = None` | Only the blit pass runs; output equals input |
| AC-LM-003 | `apply` called with `face_detected == False` | Only the blit pass runs; badge shows red cross |
| AC-LM-004 | Landmark at `(0.0, 0.0)` | NDC position is `(-1.0, 1.0)` |
| AC-LM-005 | Landmark at `(1.0, 1.0)` | NDC position is `(1.0, -1.0)` |
| AC-LM-006 | Landmark at `(0.5, 0.5)` | NDC position is `(0.0, 0.0)` |
| AC-LM-007 | `name` property read | Returns `"Face Landmarks"` |
| AC-LM-008 | Face detected | Badge is green with check mark |
| AC-LM-009 | No face detected | Badge is red with cross |
| AC-LM-010 | `update_face_result` called | Result is stored and used in the next `apply` call |
| AC-LM-011 | `apply` called with a valid result | Three head-pose arrows are drawn from the top-left corner of the frame |

---

## 8. Tests

- **Framework**: `pytest`; GPU objects stubbed as `None`.
- **Test file**: `tests/test_face_landmark_filter.py`
- **Coverage**: NDC conversion, face-detected/not-detected branching, badge state, head-pose arrow presence, `name` property, `update_face_result` storage.

---

## 9. Rationale

**Fixed white dots** — colour and size customisation add UI complexity with no meaningful user benefit for a diagnostic/creative overlay. White provides universally strong contrast against skin tones. 3 px is large enough to be visible without obscuring the underlying image.

**Head-pose arrows** — three coloured axes give an immediate spatial read of where the face is pointing, making the filter useful for both creative effects and tracking diagnostics.

**Always-visible badge** — showing face-detected status at all times (even when no filter effects are visible) confirms at a glance that the tracker is running, which is important during setup and debugging.

**Two-pass GPU rendering** — blit first, then overlay keeps each shader simple, reuses the standard passthrough blit pipeline, and lets the overlay pass enable alpha blending independently.

**`update_face_result` injection pattern** — keeps `BaseFilter.apply` signature stable and avoids coupling filters to `AppState` or `FaceTracker`.

---

## 9. Examples & Edge Cases

```python
# Constructing and configuring the filter
from filters.face_landmarks import FaceLandmarkFilter
from tracking.face_tracker import FaceTrackResult, Landmark

flt = FaceLandmarkFilter()
flt.set_param("dot_radius", 5.0)
flt.set_param("dot_r", 1.0)  # red dots
flt.set_param("dot_g", 0.0)
flt.set_param("dot_b", 0.0)

# Injecting a face result each frame before apply
result = FaceTrackResult(
    landmarks=[Landmark(x=0.5, y=0.5, z=0.0)],
    face_detected=True,
)
flt.update_face_result(result)
# pipeline then calls flt.apply(encoder, input_tex, output_tex)
```

**Edge cases:**
- **No face detected**: `update_face_result` receives `None` or a result with `face_detected=False` → blit pass only.
- **Partial landmark list**: If MediaPipe returns fewer than 478 landmarks (e.g. partial occlusion), only the available landmarks are drawn; the pre-allocated buffer is written up to `len(landmarks)` entries.
- **Landmark outside frame** (`x < 0` or `x > 1`): NDC conversion produces values outside `[-1, 1]`; the GPU clips the geometry automatically — no special handling required.

---

## 10. Validation Criteria

- `pytest tests/test_face_landmark_filter.py` passes with no errors.
- Filter appears in `filters.__all__` and can be added to `AppState.filters`.
- Running the application with face tracking enabled and the filter active renders visible green dots on detected face landmarks.

---

## 11. Related Specifications / Further Reading

- [spec/main.md](main.md) — high-level architecture and pipeline overview
- [MediaPipe Face Landmarker documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
- [spec/spec-design-main-screen.md](spec-design-main-screen.md) — UI and filter selection behaviour
