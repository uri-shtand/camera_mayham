---
title: Face Landmark Visualisation Filter
version: 1.0
date_created: 2026-03-05
last_updated: 2026-03-05
tags: [design, filter, face-tracking, gpu, wgpu]
---

# Introduction

This specification defines the **Face Landmark Visualisation Filter** for Camera Mayham.  The filter overlays all 478 MediaPipe face landmarks as coloured dots directly on the live camera feed using WebGPU (wgpu) GPU rendering.  It integrates with the existing filter chain architecture and the face tracking system without modifying either component's public contract.

---

## 1. Purpose & Scope

**Purpose:** Define the requirements, GPU rendering design, data contracts, and acceptance criteria for a filter that renders MediaPipe facial landmark points on the camera feed.

**Scope:** Covers the `FaceLandmarkFilter` class in `filters/face_landmarks.py`, the changes needed to the rendering pipeline to supply face tracking data to the filter, and the corresponding unit tests.

**Audience:** Engineers implementing the filter; AI systems extending Camera Mayham.

**Assumptions:**
- The `FaceTracker` produces a `FaceTrackResult` with up to 478 `Landmark` entries each with normalised `x, y ∈ [0, 1]` coordinates (0 = top/left, 1 = bottom/right).
- The rendering pipeline operates in the coordinate space described in `spec/main.md`.
- No GPU-side changes to `FilterPass` or `BaseFilter` interface are required.

---

## 2. Definitions

| Term | Definition |
|------|------------|
| **Landmark** | A normalised 3-D facial point detected by MediaPipe; `x, y` are in `[0, 1]` relative to frame size |
| **NDC** | Normalised Device Coordinates — the WebGPU clip-space coordinate system where x ∈ [-1, 1] (left→right) and y ∈ [-1, 1] (bottom→top) |
| **Ping-pong textures** | Two alternating GPU textures used by the filter chain so each filter reads from one and writes to the other |
| **Instanced draw** | A single GPU draw call that renders the same geometry (a quad) once per landmark instance |
| **SDF** | Signed Distance Field — a technique used in the fragment shader to produce smooth circular dots |
| **Blit pass** | A full-screen triangle draw that copies one texture to another, acting as a pass-through |
| **Alpha blending** | GPU compositing mode that blends a semi-transparent source over an existing pixel based on the source alpha value |
| **FaceTrackResult** | Dataclass returned by `FaceTracker.process()`, containing landmark list and a `face_detected` flag |

---

## 3. Requirements, Constraints & Guidelines

### Functional Requirements

- **REQ-LM-001**: The filter shall render a dot at each detected MediaPipe face landmark position on the camera frame.
- **REQ-LM-002**: When no face is detected (`FaceTrackResult.face_detected == False` or `face_result is None`) the filter shall act as a pass-through, leaving the frame unmodified.
- **REQ-LM-003**: The filter shall expose runtime-adjustable parameters: `dot_radius` (float, pixels), `dot_r`, `dot_g`, `dot_b`, `dot_a` (float, colour components `[0, 1]`).
- **REQ-LM-004**: The filter shall accept updated face tracking data each frame via a dedicated `update_face_result` method.
- **REQ-LM-005**: Landmark dots shall be rendered with smooth circular edges using a fragment-shader SDF and alpha blending over the camera frame.
- **REQ-LM-006**: The filter name reported by the `name` property shall be `"Face Landmarks"`.

### Constraints

- **CON-LM-001**: The filter must conform to the `BaseFilter` interface — `setup`, `_build_pipeline`, `apply`, `teardown`.
- **CON-LM-002**: `apply` must not allocate new GPU textures or buffers; vertex and uniform buffers are pre-allocated in `_build_pipeline`.
- **CON-LM-003**: The maximum number of landmarks supported is 478 (`MAX_LANDMARKS`); the vertex buffer is pre-allocated to this capacity.
- **CON-LM-004**: The filter uses two render passes within a single `apply` call: a blit pass (input → output) followed by a landmark overlay pass (load output, draw circles).
- **CON-LM-005**: The face tracker result must be injected via `update_face_result`; the filter does not hold a reference to `AppState` or `FaceTracker`.

### Guidelines

- **GUD-LM-001**: Default colour should be bright green (`dot_r=0.0, dot_g=1.0, dot_b=0.0, dot_a=1.0`) for maximum contrast.
- **GUD-LM-002**: Default dot radius should be `3.0` pixels.
- **GUD-LM-003**: Coordinate conversion from MediaPipe normalised space to NDC: `ndc_x = x * 2 - 1`, `ndc_y = 1 - y * 2` (flip Y because MediaPipe y increases downward, NDC y increases upward).
- **GUD-LM-004**: NDC radius must be aspect-ratio corrected: `radius_x = 2 * dot_radius / width`, `radius_y = 2 * dot_radius / height`, derived from the input texture dimensions at apply time.

---

## 4. Interfaces & Data Contracts

### 4.1 Class: `FaceLandmarkFilter(BaseFilter)`

File: `filters/face_landmarks.py`

```python
class FaceLandmarkFilter(BaseFilter):
    name: str  # "Face Landmarks"
    params: Dict[str, Any]  # see Parameter Table below

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

### 4.2 Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `dot_radius` | float | `3.0` | `1.0 – 20.0` | Dot radius in pixels |
| `dot_r` | float | `0.0` | `0.0 – 1.0` | Red channel of dot colour |
| `dot_g` | float | `1.0` | `0.0 – 1.0` | Green channel of dot colour |
| `dot_b` | float | `0.0` | `0.0 – 1.0` | Blue channel of dot colour |
| `dot_a` | float | `1.0` | `0.0 – 1.0` | Alpha of dot colour |

### 4.3 GPU Resource Layout

| Resource | Type | Size | Usage |
|----------|------|------|-------|
| `_blit_bgl` | BindGroupLayout | — | Bindings 0 (texture), 1 (sampler) |
| `_landmark_bgl` | BindGroupLayout | — | Binding 0 (uniform buffer) |
| `_landmark_param_buffer` | GPUBuffer | 32 bytes | `UNIFORM \| COPY_DST` — LandmarkParams struct |
| `_landmark_vertex_buffer` | GPUBuffer | 478 × 8 bytes | `VERTEX \| COPY_DST` — per-instance NDC positions |
| `_blit_pipeline` | GPURenderPipeline | — | Full-screen blit, no blend |
| `_landmark_pipeline` | GPURenderPipeline | — | Instanced circles, alpha blend |

### 4.4 WGSL Uniform: `LandmarkParams` (32 bytes, binding 0, group 0)

```wgsl
struct LandmarkParams {
    dot_r    : f32,   // offset  0
    dot_g    : f32,   // offset  4
    dot_b    : f32,   // offset  8
    dot_a    : f32,   // offset 12
    radius_x : f32,   // offset 16 — NDC half-width
    radius_y : f32,   // offset 20 — NDC half-height
    _pad0    : f32,   // offset 24
    _pad1    : f32,   // offset 28
}
```

### 4.5 Pipeline Integration (`rendering/pipeline.py`)

Before recording the filter pass, `render_frame` must call `update_face_result` on every enabled filter that exposes the method (duck-typing; no interface change to `BaseFilter`):

```python
for flt in state.enabled_filters():
    if hasattr(flt, "update_face_result"):
        flt.update_face_result(state.face_result)
```

---

## 5. Acceptance Criteria

- **AC-LM-001**: Given a valid `FaceTrackResult` with 478 landmarks, When `apply` is called, Then 478 instanced quads are submitted in the landmark render pass.
- **AC-LM-002**: Given `face_result is None`, When `apply` is called, Then only the blit pass executes and `output_texture` equals `input_texture` content.
- **AC-LM-003**: Given `FaceTrackResult.face_detected == False`, When `apply` is called, Then only the blit pass executes.
- **AC-LM-004**: Given a landmark with normalised coordinates `(x=0.0, y=0.0)`, Then its NDC position shall be `(-1.0, 1.0)` (top-left corner).
- **AC-LM-005**: Given a landmark with normalised coordinates `(x=1.0, y=1.0)`, Then its NDC position shall be `(1.0, -1.0)` (bottom-right corner).
- **AC-LM-006**: Given a landmark with normalised coordinates `(x=0.5, y=0.5)`, Then its NDC position shall be `(0.0, 0.0)` (screen centre).
- **AC-LM-007**: The filter `name` property shall return `"Face Landmarks"`.
- **AC-LM-008**: All five default parameters (`dot_radius`, `dot_r`, `dot_g`, `dot_b`, `dot_a`) shall be present immediately after construction.
- **AC-LM-009**: `set_param` shall accept valid parameter keys and raise `KeyError` for unknown keys.
- **AC-LM-010**: `update_face_result` shall store the provided `FaceTrackResult` for use in the next `apply` call.

---

## 6. Test Automation Strategy

- **Test Levels**: Unit only (GPU is not available in CI).
- **Frameworks**: `pytest`; GPU resources stubbed out as `None`.
- **Test file**: `tests/test_face_landmark_filter.py`
- **Test Data Management**: `FaceTrackResult` and `Landmark` instances constructed inline.
- **CI/CD Integration**: All tests run via `pytest` in the existing test suite.
- **Coverage Requirements**: All public methods and the NDC conversion logic must be covered.

---

## 7. Rationale & Context

The face landmark visualisation filter serves as both a diagnostic tool (verifying tracker accuracy) and a creative visual effect.  Rendering is GPU-side via WebGPU to avoid adding CPU-side compositing overhead to the hot render loop.

A two-pass approach within `apply` (blit then overlay) was chosen over a combined single-pass shader to reuse the standard passthrough blit pipeline, keep each shader simple, and allow the landmark pass to independently enable alpha blending without complications.

Per-instance vertex buffer data (two floats per landmark) is the most GPU-efficient way to handle ordered instanced draw calls in WGSL, following the same pattern established in `games/bubble_pop.py`.

Injecting face data via `update_face_result` keeps `BaseFilter.apply` signature unchanged and avoids coupling the filter chain to the face tracking subsystem.

---

## 8. Dependencies & External Integrations

### Technology Platform Dependencies

- **PLT-LM-001**: `wgpu` Python bindings — WebGPU GPU pipeline, texture, buffer management.
- **PLT-LM-002**: `mediapipe` — source of `FaceTrackResult` and `Landmark` dataclasses via `tracking.face_tracker`.

### Internal Architectural Dependencies

- **INF-LM-001**: `filters.base.BaseFilter` — base class; must not modify its interface.
- **INF-LM-002**: `tracking.face_tracker.FaceTrackResult` — input data contract for landmark positions.
- **INF-LM-003**: `rendering.pipeline.RenderPipeline.render_frame` — must be updated to call `update_face_result` on face-aware filters before the filter pass.

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
