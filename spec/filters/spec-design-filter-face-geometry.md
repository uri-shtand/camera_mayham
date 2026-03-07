---
title: Face Geometry Mask Filter
version: 3.0
date_created: 2026-03-07
last_updated: 2026-03-07
tags: [design, filter, face-tracking, gpu, wgpu, 3d]
---

# Face Geometry Mask Filter

The Face Geometry Vertex Filter replaces the live camera feed with a
GPU-rendered Mask of the detected face. 

* Each of the 478 MediaPipe face landmarks is drawn as a small anti-aliased dot (vertex)
at its 3-D NDC position, producing a bright white point. 
* The eyes are drawn as a 3d structure of meshes that are created from the dots that define the eyes
* The eye-brows are drawn as a 3d structure of meshes that are created from the dots that define the eye-brows
* The nose is drawn as a 3d structure of meshes that are created from the dots that define the nose
* The mouth is drawn as a 3d structure of meshes that are created from the dots that define the mouth
* Meshes are also created for the chin and the cheeks.

Each face part gets a different shade of color. 

The result is a 3d mask that represents the users face

---

## 1. What the User Sees

When the filter is active the camera feed is replaced entirely:

1. **Black background** — the original camera image is never shown; the
   output is a solid black canvas.
2. **Coloured region meshes** — when a face is detected, each facial
   region is filled with a distinct neon/cyber colour using
   fan-triangulated triangle meshes, drawn back-to-front:
   - **Cheeks** — teal `(0.0, 0.9, 0.5)`
   - **Chin** — violet `(0.4, 0.2, 1.0)`
   - **Nose** — orange `(1.0, 0.6, 0.0)`
   - **Eyebrows** — magenta `(1.0, 0.0, 0.8)`
   - **Eyes** — cyan `(0.0, 0.8, 1.0)`
   - **Mouth** — rose `(1.0, 0.1, 0.3)`
3. **White vertex dots** — all 478 3-D landmark positions are each
   rendered as a small anti-aliased white SDF-circle dot on top of the
   coloured meshes.
4. **3-D perspective** — moving or tilting the head changes the
   projected positions of all mesh vertices and dots because the z-depth
   coordinate from MediaPipe is used as the clip-space depth.
5. **Transformation matrix integration** — when MediaPipe provides a
   `facial_transformation_matrixes` result, it is stored in the shared
   `FaceTrackResult.face_matrix` field for potential future use; the
   filter uses landmark positions directly for rendering.
6. **No face detected** — output is a solid black frame with no meshes
   and no dots.

---

## 2. Scope

| In scope | Out of scope |
|----------|--------------|
| `FaceGeometryFilter` class in `filters/face_geometry.py` | Changes to `BaseFilter` or `FilterPass` interfaces |
| Per-region coloured triangle meshes (fan triangulation) | Wireframe (line-list) face mesh rendering |
| White SDF-circle vertex dots on top of meshes | UV-textured or lit face shading |
| `_fan_triangles` helper and `_REGION_DATA` list | Multi-face geometry rendering |
| Mesh pipeline (`step_mode=vertex`, indexed) + dot pipeline (`step_mode=instance`) | User-adjustable region colours or dot size |
| `face_matrix` field added to `FaceTrackResult` in `tracking/face_tracker.py` | Changes to `AppState`, `FilterPass`, or any other core module |
| `FaceTracker._extract_face_result` extracting the matrix | |
| Pipeline integration in `app/application.py` | |
| Unit tests in `tests/test_face_geometry_filter.py` | |

---

## 3. Definitions

| Term | Meaning |
|------|---------|
| **Landmark** | A normalised 3-D facial point from MediaPipe; `x, y ∈ [0, 1]` with `(0, 0)` at top-left; `z` is relative depth (negative = closer to camera) |
| **NDC** | Normalised Device Coordinates — WebGPU clip space where `x, y ∈ [-1, 1]` with `(0, 0)` at screen centre |
| **Vertex dot** | A single landmark rendered as a small anti-aliased white circular quad on the GPU; one dot per landmark, drawn on top of region meshes |
| **SDF circle** | Signed-distance-field circle: the fragment shader discards corners beyond radius 1 of a unit square, producing a smooth circular dot |
| **Fan triangulation** | A method of filling a polygon by connecting every consecutive ring edge pair to a central apex, producing N triangles for a ring of N points |
| **Region mesh** | A set of fan-triangulated triangles filling one facial region (eye, brow, nose, mouth, chin, or cheek), drawn flat-coloured |
| **_REGION_DATA** | Module-level list of `_RegionSpec(indices, color)` pairs defining all face regions in back-to-front draw order |
| **Mesh pipeline** | A `triangle-list` render pipeline with `step_mode=vertex` that draws indexed triangles from the shared vertex buffer |
| **Instanced draw** | A single GPU draw call that renders the same 6-vertex quad geometry once per landmark instance, reading per-instance position from the vertex buffer |
| **face_matrix** | A 4 × 4 float32 numpy array from MediaPipe's `facial_transformation_matrixes[i]`; transforms canonical face model coordinates to camera/world space |
| **Vertex buffer** | A per-frame-updated GPU buffer containing NDC (x, y, z) positions for all 478 landmarks; shared by both the mesh pipeline (step_mode=vertex) and the dot pipeline (step_mode=instance) |
| **Params buffer** | A static uniform buffer holding `radius_x` and `radius_y` in NDC space (calibrated for ~4 px radius dots at 1280 × 720) |
| **triangle-list** | A wgpu primitive topology where each group of three vertices defines one independent filled triangle |
| **Blit pass** | Not used by this filter — the output is cleared to black and drawn fresh each frame |

---

## 4. Requirements

### 4.1 Vertex Dot Rendering

- **REQ-FG-001** When a face is detected the filter draws a bright white vertex dot at each of the 478 landmark 3-D positions on top of the coloured region meshes on a black background.
- **REQ-FG-002** Dot colour is hard-coded as `rgba(1.0, 1.0, 1.0, alpha)` (bright white) with anti-aliased SDF edges. No user-adjustable colour parameter is exposed.
- **REQ-FG-003** All 478 landmarks are rendered as white dots, producing clearly distinct accent points for every facial feature.
- **REQ-FG-004** Landmark positions are converted to NDC before upload to the shared vertex buffer: `ndc_x = x * 2 - 1`, `ndc_y = 1 - y * 2`, `ndc_z = z * 2` (scaled for visible depth variation).
- **REQ-FG-005** When no face is detected the output texture is cleared to solid black (RGBA `0, 0, 0, 255`) and neither meshes nor dots are drawn.
- **REQ-FG-010** Each dot is rendered as an instanced quad: 6 vertices (2 triangles) per landmark instance, drawn with a single `draw(6, 478)` call.
- **REQ-FG-011** Dot radii are stored in a static uniform buffer (`_params_buffer`) using NDC-space constants `_DOT_RADIUS_X ≈ 0.00625` and `_DOT_RADIUS_Y ≈ 0.01111`, calibrated to produce circular ~4-pixel-radius dots at 1280 × 720.

### 4.2 Coloured Region Meshes

- **REQ-FG-012** When a face is detected, each facial region (cheeks, chin, nose, eyebrows, eyes, outer mouth, inner mouth) is filled with a flat-coloured triangle mesh using the neon/cyber palette defined in `_REGION_DATA`.
- **REQ-FG-013** Region meshes are produced by fan-triangulation: the `_fan_triangles(line_pairs, center)` helper extracts the ordered contour ring from edge-pair connectivity constants (`[0::2]`) and generates N fan triangles (one per ring edge), all sharing the `center` apex.
- **REQ-FG-014** Regions are drawn back-to-front within a single render pass: cheeks first, then chin, nose, eyebrows, eyes, and mouth last (before the dot layer). This ensures foreground features occlude background regions without a depth buffer.
- **REQ-FG-015** Each region uses a static per-region index buffer (`uint32`, triangle-list) and a static per-region RGBA colour uniform buffer (16 bytes, `vec4f`). Both are created once in `_build_pipeline` and never modified at runtime.
- **REQ-FG-016** The mesh pipeline uses `step_mode=vertex` with indexed draw (`draw_indexed`); the same shared vertex buffer provides NDC positions addressed by the index buffer. No per-frame buffer allocations are made.

### 4.3 Transformation Matrix

- **REQ-FG-006** The `FaceTrackResult` dataclass gains an optional field `face_matrix: Optional[np.ndarray]` that holds the 4 × 4 float32 row-major transformation matrix from MediaPipe's `facial_transformation_matrixes` result when available, and `None` otherwise.
- **REQ-FG-007** `FaceTracker._extract_face_result` populates `face_matrix` from `detection.facial_transformation_matrixes[index]` when the detection data is present; guards against missing or out-of-bounds access.

### 4.4 Integration

- **REQ-FG-008** The filter receives face tracking data each frame via `update_face_result(result)`. It does not hold a reference to `AppState` or `FaceTracker`.
- **REQ-FG-009** The filter name property returns `"Face Geometry"`.

---

## 5. Constraints

- **CON-FG-001** The filter must implement the `BaseFilter` interface: `setup`, `_build_pipeline`, `apply`, `teardown`.
- **CON-FG-002** `apply` must not allocate new GPU buffers or textures; all buffers are pre-allocated in `_build_pipeline`.
- **CON-FG-003** The shared vertex buffer is sized for exactly 478 landmarks (`NUM_LANDMARKS`), 12 bytes per position (vec3f). It is re-uploaded every frame when a face is detected. It is bound with `step_mode=vertex` for the mesh pipeline and `step_mode=instance` for the dot pipeline; the step mode is declared per-pipeline, not per-buffer.
- **CON-FG-004** All static per-region index buffers and colour uniform buffers are uploaded once in `_build_pipeline` and never modified at runtime.
- **CON-FG-005** No depth-stencil attachment is used. All depth information is encoded in the vertex z coordinate; draw order approximates correct occlusion.
- **CON-FG-006** No input texture sampling occurs; the filter ignores `input_texture` entirely. The output texture is cleared to black and then drawn into.

---

## 6. Data Contracts

### 6.1 FaceTrackResult Extension

File: `tracking/face_tracker.py`

```python
@dataclass
class FaceTrackResult:
    landmarks: List[Landmark] = field(default_factory=list)
    head_pose: HeadPose = field(default_factory=HeadPose)
    blendshapes: dict = field(default_factory=dict)
    face_detected: bool = False
    face_matrix: Optional[np.ndarray] = None
    # 4x4 float32 row-major matrix from MediaPipe
    # facial_transformation_matrixes; None when unavailable.
```

### 6.2 Class Interface

File: `filters/face_geometry.py`

```python
def _fan_triangles(
    line_pairs: tuple[int, ...], center: int
) -> tuple[int, ...]: ...
# Extracts ordered ring from edge-pair constant ([0::2]) and produces
# N fan triangles all sharing the center apex.

_RegionSpec = namedtuple("_RegionSpec", ["indices", "color"])
_REGION_DATA: List[_RegionSpec]  # 10 regions in back-to-front order

class FaceGeometryFilter(BaseFilter):
    @property
    def name(self) -> str: ...          # returns "Face Geometry"

    def update_face_result(
        self, result: Optional[FaceTrackResult]
    ) -> None: ...
    # Stores result for use in next apply() call.

    def _build_pipeline(
        self,
        device: wgpu.GPUDevice,
        texture_format: wgpu.TextureFormat,
    ) -> None: ...
    # Creates shared vertex buffer (478 × 12 bytes).
    # Creates params buffer (16 bytes: radius_x, radius_y, pad, pad).
    # Creates per-region index buffer and colour uniform for each
    # entry in _REGION_DATA; creates mesh pipeline (step_mode=vertex,
    # indexed) and dot pipeline (step_mode=instance).

    def apply(
        self,
        encoder: wgpu.GPUCommandEncoder,
        input_texture: wgpu.GPUTexture,
        output_texture: wgpu.GPUTexture,
    ) -> None: ...
    # Clears output to black; if face detected, uploads vertex data,
    # draws each region mesh back-to-front via draw_indexed, then
    # draws white SDF-circle dots with draw(6, 478).

    def teardown(self) -> None: ...
    # Releases vertex buffer, params buffer, bind groups, all per-region
    # index buffers, colour buffers, and bind group lists.
```

### 6.3 GPU Resources

| Resource | Type | Size | Lifetime |
|----------|------|------|----------|
| `_vertex_buffer` | `VERTEX \| COPY_DST` | `478 × 12` bytes (vec3f per vertex) | Created in `_build_pipeline`, updated every frame when face detected, destroyed in `teardown` |
| `_params_buffer` | `UNIFORM \| COPY_DST` | `16` bytes (radius_x, radius_y, pad, pad) | Created and uploaded once in `_build_pipeline`; never modified at runtime |
| `_params_bind_group` | `GPUBindGroup` | N/A | Created once in `_build_pipeline`, reused every frame |
| `_render_pipeline` | `GPURenderPipeline` | N/A | Dot pipeline; created in `_build_pipeline` |
| `_mesh_render_pipeline` | `GPURenderPipeline` | N/A | Mesh pipeline; created in `_build_pipeline` |
| `_bind_group_layout` | `GPUBindGroupLayout` | N/A | Params layout (VERTEX stage); created in `_build_pipeline` |
| `_mesh_bind_group_layout` | `GPUBindGroupLayout` | N/A | Mesh colour layout (FRAGMENT stage); created in `_build_pipeline` |
| `_region_index_buffers[i]` | `INDEX \| COPY_DST` | `len(region.indices) × 4` bytes | One per region; created once, static; destroyed in `teardown` |
| `_region_color_buffers[i]` | `UNIFORM \| COPY_DST` | `16` bytes (vec4f RGBA) | One per region; created once, static; destroyed in `teardown` |
| `_region_bind_groups[i]` | `GPUBindGroup` | N/A | One per region; created once, reused every frame |

### 6.4 WGSL Shader Summary

#### Mesh shader (`_WGSL_MESH`)

**Vertex stage** (`vs_mesh`): receives `@location(0) pos: vec3f` (per-vertex NDC
x, y, z from the shared vertex buffer, addressed by the index buffer).
Outputs `vec4f(pos.x, pos.y, pos.z, 1.0)` directly.

**Fragment stage** (`fs_mesh`): reads `@group(0) @binding(0) var<uniform>
mesh_color: MeshColor` (a `vec4f`) and returns it unchanged.

#### Dot shader (`_WGSL_DOT`)

**Vertex stage** (`vs_main`): receives `@location(0) lm_pos: vec3f`
(per-instance NDC x, y, depth z). Builds a quad corner from
`@builtin(vertex_index)` and the uniform radii. Outputs clip position
`vec4f(lm_pos.xy + offset, lm_pos.z, 1.0)` and `local_uv: vec2f`
(corner offset ∈ [-1,1]) for the SDF computation.

**Fragment stage** (`fs_main`): computes `d = length(local_uv)`; discards if
`d > 1`; applies `smoothstep(0.7, 1.0, d)` for anti-aliased edges; outputs
`vec4f(1.0, 1.0, 1.0, alpha)` (bright white).

### 6.5 Dot Radius Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `_DOT_RADIUS_X` | `0.00625` | NDC half-width ≈ 4 px at 1280 px wide |
| `_DOT_RADIUS_Y` | `0.01111` | NDC half-height ≈ 4 px at 720 px tall |

### 6.6 Region Colour Palette

| Region | Colour name | RGBA |
|--------|-------------|------|
| Left cheek | Teal | `(0.0, 0.9, 0.5, 0.85)` |
| Right cheek | Teal | `(0.0, 0.9, 0.5, 0.85)` |
| Chin | Violet | `(0.4, 0.2, 1.0, 0.85)` |
| Nose | Orange | `(1.0, 0.6, 0.0, 0.90)` |
| Left eyebrow | Magenta | `(1.0, 0.0, 0.8, 0.90)` |
| Right eyebrow | Magenta | `(1.0, 0.0, 0.8, 0.90)` |
| Left eye | Cyan | `(0.0, 0.8, 1.0, 0.92)` |
| Right eye | Cyan | `(0.0, 0.8, 1.0, 0.92)` |
| Mouth outer | Rose | `(1.0, 0.1, 0.3, 0.90)` |
| Mouth inner | Rose | `(1.0, 0.1, 0.3, 0.95)` |

---

## 7. Acceptance Criteria

- **AC-FG-001** When the filter is enabled and a face is detected the output frame contains no camera image pixels; all non-dot, non-mesh pixels are black.
- **AC-FG-002** When no face is detected the entire output frame is black.
- **AC-FG-003** `FaceGeometryFilter().name == "Face Geometry"`.
- **AC-FG-004** `FaceGeometryFilter()._face_result is None` immediately after construction.
- **AC-FG-005** After `update_face_result(result)`, `_face_result` is the supplied result.
- **AC-FG-006** `FaceTrackResult().face_matrix is None` (default).
- **AC-FG-007** When `facial_transformation_matrixes` is present in the MediaPipe detection result, `FaceTracker` populates `face_matrix` as a numpy array of dtype float32.
- **AC-FG-008** `apply()` raises no exception when called with a stub GPU encoder regardless of whether a face is detected.
- **AC-FG-009** `_DOT_RADIUS_X` and `_DOT_RADIUS_Y` are positive floats < 0.1.
- **AC-FG-010** `_REGION_DATA` contains at least one entry; every entry's `indices` length is a multiple of 3 and all index values are in `[0, NUM_LANDMARKS)`.
- **AC-FG-011** All region RGBA colour components are in `[0.0, 1.0]`.
- **AC-FG-012** `_fan_triangles(ring, center)` returns a tuple with length `3 × (len(ring) // 2)`; the center index appears at position 0 of every triangle; the last triangle wraps to the first ring point.

---

## 8. Tests

| Test class | Covers |
|------------|--------|
| `TestFaceGeometryFilterIdentity` | AC-FG-003, AC-FG-004 — name, default state |
| `TestUpdateFaceResult` | AC-FG-005 — stores / replaces / accepts None |
| `TestHasVisibleLandmarks` | REQ-FG-005 — guard logic |
| `TestNDCConversion` | REQ-FG-004 — coordinate conversion formula |
| `TestDotRadiusConstants` | AC-FG-009 — `_DOT_RADIUS_X`, `_DOT_RADIUS_Y` are valid |
| `TestNumLandmarksConstant` | `NUM_LANDMARKS == 478` |
| `TestFaceTrackResultMatrix` | AC-FG-006 — default field is None |
| `TestFaceTrackerMatrixExtraction` | AC-FG-007 — matrix populated from detection |
| `TestFanTriangles` | AC-FG-012 — `_fan_triangles` output count, center apex, wrap-around |
| `TestRegionData` | AC-FG-010, AC-FG-011 — index divisibility, range, colour validity |
| `TestRegionColors` | Palette spot-checks — cyan eyes, magenta brows, orange nose, rose mouth, violet chin, teal cheeks |

Test framework: `pytest`.
Tests file: `tests/test_face_geometry_filter.py`.
GPU not required — all tests work on pure Python data structures.
