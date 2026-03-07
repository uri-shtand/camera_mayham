"""
Face Geometry Mask Filter.

Replaces the live camera feed with a GPU-rendered 3-D mask of the
detected face.  When a face is tracked the output is:

1. **Coloured region meshes** — each facial region (eyes, eyebrows,
   nose, mouth, chin, cheeks) is filled with a distinct neon/cyber
   colour using fan-triangulated triangle meshes.
2. **White vertex dots** — all 478 MediaPipe landmark positions are
   each drawn as a small anti-aliased SDF circle on top of the meshes
   to accent every landmark position.

Rendering uses two wgpu pipelines recorded into one render pass:

1. **Mesh pipeline** (``triangle-list``, indexed, ``step_mode=vertex``)
   — draws flat-coloured filled triangles for each facial region.  Each
   region has a static index buffer and a static uniform colour buffer.
   The same vertex buffer shared with the dot pipeline is used here; the
   per-vertex NDC positions are indexed directly.
2. **Dot pipeline** (``triangle-list``, instanced, ``step_mode=instance``)
   — draws a bright white SDF-circle quad on top of each landmark
   position.

There is no input-texture sampling; the output texture is cleared to
black each frame (spec CON-FG-006).

NDC coordinate conversion (MediaPipe → clip space):
    ndc_x = landmark.x * 2 - 1
    ndc_y = 1 - landmark.y * 2     (flip Y: MediaPipe y grows downward)
    ndc_z = landmark.z * 2         (scale relative depth for clip space)

Dot radii (``_DOT_RADIUS_X``, ``_DOT_RADIUS_Y``) are expressed in NDC
units calibrated for a 1280 × 720 frame to produce circular ~4-pixel-
radius dots.  They are written once to a uniform buffer at build time.

Region colours are hard-coded per the neon/cyber palette:
    eyes:    cyan    (0.0, 0.8, 1.0)
    brows:   magenta (1.0, 0.0, 0.8)
    nose:    orange  (1.0, 0.6, 0.0)
    mouth:   rose    (1.0, 0.1, 0.3)
    chin:    violet  (0.4, 0.2, 1.0)
    cheeks:  teal    (0.0, 0.9, 0.5)

The ``face_matrix`` field on ``FaceTrackResult`` stores the 4×4
facial-transformation matrix from MediaPipe when available.
"""

from __future__ import annotations

import struct
from collections import namedtuple
from typing import Any, List, Optional

import numpy as np
import wgpu

from filters.base import BaseFilter
from tracking.face_tracker import FaceTrackResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Total MediaPipe face landmarks per detected face (REQ-FG-003).
NUM_LANDMARKS: int = 478

# Scale applied to the MediaPipe z-depth coordinate.
# MediaPipe z values are small (< 0.1 typically); scaling by 2 maps
# them to a visible range within the NDC [-1, 1] cube.
_Z_SCALE: float = 2.0

# ---------------------------------------------------------------------------
# Dot-radius constants (NDC space, calibrated for 1280 × 720)
#
# To produce a circular dot with ~4 px radius:
#   radius_x = 2 * 4 / 1280 ≈ 0.00625   (NDC width  per pixel = 2/W)
#   radius_y = 2 * 4 /  720 ≈ 0.01111   (NDC height per pixel = 2/H)
# ---------------------------------------------------------------------------
_DOT_RADIUS_X: float = 0.00625
_DOT_RADIUS_Y: float = 0.01111

# ---------------------------------------------------------------------------
# Face contour edge connectivity (line-list pairs used to derive rings)
#
# Each tuple stores alternating src/dst landmark indices.  Taking every
# other element (``[0::2]``) yields the ordered contour ring used by
# ``_fan_triangles`` to generate filled triangle meshes.
# ---------------------------------------------------------------------------

_FACE_OVAL: tuple[int, ...] = (
    10, 338, 338, 297, 297, 332, 332, 284, 284, 251, 251, 389,
    389, 356, 356, 454, 454, 323, 323, 361, 361, 288, 288, 397,
    397, 365, 365, 379, 379, 378, 378, 400, 400, 377, 377, 152,
    152, 148, 148, 176, 176, 149, 149, 150, 150, 136, 136, 172,
    172, 58, 58, 132, 132, 93, 93, 234, 234, 127, 127, 162,
    162, 21, 21, 54, 54, 103, 103, 67, 67, 109, 109, 10,
)

_LEFT_EYE: tuple[int, ...] = (
    33, 7, 7, 163, 163, 144, 144, 145, 145, 153, 153, 154,
    154, 155, 155, 133, 133, 173, 173, 157, 157, 158, 158, 159,
    159, 160, 160, 161, 161, 246, 246, 33,
)

_RIGHT_EYE: tuple[int, ...] = (
    263, 249, 249, 390, 390, 373, 373, 374, 374, 382, 382, 381,
    381, 380, 380, 362, 362, 398, 398, 384, 384, 385, 385, 386,
    386, 387, 387, 388, 388, 466, 466, 263,
)

_LEFT_EYEBROW: tuple[int, ...] = (
    46, 53, 53, 52, 52, 65, 65, 55,
    55, 70, 70, 63, 63, 105, 105, 66, 66, 107, 107, 46,
)

_RIGHT_EYEBROW: tuple[int, ...] = (
    276, 283, 283, 282, 282, 295, 295, 285,
    285, 300, 300, 293, 293, 334, 334, 296, 296, 336, 336, 276,
)

# Nose bridge ring only (the full _NOSE constant branches at the
# nostrils and cannot be reduced to a single closed contour).  The
# bridge runs from the glabella (168) down to the nose tip (4) and
# back via the lateral cartilage points, forming a closed loop.
_NOSE_RING: tuple[int, ...] = (
    168, 6, 6, 197, 197, 195, 195, 5, 5, 4,
    4, 45, 45, 220, 220, 115, 115, 48, 48, 168,
)

_LIPS_OUTER: tuple[int, ...] = (
    61, 146, 146, 91, 91, 181, 181, 84, 84, 17, 17, 314,
    314, 405, 405, 321, 321, 375, 375, 291, 291, 409, 409, 270,
    270, 269, 269, 267, 267, 0, 0, 37, 37, 39, 39, 40,
    40, 185, 185, 61,
)

_LIPS_INNER: tuple[int, ...] = (
    78, 95, 95, 88, 88, 178, 178, 87, 87, 14, 14, 317,
    317, 402, 402, 318, 318, 324, 324, 308, 308, 415, 415, 310,
    310, 311, 311, 312, 312, 13, 13, 82, 82, 81, 81, 80,
    80, 191, 191, 78,
)

# ---------------------------------------------------------------------------
# Triangle mesh generation helpers
# ---------------------------------------------------------------------------


def _fan_triangles(
    line_pairs: tuple[int, ...], center: int
) -> tuple[int, ...]:
    """
    Generate fan-triangulation indices from an ordered contour ring.

    The ``line_pairs`` tuple stores alternating src/dst landmark index
    pairs produced by a line-list edge connectivity constant. Taking
    every other element (``[0::2]``) extracts the ordered contour ring.

    A fan triangulation is produced by connecting each consecutive pair
    of ring points back to ``center``, forming ``N`` triangles (where
    ``N`` is the ring length).  The last triangle wraps around to close
    the fan.

    Parameters:
        line_pairs (tuple[int, ...]): Alternating edge-pair connectivity
            constant (e.g. ``_LEFT_EYE``).
        center (int): Landmark index used as the triangle fan apex.

    Returns:
        tuple[int, ...]: Flat index list suitable for a triangle-list
            index buffer (3 indices per triangle).
    """
    ring = line_pairs[0::2]  # extract ordered contour points
    n = len(ring)
    result: list[int] = []
    for i in range(n):
        result.extend(
            [center, ring[i], ring[(i + 1) % n]]
        )
    return tuple(result)


# ---------------------------------------------------------------------------
# Region-specific triangle index tuples (computed once at import time)
#
# Each constant contains a flat list of uint32 indices (3 per triangle)
# ready for a wgpu INDEX buffer with triangle-list topology.
#
# Center landmark choices:
#   Eyes:      iris landmarks  468 (left)  /  473 (right)
#   Eyebrows:  mid-brow point  105 (left)  /  334 (right)
#   Nose:      nose tip        4
#   Mouth:     lip centre      13 (inner, between inner lips)
#              top lip point   0  (outer ring)
#   Chin:      chin tip        152
#   Cheeks:    jaw landmarks   234 (left)  /  454 (right)
# ---------------------------------------------------------------------------

_TRIANGLES_LEFT_EYE: tuple[int, ...] = _fan_triangles(_LEFT_EYE, 468)
_TRIANGLES_RIGHT_EYE: tuple[int, ...] = _fan_triangles(_RIGHT_EYE, 473)
_TRIANGLES_LEFT_EYEBROW: tuple[int, ...] = _fan_triangles(
    _LEFT_EYEBROW, 105
)
_TRIANGLES_RIGHT_EYEBROW: tuple[int, ...] = _fan_triangles(
    _RIGHT_EYEBROW, 334
)
_TRIANGLES_NOSE: tuple[int, ...] = _fan_triangles(_NOSE_RING, 4)
# Outer mouth ring fans from the top-lip centre (0).
_TRIANGLES_MOUTH_OUTER: tuple[int, ...] = _fan_triangles(
    _LIPS_OUTER, 0
)
# Inner mouth ring fans from the mid-tooth centre (13).
_TRIANGLES_MOUTH_INNER: tuple[int, ...] = _fan_triangles(
    _LIPS_INNER, 13
)

# Chin: lower half of the oval arc (from 152 going left to 234,
# then continuing right to 454, closed back to 152).
# Extract from _FACE_OVAL the ordered ring and find the arc from
# 234 to 454 passing through 152.  The oval ring (via [0::2]) is:
# [10,338,297,332,284,251,389,356,454,323,361,288,397,365,
#  379,378,400,377,152,148,176,149,150,136,172,58,132,93,
#  234,127,162,21,54,103,67,109]
# Chin arc: 234 → 93 → 132 → 58 → 172 → 136 → 150 → 149 → 176
#           → 148 → 152 → 377 → 400 → 378 → 379 → 365 → 397
#           → 288 → 361 → 323 → 454
_CHIN_RING: tuple[int, ...] = (
    234, 93, 93, 132, 132, 58, 58, 172, 172, 136,
    136, 150, 150, 149, 149, 176, 176, 148, 148, 152,
    152, 377, 377, 400, 400, 378, 378, 379, 379, 365,
    365, 397, 397, 288, 288, 361, 361, 323, 323, 454,
    454, 234,
)
_TRIANGLES_CHIN: tuple[int, ...] = _fan_triangles(_CHIN_RING, 152)

# Left cheek: left side upper arc from 234 to 10.
_LEFT_CHEEK_RING: tuple[int, ...] = (
    234, 127, 127, 162, 162, 21, 21, 54, 54, 103,
    103, 67, 67, 109, 109, 10, 10, 338, 338, 297,
    297, 332, 332, 284, 284, 251, 251, 389, 389, 356,
    356, 454, 454, 234,
)
_TRIANGLES_LEFT_CHEEK: tuple[int, ...] = _fan_triangles(
    _LEFT_CHEEK_RING, 234
)

# Right cheek: right side upper arc from 454 back around to 234.
_RIGHT_CHEEK_RING: tuple[int, ...] = (
    454, 323, 323, 361, 361, 288, 288, 397, 397, 365,
    365, 379, 379, 378, 378, 400, 400, 377, 377, 152,
    152, 148, 148, 176, 176, 149, 149, 150, 150, 136,
    136, 172, 172, 58, 58, 132, 132, 93, 93, 234,
    234, 454,
)
_TRIANGLES_RIGHT_CHEEK: tuple[int, ...] = _fan_triangles(
    _RIGHT_CHEEK_RING, 454
)

# ---------------------------------------------------------------------------
# Region data: ordered list of (triangle indices, RGBA colour).
#
# Draw order is back-to-front so later regions occlude earlier ones:
#   cheeks → chin → nose → eyebrows → eyes → mouth (outer then inner)
# White vertex dots are drawn last (separate dot pipeline pass).
#
# Neon/cyber colour palette — symmetric (L=R same hue):
#   cheeks   teal    (0.0, 0.9, 0.5)
#   chin     violet  (0.4, 0.2, 1.0)
#   nose     orange  (1.0, 0.6, 0.0)
#   eyebrows magenta (1.0, 0.0, 0.8)
#   eyes     cyan    (0.0, 0.8, 1.0)
#   mouth    rose    (1.0, 0.1, 0.3)
# ---------------------------------------------------------------------------

_RegionSpec = namedtuple("_RegionSpec", ["indices", "color"])

_REGION_DATA: List[_RegionSpec] = [
    # ── Background layers first ────────────────────────────────────
    _RegionSpec(_TRIANGLES_LEFT_CHEEK,    (0.0, 0.9, 0.5, 0.85)),
    _RegionSpec(_TRIANGLES_RIGHT_CHEEK,   (0.0, 0.9, 0.5, 0.85)),
    _RegionSpec(_TRIANGLES_CHIN,          (0.4, 0.2, 1.0, 0.85)),
    # ── Mid layers ─────────────────────────────────────────────────
    _RegionSpec(_TRIANGLES_NOSE,          (1.0, 0.6, 0.0, 0.90)),
    _RegionSpec(_TRIANGLES_LEFT_EYEBROW,  (1.0, 0.0, 0.8, 0.90)),
    _RegionSpec(_TRIANGLES_RIGHT_EYEBROW, (1.0, 0.0, 0.8, 0.90)),
    # ── Foreground layers ──────────────────────────────────────────
    _RegionSpec(_TRIANGLES_LEFT_EYE,      (0.0, 0.8, 1.0, 0.92)),
    _RegionSpec(_TRIANGLES_RIGHT_EYE,     (0.0, 0.8, 1.0, 0.92)),
    _RegionSpec(_TRIANGLES_MOUTH_OUTER,   (1.0, 0.1, 0.3, 0.90)),
    _RegionSpec(_TRIANGLES_MOUTH_INNER,   (1.0, 0.1, 0.3, 0.95)),
]

# ---------------------------------------------------------------------------
# WGSL shaders
# ---------------------------------------------------------------------------

# Flat-colour mesh shader.
#
# Vertex stage: receives per-vertex NDC position from the shared vertex
# buffer (step_mode=vertex); the GPU reads the buffer value at the index
# supplied via the index buffer (``draw_indexed``).
#
# Fragment stage: reads the region colour from a per-region uniform
# buffer (group 0, binding 0) and outputs it directly.
_WGSL_MESH: str = """
struct MeshColor {
    color : vec4f,
}

@group(0) @binding(0) var<uniform> mesh_color : MeshColor;

struct MeshOut {
    @builtin(position) clip_pos : vec4f,
}

@vertex
fn vs_mesh(@location(0) pos : vec3f) -> MeshOut {
    var out: MeshOut;
    out.clip_pos = vec4f(pos.x, pos.y, pos.z, 1.0);
    return out;
}

@fragment
fn fs_mesh(in: MeshOut) -> @location(0) vec4f {
    return mesh_color.color;
}
"""

# Instanced quad with SDF circle fragment shading (dot overlay).
#
# Vertex stage: one instance = one landmark.  ``@location(0) lm_pos``
# is the per-instance NDC (x, y, z) position (step_mode=instance).
# ``@builtin(vertex_index)`` selects one of the six corners of the
# two-triangle quad for this landmark.  The quad is expanded around the
# landmark centre by (radius_x, radius_y) in NDC space; the corner
# offset is passed to the fragment shader as ``local_uv`` for SDF.
#
# Fragment stage: SDF circle — discards corners beyond radius 1,
# smoothstep anti-aliasing in outer 30 % of radius.  Bright white.
_WGSL_DOT: str = """
struct Params {
    radius_x : f32,
    radius_y : f32,
    _pad0    : f32,
    _pad1    : f32,
}

@group(0) @binding(0) var<uniform> params : Params;

struct VertexOutput {
    @builtin(position) clip_pos : vec4f,
    @location(0)       local_uv : vec2f,
}

@vertex
fn vs_main(
    @builtin(vertex_index)  vi     : u32,
    @location(0)            lm_pos : vec3f,
) -> VertexOutput {
    // Six corners forming two triangles (one quad per instance).
    var offsets = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f(-1.0,  1.0),
        vec2f(-1.0,  1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0),
    );
    let off = offsets[vi];
    let xy  = lm_pos.xy + vec2f(
        off.x * params.radius_x,
        off.y * params.radius_y,
    );
    var out: VertexOutput;
    out.clip_pos = vec4f(xy, lm_pos.z, 1.0);
    out.local_uv = off;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let d = length(in.local_uv);
    if d > 1.0 { discard; }
    // Anti-aliased edge in outer 30 % of radius.
    let alpha = 1.0 - smoothstep(0.7, 1.0, d);
    // Bright white vertex dots sit on top of the coloured mesh.
    return vec4f(1.0, 1.0, 1.0, alpha);
}
"""


class FaceGeometryFilter(BaseFilter):
    """
    GPU filter that renders a coloured 3-D face mask on a black background.

    When a face is detected the output shows:

    1. **Coloured region meshes** — each facial region (eyes, eyebrows,
       nose, mouth, chin, cheeks) is filled with a distinct neon/cyber
       colour using fan-triangulated triangle meshes drawn back-to-front.
    2. **White vertex dots** — all 478 MediaPipe landmark positions are
       each drawn as a small anti-aliased SDF circle on top of the meshes.

    Two wgpu pipelines are used in a single render pass per frame:

    * **Mesh pipeline** (``triangle-list``, indexed, ``step_mode=vertex``)
      draws flat-coloured filled triangles for each region.  Each region
      has a static index buffer and a static RGBA colour uniform.  The
      shared vertex buffer provides NDC positions indexed by the index buf.
    * **Dot pipeline** (``triangle-list``, instanced, ``step_mode=instance``)
      draws a white SDF-circle quad over every landmark position.

    The filter does **not** sample the input camera texture; it ignores
    ``input_texture`` entirely (CON-FG-006).

    Face tracking data is supplied each frame via
    :py:meth:`update_face_result` before :py:meth:`apply` is called.
    """

    def __init__(self) -> None:
        """Initialise the filter.  All visual constants are hard-coded."""
        super().__init__()

        # Latest face tracking result; injected each frame before apply().
        self._face_result: Optional[FaceTrackResult] = None

        # GPU resources — allocated in _build_pipeline.
        # Dot pipeline (triangle-list, step_mode=instance).
        self._render_pipeline: Any = None
        # Mesh pipeline (triangle-list, indexed, step_mode=vertex).
        self._mesh_render_pipeline: Any = None
        self._mesh_bind_group_layout: Any = None

        # Shared vertex buffer: NDC (x, y, z) per landmark, float32x3.
        # Updated every frame when a face is detected.
        self._vertex_buffer: Any = None

        # Static uniform for dot radii (radius_x, radius_y, pad, pad).
        self._params_buffer: Any = None
        self._params_bind_group: Any = None

        # Per-region GPU resources (one entry per _REGION_DATA item).
        # _region_index_buffers[i]  — static INDEX buffer
        # _region_color_buffers[i]  — static UNIFORM buffer (vec4f)
        # _region_bind_groups[i]    — bind group for mesh pipeline
        self._region_index_buffers: List[Any] = []
        self._region_color_buffers: List[Any] = []
        self._region_bind_groups: List[Any] = []

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """
        Unique filter name.

        Returns:
            str: ``"Face Geometry"``
        """
        return "Face Geometry"

    # ------------------------------------------------------------------
    # Face data injection
    # ------------------------------------------------------------------

    def update_face_result(
        self, result: Optional[FaceTrackResult]
    ) -> None:
        """
        Store the latest face tracking result for use in the next frame.

        Called by the rendering pipeline each frame before
        :py:meth:`apply` is invoked.

        Parameters:
            result (Optional[FaceTrackResult]): The tracking result for
                the current frame, or ``None`` when unavailable.
        """
        self._face_result = result

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _build_pipeline(
        self,
        device: wgpu.GPUDevice,
        texture_format: wgpu.TextureFormat,
    ) -> None:
        """
        Compile shaders and create all wgpu pipeline objects.

        Creates the shared vertex buffer, per-region static index and
        colour buffers, the mesh pipeline (``step_mode=vertex``) and the
        dot pipeline (``step_mode=instance``).  No GPU resources are
        allocated at render time (CON-FG-002).

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            texture_format (wgpu.TextureFormat): Output texture format.
        """
        # ── Shared vertex buffer ────────────────────────────────────
        # 478 landmarks × vec3f (ndc_x, ndc_y, ndc_z) = 478 × 12 bytes.
        # Updated every frame; used by both mesh (indexed/vertex) and
        # dot (instanced) pipelines.
        self._vertex_buffer = device.create_buffer(
            size=NUM_LANDMARKS * 3 * 4,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
        )

        # ── Dot-radius uniform buffer (static) ──────────────────────
        # 16 bytes: radius_x f32, radius_y f32, pad f32, pad f32.
        self._params_buffer = device.create_buffer_with_data(
            data=struct.pack(
                "4f", _DOT_RADIUS_X, _DOT_RADIUS_Y, 0.0, 0.0
            ),
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        # Params bind group layout — visible to VERTEX stage (dot shader).
        self._bind_group_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": "uniform"},
                }
            ]
        )
        self._params_bind_group = device.create_bind_group(
            layout=self._bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self._params_buffer,
                        "offset": 0,
                        "size": 16,
                    },
                }
            ],
        )

        # ── Mesh bind group layout (fragment stage: region colour) ───
        self._mesh_bind_group_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "uniform"},
                }
            ]
        )

        # ── Per-region index buffers, colour buffers, bind groups ────
        # All region GPU resources are static; created once here.
        self._region_index_buffers = []
        self._region_color_buffers = []
        self._region_bind_groups = []

        for region in _REGION_DATA:
            # Index buffer: flat uint32 triangle indices.
            idx_data = np.array(region.indices, dtype=np.uint32)
            idx_buf = device.create_buffer_with_data(
                data=idx_data.tobytes(),
                usage=(
                    wgpu.BufferUsage.INDEX
                    | wgpu.BufferUsage.COPY_DST
                ),
            )
            self._region_index_buffers.append(idx_buf)

            # Colour uniform buffer: vec4f (r, g, b, a) = 16 bytes.
            col_buf = device.create_buffer_with_data(
                data=struct.pack("4f", *region.color),
                usage=(
                    wgpu.BufferUsage.UNIFORM
                    | wgpu.BufferUsage.COPY_DST
                ),
            )
            self._region_color_buffers.append(col_buf)

            # Bind group for this region's colour uniform.
            bg = device.create_bind_group(
                layout=self._mesh_bind_group_layout,
                entries=[
                    {
                        "binding": 0,
                        "resource": {
                            "buffer": col_buf,
                            "offset": 0,
                            "size": 16,
                        },
                    }
                ],
            )
            self._region_bind_groups.append(bg)

        # ── Mesh render pipeline ─────────────────────────────────────
        # triangle-list, indexed draw, step_mode=vertex.
        mesh_shader = device.create_shader_module(code=_WGSL_MESH)
        mesh_layout = device.create_pipeline_layout(
            bind_group_layouts=[self._mesh_bind_group_layout]
        )
        self._mesh_render_pipeline = device.create_render_pipeline(
            layout=mesh_layout,
            vertex={
                "module": mesh_shader,
                "entry_point": "vs_mesh",
                "buffers": [
                    {
                        # Per-vertex NDC position from shared vertex buf.
                        "array_stride": 12,
                        "step_mode": "vertex",
                        "attributes": [
                            {
                                "format": "float32x3",
                                "offset": 0,
                                "shader_location": 0,
                            }
                        ],
                    }
                ],
            },
            primitive={
                "topology": "triangle-list",
                "strip_index_format": None,
                "front_face": "ccw",
                "cull_mode": "none",
            },
            depth_stencil=None,
            multisample={"count": 1, "mask": 0xFFFFFFFF},
            fragment={
                "module": mesh_shader,
                "entry_point": "fs_mesh",
                "targets": [
                    {
                        "format": texture_format,
                        "blend": {
                            "color": {
                                "src_factor": "src-alpha",
                                "dst_factor": "one-minus-src-alpha",
                                "operation": "add",
                            },
                            "alpha": {
                                "src_factor": "one",
                                "dst_factor": "one-minus-src-alpha",
                                "operation": "add",
                            },
                        },
                        "write_mask": wgpu.ColorWrite.ALL,
                    }
                ],
            },
        )

        # ── Dot render pipeline ──────────────────────────────────────
        # triangle-list, instanced (one instance per landmark).
        # step_mode=instance: each landmark advances the vertex buffer
        # by one element (12 bytes) per instance.
        dot_shader = device.create_shader_module(code=_WGSL_DOT)
        dot_layout = device.create_pipeline_layout(
            bind_group_layouts=[self._bind_group_layout]
        )
        self._render_pipeline = device.create_render_pipeline(
            layout=dot_layout,
            vertex={
                "module": dot_shader,
                "entry_point": "vs_main",
                "buffers": [
                    {
                        # Per-instance landmark position (vec3f = 12 B).
                        "array_stride": 12,
                        "step_mode": "instance",
                        "attributes": [
                            {
                                "format": "float32x3",
                                "offset": 0,
                                "shader_location": 0,
                            }
                        ],
                    }
                ],
            },
            primitive={
                "topology": "triangle-list",
                "strip_index_format": None,
                "front_face": "ccw",
                "cull_mode": "none",
            },
            depth_stencil=None,
            multisample={"count": 1, "mask": 0xFFFFFFFF},
            fragment={
                "module": dot_shader,
                "entry_point": "fs_main",
                "targets": [
                    {
                        "format": texture_format,
                        # Alpha blending for smooth SDF circle edges.
                        "blend": {
                            "color": {
                                "src_factor": "src-alpha",
                                "dst_factor": "one-minus-src-alpha",
                                "operation": "add",
                            },
                            "alpha": {
                                "src_factor": "one",
                                "dst_factor": "one-minus-src-alpha",
                                "operation": "add",
                            },
                        },
                        "write_mask": wgpu.ColorWrite.ALL,
                    }
                ],
            },
        )

    def teardown(self) -> None:
        """
        Release all GPU resources held by this filter.

        Calls the base-class teardown then nulls all filter-specific refs
        including the per-region resource lists.
        """
        super().teardown()
        self._render_pipeline = None
        self._mesh_render_pipeline = None
        self._mesh_bind_group_layout = None
        self._vertex_buffer = None
        self._params_buffer = None
        self._params_bind_group = None
        self._region_index_buffers = []
        self._region_color_buffers = []
        self._region_bind_groups = []

    # ------------------------------------------------------------------
    # Per-frame rendering
    # ------------------------------------------------------------------

    def apply(
        self,
        encoder: wgpu.GPUCommandEncoder,
        input_texture: wgpu.GPUTexture,
        output_texture: wgpu.GPUTexture,
    ) -> None:
        """
        Record GPU commands to render the face mask for one frame.

        The output texture is always cleared to solid black first.  When
        a face is detected the landmark positions are converted to NDC
        and written to the shared vertex buffer; then each region mesh is
        drawn back-to-front using indexed draw calls, followed by white
        SDF-circle dots drawn instanced on top.

        ``input_texture`` is not sampled; the camera image is discarded
        (CON-FG-006).

        Parameters:
            encoder (wgpu.GPUCommandEncoder): Current frame encoder.
            input_texture (wgpu.GPUTexture): Ignored by this filter.
            output_texture (wgpu.GPUTexture): Target for mask output.
        """
        face_visible = self._has_visible_landmarks()

        # Upload landmark positions as NDC whenever a face is present.
        if face_visible:
            self._upload_vertex_data()

        # Single render pass — clear to black, then draw mask + dots.
        pass_enc = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": output_texture.create_view(),
                    "load_op": "clear",
                    "store_op": "store",
                    "clear_value": (0.0, 0.0, 0.0, 1.0),
                }
            ]
        )

        if face_visible:
            # -- Mesh pass: draw each region back-to-front ─────────────
            pass_enc.set_pipeline(self._mesh_render_pipeline)
            pass_enc.set_vertex_buffer(0, self._vertex_buffer)
            for i, region in enumerate(_REGION_DATA):
                pass_enc.set_bind_group(
                    0, self._region_bind_groups[i]
                )
                pass_enc.set_index_buffer(
                    self._region_index_buffers[i], "uint32"
                )
                pass_enc.draw_indexed(len(region.indices))

            # -- Dot pass: white SDF circles on top of meshes ──────────
            pass_enc.set_pipeline(self._render_pipeline)
            pass_enc.set_bind_group(0, self._params_bind_group)
            pass_enc.set_vertex_buffer(0, self._vertex_buffer)
            # 6 vertices per quad (2 triangles), 478 instances total.
            pass_enc.draw(6, NUM_LANDMARKS)

        pass_enc.end()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _has_visible_landmarks(self) -> bool:
        """
        Return True when the stored face result contains valid landmarks.

        Guards the vertex-upload and draw calls in :py:meth:`apply`.

        Returns:
            bool: True when a face is detected and landmarks present.
        """
        if self._face_result is None:
            return False
        if not self._face_result.face_detected:
            return False
        return bool(self._face_result.landmarks)

    def _upload_vertex_data(self) -> None:
        """
        Convert landmark positions to NDC and upload to the vertex buffer.

        NDC conversion::

            ndc_x = x * 2 - 1
            ndc_y = 1 - y * 2   (flip Y so MediaPipe top maps to NDC top)
            ndc_z = z * _Z_SCALE

        Uses at most ``NUM_LANDMARKS`` landmarks; pads with zeros when
        fewer are present (guard against partial results).
        """
        landmarks = self._face_result.landmarks[:NUM_LANDMARKS]
        n = len(landmarks)

        # Build float32 array: (ndc_x, ndc_y, ndc_z) per vertex.
        data = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
        if n > 0:
            xs = np.array([lm.x for lm in landmarks], dtype=np.float32)
            ys = np.array([lm.y for lm in landmarks], dtype=np.float32)
            zs = np.array([lm.z for lm in landmarks], dtype=np.float32)
            data[:n, 0] = xs * 2.0 - 1.0        # ndc_x
            data[:n, 1] = 1.0 - ys * 2.0        # ndc_y (Y-flip)
            data[:n, 2] = zs * _Z_SCALE          # ndc_z

        self._device.queue.write_buffer(
            self._vertex_buffer, 0, data.tobytes()
        )
