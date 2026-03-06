"""
Face Landmark Visualisation Filter.

Renders three visual layers on the live camera feed each frame:

1. **Landmark dots** — 478 white SDF circles at each MediaPipe keypoint
   (GPU instanced draw, alpha-blended).
2. **Head-pose arrows** — three coloured arrows drawn from the top-left
   corner of the frame showing yaw (blue), pitch (green), and roll (red)
   angle magnitudes and directions (CPU-drawn, uploaded per frame).
3. **Face-detected badge** — a green ✓ or red ✗ badge in the top-right
   corner, always visible regardless of face detection state (CPU-drawn,
   uploaded per frame).

GPU render pass order per frame
--------------------------------
1. **Blit pass**: ``input_texture`` → ``output_texture`` unchanged.
2. **Landmark pass** (only when face detected): instanced white dots,
   alpha-blended over output.
3. **Overlay pass** (always): CPU-drawn arrows + badge blended onto
   output from a pre-allocated RGBA overlay texture.

Coordinate conversion (MediaPipe → NDC):
    ndc_x = landmark.x * 2 - 1
    ndc_y = 1 - landmark.y * 2   (flip Y: MediaPipe y grows downward)

Dot style is fixed: white (1, 1, 1, 1), 3 px radius. No user-adjustable
parameters exist on this filter (REQ-LM-002).
"""

from __future__ import annotations

import math
import struct
from typing import Any, Optional

import cv2
import numpy as np
import wgpu

from filters.base import BaseFilter
from tracking.face_tracker import FaceTrackResult

# Maximum number of landmarks MediaPipe FaceLandmarker produces.
MAX_LANDMARKS: int = 478

# ---------------------------------------------------------------------------
# Visual constants (hard-coded, not user-adjustable — REQ-LM-002)
# ---------------------------------------------------------------------------

# Dot style
_DOT_RADIUS_PX: float = 3.0

# Arrow drawing — all arrows originate from the top-left corner
_ARROW_ORIGIN: tuple[int, int] = (50, 50)
_ARROW_MAX_LEN_PX: int = 80     # pixel length at 90° rotation
_ARROW_MAX_ANGLE: float = 90.0  # degrees that maps to max length
_ARROW_THICKNESS: int = 2

# Badge (face-detected indicator) — top-right corner
_BADGE_RADIUS_PX: int = 18
_BADGE_MARGIN_PX: int = 30

# OpenCV draws with BGRA channel order
_YAW_BGRA: tuple[int, int, int, int] = (255, 0, 0, 255)    # blue
_PITCH_BGRA: tuple[int, int, int, int] = (0, 255, 0, 255)  # green
_ROLL_BGRA: tuple[int, int, int, int] = (0, 0, 255, 255)   # red
_GREEN_BADGE_BGRA: tuple[int, int, int, int] = (0, 180, 0, 255)
_RED_BADGE_BGRA: tuple[int, int, int, int] = (0, 0, 200, 255)
_WHITE_BGRA: tuple[int, int, int, int] = (255, 255, 255, 255)

# ---------------------------------------------------------------------------
# WGSL — Blit pass (full-screen triangle, pass-through texture copy)
# ---------------------------------------------------------------------------
_WGSL_BLIT = """
struct BlitVertexOutput {
    @builtin(position) clip_pos : vec4f,
    @location(0)       uv       : vec2f,
}

// Full-screen triangle covering NDC [-1, 1] space; no vertex buffer.
@vertex
fn vs_blit(@builtin(vertex_index) vi : u32) -> BlitVertexOutput {
    var pos = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0),
    );
    // UV (0,1) = bottom-left → matches wgpu texture origin (top-left)
    var uv = array<vec2f, 3>(
        vec2f(0.0, 1.0),
        vec2f(2.0, 1.0),
        vec2f(0.0, -1.0),
    );
    var out: BlitVertexOutput;
    out.clip_pos = vec4f(pos[vi], 0.0, 1.0);
    out.uv       = uv[vi];
    return out;
}

@group(0) @binding(0) var blit_tex : texture_2d<f32>;
@group(0) @binding(1) var blit_smp : sampler;

@fragment
fn fs_blit(in: BlitVertexOutput) -> @location(0) vec4f {
    return textureSample(blit_tex, blit_smp, in.uv);
}
"""

# ---------------------------------------------------------------------------
# WGSL — Landmark overlay pass (instanced, alpha-blended circular quads)
# ---------------------------------------------------------------------------
_WGSL_LANDMARK = """
// Per-frame colour and aspect-corrected radius in NDC units.
struct LandmarkParams {
    dot_r    : f32,
    dot_g    : f32,
    dot_b    : f32,
    dot_a    : f32,
    radius_x : f32,   // NDC half-width  (2 * pixel_radius / frame_width)
    radius_y : f32,   // NDC half-height (2 * pixel_radius / frame_height)
    _pad0    : f32,
    _pad1    : f32,
}

@group(0) @binding(0) var<uniform> lm_params : LandmarkParams;

struct LandmarkVertexOutput {
    @builtin(position) clip_pos  : vec4f,
    @location(0)       local_uv  : vec2f,  // unit square [-1, 1] per quad
}

// Per-instance position comes from the VERTEX buffer (step_mode=instance).
// @location(0) maps to the first (and only) vertex attribute slot.
@vertex
fn vs_landmark(
    @builtin(vertex_index)    vi : u32,
    @location(0) lm_ndc : vec2f,   // landmark centre in NDC
) -> LandmarkVertexOutput {
    // Six vertices forming two triangles (one quad per landmark instance).
    var offsets = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f(-1.0,  1.0),
        vec2f(-1.0,  1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0),
    );
    let off = offsets[vi];
    let pos = lm_ndc + vec2f(
        off.x * lm_params.radius_x,
        off.y * lm_params.radius_y,
    );

    var out: LandmarkVertexOutput;
    out.clip_pos = vec4f(pos, 0.0, 1.0);
    out.local_uv = off;  // passed to fragment for SDF circle
    return out;
}

@fragment
fn fs_landmark(in: LandmarkVertexOutput) -> @location(0) vec4f {
    // Signed-distance-field circle: discard corners outside unit circle.
    let d = length(in.local_uv);
    if d > 1.0 { discard; }

    // Smooth anti-aliased edge in the outer 30 % of the radius.
    let alpha = lm_params.dot_a * (1.0 - smoothstep(0.7, 1.0, d));
    return vec4f(lm_params.dot_r, lm_params.dot_g, lm_params.dot_b, alpha);
}
"""


class FaceLandmarkFilter(BaseFilter):
    """
    GPU filter that draws face landmarks, head-pose arrows, and a badge.

    Three render passes per :py:meth:`apply` call:

    1. **Blit pass**: copies ``input_texture`` to ``output_texture``.
    2. **Landmark pass** (face detected only): 478 instanced white SDF
       circles, alpha-blended onto the output.
    3. **Overlay pass** (always): CPU-drawn arrows and face-detected
       badge alpha-blended onto the output.

    All visual properties (dot colour, dot size, arrow colours) are
    fixed constants.  The filter has no user-adjustable parameters
    (REQ-LM-002).

    Face tracking data is supplied each frame via
    :py:meth:`update_face_result` before :py:meth:`apply` is called.
    """

    def __init__(self) -> None:
        """Initialise the filter.  All visual constants are hard-coded."""
        super().__init__()

        # Latest face tracking result; injected each frame before apply().
        self._face_result: Optional[FaceTrackResult] = None

        # GPU pipeline resources — created in _build_pipeline.
        self._blit_pipeline: Any = None
        self._blit_bgl: Any = None
        self._landmark_pipeline: Any = None
        self._landmark_bgl: Any = None
        self._landmark_param_buffer: Any = None
        self._landmark_vertex_buffer: Any = None

        # Overlay (arrows + badge) pipeline resources
        self._overlay_pipeline: Any = None
        self._overlay_bgl: Any = None
        # Overlay texture is lazy-initialised on the first apply() call
        # because texture dimensions are only known then.
        self._overlay_texture: Any = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """
        Unique filter name.

        Returns:
            str: 'Face Landmarks'
        """
        return "Face Landmarks"

    # ------------------------------------------------------------------
    # Face tracking data injection
    # ------------------------------------------------------------------

    def update_face_result(
        self, result: Optional[FaceTrackResult]
    ) -> None:
        """
        Store the latest face tracking result for use in the next frame.

        Called by the rendering pipeline each frame before
        :py:meth:`apply`.  When ``result`` is ``None`` or
        ``face_detected`` is ``False``, the landmark and arrow passes
        are skipped and the badge shows a red cross.

        Parameters:
            result (Optional[FaceTrackResult]): Latest face tracking
                output, or ``None`` when no inference was performed.
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
        Compile all WGSL shaders and pre-allocate fixed GPU resources.

        Creates:
        * Blit render pipeline (pass-through, no blending).
        * Landmark render pipeline (instanced SDF circles, alpha blend).
        * Uniform buffer for LandmarkParams (32 bytes).
        * Vertex buffer for MAX_LANDMARKS NDC positions (478 × 8 bytes).
        * Overlay render pipeline (alpha-blended full-screen blit).

        The overlay texture is not allocated here because texture
        dimensions are not yet available; it is lazy-initialised on the
        first :py:meth:`apply` call.

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            texture_format (wgpu.TextureFormat): Target texture format.
        """
        self._build_blit_pipeline(device, texture_format)
        self._build_landmark_pipeline(device, texture_format)
        self._build_overlay_pipeline(device, texture_format)

    def _build_overlay_pipeline(
        self,
        device: wgpu.GPUDevice,
        texture_format: wgpu.TextureFormat,
    ) -> None:
        """
        Build the alpha-blended overlay blit pipeline.

        Reuses the blit WGSL but enables alpha blending so the
        CPU-drawn overlay (arrows and badge) composites correctly onto
        the output texture.

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            texture_format (wgpu.TextureFormat): Target texture format.
        """
        shader = device.create_shader_module(code=_WGSL_BLIT)

        bgl = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": "float",
                        "view_dimension": "2d",
                        "multisampled": False,
                    },
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": "filtering"},
                },
            ]
        )
        self._overlay_bgl = bgl

        # Alpha blending: src*src_alpha + dst*(1-src_alpha)
        blend_state = {
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
        }

        layout = device.create_pipeline_layout(
            bind_group_layouts=[bgl]
        )
        self._overlay_pipeline = device.create_render_pipeline(
            layout=layout,
            vertex={
                "module": shader,
                "entry_point": "vs_blit",
                "buffers": [],
            },
            fragment={
                "module": shader,
                "entry_point": "fs_blit",
                "targets": [
                    {"format": texture_format, "blend": blend_state}
                ],
            },
            primitive={"topology": "triangle-list"},
            depth_stencil=None,
            multisample=None,
        )

    def _build_blit_pipeline(
        self,
        device: wgpu.GPUDevice,
        texture_format: wgpu.TextureFormat,
    ) -> None:
        """
        Compile and store the pass-through blit render pipeline.

        Reads from an input texture and writes pixel values unchanged
        to the output texture.

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            texture_format (wgpu.TextureFormat): Target texture format.
        """
        shader = device.create_shader_module(code=_WGSL_BLIT)

        bgl = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": "float",
                        "view_dimension": "2d",
                        "multisampled": False,
                    },
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": "filtering"},
                },
            ]
        )
        self._blit_bgl = bgl

        layout = device.create_pipeline_layout(
            bind_group_layouts=[bgl]
        )
        self._blit_pipeline = device.create_render_pipeline(
            layout=layout,
            vertex={
                "module": shader,
                "entry_point": "vs_blit",
                "buffers": [],
            },
            fragment={
                "module": shader,
                "entry_point": "fs_blit",
                "targets": [{"format": texture_format}],
            },
            primitive={"topology": "triangle-list"},
            depth_stencil=None,
            multisample=None,
        )

    def _build_landmark_pipeline(
        self,
        device: wgpu.GPUDevice,
        texture_format: wgpu.TextureFormat,
    ) -> None:
        """
        Compile and store the instanced landmark dot render pipeline.

        Also pre-allocates:
        * A 32-byte uniform buffer for ``LandmarkParams``.
        * A vertex buffer sized for ``MAX_LANDMARKS`` NDC positions.

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            texture_format (wgpu.TextureFormat): Target texture format.
        """
        # Uniform buffer: 8 × f32 = 32 bytes.
        self._landmark_param_buffer = device.create_buffer(
            size=32,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        # Vertex buffer: MAX_LANDMARKS × vec2f (8 bytes each).
        self._landmark_vertex_buffer = device.create_buffer(
            size=MAX_LANDMARKS * 8,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
        )

        shader = device.create_shader_module(code=_WGSL_LANDMARK)

        bgl = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX
                    | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "uniform"},
                },
            ]
        )
        self._landmark_bgl = bgl

        layout = device.create_pipeline_layout(
            bind_group_layouts=[bgl]
        )

        # Alpha blending: src*src_alpha + dst*(1-src_alpha)
        blend_state = {
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
        }

        self._landmark_pipeline = device.create_render_pipeline(
            layout=layout,
            vertex={
                "module": shader,
                "entry_point": "vs_landmark",
                "buffers": [
                    {
                        # 8 bytes per instance: two f32 (x, y NDC)
                        "array_stride": 8,
                        "step_mode": "instance",
                        "attributes": [
                            {
                                "format": "float32x2",
                                "offset": 0,
                                "shader_location": 0,
                            }
                        ],
                    }
                ],
            },
            fragment={
                "module": shader,
                "entry_point": "fs_landmark",
                "targets": [
                    {"format": texture_format, "blend": blend_state}
                ],
            },
            primitive={"topology": "triangle-list"},
            depth_stencil=None,
            multisample=None,
        )

    def teardown(self) -> None:
        """Release all GPU buffer and pipeline references."""
        self._blit_pipeline = None
        self._blit_bgl = None
        self._landmark_pipeline = None
        self._landmark_bgl = None
        self._landmark_param_buffer = None
        self._landmark_vertex_buffer = None
        self._overlay_pipeline = None
        self._overlay_bgl = None
        self._overlay_texture = None
        super().teardown()

    # ------------------------------------------------------------------
    # Per-frame application
    # ------------------------------------------------------------------

    def apply(
        self,
        encoder: wgpu.GPUCommandEncoder,
        input_texture: wgpu.GPUTexture,
        output_texture: wgpu.GPUTexture,
    ) -> None:
        """
        Record GPU commands for one frame.

        Executes up to three render passes:

        1. **Blit pass**: copies ``input_texture`` to ``output_texture``.
        2. **Landmark pass** (face detected only): 478 instanced white
           SDF circles, alpha-blended over the output.
        3. **Overlay pass** (always): CPU-drawn arrows and badge
           composited over the output.

        Parameters:
            encoder (wgpu.GPUCommandEncoder): Current frame command
                encoder.
            input_texture (wgpu.GPUTexture): Source camera texture.
            output_texture (wgpu.GPUTexture): Destination texture.
        """
        device = self._device
        assert device is not None, "apply() called before setup()"

        w, h = input_texture.size[0], input_texture.size[1]

        # Pass 1: blit input → output
        self._record_blit_pass(encoder, input_texture, output_texture)

        # Pass 2: landmark dots (only when a face is present)
        if self._has_visible_landmarks():
            landmarks = (
                self._face_result.landmarks  # type: ignore[union-attr]
            )
            num = len(landmarks)
            self._upload_landmark_positions(device, landmarks)
            self._upload_landmark_params(device, input_texture)
            self._record_landmark_pass(
                encoder, output_texture, device, num
            )

        # Pass 3: overlay (arrows + badge — badge always visible)
        self._ensure_overlay_texture(device, w, h)
        self._update_overlay_texture(device, w, h)
        self._record_overlay_pass(encoder, output_texture, device)

    # ------------------------------------------------------------------
    # Private helpers — render passes
    # ------------------------------------------------------------------

    def _has_visible_landmarks(self) -> bool:
        """
        Return True when the stored result contains visible landmarks.

        Returns:
            bool: True if landmark dots should be drawn this frame.
        """
        return (
            self._face_result is not None
            and self._face_result.face_detected
            and bool(self._face_result.landmarks)
        )

    def _record_blit_pass(
        self,
        encoder: wgpu.GPUCommandEncoder,
        input_texture: wgpu.GPUTexture,
        output_texture: wgpu.GPUTexture,
    ) -> None:
        """
        Record a full-screen blit from ``input_texture`` to
        ``output_texture``.

        Parameters:
            encoder (wgpu.GPUCommandEncoder): Current command encoder.
            input_texture (wgpu.GPUTexture): Texture to copy from.
            output_texture (wgpu.GPUTexture): Texture to write into.
        """
        device = self._device
        bind_group = device.create_bind_group(
            layout=self._blit_bgl,
            entries=[
                {
                    "binding": 0,
                    "resource": input_texture.create_view(),
                },
                {"binding": 1, "resource": self._sampler},
            ],
        )
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
        pass_enc.set_pipeline(self._blit_pipeline)
        pass_enc.set_bind_group(0, bind_group, [], 0, 0)
        pass_enc.draw(3, 1, 0, 0)  # full-screen triangle
        pass_enc.end()

    def _upload_landmark_positions(
        self,
        device: wgpu.GPUDevice,
        landmarks: list,
    ) -> None:
        """
        Convert landmarks to NDC and upload to the vertex buffer.

        Coordinate conversion (section 6.4 of spec):
            ndc_x = landmark.x * 2 - 1
            ndc_y = 1 - landmark.y * 2   (flip Y axis)

        Only the first ``MAX_LANDMARKS`` entries are written.

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            landmarks (list): List of :py:class:`Landmark` instances.
        """
        count = min(len(landmarks), MAX_LANDMARKS)
        positions: list[float] = []
        for lm in landmarks[:count]:
            positions.append(lm.x * 2.0 - 1.0)   # ndc_x
            positions.append(1.0 - lm.y * 2.0)    # ndc_y (flip)

        data = struct.pack(f"{count * 2}f", *positions)
        device.queue.write_buffer(
            self._landmark_vertex_buffer, 0, data
        )

    def _upload_landmark_params(
        self,
        device: wgpu.GPUDevice,
        input_texture: wgpu.GPUTexture,
    ) -> None:
        """
        Upload the LandmarkParams uniform buffer using fixed white values.

        NDC radius is aspect-ratio corrected from the texture dimensions:
            radius_x = 2 * DOT_RADIUS_PX / texture_width
            radius_y = 2 * DOT_RADIUS_PX / texture_height

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            input_texture (wgpu.GPUTexture): Provides frame dimensions.
        """
        tex_w, tex_h = input_texture.size[0], input_texture.size[1]
        radius_x = 2.0 * _DOT_RADIUS_PX / tex_w
        radius_y = 2.0 * _DOT_RADIUS_PX / tex_h

        data = struct.pack(
            "ffffffff",
            1.0, 1.0, 1.0, 1.0,   # white (dot_r, dot_g, dot_b, dot_a)
            radius_x,
            radius_y,
            0.0,   # _pad0
            0.0,   # _pad1
        )
        device.queue.write_buffer(
            self._landmark_param_buffer, 0, data
        )

    def _record_landmark_pass(
        self,
        encoder: wgpu.GPUCommandEncoder,
        output_texture: wgpu.GPUTexture,
        device: wgpu.GPUDevice,
        num_landmarks: int,
    ) -> None:
        """
        Record the instanced landmark circle draw pass.

        Uses ``load_op="load"`` to preserve the blit result and
        composites dots with alpha blending.

        Parameters:
            encoder (wgpu.GPUCommandEncoder): Current command encoder.
            output_texture (wgpu.GPUTexture): Render target.
            device (wgpu.GPUDevice): The active WebGPU device.
            num_landmarks (int): Number of instances to draw.
        """
        bind_group = device.create_bind_group(
            layout=self._landmark_bgl,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self._landmark_param_buffer,
                        "offset": 0,
                        "size": 32,
                    },
                }
            ],
        )
        pass_enc = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": output_texture.create_view(),
                    "load_op": "load",   # preserve blit result
                    "store_op": "store",
                }
            ]
        )
        pass_enc.set_pipeline(self._landmark_pipeline)
        pass_enc.set_bind_group(0, bind_group, [], 0, 0)
        pass_enc.set_vertex_buffer(0, self._landmark_vertex_buffer)
        # 6 vertices per quad (2 triangles), one instance per landmark
        pass_enc.draw(6, num_landmarks, 0, 0)
        pass_enc.end()

    # ------------------------------------------------------------------
    # Private helpers — overlay (arrows + badge)
    # ------------------------------------------------------------------

    def _ensure_overlay_texture(
        self,
        device: wgpu.GPUDevice,
        width: int,
        height: int,
    ) -> None:
        """
        Lazy-initialise the overlay RGBA texture on the first apply call.

        Not allocated in ``_build_pipeline`` because texture dimensions
        are not available at that stage.  After the first call the
        texture is reused every frame without reallocation.

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            width (int): Frame width in pixels.
            height (int): Frame height in pixels.
        """
        if self._overlay_texture is not None:
            return
        self._overlay_texture = device.create_texture(
            size=(width, height, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            usage=(
                wgpu.TextureUsage.TEXTURE_BINDING
                | wgpu.TextureUsage.COPY_DST
            ),
        )

    def _draw_overlay(
        self,
        face_result: Optional[FaceTrackResult],
        width: int,
        height: int,
    ) -> np.ndarray:
        """
        Draw arrows and the face-detected badge into an RGBA numpy array.

        The canvas is transparent everywhere except for the drawn
        elements.  Arrows are omitted when no face is detected; the
        badge is always present.

        Arrow layout (all from ``_ARROW_ORIGIN`` at the top-left corner):

        * **Blue** (yaw)   — horizontal; positive yaw → right.
        * **Green** (pitch) — vertical; positive pitch → downward.
        * **Red** (roll)   — direction of the face's ``up`` axis after
          rotation; positive roll tilts clockwise → arrow tilts right.

        Badge layout (top-right corner):

        * Green circle + white ✓ when a face is detected.
        * Red circle + white ✗ when no face is detected.

        Parameters:
            face_result (Optional[FaceTrackResult]): Latest tracking
                result, or ``None`` if tracking has not run.
            width (int): Frame width in pixels.
            height (int): Frame height in pixels.

        Returns:
            np.ndarray: RGBA uint8 array of shape
            ``(height, width, 4)``.
        """
        face_detected = (
            face_result is not None and face_result.face_detected
        )

        # Transparent BGRA canvas; cv2 uses BGRA channel order.
        canvas = np.zeros((height, width, 4), dtype=np.uint8)

        # -- Arrows (only when face detected) -------------------------
        if face_detected and face_result is not None:
            hp = face_result.head_pose
            ox, oy = _ARROW_ORIGIN

            # Clamp each angle to keep arrows within the frame.
            yaw = max(
                -_ARROW_MAX_ANGLE, min(_ARROW_MAX_ANGLE, hp.yaw)
            )
            pitch = max(
                -_ARROW_MAX_ANGLE, min(_ARROW_MAX_ANGLE, hp.pitch)
            )
            roll = max(
                -_ARROW_MAX_ANGLE, min(_ARROW_MAX_ANGLE, hp.roll)
            )

            # Yaw — horizontal (positive yaw → rightward arrow)
            yaw_end_x = ox + int(
                yaw / _ARROW_MAX_ANGLE * _ARROW_MAX_LEN_PX
            )
            if yaw_end_x != ox:
                cv2.arrowedLine(
                    canvas, (ox, oy), (yaw_end_x, oy),
                    _YAW_BGRA, _ARROW_THICKNESS, tipLength=0.3,
                )

            # Pitch — vertical (positive pitch → downward arrow)
            pitch_end_y = oy + int(
                pitch / _ARROW_MAX_ANGLE * _ARROW_MAX_LEN_PX
            )
            if pitch_end_y != oy:
                cv2.arrowedLine(
                    canvas, (ox, oy), (ox, pitch_end_y),
                    _PITCH_BGRA, _ARROW_THICKNESS, tipLength=0.3,
                )

            # Roll — shows direction the top-of-head faces.
            # At roll=0 the arrow points straight up (-y in screen
            # coordinates).  Positive roll tilts it clockwise.
            roll_rad = math.radians(roll)
            roll_len = int(
                abs(roll) / _ARROW_MAX_ANGLE * _ARROW_MAX_LEN_PX
            )
            if roll_len > 0:
                roll_dx = int(math.sin(roll_rad) * roll_len)
                # Negative because screen y increases downward.
                roll_dy = int(-math.cos(roll_rad) * roll_len)
                cv2.arrowedLine(
                    canvas, (ox, oy),
                    (ox + roll_dx, oy + roll_dy),
                    _ROLL_BGRA, _ARROW_THICKNESS, tipLength=0.3,
                )

        # -- Badge (always visible) -----------------------------------
        bx = width - _BADGE_MARGIN_PX
        by = _BADGE_MARGIN_PX
        badge_colour = (
            _GREEN_BADGE_BGRA if face_detected else _RED_BADGE_BGRA
        )
        cv2.circle(canvas, (bx, by), _BADGE_RADIUS_PX, badge_colour, -1)

        if face_detected:
            # Check mark ✓: two line segments
            cv2.line(canvas, (bx - 7, by), (bx - 2, by + 6),
                     _WHITE_BGRA, 2)
            cv2.line(canvas, (bx - 2, by + 6), (bx + 8, by - 8),
                     _WHITE_BGRA, 2)
        else:
            # Cross ✗: two diagonal lines
            cv2.line(canvas, (bx - 7, by - 7), (bx + 7, by + 7),
                     _WHITE_BGRA, 2)
            cv2.line(canvas, (bx + 7, by - 7), (bx - 7, by + 7),
                     _WHITE_BGRA, 2)

        # Convert BGRA → RGBA for GPU upload.
        return cv2.cvtColor(canvas, cv2.COLOR_BGRA2RGBA)

    def _update_overlay_texture(
        self,
        device: wgpu.GPUDevice,
        width: int,
        height: int,
    ) -> None:
        """
        Draw the current frame's overlay and upload it to the GPU.

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            width (int): Frame width in pixels.
            height (int): Frame height in pixels.
        """
        rgba = self._draw_overlay(self._face_result, width, height)
        rgba_c = np.ascontiguousarray(rgba)
        device.queue.write_texture(
            {
                "texture": self._overlay_texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            rgba_c.tobytes(),
            {
                "offset": 0,
                "bytes_per_row": width * 4,
                "rows_per_image": height,
            },
            (width, height, 1),
        )

    def _record_overlay_pass(
        self,
        encoder: wgpu.GPUCommandEncoder,
        output_texture: wgpu.GPUTexture,
        device: wgpu.GPUDevice,
    ) -> None:
        """
        Record the alpha-blended overlay blit onto ``output_texture``.

        Uses ``load_op="load"`` so the existing output content (blit +
        optional landmark dots) is preserved underneath the overlay.

        Parameters:
            encoder (wgpu.GPUCommandEncoder): Current command encoder.
            output_texture (wgpu.GPUTexture): Render target.
            device (wgpu.GPUDevice): The active WebGPU device.
        """
        bind_group = device.create_bind_group(
            layout=self._overlay_bgl,
            entries=[
                {
                    "binding": 0,
                    "resource": self._overlay_texture.create_view(),
                },
                {"binding": 1, "resource": self._sampler},
            ],
        )
        pass_enc = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": output_texture.create_view(),
                    "load_op": "load",   # preserve blit + dots
                    "store_op": "store",
                }
            ]
        )
        pass_enc.set_pipeline(self._overlay_pipeline)
        pass_enc.set_bind_group(0, bind_group, [], 0, 0)
        pass_enc.draw(3, 1, 0, 0)  # full-screen triangle
        pass_enc.end()
