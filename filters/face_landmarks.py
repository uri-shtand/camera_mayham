"""
Face Landmark Visualisation Filter.

Renders the 478 MediaPipe face landmarks as coloured dots overlaid on
the live camera feed using two WebGPU render passes per frame:

1. **Blit pass** — copies ``input_texture`` to ``output_texture`` using
   a full-screen triangle (pass-through).
2. **Landmark overlay pass** — draws instanced circular quads at each
   detected landmark position with alpha blending, preserving the blit
   result underneath (``load_op="load"``).

Coordinate conversion (MediaPipe → NDC):
    ndc_x = landmark.x * 2 - 1          (x: 0=left,  1=right)
    ndc_y = 1 - landmark.y * 2          (y: 0=top,   1=bottom → flip)

Face tracking data is injected each frame via
:py:meth:`update_face_result` before :py:meth:`apply` is called by the
render pipeline.  The filter holds no reference to ``AppState`` or
``FaceTracker`` directly (CON-LM-005).

Design decisions
----------------
* Two separate pipelines (blit + landmark) keep each WGSL shader small
  and allow the landmark pipeline to enable alpha blending independently.
* A pre-allocated vertex buffer (``MAX_LANDMARKS × 8`` bytes) holds one
  ``vec2f`` NDC position per landmark instance (CON-LM-002).
* Dot dimensions are aspect-ratio corrected at apply time from the
  input texture dimensions to avoid elliptical dots (GUD-LM-004).
"""

from __future__ import annotations

import struct
from typing import Any, Dict, Optional

import wgpu

from filters.base import BaseFilter
from tracking.face_tracker import FaceTrackResult

# Maximum number of landmarks MediaPipe FaceLandmarker produces.
MAX_LANDMARKS: int = 478

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
    GPU filter that draws MediaPipe face landmarks on the camera frame.

    Uses two render passes per :py:meth:`apply` call:

    1. **Blit pass**: copies ``input_texture`` to ``output_texture``.
    2. **Landmark pass**: draws instanced circles on top with alpha
       blending.

    Face tracking data must be supplied each frame via
    :py:meth:`update_face_result` before ``apply`` is called.

    Parameters
    ----------
    ``dot_radius`` : float [1.0 – 20.0]
        Dot radius in pixels.  Defaults to ``3.0``.
    ``dot_r`` : float [0.0 – 1.0]
        Red channel of dot colour.  Defaults to ``0.0``.
    ``dot_g`` : float [0.0 – 1.0]
        Green channel of dot colour.  Defaults to ``1.0`` (bright green).
    ``dot_b`` : float [0.0 – 1.0]
        Blue channel of dot colour.  Defaults to ``0.0``.
    ``dot_a`` : float [0.0 – 1.0]
        Dot opacity.  Defaults to ``1.0`` (fully opaque).
    """

    def __init__(self) -> None:
        """Initialise the face landmark filter with default parameters."""
        super().__init__()
        # Runtime-adjustable parameters exposed to the widget panel.
        # Default: bright green, 3-pixel radius.
        self.params: Dict[str, Any] = {
            "dot_radius": 3.0,
            "dot_r":      0.0,
            "dot_g":      1.0,
            "dot_b":      0.0,
            "dot_a":      1.0,
        }

        # Latest face tracking result; updated each frame via
        # update_face_result() before apply() is called.
        self._face_result: Optional[FaceTrackResult] = None

        # GPU resources — created in _build_pipeline, released in teardown.
        self._blit_pipeline: Any = None
        self._blit_bgl: Any = None
        self._landmark_pipeline: Any = None
        self._landmark_bgl: Any = None
        self._landmark_param_buffer: Any = None
        self._landmark_vertex_buffer: Any = None

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
        :py:meth:`apply`.  Passing ``None`` or a result whose
        ``face_detected`` flag is ``False`` causes only the blit pass
        to run in ``apply``, leaving the frame unmodified.

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
        Compile both WGSL shaders and pre-allocate all GPU resources.

        Creates:
        * Blit render pipeline (full-screen triangle, no blending).
        * Landmark render pipeline (instanced quads, alpha blending).
        * Uniform buffer for ``LandmarkParams`` (32 bytes).
        * Vertex buffer pre-allocated for ``MAX_LANDMARKS`` NDC
          positions (``MAX_LANDMARKS × 8`` bytes).

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            texture_format (wgpu.TextureFormat): Target texture format.
        """
        self._build_blit_pipeline(device, texture_format)
        self._build_landmark_pipeline(device, texture_format)

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
                "dst_factor": "zero",
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

        Executes two render passes:

        1. Blit pass: copies ``input_texture`` to ``output_texture``
           using a full-screen triangle.
        2. Landmark overlay pass (only when a face is detected): draws
           instanced circular dots at each landmark position with alpha
           blending, loading and preserving the blit result.

        Landmark NDC positions are uploaded to the pre-allocated vertex
        buffer before drawing.  Dot colour and NDC radius are uploaded
        to the uniform buffer.

        Parameters:
            encoder (wgpu.GPUCommandEncoder): Current frame command
                encoder.
            input_texture (wgpu.GPUTexture): Source camera texture.
            output_texture (wgpu.GPUTexture): Destination texture.
        """
        device = self._device
        assert device is not None, "apply() called before setup()"

        # --- Pass 1: Blit input → output ----------------------------
        self._record_blit_pass(encoder, input_texture, output_texture)

        # --- Pass 2: Landmark overlay (only if face detected) --------
        if not self._has_visible_landmarks():
            return

        landmarks = self._face_result.landmarks  # type: ignore[union-attr]
        num_landmarks = len(landmarks)

        self._upload_landmark_positions(device, landmarks)
        self._upload_landmark_params(device, input_texture, num_landmarks)
        self._record_landmark_pass(
            encoder, output_texture, device, num_landmarks
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _has_visible_landmarks(self) -> bool:
        """
        Return True when the stored face result contains visible landmarks.

        Returns:
            bool: True if landmarks should be drawn this frame.
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
        Record a full-screen blit from input_texture to output_texture.

        Parameters:
            encoder (wgpu.GPUCommandEncoder): Current command encoder.
            input_texture (wgpu.GPUTexture): Texture to copy from.
            output_texture (wgpu.GPUTexture): Texture to write into.
        """
        device = self._device
        bind_group = device.create_bind_group(
            layout=self._blit_bgl,
            entries=[
                {"binding": 0, "resource": input_texture.create_view()},
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
        pass_enc.draw(3, 1, 0, 0)  # Full-screen triangle
        pass_enc.end()

    def _upload_landmark_positions(
        self,
        device: wgpu.GPUDevice,
        landmarks: list,
    ) -> None:
        """
        Convert landmarks to NDC and upload to the vertex buffer.

        Coordinate conversion (GUD-LM-003):
            ndc_x = landmark.x * 2 - 1
            ndc_y = 1 - landmark.y * 2   (flip Y axis)

        Only landmarks within the pre-allocated buffer capacity
        (``MAX_LANDMARKS``) are written.

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            landmarks (list): List of :py:class:`Landmark` instances.
        """
        count = min(len(landmarks), MAX_LANDMARKS)
        positions: list[float] = []
        for lm in landmarks[:count]:
            positions.append(lm.x * 2.0 - 1.0)       # ndc_x
            positions.append(1.0 - lm.y * 2.0)        # ndc_y (flip)

        data = struct.pack(f"{count * 2}f", *positions)
        device.queue.write_buffer(self._landmark_vertex_buffer, 0, data)

    def _upload_landmark_params(
        self,
        device: wgpu.GPUDevice,
        input_texture: wgpu.GPUTexture,
        num_landmarks: int,
    ) -> None:
        """
        Build and upload the LandmarkParams uniform buffer.

        NDC radius is computed from the input texture dimensions to
        produce aspect-ratio-correct circular dots (GUD-LM-004):
            radius_x = 2 * dot_radius / texture_width
            radius_y = 2 * dot_radius / texture_height

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            input_texture (wgpu.GPUTexture): Used to obtain frame
                dimensions for aspect-ratio correction.
            num_landmarks (int): Number of landmarks to render (unused
                here but kept for call-site symmetry).
        """
        tex_w, tex_h = input_texture.size[0], input_texture.size[1]
        px_radius = float(self.params["dot_radius"])
        radius_x = 2.0 * px_radius / tex_w
        radius_y = 2.0 * px_radius / tex_h

        data = struct.pack(
            "ffffffff",
            float(self.params["dot_r"]),
            float(self.params["dot_g"]),
            float(self.params["dot_b"]),
            float(self.params["dot_a"]),
            radius_x,
            radius_y,
            0.0,   # _pad0
            0.0,   # _pad1
        )
        device.queue.write_buffer(self._landmark_param_buffer, 0, data)

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
        composites the dots with alpha blending.

        Parameters:
            encoder (wgpu.GPUCommandEncoder): Current command encoder.
            output_texture (wgpu.GPUTexture): Render target (preserving
                blit output underneath the dots).
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
        pass_enc.set_vertex_buffer(
            0, self._landmark_vertex_buffer
        )
        # 6 vertices per quad, one instance per landmark
        pass_enc.draw(6, num_landmarks, 0, 0)
        pass_enc.end()
