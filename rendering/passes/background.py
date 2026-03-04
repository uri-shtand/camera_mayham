"""
Background render pass.

Uploads the raw camera frame (BGR uint8 NumPy array) to a GPU texture
and renders it as a full-screen quad.  This is the first pass in the
pipeline; subsequent filter passes read from the texture this pass
writes to.

Design note: the texture is pre-allocated at pipeline setup time at the
camera resolution.  Each frame only calls write_texture (a DMA-style
upload from a staging buffer) — no texture reallocation occurs
(CON-002).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import wgpu

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WGSL shader — full-screen textured quad (no vertex buffer)
# ---------------------------------------------------------------------------
_WGSL = """
struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0)       uv       : vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> VertexOutput {
    var pos = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0),
    );
    var uv = array<vec2f, 3>(
        vec2f(0.0, 1.0),
        vec2f(2.0, 1.0),
        vec2f(0.0, -1.0),
    );
    var out: VertexOutput;
    out.position = vec4f(pos[vi], 0.0, 1.0);
    out.uv       = uv[vi];
    return out;
}

@group(0) @binding(0) var tex : texture_2d<f32>;
@group(0) @binding(1) var smp : sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    return textureSample(tex, smp, in.uv);
}
"""


class BackgroundPass:
    """
    Uploads camera frames to the GPU and renders a full-screen quad
    (REQ-001, §4.1 Background pass).

    The pass owns a camera-resolution RGBA8 texture (``frame_texture``).
    After :py:meth:`record`, use ``frame_texture`` as the input to the
    first filter in the filter chain.
    """

    def __init__(self) -> None:
        """Initialise the background pass descriptor."""
        self._device: Optional[wgpu.GPUDevice] = None
        self._pipeline: Optional[wgpu.GPURenderPipeline] = None
        self._sampler: Optional[wgpu.GPUSampler] = None
        self._bind_group_layout: Optional[wgpu.GPUBindGroupLayout] = None
        self.frame_texture: Optional[wgpu.GPUTexture] = None
        self._width: int = 0
        self._height: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(
        self,
        device: wgpu.GPUDevice,
        width: int,
        height: int,
        texture_format: wgpu.TextureFormat,
    ) -> None:
        """
        Allocate the camera frame texture and compile the blit shader.

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            width (int): Camera frame width in pixels.
            height (int): Camera frame height in pixels.
            texture_format (wgpu.TextureFormat): Render target format.
        """
        self._device = device
        self._width = width
        self._height = height

        # Camera frame texture — RGBA8 for colour upload + sampling
        self.frame_texture = device.create_texture(
            size=(width, height, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            usage=(
                wgpu.TextureUsage.TEXTURE_BINDING
                | wgpu.TextureUsage.COPY_DST
                | wgpu.TextureUsage.RENDER_ATTACHMENT
            ),
        )

        self._sampler = device.create_sampler(
            address_mode_u="clamp-to-edge",
            address_mode_v="clamp-to-edge",
            mag_filter="linear",
            min_filter="linear",
        )

        shader = device.create_shader_module(code=_WGSL)

        self._bind_group_layout = device.create_bind_group_layout(
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

        pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[self._bind_group_layout]
        )
        self._pipeline = device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": shader,
                "entry_point": "vs_main",
                "buffers": [],
            },
            fragment={
                "module": shader,
                "entry_point": "fs_main",
                "targets": [{"format": texture_format}],
            },
            primitive={"topology": "triangle-list"},
            depth_stencil=None,
            multisample=None,
        )
        logger.debug(
            "BackgroundPass ready (%dx%d %s)", width, height, texture_format
        )

    def teardown(self) -> None:
        """Release GPU resources."""
        self.frame_texture = None
        self._pipeline = None
        self._sampler = None
        self._bind_group_layout = None
        self._device = None

    # ------------------------------------------------------------------
    # Per-frame record
    # ------------------------------------------------------------------

    def upload_frame(self, bgr_frame: np.ndarray) -> None:
        """
        Upload a BGR camera frame to the GPU texture (DMA write).

        Converts BGR → RGBA in CPU before upload.  This conversion is
        the only per-frame CPU→GPU data transfer for the camera feed.

        Parameters:
            bgr_frame (np.ndarray): (H, W, 3) uint8 array in BGR order
                                    from OpenCV.
        """
        assert self._device is not None

        h, w = bgr_frame.shape[:2]

        # Convert BGR → RGBA (add alpha=255)
        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0] = bgr_frame[:, :, 2]  # R
        rgba[:, :, 1] = bgr_frame[:, :, 1]  # G
        rgba[:, :, 2] = bgr_frame[:, :, 0]  # B
        rgba[:, :, 3] = 255                  # A

        # Flip vertically: OpenCV origin is top-left; WebGPU textures
        # also expect top-left but the NDC Y-axis is inverted in the VS,
        # so we flip here once to keep the blit shader simple.
        rgba = np.ascontiguousarray(rgba[::-1])

        self._device.queue.write_texture(
            {
                "texture": self.frame_texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            rgba.tobytes(),
            {
                "offset": 0,
                "bytes_per_row": w * 4,
                "rows_per_image": h,
            },
            (w, h, 1),
        )

    def record(
        self,
        encoder: wgpu.GPUCommandEncoder,
        output_texture: wgpu.GPUTexture,
    ) -> None:
        """
        Record a render pass that blits the camera frame to
        ``output_texture``.

        Parameters:
            encoder (wgpu.GPUCommandEncoder): Current frame command
                encoder.
            output_texture (wgpu.GPUTexture): Destination texture (the
                first ping-pong buffer).
        """
        assert self._device is not None

        bind_group = self._device.create_bind_group(
            layout=self._bind_group_layout,
            entries=[
                {"binding": 0, "resource": self.frame_texture.create_view()},
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
        pass_enc.set_pipeline(self._pipeline)
        pass_enc.set_bind_group(0, bind_group, [], 0, 0)
        pass_enc.draw(3, 1, 0, 0)
        pass_enc.end()
