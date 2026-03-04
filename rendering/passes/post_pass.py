"""
Post-processing render pass.

Final pipeline stage: blits the composited frame texture to the swap
chain surface texture (§4.1 'Post-processing pass').

Currently implements a direct blit with optional vignette.  This pass
is the only one that writes to the swap chain surface — all preceding
passes write into intermediate textures.
"""

from __future__ import annotations

import logging
import struct
from typing import Optional

import wgpu

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WGSL shader — fullscreen blit with optional vignette
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

@group(0) @binding(0) var tex     : texture_2d<f32>;
@group(0) @binding(1) var smp     : sampler;
@group(0) @binding(2) var<uniform> params : PostParams;

struct PostParams {
    vignette_strength : f32,
    vignette_radius   : f32,
    _pad0             : f32,
    _pad1             : f32,
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let colour = textureSample(tex, smp, in.uv);

    // Vignette: darken edges based on distance from screen centre
    let uv_centred = in.uv - vec2f(0.5);
    let dist = length(uv_centred);
    let vign = 1.0 - smoothstep(
        params.vignette_radius,
        params.vignette_radius + 0.3,
        dist,
    ) * params.vignette_strength;

    return vec4f(colour.rgb * vign, colour.a);
}
"""


class PostPass:
    """
    Final compositing pass: blits the pipeline output to the swap chain
    surface (§4.1 Post-processing pass).

    Exposes a ``vignette_strength`` parameter (0 = none, 1 = full)
    for optional edge-darkening.
    """

    def __init__(self) -> None:
        """Initialise the post pass descriptor."""
        self._device: Optional[wgpu.GPUDevice] = None
        self._pipeline: Optional[wgpu.GPURenderPipeline] = None
        self._sampler: Optional[wgpu.GPUSampler] = None
        self._bind_group_layout: Optional[wgpu.GPUBindGroupLayout] = None
        self._param_buffer: Optional[wgpu.GPUBuffer] = None
        self.vignette_strength: float = 0.4
        self.vignette_radius: float = 0.45

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(
        self,
        device: wgpu.GPUDevice,
        surface_format: wgpu.TextureFormat,
    ) -> None:
        """
        Compile the post-process shader targeting the swap chain format.

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            surface_format (wgpu.TextureFormat): The swap chain surface
                texture format.
        """
        self._device = device

        self._param_buffer = device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
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
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "uniform"},
                },
            ]
        )

        layout = device.create_pipeline_layout(
            bind_group_layouts=[self._bind_group_layout]
        )
        self._pipeline = device.create_render_pipeline(
            layout=layout,
            vertex={
                "module": shader,
                "entry_point": "vs_main",
                "buffers": [],
            },
            fragment={
                "module": shader,
                "entry_point": "fs_main",
                "targets": [{"format": surface_format}],
            },
            primitive={"topology": "triangle-list"},
            depth_stencil=None,
            multisample=None,
        )
        logger.debug("PostPass ready (surface format: %s).", surface_format)

    def teardown(self) -> None:
        """Release GPU resources."""
        self._pipeline = None
        self._sampler = None
        self._bind_group_layout = None
        self._param_buffer = None
        self._device = None

    # ------------------------------------------------------------------
    # Per-frame record
    # ------------------------------------------------------------------

    def record(
        self,
        encoder: wgpu.GPUCommandEncoder,
        input_texture: wgpu.GPUTexture,
        surface_view: wgpu.GPUTextureView,
    ) -> None:
        """
        Record the final blit from ``input_texture`` to the swap chain.

        Parameters:
            encoder (wgpu.GPUCommandEncoder): Current frame command
                encoder.
            input_texture (wgpu.GPUTexture): The composited pipeline
                output texture.
            surface_view (wgpu.GPUTextureView): The current swap chain
                surface texture view to render into.
        """
        assert self._device is not None

        # Upload uniform params
        data = struct.pack(
            "ffff",
            self.vignette_strength,
            self.vignette_radius,
            0.0,
            0.0,
        )
        self._device.queue.write_buffer(self._param_buffer, 0, data)

        bind_group = self._device.create_bind_group(
            layout=self._bind_group_layout,
            entries=[
                {"binding": 0, "resource": input_texture.create_view()},
                {"binding": 1, "resource": self._sampler},
                {"binding": 2, "resource": {
                    "buffer": self._param_buffer,
                    "offset": 0,
                    "size": 16,
                }},
            ],
        )

        pass_enc = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": surface_view,
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
