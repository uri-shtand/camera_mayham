"""
Grayscale filter.

Converts each frame to grayscale using the standard luminance formula::

    L = 0.2126 R + 0.7152 G + 0.0722 B

The result maintains the original texture format (RGBA) with all three
colour channels set to the same luminance value.

The ``strength`` parameter blends between the original colour frame
(0.0) and full grayscale (1.0), allowing partial desaturation effects.
"""

from __future__ import annotations

import wgpu

from filters.base import BaseFilter

# ---------------------------------------------------------------------------
# WGSL shader source
# ---------------------------------------------------------------------------
_WGSL = """
struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0)       uv       : vec2f,
}

// Full-screen triangle vertex shader — no vertex buffer needed
@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> VertexOutput {
    // Three vertices that cover the NDC [-1, 1] space
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
@group(0) @binding(2) var<uniform> params : GrayscaleParams;

struct GrayscaleParams {
    strength : f32,
    _pad0    : f32,
    _pad1    : f32,
    _pad2    : f32,
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let colour = textureSample(tex, smp, in.uv);

    // Luminance (ITU-R BT.709 coefficients)
    let lum = dot(colour.rgb, vec3f(0.2126, 0.7152, 0.0722));
    let grey = vec3f(lum);

    // Blend between original and grey based on strength
    let out_colour = mix(colour.rgb, grey, params.strength);
    return vec4f(out_colour, colour.a);
}
"""


class GrayscaleFilter(BaseFilter):
    """
    GPU greyscale desaturation filter (REQ-002, §4.4).

    Parameters
    ----------
    ``strength`` : float [0.0 – 1.0]
        Controls the mix between the original colour frame and full
        greyscale.  Defaults to ``1.0`` (full greyscale).
    """

    def __init__(self) -> None:
        """Initialise the grayscale filter with default parameters."""
        super().__init__()
        # Runtime-adjustable parameters exposed to the widget panel
        self.params = {"strength": 1.0}
        self._param_buffer: wgpu.GPUBuffer | None = None
        self._render_pipeline: wgpu.GPURenderPipeline | None = None

    @property
    def name(self) -> str:
        """
        Unique filter name.

        Returns:
            str: 'Grayscale'
        """
        return "Grayscale"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _build_pipeline(
        self,
        device: wgpu.GPUDevice,
        texture_format: wgpu.TextureFormat,
    ) -> None:
        """
        Compile the WGSL shader and create the wgpu render pipeline.

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            texture_format (wgpu.TextureFormat): Target texture format.
        """
        # Uniform buffer — 4× f32 (strength + 3 padding bytes)
        self._param_buffer = device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        shader = device.create_shader_module(code=_WGSL)

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
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "uniform"},
                },
            ]
        )
        self._bind_group_layout = bgl

        pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[bgl]
        )
        self._render_pipeline = device.create_render_pipeline(
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

    def teardown(self) -> None:
        """Release GPU buffer references."""
        self._param_buffer = None
        self._render_pipeline = None
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
        Record grayscale render commands for the current frame.

        Writes the desaturated image into ``output_texture``.

        Parameters:
            encoder (wgpu.GPUCommandEncoder): Current command encoder.
            input_texture (wgpu.GPUTexture): Source colour texture.
            output_texture (wgpu.GPUTexture): Destination texture.
        """
        import struct  # noqa: PLC0415

        device = self._device
        assert device is not None, "apply() called before setup()"

        # Upload uniform data (strength + padding)
        strength = float(self.params["strength"])
        data = struct.pack("ffff", strength, 0.0, 0.0, 0.0)
        device.queue.write_buffer(self._param_buffer, 0, data)

        # Build bind group pointing to the input texture
        bind_group = device.create_bind_group(
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

        # Render pass → output texture
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
        pass_enc.set_pipeline(self._render_pipeline)
        pass_enc.set_bind_group(0, bind_group, [], 0, 0)
        pass_enc.draw(3, 1, 0, 0)  # Full-screen triangle
        pass_enc.end()
