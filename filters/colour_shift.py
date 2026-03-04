"""
Colour shift (hue rotation) filter.

Rotates the hue of each pixel by a configurable angle while preserving
saturation and value.  Optionally boosts saturation via a ``saturation``
multiplier.  The conversion path is RGB → HSV → (hue rotate) → RGB.
"""

from __future__ import annotations

import struct

import wgpu

from filters.base import BaseFilter

# ---------------------------------------------------------------------------
# WGSL shader
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
@group(0) @binding(2) var<uniform> params : ColourShiftParams;

struct ColourShiftParams {
    hue_shift  : f32,   // degrees [0, 360)
    saturation : f32,   // multiplier  [0, 4]
    _pad0      : f32,
    _pad1      : f32,
}

// Convert RGB → HSV
fn rgb_to_hsv(c: vec3f) -> vec3f {
    let K = vec4f(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    let p = mix(vec4f(c.bg, K.wz), vec4f(c.gb, K.xy), step(c.b, c.g));
    let q = mix(vec4f(p.xyw, c.r), vec4f(c.r, p.yzx), step(p.x, c.r));
    let d = q.x - min(q.w, q.y);
    let e = 1.0e-10;
    return vec3f(abs(q.z + (q.w - q.y) / (6.0 * d + e)),
                 d / (q.x + e),
                 q.x);
}

// Convert HSV → RGB
fn hsv_to_rgb(c: vec3f) -> vec3f {
    let K = vec4f(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3f(0.0), vec3f(1.0)), c.y);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let colour = textureSample(tex, smp, in.uv);
    var hsv = rgb_to_hsv(colour.rgb);

    // Rotate hue (normalise shift to [0, 1])
    hsv.x = fract(hsv.x + params.hue_shift / 360.0);
    // Scale saturation and clamp
    hsv.y = clamp(hsv.y * params.saturation, 0.0, 1.0);

    let out_rgb = hsv_to_rgb(hsv);
    return vec4f(out_rgb, colour.a);
}
"""


class ColourShiftFilter(BaseFilter):
    """
    GPU hue-rotation and saturation filter (REQ-002, §4.4).

    Parameters
    ----------
    ``hue_shift`` : float [0.0 – 360.0]
        Degrees to rotate the hue of every pixel.  Default: ``0.0``
        (no shift).
    ``saturation`` : float [0.0 – 4.0]
        Saturation multiplier.  1.0 = unchanged, 0.0 = grayscale,
        2.0 = double saturation.  Default: ``1.0``.
    """

    def __init__(self) -> None:
        """Initialise the colour shift filter with default parameters."""
        super().__init__()
        self.params = {"hue_shift": 90.0, "saturation": 1.5}
        self._param_buffer: wgpu.GPUBuffer | None = None
        self._render_pipeline: wgpu.GPURenderPipeline | None = None

    @property
    def name(self) -> str:
        """
        Unique filter name.

        Returns:
            str: 'ColourShift'
        """
        return "ColourShift"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _build_pipeline(
        self,
        device: wgpu.GPUDevice,
        texture_format: wgpu.TextureFormat,
    ) -> None:
        """
        Compile the hue-rotation WGSL shader and create the pipeline.

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            texture_format (wgpu.TextureFormat): Target texture format.
        """
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

        layout = device.create_pipeline_layout(bind_group_layouts=[bgl])
        self._render_pipeline = device.create_render_pipeline(
            layout=layout,
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
        """Release GPU resource references."""
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
        Record hue-rotation render commands for the current frame.

        Parameters:
            encoder (wgpu.GPUCommandEncoder): Current command encoder.
            input_texture (wgpu.GPUTexture): Source colour texture.
            output_texture (wgpu.GPUTexture): Destination texture.
        """
        device = self._device
        assert device is not None, "apply() called before setup()"

        hue_shift = float(self.params["hue_shift"])
        saturation = float(self.params["saturation"])
        data = struct.pack("ffff", hue_shift, saturation, 0.0, 0.0)
        device.queue.write_buffer(self._param_buffer, 0, data)

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
        pass_enc.draw(3, 1, 0, 0)
        pass_enc.end()
