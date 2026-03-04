"""
Edge detection filter.

Applies a Sobel edge-detection kernel to the luminance of the input
frame and composites the result over the original image.  The
``intensity`` parameter scales the edge prominence, and ``colour``
allows tinting the detected edges with a custom RGBA colour.
"""

from __future__ import annotations

import struct

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
@group(0) @binding(2) var<uniform> params : EdgeParams;

struct EdgeParams {
    intensity  : f32,
    edge_r     : f32,
    edge_g     : f32,
    edge_b     : f32,
}

fn luminance(c: vec3f) -> f32 {
    return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let tex_size = vec2f(textureDimensions(tex, 0));
    let d = 1.0 / tex_size;

    // Sample 3×3 neighbourhood
    let tl = luminance(textureSample(tex, smp, in.uv + vec2f(-d.x,  d.y)).rgb);
    let tm = luminance(textureSample(tex, smp, in.uv + vec2f( 0.0,  d.y)).rgb);
    let tr = luminance(textureSample(tex, smp, in.uv + vec2f( d.x,  d.y)).rgb);
    let ml = luminance(textureSample(tex, smp, in.uv + vec2f(-d.x,  0.0)).rgb);
    let mr = luminance(textureSample(tex, smp, in.uv + vec2f( d.x,  0.0)).rgb);
    let bl = luminance(textureSample(tex, smp, in.uv + vec2f(-d.x, -d.y)).rgb);
    let bm = luminance(textureSample(tex, smp, in.uv + vec2f( 0.0, -d.y)).rgb);
    let br = luminance(textureSample(tex, smp, in.uv + vec2f( d.x, -d.y)).rgb);

    // Sobel kernels
    let gx = (-tl + tr) + 2.0*(-ml + mr) + (-bl + br);
    let gy = (tl + 2.0*tm + tr) - (bl + 2.0*bm + br);
    let edge = clamp(sqrt(gx*gx + gy*gy) * params.intensity, 0.0, 1.0);

    let original = textureSample(tex, smp, in.uv);
    let edge_colour = vec3f(params.edge_r, params.edge_g, params.edge_b);

    // Composite: blend edge colour over original
    let out_rgb = mix(original.rgb, edge_colour, edge);
    return vec4f(out_rgb, original.a);
}
"""


class EdgeDetectionFilter(BaseFilter):
    """
    GPU Sobel edge-detection filter (REQ-002, §4.4).

    Parameters
    ----------
    ``intensity`` : float [0.0 – 10.0]
        Scales edge magnitude; higher values make faint edges visible.
        Default: ``3.0``.
    ``edge_colour`` : tuple[float, float, float] (r, g, b) [0.0 – 1.0]
        RGB tint applied to detected edges.  Default: ``(1.0, 1.0, 1.0)``
        (white edges).
    """

    def __init__(self) -> None:
        """Initialise the edge detection filter with default parameters."""
        super().__init__()
        self.params = {
            "intensity": 3.0,
            "edge_colour": (1.0, 1.0, 1.0),  # white
        }
        self._param_buffer: wgpu.GPUBuffer | None = None
        self._render_pipeline: wgpu.GPURenderPipeline | None = None

    @property
    def name(self) -> str:
        """
        Unique filter name.

        Returns:
            str: 'EdgeDetection'
        """
        return "EdgeDetection"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _build_pipeline(
        self,
        device: wgpu.GPUDevice,
        texture_format: wgpu.TextureFormat,
    ) -> None:
        """
        Compile the Sobel WGSL shader and create the render pipeline.

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            texture_format (wgpu.TextureFormat): Target texture format.
        """
        # 4× f32 uniform: intensity, r, g, b
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
        Record Sobel edge detection render commands for the current frame.

        Parameters:
            encoder (wgpu.GPUCommandEncoder): Current command encoder.
            input_texture (wgpu.GPUTexture): Source colour texture.
            output_texture (wgpu.GPUTexture): Destination texture.
        """
        device = self._device
        assert device is not None, "apply() called before setup()"

        intensity = float(self.params["intensity"])
        r, g, b = self.params["edge_colour"]
        data = struct.pack("ffff", intensity, float(r), float(g), float(b))
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
