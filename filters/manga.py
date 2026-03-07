"""
Manga filter.

Transforms the live camera feed into a black-and-white manga /
comic-book style image using a single WGSL fragment shader pass.

The shader pipeline:

1. **Grayscale + posterization** — luminance is extracted with the
   BT.709 formula and quantised to a small number of discrete levels.
2. **White-paper boost** — any posterized value above 0.80 is clamped
   to pure white, replicating the bright paper of printed manga.
3. **Ink lines** — a Sobel edge-magnitude is computed and
   morphologically dilated (1-pixel 8-neighbour max-pool) to produce
   bold ~2-pixel-wide contour lines.
4. **Screentone** — non-ink pixels receive a dot-grid / diagonal
   crosshatch pattern whose density scales with the posterized
   luminance, using screen-space FragCoord for a stable, camera-
   motion-independent grid.
5. **Vignette** — a fixed mild radial darkening is applied last,
   simulating aged printing paper.

Runtime-adjustable parameters:
    ``edge_threshold`` — controls ink sensitivity [0.0, 1.0]
    ``posterize_levels`` — number of discrete luminance steps [2, 8]
    ``dot_scale`` — screentone cell size in pixels [2, 12]
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

@group(0) @binding(0) var tex     : texture_2d<f32>;
@group(0) @binding(1) var smp     : sampler;
@group(0) @binding(2) var<uniform> params : MangaParams;

struct MangaParams {
    edge_threshold   : f32,
    posterize_levels : f32,
    dot_scale        : f32,
    _pad             : f32,
}

// BT.709 luminance
fn luminance(c: vec3f) -> f32 {
    return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

// Sobel edge magnitude at an arbitrary UV position.
// Used both at the current pixel and its 8 neighbours for dilation.
fn sobel_mag(uv: vec2f, d: vec2f) -> f32 {
    let tl = luminance(textureSample(tex, smp, uv + vec2f(-d.x,  d.y)).rgb);
    let tm = luminance(textureSample(tex, smp, uv + vec2f( 0.0,  d.y)).rgb);
    let tr = luminance(textureSample(tex, smp, uv + vec2f( d.x,  d.y)).rgb);
    let ml = luminance(textureSample(tex, smp, uv + vec2f(-d.x,  0.0)).rgb);
    let mr = luminance(textureSample(tex, smp, uv + vec2f( d.x,  0.0)).rgb);
    let bl = luminance(textureSample(tex, smp, uv + vec2f(-d.x, -d.y)).rgb);
    let bm = luminance(textureSample(tex, smp, uv + vec2f( 0.0, -d.y)).rgb);
    let br = luminance(textureSample(tex, smp, uv + vec2f( d.x, -d.y)).rgb);
    let gx = (-tl + tr) + 2.0 * (-ml + mr) + (-bl + br);
    let gy = (tl + 2.0 * tm + tr) - (bl + 2.0 * bm + br);
    return sqrt(gx * gx + gy * gy);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let tex_size = vec2f(textureDimensions(tex, 0));
    // One-pixel step in UV space
    let d = 1.0 / tex_size;

    // ----------------------------------------------------------------
    // 1. Grayscale + posterization
    // ----------------------------------------------------------------
    let colour  = textureSample(tex, smp, in.uv);
    let lum     = luminance(colour.rgb);
    let levels  = params.posterize_levels;
    var L       = floor(lum * levels) / levels;
    // White-paper boost: crush near-whites to pure white (REQ-MNG-003)
    if (L > 0.80) { L = 1.0; }

    // ----------------------------------------------------------------
    // 2. Ink lines: Sobel with 8-neighbour max-pool dilation
    //    (REQ-MNG-004, REQ-MNG-005)
    // ----------------------------------------------------------------
    var edge = sobel_mag(in.uv, d);
    edge = max(edge, sobel_mag(in.uv + vec2f(-d.x,  0.0), d));
    edge = max(edge, sobel_mag(in.uv + vec2f( d.x,  0.0), d));
    edge = max(edge, sobel_mag(in.uv + vec2f( 0.0, -d.y), d));
    edge = max(edge, sobel_mag(in.uv + vec2f( 0.0,  d.y), d));
    edge = max(edge, sobel_mag(in.uv + vec2f(-d.x, -d.y), d));
    edge = max(edge, sobel_mag(in.uv + vec2f( d.x, -d.y), d));
    edge = max(edge, sobel_mag(in.uv + vec2f(-d.x,  d.y), d));
    edge = max(edge, sobel_mag(in.uv + vec2f( d.x,  d.y), d));
    let ink = edge > params.edge_threshold;

    // ----------------------------------------------------------------
    // 3. Screentone pattern (REQ-MNG-008, REQ-MNG-009)
    //    Grid is anchored to screen coordinates (FragCoord), not UVs.
    // ----------------------------------------------------------------
    let px = in.position.x;
    let py = in.position.y;
    let sc = params.dot_scale;

    // Position within each dot cell mapped to [-1, 1]
    let cell = fract(vec2f(px, py) / sc) * 2.0 - vec2f(1.0);
    let dist = length(cell);

    // Fractional positions along the two diagonal axes
    let diag_fwd = fract((px + py) / sc);   // "/" diagonal
    let diag_bwd = fract((px - py) / sc);   // "\" diagonal

    var tone: f32;
    if (L > 0.80) {
        // Bright zone: pure white paper — no dots
        tone = 1.0;
    } else if (L > 0.55) {
        // Light zone: fine dot grid, radius grows toward darker end
        let r = 0.55 * (1.0 - (L - 0.55) / 0.25);
        tone = select(0.0, 1.0, dist > r);
    } else if (L > 0.30) {
        // Mid zone: denser dots + single "/" diagonal crosshatch
        let r       = 0.55 + 0.20 * (0.55 - L) / 0.25;
        let in_dot  = dist < r;
        let in_line = diag_fwd < 0.18;
        tone = select(1.0, 0.0, in_dot || in_line);
    } else {
        // Dark zone: dense dots + dual crosshatch → near black
        let in_dot   = dist < 0.75;
        let in_line1 = diag_fwd < 0.22;
        let in_line2 = diag_bwd < 0.22;
        tone = select(1.0, 0.0, in_dot || in_line1 || in_line2);
    }

    // ----------------------------------------------------------------
    // 4. Composite: ink overrides screentone (REQ-MNG-007, REQ-MNG-010)
    // ----------------------------------------------------------------
    let manga_val = select(tone, 0.0, ink);

    // ----------------------------------------------------------------
    // 5. Vignette: mild fixed radial darkening (REQ-MNG-011)
    //    v = 1 - 0.45 * |uv - 0.5|^2 * 2.56
    // ----------------------------------------------------------------
    let uv_c    = in.uv - vec2f(0.5);
    let vig     = 1.0 - 0.45 * dot(uv_c * 1.6, uv_c * 1.6);
    let out_val = clamp(manga_val * vig, 0.0, 1.0);

    return vec4f(vec3f(out_val), colour.a);
}
"""


class MangaFilter(BaseFilter):
    """
    GPU manga / comic-book visual filter.

    Converts the live camera feed to a black-and-white manga illustration
    using Sobel ink lines, posterization, screen-aligned screentone
    patterns, and a subtle vignette — all in a single WGSL shader pass.

    Parameters
    ----------
    ``edge_threshold`` : float [0.0, 1.0]
        Minimum dilated Sobel magnitude required to render a pixel as
        solid ink.  Lower values produce more (and finer) ink lines;
        higher values yield bolder, sparser outlines.  Default: ``0.15``.
    ``posterize_levels`` : float [2.0, 8.0]
        Number of discrete luminance steps.  Fewer levels (e.g. 2 or 3)
        give a stark high-contrast look; more levels (e.g. 6–8) produce
        smoother tonal transitions.  Default: ``4.0``.
    ``dot_scale`` : float [2.0, 12.0]
        Screentone dot-cell size in pixels.  Smaller values create a
        finer, denser halftone; larger values yield a coarser, more
        visible dot pattern.  Default: ``4.0``.
    """

    def __init__(self) -> None:
        """Initialise the manga filter with default parameters."""
        super().__init__()
        self.params = {
            "edge_threshold": 0.15,
            "posterize_levels": 4.0,
            "dot_scale": 4.0,
        }
        self._param_buffer: wgpu.GPUBuffer | None = None
        self._render_pipeline: wgpu.GPURenderPipeline | None = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """
        Unique filter name.

        Returns:
            str: 'Manga'
        """
        return "Manga"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _build_pipeline(
        self,
        device: wgpu.GPUDevice,
        texture_format: wgpu.TextureFormat,
    ) -> None:
        """
        Compile the manga WGSL shader and create the render pipeline.

        Allocates the 16-byte uniform buffer that holds the three
        runtime parameters plus one padding scalar.

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            texture_format (wgpu.TextureFormat): Target texture format.
        """
        # 4 × f32: edge_threshold, posterize_levels, dot_scale, _pad
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
        Record manga-effect render commands for the current frame.

        Uploads the current parameter values to the uniform buffer,
        constructs a bind group, and records a full-screen triangle draw
        into ``output_texture``.

        Parameters:
            encoder (wgpu.GPUCommandEncoder): Current command encoder.
            input_texture (wgpu.GPUTexture): Source colour texture.
            output_texture (wgpu.GPUTexture): Destination texture.
        """
        device = self._device
        assert device is not None, "apply() called before setup()"

        edge_threshold = float(self.params["edge_threshold"])
        posterize_levels = float(self.params["posterize_levels"])
        dot_scale = float(self.params["dot_scale"])
        data = struct.pack(
            "ffff", edge_threshold, posterize_levels, dot_scale, 0.0
        )
        device.queue.write_buffer(self._param_buffer, 0, data)

        bind_group = device.create_bind_group(
            layout=self._bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": input_texture.create_view(),
                },
                {
                    "binding": 1,
                    "resource": self._sampler,
                },
                {
                    "binding": 2,
                    "resource": {
                        "buffer": self._param_buffer,
                        "offset": 0,
                        "size": 16,
                    },
                },
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
