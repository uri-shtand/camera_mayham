---
title: Manga Filter
version: 1.0
date_created: 2026-03-08
last_updated: 2026-03-08
tags: [design, filter, gpu, wgpu, manga, screentone]
---

# Manga Filter

The Manga Filter transforms the live camera feed into a black-and-white
manga / comic-book style image entirely on the GPU via a single WGSL
fragment shader — no extra render passes and no changes to the core
pipeline are required.

The effect combines five artistic techniques to simulate traditional
manga printing: **luminance posterization**, **white-paper boost**,
**Sobel ink-line detection with morphological dilation**, a **layered
screentone pattern** (dot grid + diagonal crosshatch), and a fixed
**vignette**.

---

## 1. What the User Sees

When the filter is active the camera feed is converted to a
black-and-white image with the following visual properties:

1. **Ink outlines** — bold black contour lines trace the edges of the
   face, background objects, and any high-contrast border. Lines are
   approximately 2 pixels wide due to morphological dilation.
2. **Posterized gray levels** — smooth colour gradients are quantised
   into a small number of discrete gray bands (default: 4 levels). The
   number of bands is user-adjustable via `posterize_levels`.
3. **White-paper boost** — areas brighter than approximately 80 % of
   maximum luminance are crushed to pure white, replicating the
   bright-white paper of printed manga.
4. **Screentone shading** — non-ink pixels are rendered with a
   printing-press dot/crosshatch pattern whose density matches the
   underlying gray level:
   - **Bright zones** (L > 0.80): pure white — no dots.
   - **Light zones** (0.55 < L ≤ 0.80): fine black dot grid on white.
   - **Mid zones** (0.30 < L ≤ 0.55): denser dots and a single
     diagonal slash (`/`) crosshatch.
   - **Dark zones** (L ≤ 0.30): dense dots and dual crosshatch (`/`
     and `\`), giving a near-black impression.
5. **Dot grid is screen-aligned** — the screentone pattern is computed
   from fragment-shader screen coordinates (FragCoord) so the dots
   stay fixed on screen and do not "swim" with camera motion.
6. **Vignette** — a fixed, mild radial darkening toward the frame
   edges focuses the eye toward the centre and simulates aged paper.
7. **No face required** — the filter is purely image-based; it does
   not require face tracking to be active or a face to be detected.

---

## 2. Scope

| In scope | Out of scope |
|---|---|
| `MangaFilter` class in `filters/manga.py` | Changes to `BaseFilter` or `FilterPass` interfaces |
| Single-pass WGSL fragment shader (`_WGSL` string constant) | Multi-pass or compute-pipeline implementations |
| Posterization, white-paper boost, Sobel ink detection | Colour tinting or sepia tone variants |
| 8-neighbour Sobel max-pooling for ink dilation | Separate morphological compute pass |
| Layered screentone (dot grid + diagonal crosshatch) | Configurable dot shape (diamonds, squares, etc.) |
| Fixed vignette | User-adjustable vignette strength |
| Three runtime params: `edge_threshold`, `posterize_levels`, `dot_scale` | Additional params beyond the three listed |
| `filters/__init__.py` export | Changes to `AppState`, `FilterPass`, or `RenderPipeline` |
| `app/application.py` registration | Any UI controls beyond the filter on/off toggle |
| Unit tests in `tests/test_manga_filter.py` | Visual regression / pixel-output tests |

---

## 3. Definitions

| Term | Meaning |
|---|---|
| **BT.709 luminance** | Weighted sum of RGB channels: `Y = 0.2126 R + 0.7152 G + 0.0722 B` — the standard used across all filters in this project |
| **Posterization** | Quantising a continuous luminance value into N discrete steps: $L_{out} = \lfloor L_{in} \cdot N \rfloor / N$ |
| **Sobel operator** | A 3×3 separable kernel that estimates the horizontal (`gx`) and vertical (`gy`) luminance gradient; magnitude = `√(gx² + gy²)` |
| **Morphological dilation** | Expanding a binary mask by taking the pixel-wise maximum of a value and its neighbours; here approximated by computing Sobel at the current pixel and its 8 cardinal/diagonal neighbours, then taking the max |
| **Screentone** | Traditional manga printing technique that represents gray tones using patterns of small black dots; denser dots = darker tone |
| **CrossHatch** | Overlapping diagonal line patterns used in dark areas — one set of `/` diagonals plus one set of `\` diagonals |
| **FragCoord** | WGSL `@builtin(position)`: the fragment's window-space (x, y) coordinates in pixels, used as the screentone grid coordinate |
| **Vignette** | Radial darkening applied after compositing; computed as `1 − k · |uv − 0.5|²` where `k` is a fixed constant |
| **Full-screen triangle** | A single three-vertex draw call that covers the entire render target without a vertex buffer |

---

## 4. Requirements

### 4.1 Grayscale and Posterization

- **REQ-MNG-001** The filter extracts luminance from each input pixel
  using the BT.709 formula.
- **REQ-MNG-002** Luminance is posterized to `posterize_levels`
  discrete steps using the formula
  $L_{post} = \lfloor L_{in} \cdot N \rfloor / N$.
- **REQ-MNG-003** After posterization, any pixel with `L_post > 0.80`
  is clamped to `1.0` (white-paper boost).

### 4.2 Ink Lines

- **REQ-MNG-004** The filter computes a Sobel edge magnitude at each
  pixel using a 3×3 luminance neighbourhood.
- **REQ-MNG-005** The edge magnitude is dilated over a 1-pixel
  neighbourhood by computing the Sobel magnitude at the current pixel
  and each of its 8 cardinal and diagonal neighbours (9 total) and
  taking the maximum value. This produces ink lines approximately 2
  pixels wide.
- **REQ-MNG-006** A pixel is marked as "ink" when its dilated edge
  magnitude exceeds `edge_threshold`.
- **REQ-MNG-007** Ink pixels are rendered as solid black (`0.0`),
  overriding any screentone value.

### 4.3 Screentone

- **REQ-MNG-008** Non-ink pixels receive a screentone tone value
  derived from the posterized luminance:

  | Luminance range | Pattern |
  |---|---|
  | L > 0.80 | Pure white (1.0) |
  | 0.55 < L ≤ 0.80 | Fine dot grid; dot radius scales with darkness |
  | 0.30 < L ≤ 0.55 | Dense dot grid + single `/` diagonal crosshatch |
  | L ≤ 0.30 | Dense dot grid + dual `/` and `\` crosshatch |

- **REQ-MNG-009** The screentone grid is computed from WGSL
  `@builtin(position)` screen coordinates divided by `dot_scale`,
  ensuring the pattern is screen-aligned and independent of camera
  motion.
- **REQ-MNG-010** Black ink pixels (`REQ-MNG-007`) take precedence
  over the screentone value — the ink mask is applied after the tone
  is computed.

### 4.4 Vignette

- **REQ-MNG-011** After compositing ink and screentone, a mild radial
  vignette is applied:
  $out = clamp(tone \cdot (1 - 0.45 \cdot |uv - 0.5|^2 \cdot 2.56), 0, 1)$
  The strength is hard-coded and not user-configurable.

### 4.5 Parameters

- **REQ-MNG-012** The filter exposes exactly three runtime-adjustable
  parameters:

  | Parameter | Type | Default | Range | Description |
  |---|---|---|---|---|
  | `edge_threshold` | `float` | `0.15` | [0.0, 1.0] | Minimum dilated Sobel magnitude to render as ink |
  | `posterize_levels` | `float` | `4.0` | [2.0, 8.0] | Number of discrete luminance steps |
  | `dot_scale` | `float` | `4.0` | [2.0, 12.0] | Screentone dot-grid cell size in pixels |

### 4.6 Integration

- **REQ-MNG-013** The filter name property returns `"Manga"`.
- **REQ-MNG-014** The filter is registered in
  `app/application.py` with `enabled = False` (off by default, like
  all built-in filters).
- **REQ-MNG-015** The filter class is exported from
  `filters/__init__.py`.

---

## 5. Constraints

- **CON-MNG-001** The filter implements the `BaseFilter` interface:
  `name` property, `setup`, `_build_pipeline`, `apply`, `teardown`.
- **CON-MNG-002** `apply` must not allocate new GPU buffers or
  textures; all resources are pre-allocated in `_build_pipeline`.
- **CON-MNG-003** The shader is a single-pass WGSL render pipeline
  (no compute shaders, no additional render passes).
- **CON-MNG-004** No CPU face tracking data is required; the filter
  operates solely on the input texture.
- **CON-MNG-005** The uniform buffer is exactly 16 bytes (four `f32`
  values with one padding scalar) to satisfy WebGPU 16-byte alignment.

---

## 6. Data Contracts

### 6.1 Class Interface

File: `filters/manga.py`

```python
class MangaFilter(BaseFilter):

    @property
    def name(self) -> str: ...           # returns "Manga"

    def _build_pipeline(
        self,
        device: wgpu.GPUDevice,
        texture_format: wgpu.TextureFormat,
    ) -> None: ...

    def apply(
        self,
        encoder: wgpu.GPUCommandEncoder,
        input_texture: wgpu.GPUTexture,
        output_texture: wgpu.GPUTexture,
    ) -> None: ...
```

### 6.2 Uniform Buffer Layout

The uniform buffer is 16 bytes packed with `struct.pack("4f", ...)`:

| Offset (bytes) | Field | Type | Description |
|---|---|---|---|
| 0 | `edge_threshold` | `f32` | Ink threshold |
| 4 | `posterize_levels` | `f32` | Gray quantisation steps |
| 8 | `dot_scale` | `f32` | Screentone cell size |
| 12 | `_pad` | `f32` | Padding for 16-byte alignment |

### 6.3 WGSL Shader Pipeline Structure

```
vs_main (full-screen triangle, 3 vertices)
    └── fs_main
            ├── luminance()          — BT.709 helper
            ├── sobel_mag()          — 3×3 Sobel magnitude helper
            ├── Posterize + boost    — quantise luminance
            ├── Dilation loop        — max of 9 sobel_mag() calls
            ├── Screentone logic     — FragCoord-based dot/line grid
            ├── Ink composite        — ink overrides tone
            └── Vignette             — radial darkening
```
