"""
2D Moustache Overlay Filter.

Composites a moustache sprite from the sprite-sheet
``2dmodels/moustaches2.jpg`` onto the live camera feed each frame.
The moustache is anchored between the detected nose tip and upper lip,
scaled to match face size, and rotated to track head roll.

Sprite sheet layout (3 columns × 2 rows)::

    ┌─────────┬─────────┬─────────┐
    │  idx 0  │  idx 1  │  idx 2  │
    ├─────────┼─────────┼─────────┤
    │  idx 3  │  idx 4  │  idx 5  │
    └─────────┴─────────┴─────────┘

The white background of each cell is converted to full transparency via
a luminance threshold at load time (REQ-MS-002).

GPU render pass order per frame
--------------------------------
1. **Blit pass**: ``input_texture`` → ``output_texture`` unchanged.
2. **Overlay pass** (only when face detected): the CPU-drawn BGRA canvas
   (containing the scaled and rotated sprite) is uploaded to the GPU as
   an RGBA texture and alpha-blended over the output.

Coordinate anchor points (landmark indices)
-------------------------------------------
- 1  — nose tip
- 13 — upper lip centre
- 61 — left mouth corner
- 291 — right mouth corner

Anchor computation (spec §6.5)::

    cx = (lm[61].x + lm[291].x) / 2 × frame_width
    cy = (lm[1].y  + lm[13].y)  / 2 × frame_height
    rw = |lm[291].x − lm[61].x| × frame_width × SCALE_FACTOR
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import wgpu

from filters.base import BaseFilter
from tracking.face_tracker import FaceTrackResult, Landmark

# ---------------------------------------------------------------------------
# Sprite sheet configuration
# ---------------------------------------------------------------------------

# Path is resolved relative to this file's parent-parent (project root).
_SPRITE_SHEET_PATH: Path = (
    Path(__file__).parent.parent / "2dmodels" / "moustaches2.jpg"
)

# Sprite grid dimensions (3 columns × 2 rows = 6 moustaches).
_GRID_COLS: int = 3
_GRID_ROWS: int = 2
_NUM_SPRITES: int = _GRID_COLS * _GRID_ROWS  # 6

# Pixels with luminance ≥ this value are considered background (white).
_ALPHA_THRESHOLD: int = 220

# Scale factor applied to the mouth-corner distance to determine the
# rendered moustache width.  1.4 gives a natural-looking width that
# extends slightly past the mouth corners.
_WIDTH_SCALE: float = 1.4

# Landmark indices used for anchor computation (MediaPipe 478-point model).
_LM_NOSE_TIP: int = 1
_LM_UPPER_LIP: int = 13
_LM_MOUTH_LEFT: int = 61
_LM_MOUTH_RIGHT: int = 291

# Minimum rendered width (px) — prevents zero-size sprites on degenerate
# landmark positions.
_MIN_RENDER_WIDTH: int = 10

# ---------------------------------------------------------------------------
# WGSL — Blit pass (full-screen triangle, pass-through texture copy)
# ---------------------------------------------------------------------------
_WGSL_BLIT = """
struct BlitVertexOutput {
    @builtin(position) clip_pos : vec4f,
    @location(0)       uv       : vec2f,
}

@vertex
fn vs_blit(@builtin(vertex_index) vi : u32) -> BlitVertexOutput {
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


class MoustacheFilter(BaseFilter):
    """
    2D moustache sprite overlay filter (spec §4, CON-MS-001).

    Loads six moustache sprites from a 3×2 JPEG sprite sheet, removes
    the white background via alpha masking, and composites the selected
    sprite onto the live camera feed anchored to the detected face.

    The user-adjustable parameter ``moustache_index`` (0–5) selects
    which sprite to render.  Changing it takes effect on the very next
    frame (REQ-MS-006).

    Face tracking data is supplied each frame via
    :py:meth:`update_face_result` before :py:meth:`apply` is called.
    """

    def __init__(self) -> None:
        """Initialise the filter; sprite and GPU resources are deferred."""
        super().__init__()

        # Validated moustache index; written via params property setter.
        self._moustache_index: int = 0
        self.params["moustache_index"] = 0

        # Pre-processed BGRA sprites loaded at setup() time.
        self._sprites: List[np.ndarray] = []

        # Latest face tracking result injected each frame.
        self._face_result: Optional[FaceTrackResult] = None

        # GPU pipeline resources (created in _build_pipeline).
        self._blit_pipeline: Any = None
        self._blit_bgl: Any = None
        self._overlay_pipeline: Any = None
        self._overlay_bgl: Any = None

        # Overlay texture — lazy-initialised on first apply() call
        # because dimensions are only known then.
        self._overlay_texture: Any = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """
        Unique filter name.

        Returns:
            str: ``'Moustache'``
        """
        return "Moustache"

    # ------------------------------------------------------------------
    # Param access with validation
    # ------------------------------------------------------------------

    def _get_index(self) -> int:
        """
        Read and validate ``moustache_index`` from ``self.params``.

        Clamps out-of-range values silently to ``[0, _NUM_SPRITES - 1]``
        (REQ-MS-005).

        Returns:
            int: Valid moustache index in ``[0, 5]``.
        """
        raw = self.params.get("moustache_index", 0)
        try:
            idx = int(raw)
        except (TypeError, ValueError):
            idx = 0
        return max(0, min(_NUM_SPRITES - 1, idx))

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
        ``face_detected`` is ``False``, no moustache is drawn and the
        frame passes through unchanged.

        Parameters:
            result (Optional[FaceTrackResult]): Latest face tracking
                output, or ``None`` when no inference was performed.
        """
        self._face_result = result

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(
        self,
        device: wgpu.GPUDevice,
        texture_format: wgpu.TextureFormat,
    ) -> None:
        """
        Load sprite sheet, build alpha masks, and create GPU resources.

        Extends :py:meth:`BaseFilter.setup` by loading sprites before
        the pipeline is built so that ``_build_pipeline`` has access to
        sprite dimensions if needed.

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            texture_format (wgpu.TextureFormat): Target texture format.
        """
        self._load_sprites()
        super().setup(device, texture_format)

    def _load_sprites(self) -> None:
        """
        Load ``moustaches2.jpg``, split into six cells, and alpha-mask
        each cell's white background (REQ-MS-001, REQ-MS-002).

        Sprites are stored in ``self._sprites`` as BGRA uint8 arrays of
        shape ``(cell_height, cell_width, 4)`` (REQ-MS-003).

        Raises:
            FileNotFoundError: If the sprite sheet cannot be read.
        """
        sheet_bgr = cv2.imread(str(_SPRITE_SHEET_PATH))
        if sheet_bgr is None:
            raise FileNotFoundError(
                f"Sprite sheet not found: {_SPRITE_SHEET_PATH}"
            )

        # Convert BGR → BGRA so we can write per-pixel alpha.
        sheet_bgra = cv2.cvtColor(sheet_bgr, cv2.COLOR_BGR2BGRA)

        sheet_h, sheet_w = sheet_bgra.shape[:2]
        cell_h = sheet_h // _GRID_ROWS
        cell_w = sheet_w // _GRID_COLS

        self._sprites = []
        for row in range(_GRID_ROWS):
            for col in range(_GRID_COLS):
                y0 = row * cell_h
                y1 = y0 + cell_h
                x0 = col * cell_w
                x1 = x0 + cell_w
                cell = sheet_bgra[y0:y1, x0:x1].copy()
                cell = self._apply_alpha_mask(cell)
                self._sprites.append(cell)

    @staticmethod
    def _apply_alpha_mask(sprite_bgra: np.ndarray) -> np.ndarray:
        """
        Zero out the alpha channel wherever the pixel is near-white.

        Luminance is computed as:
            L = 0.114*B + 0.587*G + 0.299*R  (BT.601 approximation)

        Pixels with L ≥ ``_ALPHA_THRESHOLD`` are set to ``alpha = 0``
        (fully transparent); all other pixels get ``alpha = 255``
        (fully opaque) (REQ-MS-002).

        Parameters:
            sprite_bgra (np.ndarray): BGRA uint8 array of shape
                ``(H, W, 4)``.

        Returns:
            np.ndarray: Modified BGRA array with white background
                replaced by transparency.
        """
        b = sprite_bgra[:, :, 0].astype(np.float32)
        g = sprite_bgra[:, :, 1].astype(np.float32)
        r = sprite_bgra[:, :, 2].astype(np.float32)
        luminance = 0.114 * b + 0.587 * g + 0.299 * r

        mask = luminance >= _ALPHA_THRESHOLD
        sprite_bgra[:, :, 3] = np.where(mask, 0, 255).astype(np.uint8)
        return sprite_bgra

    def _build_pipeline(
        self,
        device: wgpu.GPUDevice,
        texture_format: wgpu.TextureFormat,
    ) -> None:
        """
        Compile WGSL shaders and allocate GPU pipeline objects.

        Creates:
        * Pass-through blit pipeline (no blending).
        * Alpha-blended overlay blit pipeline (for moustache sprite).

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            texture_format (wgpu.TextureFormat): Target texture format.
        """
        self._build_blit_pipeline(device, texture_format)
        self._build_overlay_pipeline(device, texture_format)

    def _build_blit_pipeline(
        self,
        device: wgpu.GPUDevice,
        texture_format: wgpu.TextureFormat,
    ) -> None:
        """
        Compile and store the pass-through blit render pipeline.

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

    def _build_overlay_pipeline(
        self,
        device: wgpu.GPUDevice,
        texture_format: wgpu.TextureFormat,
    ) -> None:
        """
        Build the alpha-blended overlay blit pipeline.

        Enables ``src-alpha / one-minus-src-alpha`` blending so the
        moustache sprite composites correctly over the camera frame.

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

    def teardown(self) -> None:
        """Release all GPU pipeline references and sprite cache."""
        self._blit_pipeline = None
        self._blit_bgl = None
        self._overlay_pipeline = None
        self._overlay_bgl = None
        self._overlay_texture = None
        self._sprites = []
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

        1. **Blit pass**: copies ``input_texture`` to ``output_texture``.
        2. **Overlay pass** (face detected only): the selected moustache
           sprite, scaled and rotated to match the face, is composited
           alpha-blended over the output.

        Parameters:
            encoder (wgpu.GPUCommandEncoder): Current frame command
                encoder.
            input_texture (wgpu.GPUTexture): Source camera texture.
            output_texture (wgpu.GPUTexture): Destination texture.
        """
        device = self._device
        assert device is not None, "apply() called before setup()"

        w, h = input_texture.size[0], input_texture.size[1]

        # Pass 1: blit input → output (always)
        self._record_blit_pass(encoder, input_texture, output_texture)

        # Pass 2: moustache overlay (only when face detected)
        if not self._has_face():
            return

        self._ensure_overlay_texture(device, w, h)
        self._update_overlay_texture(device, w, h)
        self._record_overlay_pass(encoder, output_texture, device)

    # ------------------------------------------------------------------
    # Private helpers — face detection guard
    # ------------------------------------------------------------------

    def _has_face(self) -> bool:
        """
        Return True when the stored result contains a detected face
        with sufficient landmarks for anchor computation.

        Returns:
            bool: True if the moustache should be drawn this frame.
        """
        return (
            self._face_result is not None
            and self._face_result.face_detected
            and len(self._face_result.landmarks) > max(
                _LM_NOSE_TIP, _LM_UPPER_LIP,
                _LM_MOUTH_LEFT, _LM_MOUTH_RIGHT,
            )
        )

    # ------------------------------------------------------------------
    # Private helpers — anchor computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_anchor(
        landmarks: List[Landmark],
        frame_width: int,
        frame_height: int,
        roll: float,
    ) -> Tuple[int, int, int, float]:
        """
        Compute the moustache placement parameters from face landmarks.

        The anchor is placed midway between the nose tip and the upper
        lip centre.  Width is proportional to the inter-mouth-corner
        distance multiplied by ``_WIDTH_SCALE`` (spec §6.5).

        Parameters:
            landmarks (List[Landmark]): 478-point landmark list.
            frame_width (int): Frame width in pixels.
            frame_height (int): Frame height in pixels.
            roll (float): Head roll angle in degrees.

        Returns:
            Tuple[int, int, int, float]:
                ``(centre_x, centre_y, render_width, roll_degrees)``
        """
        cx = (
            (landmarks[_LM_MOUTH_LEFT].x + landmarks[_LM_MOUTH_RIGHT].x)
            / 2.0
            * frame_width
        )
        cy = (
            (landmarks[_LM_NOSE_TIP].y + landmarks[_LM_UPPER_LIP].y)
            / 2.0
            * frame_height
        )
        rw = max(
            _MIN_RENDER_WIDTH,
            int(
                abs(
                    landmarks[_LM_MOUTH_RIGHT].x
                    - landmarks[_LM_MOUTH_LEFT].x
                )
                * frame_width
                * _WIDTH_SCALE
            ),
        )
        return int(cx), int(cy), rw, roll

    # ------------------------------------------------------------------
    # Private helpers — render passes
    # ------------------------------------------------------------------

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
        pass_enc.draw(3, 1, 0, 0)
        pass_enc.end()

    def _ensure_overlay_texture(
        self,
        device: wgpu.GPUDevice,
        width: int,
        height: int,
    ) -> None:
        """
        Lazy-initialise the overlay RGBA texture on the first apply call.

        Not allocated in ``_build_pipeline`` because texture dimensions
        are only known at apply time.  Reused every frame after the
        first (CON-MS-002).

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

    def _draw_moustache(self, width: int, height: int) -> np.ndarray:
        """
        Render the selected moustache sprite onto a transparent BGRA
        canvas matching the frame dimensions.

        The sprite is scaled to the computed render width, rotated by
        the head roll angle, and blended onto the canvas at the anchor
        position using the alpha channel derived from
        :py:meth:`_apply_alpha_mask`.

        Parameters:
            width (int): Frame width in pixels.
            height (int): Frame height in pixels.

        Returns:
            np.ndarray: RGBA uint8 array of shape ``(height, width, 4)``
                with the moustache composited at the correct position.
        """
        # Transparent canvas (BGRA, matching OpenCV convention).
        canvas = np.zeros((height, width, 4), dtype=np.uint8)

        result = self._face_result
        if result is None or not result.face_detected:
            return cv2.cvtColor(canvas, cv2.COLOR_BGRA2RGBA)

        cx, cy, rw, roll = self._compute_anchor(
            result.landmarks,
            width,
            height,
            result.head_pose.roll,
        )

        idx = self._get_index()
        if not self._sprites or idx >= len(self._sprites):
            return cv2.cvtColor(canvas, cv2.COLOR_BGRA2RGBA)

        sprite = self._sprites[idx]
        sh, sw = sprite.shape[:2]
        if sw == 0 or sh == 0:
            return cv2.cvtColor(canvas, cv2.COLOR_BGRA2RGBA)

        # Scale sprite to the computed render width (preserve aspect).
        rh = max(1, int(rw * sh / sw))
        scaled = cv2.resize(sprite, (rw, rh), interpolation=cv2.INTER_AREA)

        # Rotate around the sprite centre by the head roll angle.
        # Negate roll: cv2 positive angle = counter-clockwise in image
        # coordinates (y-down), which is opposite to MediaPipe's roll
        # convention, so we flip the sign to match head tilt direction.
        rot_mat = cv2.getRotationMatrix2D(
            (rw / 2.0, rh / 2.0), -roll, 1.0
        )
        # Expand bounding box to hold the rotated sprite without clipping.
        cos_a = abs(rot_mat[0, 0])
        sin_a = abs(rot_mat[0, 1])
        new_w = int(rh * sin_a + rw * cos_a)
        new_h = int(rh * cos_a + rw * sin_a)
        rot_mat[0, 2] += new_w / 2.0 - rw / 2.0
        rot_mat[1, 2] += new_h / 2.0 - rh / 2.0
        rotated = cv2.warpAffine(
            scaled, rot_mat, (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        # Compute top-left corner so the sprite is centred on (cx, cy).
        rx, ry = new_w, new_h
        x0 = cx - rx // 2
        y0 = cy - ry // 2
        x1 = x0 + rx
        y1 = y0 + ry

        # Clip sprite region to frame bounds.
        sx0 = max(0, -x0)
        sy0 = max(0, -y0)
        sx1 = rx - max(0, x1 - width)
        sy1 = ry - max(0, y1 - height)
        dx0 = max(0, x0)
        dy0 = max(0, y0)
        dx1 = dx0 + (sx1 - sx0)
        dy1 = dy0 + (sy1 - sy0)

        if sx1 <= sx0 or sy1 <= sy0:
            return cv2.cvtColor(canvas, cv2.COLOR_BGRA2RGBA)

        src_crop = rotated[sy0:sy1, sx0:sx1]
        dst_crop = canvas[dy0:dy1, dx0:dx1]

        # Alpha-blend sprite onto canvas.
        alpha = src_crop[:, :, 3:4].astype(np.float32) / 255.0
        canvas[dy0:dy1, dx0:dx1, :3] = (
            src_crop[:, :, :3] * alpha
            + dst_crop[:, :, :3] * (1.0 - alpha)
        ).astype(np.uint8)
        canvas[dy0:dy1, dx0:dx1, 3] = np.clip(
            src_crop[:, :, 3].astype(np.int32)
            + dst_crop[:, :, 3].astype(np.int32),
            0, 255,
        ).astype(np.uint8)

        # Convert BGRA → RGBA for GPU upload.
        return cv2.cvtColor(canvas, cv2.COLOR_BGRA2RGBA)

    def _update_overlay_texture(
        self,
        device: wgpu.GPUDevice,
        width: int,
        height: int,
    ) -> None:
        """
        Draw the current frame's moustache and upload it to the GPU.

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            width (int): Frame width in pixels.
            height (int): Frame height in pixels.
        """
        rgba = self._draw_moustache(width, height)
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
        Record the alpha-blended moustache overlay blit.

        Uses ``load_op="load"`` to preserve the blit result underneath.

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
                    "load_op": "load",
                    "store_op": "store",
                }
            ]
        )
        pass_enc.set_pipeline(self._overlay_pipeline)
        pass_enc.set_bind_group(0, bind_group, [], 0, 0)
        pass_enc.draw(3, 1, 0, 0)
        pass_enc.end()
