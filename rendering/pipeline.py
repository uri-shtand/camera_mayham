"""
GPU rendering pipeline.

Owns the wgpu canvas surface, device, and all render pass objects.
The application orchestrator calls :py:meth:`RenderPipeline.render_frame`
once per main-loop iteration with the current :py:class:`AppState`.

Pipeline order per frame (§4.1):
    1. Background pass  — upload camera frame → ping-pong tex[0]
    2. Filter chain     — tex[0] → filters → tex[n]
    3. Overlay pass     — composite 3D model onto tex[n]
    4. Game pass        — composite game elements onto tex[n]
    5. Post pass        — blit tex[n] → swap chain surface

Architecture notes
------------------
* All textures used between passes are in ``rgba8unorm`` format.
* The swap chain surface may be in a different format (e.g.,
  ``bgra8unorm-srgb`` on Windows DX12); the post pass shader targets
  the surface format directly.
* Frame timing is measured via Python's ``time.perf_counter``;
  GPU-side timestamps are out of scope for the initial release.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional

import numpy as np
import wgpu
from rendercanvas import BaseRenderCanvas

if TYPE_CHECKING:
    from app.state import AppState

from rendering.passes.background import BackgroundPass
from rendering.passes.filter_pass import FilterPass
from rendering.passes.game_pass import GamePass
from rendering.passes.overlay_pass import OverlayPass
from rendering.passes.post_pass import PostPass

logger = logging.getLogger(__name__)

# Bytes-per-row in GPU texture copies must be a multiple of this value.
_COPY_BYTES_PER_ROW_ALIGN: int = 256


class RenderPipeline:
    """
    Manages the wgpu device and orchestrates all render passes
    (§4.1, CON-001 – CON-003).

    Two operating modes are supported:

    **canvas mode** (``canvas`` provided)
        The pipeline renders to the swap-chain surface of the supplied
        ``BaseRenderCanvas`` window and presents each frame on screen.

    **offscreen mode** (``canvas=None``)
        The pipeline renders into an intermediate ``rgba8unorm`` texture.
        After each frame the pixels are copied to CPU memory and stored in
        :py:attr:`latest_pixels` as an ``(H, W, 4)`` uint8 NumPy array.  The
        caller reads ``latest_pixels`` each iteration to obtain the composited
        frame for display in any host UI (e.g. a single DPG window).

    Usage::

        # Offscreen (single-window DPG) mode:
        pipeline = RenderPipeline(canvas=None)
        pipeline.setup(state)

        while state.running:
            pixels = pipeline.render_frame(state)
            # pixels is an (H, W, 4) uint8 ndarray (RGBA)

        pipeline.teardown(state)

    Parameters:
        canvas: A ``BaseRenderCanvas`` instance that provides the native
                surface for swap-chain presentation, or ``None`` for
                offscreen rendering.
    """

    def __init__(self, canvas: Optional[BaseRenderCanvas] = None) -> None:
        """
        Store the canvas reference (does not touch the GPU yet).

        Parameters:
            canvas: The rendercanvas canvas/window that owns the swap chain,
                    or ``None`` to run in offscreen mode.
        """
        self._canvas = canvas
        self._device: Optional[wgpu.GPUDevice] = None
        self._adapter: Optional[wgpu.GPUAdapter] = None
        self._context: Optional[wgpu.GPUCanvasContext] = None
        self._surface_format: Optional[wgpu.TextureFormat] = None
        # Offscreen output
        self._output_texture: Optional[wgpu.GPUTexture] = None
        self._output_bytes_per_row: int = 0
        self.latest_pixels: Optional[np.ndarray] = None
        """Latest rendered frame as an (H, W, 4) uint8 RGBA array (offscreen
        mode only).  Updated after each :py:meth:`render_frame` call."""

        # Intermediate pipeline texture format — all passes except the
        # final blit use this format (rgba8unorm is universally supported
        # for sampling and as a render attachment on all adapters).
        self._pipeline_format: wgpu.TextureFormat = (
            wgpu.TextureFormat.rgba8unorm
        )

        # Render passes
        self._bg_pass = BackgroundPass()
        self._filter_pass = FilterPass()
        self._overlay_pass = OverlayPass()
        self._game_pass = GamePass()
        self._post_pass = PostPass()

        self._last_frame_time: float = 0.0
        self._is_ready: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self, state: AppState) -> None:
        """
        Initialise the wgpu adapter, device, swap chain, and all passes.

        Must be called once from the main thread before entering the
        render loop.

        Parameters:
            state (AppState): Current application state.  Used to wire
                filters, overlay, and game to their GPU resources.

        Raises:
            RuntimeError: If no compatible GPU adapter is found.
        """
        # Request a high-performance adapter (AMD / Nvidia preferred).
        # canvas is optional: pass it only in canvas mode so wgpu can
        # match the adapter to the display surface.
        adapter_kwargs: dict = {"power_preference": "high-performance"}
        if self._canvas is not None:
            adapter_kwargs["canvas"] = self._canvas
        self._adapter = wgpu.gpu.request_adapter_sync(**adapter_kwargs)
        if self._adapter is None:  # pragma: no cover
            raise RuntimeError(
                "No compatible WebGPU adapter found.  "
                "Ensure a supported GPU and driver are installed."
            )

        logger.info(
            "GPU adapter: %s (backend: %s)",
            self._adapter.info.get("device", "unknown"),
            self._adapter.info.get("backend_type", "unknown"),
        )

        self._device = self._adapter.request_device_sync()

        w, h = state.camera_width, state.camera_height

        if self._canvas is not None:
            # Canvas mode: configure the swap-chain surface.
            self._context = self._canvas.get_context("wgpu")
            self._surface_format = self._context.get_preferred_format(
                self._adapter
            )
            self._context.configure(
                device=self._device,
                format=self._surface_format,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            )
            output_format = self._surface_format
        else:
            # Offscreen mode: create an RGBA output texture for readback.
            output_format = wgpu.TextureFormat.rgba8unorm
            self._output_texture = self._device.create_texture(
                size=(w, h, 1),
                format=output_format,
                usage=(
                    wgpu.TextureUsage.RENDER_ATTACHMENT
                    | wgpu.TextureUsage.COPY_SRC
                ),
            )
            # Align bytes_per_row to GPU copy requirement (256-byte boundary)
            unaligned = w * 4
            remainder = unaligned % _COPY_BYTES_PER_ROW_ALIGN
            self._output_bytes_per_row = (
                unaligned
                if remainder == 0
                else unaligned + (_COPY_BYTES_PER_ROW_ALIGN - remainder)
            )
            logger.info(
                "RenderPipeline offscreen mode — %dx%d, "
                "bytes_per_row=%d",
                w, h, self._output_bytes_per_row,
            )

        # Set up each render pass
        self._bg_pass.setup(
            self._device, w, h, self._pipeline_format
        )
        self._filter_pass.setup(
            self._device, w, h, self._pipeline_format, state.filters
        )
        self._overlay_pass.setup(
            self._device, w, h, self._pipeline_format,
            state.active_overlay,
        )
        self._game_pass.setup(
            self._device, w, h, self._pipeline_format,
            state.active_game,
        )
        self._post_pass.setup(self._device, output_format)

        self._last_frame_time = time.perf_counter()
        self._is_ready = True
        logger.info(
            "RenderPipeline ready — %dx%d, output=%s",
            w, h, output_format,
        )

    def teardown(self, state: AppState) -> None:
        """
        Release all GPU resources in reverse setup order.

        Parameters:
            state (AppState): Current application state (needed to
                tear down per-filter and per-game resources).
        """
        self._post_pass.teardown()
        self._game_pass.teardown(state.active_game)
        self._overlay_pass.teardown(state.active_overlay)
        self._filter_pass.teardown(state.filters)
        self._bg_pass.teardown()
        self._output_texture = None
        self._context = None
        self._device = None
        self._adapter = None
        self._is_ready = False
        logger.info("RenderPipeline torn down.")

    # ------------------------------------------------------------------
    # Per-frame rendering
    # ------------------------------------------------------------------

    def render_frame(self, state: AppState) -> None:
        """
        Render one complete frame from the current application state.

        Called once per main-loop iteration.  The method:

        1. Computes delta time.
        2. Uploads the current camera frame to the GPU.
        3. Builds a command encoder and records all render passes.
        4. Submits the encoder to the queue.
        5. Presents the swap chain surface.
        6. Updates ``state.last_frame_time_ms`` and ``state.frame_count``.

        Parameters:
            state (AppState): Shared application state providing the
                camera frame, face result, active filters, overlay,
                and game.

        Returns:
            Optional[np.ndarray]: In offscreen mode, returns the rendered
                frame as an ``(H, W, 4)`` uint8 RGBA array (also stored in
                :py:attr:`latest_pixels`).  In canvas mode, returns ``None``
                (the frame is displayed directly on the swap-chain surface).
        """
        if not self._is_ready:
            return None

        now = time.perf_counter()
        dt = now - self._last_frame_time
        self._last_frame_time = now

        # -- Upload camera frame -------------------------------------
        if state.camera_frame is not None:
            self._bg_pass.upload_frame(state.camera_frame)

        # -- Determine output surface view ---------------------------
        if self._canvas is not None:
            # Canvas mode: acquire the swap-chain texture.
            surface_texture = self._context.get_current_texture()
            output_view = surface_texture.create_view()
        else:
            # Offscreen mode: render into the pre-allocated output texture.
            output_view = self._output_texture.create_view()

        # -- Build command encoder -----------------------------------
        encoder = self._device.create_command_encoder()

        # Pass 1: background — camera frame → ping-pong tex[1]
        self._bg_pass.record(
            encoder,
            self._filter_pass.textures[1],
        )

        # Inject face tracking data into face-aware filters (e.g.
        # FaceLandmarkFilter) before the filter chain records commands.
        # Duck-typing is used so BaseFilter's interface remains unchanged.
        for flt in state.enabled_filters():
            if hasattr(flt, "update_face_result"):
                flt.update_face_result(state.face_result)

        # Pass 2: filter chain — ping-pong tex[1] → filtered texture
        filtered_tex = self._filter_pass.record(
            encoder,
            self._filter_pass.textures[1],
            state.enabled_filters(),
        )

        # Pass 3: overlay — composite 3D model on top
        self._overlay_pass.record(
            encoder,
            filtered_tex,
            state.active_overlay,
            state.face_result,
        )

        # Pass 4: game pass — composite game elements
        self._game_pass.record(
            encoder,
            filtered_tex,
            state.active_game,
            state.face_result,
            dt,
        )

        # Pass 5: post-processing — blit to output surface/texture
        self._post_pass.record(encoder, filtered_tex, output_view)

        # -- Submit --------------------------------------------------
        self._device.queue.submit([encoder.finish()])

        # -- Present or read back pixels -----------------------------
        if self._canvas is not None:
            self._context._rc_present(force_sync=True)
            pixels: Optional[np.ndarray] = None
        else:
            pixels = self._readback_pixels(state)
            self.latest_pixels = pixels

        # -- Update diagnostics -------------------------------------
        frame_ms = (time.perf_counter() - now) * 1000.0
        state.last_frame_time_ms = frame_ms
        state.frame_count += 1
        return pixels

    # ------------------------------------------------------------------
    # Offscreen helpers
    # ------------------------------------------------------------------

    def _readback_pixels(
        self, state: "AppState"
    ) -> Optional[np.ndarray]:
        """
        Copy the output texture to CPU memory after a submitted frame.

        The copy uses ``queue.read_texture`` which is synchronous (it
        waits for GPU work to complete).  The result is an
        ``(H, W, 4)`` uint8 RGBA array.

        Parameters:
            state (AppState): Provides width/height dimensions.

        Returns:
            Optional[np.ndarray]: Pixel data, or ``None`` if not in
                offscreen mode.
        """
        if self._output_texture is None or self._device is None:
            return None

        w = state.camera_width
        h = state.camera_height
        bpr = self._output_bytes_per_row  # aligned bytes per row

        raw: memoryview = self._device.queue.read_texture(
            {
                "texture": self._output_texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            {
                "offset": 0,
                "bytes_per_row": bpr,
                "rows_per_image": h,
            },
            (w, h, 1),
        )

        # The readback buffer may have row padding; strip it if necessary.
        if bpr == w * 4:
            # No padding — reshape directly.
            frame = (
                np.frombuffer(raw, dtype=np.uint8)
                .reshape(h, w, 4)
                .copy()
            )
        else:
            # Padded rows: copy only the valid bytes of each row.
            buf = np.frombuffer(raw, dtype=np.uint8).reshape(h, bpr)
            frame = np.ascontiguousarray(buf[:, : w * 4]).reshape(
                h, w, 4
            )
        return frame
