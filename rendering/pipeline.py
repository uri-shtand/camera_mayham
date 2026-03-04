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


class RenderPipeline:
    """
    Manages the wgpu device and orchestrates all render passes
    (§4.1, CON-001 – CON-003).

    Usage::

        pipeline = RenderPipeline(canvas)
        pipeline.setup(state)

        # main loop:
        while state.running:
            pipeline.render_frame(state)

        pipeline.teardown(state)

    Parameters:
        canvas: A ``BaseRenderCanvas`` instance (from ``rendercanvas``)
                that provides the native surface for swap-chain presentation.
    """

    def __init__(self, canvas: BaseRenderCanvas) -> None:
        """
        Store the canvas reference (does not touch the GPU yet).

        Parameters:
            canvas: The rendercanvas canvas/window that owns the swap chain.
        """
        self._canvas = canvas
        self._device: Optional[wgpu.GPUDevice] = None
        self._adapter: Optional[wgpu.GPUAdapter] = None
        self._context: Optional[wgpu.GPUCanvasContext] = None
        self._surface_format: Optional[wgpu.TextureFormat] = None

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
        # Request a high-performance adapter (AMD / Nvidia preferred)
        self._adapter = wgpu.gpu.request_adapter_sync(
            canvas=self._canvas,
            power_preference="high-performance",
        )
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

        # Configure the canvas surface / swap chain
        self._context = self._canvas.get_context("wgpu")
        self._surface_format = self._context.get_preferred_format(
            self._adapter
        )
        self._context.configure(
            device=self._device,
            format=self._surface_format,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        )

        w, h = state.camera_width, state.camera_height

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
        self._post_pass.setup(self._device, self._surface_format)

        self._last_frame_time = time.perf_counter()
        self._is_ready = True
        logger.info(
            "RenderPipeline ready — %dx%d, surface=%s",
            w, h, self._surface_format,
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
        """
        if not self._is_ready:
            return

        now = time.perf_counter()
        dt = now - self._last_frame_time
        self._last_frame_time = now

        # -- Upload camera frame -------------------------------------
        if state.camera_frame is not None:
            self._bg_pass.upload_frame(state.camera_frame)

        # -- Acquire swap chain texture ------------------------------
        surface_texture = self._context.get_current_texture()
        surface_view = surface_texture.create_view()

        # -- Build command encoder -----------------------------------
        encoder = self._device.create_command_encoder()

        # Pass 1: background — camera frame → ping-pong tex[1]
        self._bg_pass.record(
            encoder,
            self._filter_pass.textures[1],
        )

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

        # Pass 5: post-processing — blit to swap chain
        self._post_pass.record(encoder, filtered_tex, surface_view)

        # -- Submit and present --------------------------------------
        self._device.queue.submit([encoder.finish()])
        self._context._rc_present(force_sync=True)

        # -- Update diagnostics -------------------------------------
        frame_ms = (time.perf_counter() - now) * 1000.0
        state.last_frame_time_ms = frame_ms
        state.frame_count += 1
