"""
3D overlay render pass.

Renders face-tracked 3D models on top of the filtered camera frame
(REQ-007, §4.1 '3D overlay pass').  The pass opens a single render pass
targeting the post-filter texture (load_op='load') so it composites on
top of what the filter chain produced.

The overlay model uses a depth attachment to handle self-occlusion on
complex meshes.
"""

from __future__ import annotations

import logging
from typing import Optional

import wgpu

from overlays.base import BaseOverlay
from tracking.face_tracker import FaceTrackResult

logger = logging.getLogger(__name__)


class OverlayPass:
    """
    Composites the active 3D overlay on top of the filtered frame
    (§4.1, REQ-007).

    The pass renders directly into ``base_texture`` (load_op='load'),
    preserving the filtered camera image and drawing the 3D model
    above it.
    """

    def __init__(self) -> None:
        """Initialise the overlay pass descriptor."""
        self._device: Optional[wgpu.GPUDevice] = None
        self._depth_texture: Optional[wgpu.GPUTexture] = None
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
        overlay: Optional[BaseOverlay],
    ) -> None:
        """
        Allocate the depth buffer and set up the overlay model.

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            width (int): Render target width in pixels.
            height (int): Render target height in pixels.
            texture_format (wgpu.TextureFormat): Colour target format.
            overlay (Optional[BaseOverlay]): The overlay to set up, or
                None for no overlay.
        """
        self._device = device
        self._width = width
        self._height = height

        self._depth_texture = device.create_texture(
            size=(width, height, 1),
            format=wgpu.TextureFormat.depth24plus,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        )

        if overlay is not None:
            overlay.setup(device, texture_format)

        logger.debug("OverlayPass ready (%dx%d).", width, height)

    def teardown(self, overlay: Optional[BaseOverlay]) -> None:
        """
        Release GPU depth buffer and overlay resources.

        Parameters:
            overlay (Optional[BaseOverlay]): Active overlay, or None.
        """
        if overlay is not None:
            overlay.teardown()
        self._depth_texture = None
        self._device = None

    # ------------------------------------------------------------------
    # Per-frame record
    # ------------------------------------------------------------------

    def record(
        self,
        encoder: wgpu.GPUCommandEncoder,
        base_texture: wgpu.GPUTexture,
        overlay: Optional[BaseOverlay],
        face_result: Optional[FaceTrackResult],
    ) -> None:
        """
        Record overlay draw calls into ``base_texture``.

        If no overlay is active or no face is detected, the pass is
        a no-op.

        Parameters:
            encoder (wgpu.GPUCommandEncoder): Current frame command
                encoder.
            base_texture (wgpu.GPUTexture): Texture with the filtered
                camera frame (written by the filter chain).  The render
                pass loads and preserves its contents.
            overlay (Optional[BaseOverlay]): The active overlay model,
                or None.
            face_result (Optional[FaceTrackResult]): Latest tracking
                data.  If None or face not detected, the overlay is
                skipped.
        """
        if (
            overlay is None
            or face_result is None
            or not face_result.face_detected
        ):
            return

        pass_enc = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": base_texture.create_view(),
                    "load_op": "load",    # preserve filtered background
                    "store_op": "store",
                }
            ],
            depth_stencil_attachment={
                "view": self._depth_texture.create_view(),
                "depth_load_op": "clear",
                "depth_store_op": "discard",
                "depth_clear_value": 1.0,
                "stencil_load_op": "clear",
                "stencil_store_op": "discard",
                "stencil_clear_value": 0,
            },
        )

        overlay.render(pass_enc, face_result, self._width, self._height)
        pass_enc.end()
