"""
Filter chain render pass.

Executes the ordered list of enabled filters using a ping-pong texture
ping-pong pattern: each filter reads from one texture and writes to the
other.  After all filters run, the final output texture is exposed as
``output_texture`` for the next pass.

Textures are allocated once at setup time at the camera/render
resolution (CON-002).  Filters write directly into pre-allocated
textures — no per-frame GPU allocation occurs.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import wgpu

from filters.base import BaseFilter

logger = logging.getLogger(__name__)


class FilterPass:
    """
    Orchestrates the GPU filter chain using ping-pong textures (§4.1,
    REQ-002, REQ-003).

    Attributes:
        textures (list[wgpu.GPUTexture]): Two RGBA8 render-target textures
            at the camera resolution.  Index 0 receives the background
            blit; the filter chain alternates between them.
        output_index (int): Index of the texture holding the final
            filtered output after :py:meth:`record` completes.
    """

    def __init__(self) -> None:
        """Initialise the filter pass descriptor."""
        self._device: Optional[wgpu.GPUDevice] = None
        self._texture_format: Optional[wgpu.TextureFormat] = None
        self.textures: List[wgpu.GPUTexture] = []
        self.output_index: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(
        self,
        device: wgpu.GPUDevice,
        width: int,
        height: int,
        texture_format: wgpu.TextureFormat,
        filters: List[BaseFilter],
    ) -> None:
        """
        Allocate ping-pong textures and set up each filter.

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            width (int): Render target width in pixels.
            height (int): Render target height in pixels.
            texture_format (wgpu.TextureFormat): Pipeline texture format.
            filters (List[BaseFilter]): All registered filters; each
                receives ``setup()`` here so shaders are compiled once.
        """
        self._device = device
        self._texture_format = texture_format

        tex_usage = (
            wgpu.TextureUsage.TEXTURE_BINDING
            | wgpu.TextureUsage.RENDER_ATTACHMENT
            | wgpu.TextureUsage.COPY_SRC
        )

        self.textures = [
            device.create_texture(
                size=(width, height, 1),
                format=texture_format,
                usage=tex_usage,
            )
            for _ in range(2)
        ]

        for flt in filters:
            flt.setup(device, texture_format)

        logger.debug("FilterPass ready — %d filter(s) set up.", len(filters))

    def teardown(self, filters: List[BaseFilter]) -> None:
        """
        Release filter GPU resources.

        Parameters:
            filters (List[BaseFilter]): All registered filter instances.
        """
        for flt in filters:
            flt.teardown()
        self.textures = []
        self._device = None

    # ------------------------------------------------------------------
    # Per-frame record
    # ------------------------------------------------------------------

    def record(
        self,
        encoder: wgpu.GPUCommandEncoder,
        input_texture: wgpu.GPUTexture,
        enabled_filters: List[BaseFilter],
    ) -> wgpu.GPUTexture:
        """
        Apply all enabled filters sequentially and return the final
        output texture.

        If no filters are enabled the input texture is returned
        unchanged.

        Parameters:
            encoder (wgpu.GPUCommandEncoder): Current frame command
                encoder.
            input_texture (wgpu.GPUTexture): Texture populated by the
                background pass.
            enabled_filters (List[BaseFilter]): Ordered list of filters
                to apply (already filtered for enabled state).

        Returns:
            wgpu.GPUTexture: The texture holding the final filtered
            output.
        """
        if not enabled_filters:
            # No active filters — return the input unchanged
            return input_texture

        current_input = input_texture
        ping = 0  # index into self.textures

        for flt in enabled_filters:
            output = self.textures[ping]
            flt.apply(encoder, current_input, output)
            current_input = output
            ping ^= 1  # toggle 0 ↔ 1

        return current_input
