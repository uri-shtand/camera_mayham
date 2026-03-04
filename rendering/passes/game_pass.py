"""
Game render pass.

Renders the active mini-game's visual elements on top of the composited
camera + overlay image (REQ-009, §4.1 'Game pass').  Uses load_op='load'
so the game is drawn over the existing frame contents.
"""

from __future__ import annotations

import logging
from typing import Optional

import wgpu

from games.base import BaseGame, GameState
from tracking.face_tracker import FaceTrackResult

logger = logging.getLogger(__name__)


class GamePass:
    """
    Composites the active mini-game over the current frame
    (§4.1, REQ-009).

    Like :py:class:`OverlayPass`, this pass uses ``load_op='load'`` on
    the colour attachment so it draws on top of whatever the previous
    passes produced.
    """

    def __init__(self) -> None:
        """Initialise the game pass descriptor."""
        self._device: Optional[wgpu.GPUDevice] = None
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
        game: Optional[BaseGame],
    ) -> None:
        """
        Set up the active game's GPU resources.

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            width (int): Render target width in pixels.
            height (int): Render target height in pixels.
            texture_format (wgpu.TextureFormat): Colour target format.
            game (Optional[BaseGame]): The game to set up, or None.
        """
        self._device = device
        self._width = width
        self._height = height

        if game is not None:
            game.setup(device, texture_format)

        logger.debug("GamePass ready.")

    def teardown(self, game: Optional[BaseGame]) -> None:
        """
        Release GPU resources held by the game.

        Parameters:
            game (Optional[BaseGame]): Active game, or None.
        """
        if game is not None:
            game.teardown()
        self._device = None

    # ------------------------------------------------------------------
    # Per-frame record
    # ------------------------------------------------------------------

    def record(
        self,
        encoder: wgpu.GPUCommandEncoder,
        base_texture: wgpu.GPUTexture,
        game: Optional[BaseGame],
        face_result: Optional[FaceTrackResult],
        dt: float,
    ) -> None:
        """
        Update game logic and record draw calls into ``base_texture``.

        Calls :py:meth:`BaseGame.update` before opening the render pass
        so game state is up to date for this frame.

        Parameters:
            encoder (wgpu.GPUCommandEncoder): Current frame command
                encoder.
            base_texture (wgpu.GPUTexture): Composited frame texture.
            game (Optional[BaseGame]): Active game, or None.
            face_result (Optional[FaceTrackResult]): Latest tracking
                data for game input.
            dt (float): Delta time in seconds since the last frame.
        """
        if game is None or game.state != GameState.RUNNING:
            return

        # Update game logic before recording GPU commands
        if face_result is not None:
            game.update(face_result, dt)
        else:
            from tracking.face_tracker import FaceTrackResult  # noqa: PLC0415

            game.update(FaceTrackResult(), dt)

        # Open render pass — load_op='load' preserves the frame below
        pass_enc = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": base_texture.create_view(),
                    "load_op": "load",
                    "store_op": "store",
                }
            ]
        )
        game.render(pass_enc, self._width, self._height)
        pass_enc.end()
