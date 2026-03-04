"""
Mini-game base class.

Mini-games are self-contained interactive modules driven by facial
blendshape and head-pose input (REQ-008, REQ-009, §4.5).  The rendering
pipeline calls ``update`` once per frame (game logic) and ``render``
inside the open game render pass so the game can issue its own draw
calls.

Mini-game lifecycle
-------------------
1. :py:meth:`setup`   — called once after the wgpu device is ready.
2. :py:meth:`start`   — called when the user launches the game from the
                         widget panel.  May also be called again to
                         restart a finished game.
3. :py:meth:`update`  — called every frame while the game is active.
4. :py:meth:`render`  — called every frame inside the game render pass.
5. :py:meth:`stop`    — called when the user quits or another game
                         launches.
6. :py:meth:`teardown`— called once when the application shuts down.
"""

from __future__ import annotations

import abc
from enum import Enum, auto
from typing import Any, Optional

import wgpu

from tracking.face_tracker import FaceTrackResult


class GameState(Enum):
    """Represents the current lifecycle state of a mini-game."""

    IDLE = auto()
    """Game has been set up but not yet started."""
    RUNNING = auto()
    """Game is active and receiving input."""
    FINISHED = auto()
    """Game ended (win/lose condition met); awaiting restart or stop."""


class BaseGame(abc.ABC):
    """
    Abstract base for facial-input mini-games (§4.5, REQ-008).

    Subclasses must implement:
    * :py:attr:`name`            — unique identifier
    * :py:meth:`_on_start`       — game-specific start / reset logic
    * :py:meth:`_on_stop`        — game-specific stop / cleanup logic
    * :py:meth:`_setup_pipeline` — compile shaders at setup time
    * :py:meth:`update`          — per-frame logic
    * :py:meth:`render`          — per-frame draw calls
    """

    def __init__(self) -> None:
        """Initialise shared game state."""
        self._device: Optional[wgpu.GPUDevice] = None
        self.state: GameState = GameState.IDLE
        self.score: int = 0

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Unique human-readable game identifier.

        Returns:
            str: Game name (e.g. ``"BubblePop"``).
        """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(
        self,
        device: wgpu.GPUDevice,
        texture_format: wgpu.TextureFormat,
    ) -> None:
        """
        Compile shaders and allocate GPU resources for the game.

        Called once by the render pipeline after the device is ready.

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            texture_format (wgpu.TextureFormat): Target texture format.
        """
        self._device = device
        self._setup_pipeline(device, texture_format)

    @abc.abstractmethod
    def _setup_pipeline(
        self,
        device: wgpu.GPUDevice,
        texture_format: wgpu.TextureFormat,
    ) -> None:
        """
        Compile shaders and build GPU pipeline objects.

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            texture_format (wgpu.TextureFormat): Target texture format.
        """

    def start(self) -> None:
        """
        Start or restart the game.

        Resets score and transitions to RUNNING state before calling
        game-specific :py:meth:`_on_start`.
        """
        self.score = 0
        self.state = GameState.RUNNING
        self._on_start()

    @abc.abstractmethod
    def _on_start(self) -> None:
        """Game-specific initialisation called by :py:meth:`start`."""

    def stop(self) -> None:
        """
        Stop the game and release any per-session resources.

        Transitions to IDLE state and calls :py:meth:`_on_stop`.
        """
        self.state = GameState.IDLE
        self._on_stop()

    @abc.abstractmethod
    def _on_stop(self) -> None:
        """Game-specific cleanup called by :py:meth:`stop`."""

    def teardown(self) -> None:
        """Release all GPU resources held by this game."""
        self._device = None

    # ------------------------------------------------------------------
    # Per-frame hooks
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def update(self, face_result: FaceTrackResult, dt: float) -> None:
        """
        Advance game logic for one frame.

        Called before the render pass for this frame.  Must not
        allocate GPU resources.

        Parameters:
            face_result (FaceTrackResult): Latest face tracking data
                used as game input (mouth open, head pose, etc.).
            dt (float): Elapsed seconds since the previous frame.
        """

    @abc.abstractmethod
    def render(
        self,
        pass_encoder: wgpu.GPURenderPassEncoder,
        viewport_width: int,
        viewport_height: int,
    ) -> None:
        """
        Issue GPU draw calls for this frame inside an open render pass.

        Parameters:
            pass_encoder (wgpu.GPURenderPassEncoder): Active render pass
                encoder; must NOT be ended by this method.
            viewport_width (int): Render target width in pixels.
            viewport_height (int): Render target height in pixels.
        """

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"<{self.__class__.__name__} name={self.name!r} "
            f"state={self.state.name} score={self.score}>"
        )
