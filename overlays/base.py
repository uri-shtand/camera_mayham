"""
3D overlay base class.

A BaseOverlay renders a 3D model attached to facial landmarks.  The
rendering pipeline calls ``setup`` once, then ``render`` each frame
whenever a face is detected, passing the latest head-pose data so the
overlay can update its model-view matrix before drawing (REQ-007).
"""

from __future__ import annotations

import abc
from typing import Any, Optional

import wgpu

from tracking.face_tracker import FaceTrackResult


class BaseOverlay(abc.ABC):
    """
    Abstract base for head-tracked 3D overlays (§4.4, REQ-007).

    Subclasses must implement:
    * :py:attr:`name`             — unique identifier
    * :py:meth:`_load_geometry`   — load mesh data at setup time
    * :py:meth:`render`           — issue GPU draw calls for one frame

    The overlay pass in the rendering pipeline calls :py:meth:`render`
    inside an already-open render pass encoder: overlays must NOT begin
    or end render passes themselves.
    """

    def __init__(self) -> None:
        """Initialise shared overlay state."""
        self._device: Optional[wgpu.GPUDevice] = None
        self._pipeline: Any = None

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Unique human-readable overlay identifier.

        Returns:
            str: Overlay name (e.g. ``"CoolHat"``).
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
        Load mesh data and compile the depth-aware render pipeline.

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            texture_format (wgpu.TextureFormat): Target texture format.
        """
        self._device = device
        self._load_geometry(device, texture_format)

    @abc.abstractmethod
    def _load_geometry(
        self,
        device: wgpu.GPUDevice,
        texture_format: wgpu.TextureFormat,
    ) -> None:
        """
        Load vertex/index buffers and build the GPU pipeline.

        Called once by :py:meth:`setup`.

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            texture_format (wgpu.TextureFormat): Target texture format.
        """

    def teardown(self) -> None:
        """Release all GPU resources held by this overlay."""
        self._pipeline = None
        self._device = None

    # ------------------------------------------------------------------
    # Per-frame rendering
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def render(
        self,
        pass_encoder: wgpu.GPURenderPassEncoder,
        face_result: FaceTrackResult,
        viewport_width: int,
        viewport_height: int,
    ) -> None:
        """
        Issue GPU draw calls for the current frame inside an open render
        pass.

        Parameters:
            pass_encoder (wgpu.GPURenderPassEncoder): Active render pass
                encoder provided by the overlay pass.
            face_result (FaceTrackResult): Latest face tracking data
                used to compute the model-view transform.
            viewport_width (int): Render target width in pixels.
            viewport_height (int): Render target height in pixels.
        """

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return f"<{self.__class__.__name__} name={self.name!r}>"
