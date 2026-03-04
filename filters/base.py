"""
Filter base class.

All GPU shader-based visual filters inherit from BaseFilter.  The
rendering pipeline calls ``setup`` once on pipeline initialisation and
``apply`` on every rendered frame for each enabled filter.  Filters
must be self-contained: the pipeline does not assume any shared GPU
state between filter invocations.

Design decisions
----------------
* Filters expose a ``params`` dict of named float/int/colour values so
  the widget panel can dynamically render controls for any filter
  without knowing its internals (REQ-004, §4.4).
* ``setup`` and ``teardown`` bracket the WebGPU device lifetime.
* ``apply`` receives a command encoder and two textures; the filter
  writes its output into ``output_texture``, leaving ``input_texture``
  unmodified (ping-pong pattern, CON-002).
"""

from __future__ import annotations

import abc
from typing import Any, Dict

import wgpu


class BaseFilter(abc.ABC):
    """
    Abstract base for all GPU shader-based visual filters (§4.4).

    Subclasses must implement:
    * :py:meth:`name`      (property)
    * :py:meth:`_build_pipeline` — compile wgpu pipeline at setup time
    * :py:meth:`apply`     — record GPU commands for one frame

    Attributes:
        enabled (bool): When False the pipeline skips this filter.
        params  (dict):  Named runtime-adjustable parameters exposed
                         to the widget panel.
    """

    def __init__(self) -> None:
        """Initialise shared filter state (device assigned during setup)."""
        self.enabled: bool = True
        self.params: Dict[str, Any] = {}
        self._device: wgpu.GPUDevice | None = None
        self._pipeline: Any = None  # wgpu.GPURenderPipeline or Compute
        self._bind_group_layout: Any = None
        self._sampler: Any = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Unique human-readable filter identifier.

        Returns:
            str: Filter name (e.g. ``"Grayscale"``).
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
        Compile shaders and create pipeline resources on the GPU.

        Called once by the rendering pipeline after the wgpu device is
        available.  Must not perform any per-frame allocations.

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            texture_format (wgpu.TextureFormat): The surface / swap-chain
                format so the pipeline output matches expectations.
        """
        self._device = device
        self._sampler = device.create_sampler(
            address_mode_u="clamp-to-edge",
            address_mode_v="clamp-to-edge",
            mag_filter="linear",
            min_filter="linear",
        )
        self._build_pipeline(device, texture_format)

    @abc.abstractmethod
    def _build_pipeline(
        self,
        device: wgpu.GPUDevice,
        texture_format: wgpu.TextureFormat,
    ) -> None:
        """
        Compile shaders and create wgpu pipeline objects.

        Called once from :py:meth:`setup`.  Implementations must store
        any pipeline objects they need in ``self._pipeline`` (or named
        attributes).

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            texture_format (wgpu.TextureFormat): Target texture format.
        """

    def teardown(self) -> None:
        """
        Release all GPU resources held by this filter.

        The default implementation clears references; subclasses may
        override to destroy GPU objects explicitly if wgpu requires it.
        """
        self._pipeline = None
        self._bind_group_layout = None
        self._sampler = None
        self._device = None

    # ------------------------------------------------------------------
    # Per-frame application
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def apply(
        self,
        encoder: wgpu.GPUCommandEncoder,
        input_texture: wgpu.GPUTexture,
        output_texture: wgpu.GPUTexture,
    ) -> None:
        """
        Record GPU draw/compute commands for one frame.

        Must read from ``input_texture`` and write to ``output_texture``.
        Must not allocate new GPU buffers or textures (CON-004).

        Parameters:
            encoder (wgpu.GPUCommandEncoder): Current frame command encoder.
            input_texture (wgpu.GPUTexture): Texture to read from.
            output_texture (wgpu.GPUTexture): Texture to write into.
        """

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def set_param(self, key: str, value: Any) -> None:
        """
        Update a runtime parameter by name.

        Parameters:
            key (str): Parameter name.
            value (Any): New parameter value.

        Raises:
            KeyError: If the parameter name is not registered for this
                      filter.
        """
        if key not in self.params:
            raise KeyError(
                f"Filter '{self.name}' has no parameter '{key}'. "
                f"Available: {list(self.params.keys())}"
            )
        self.params[key] = value

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"<{self.__class__.__name__} name={self.name!r} "
            f"enabled={self.enabled}>"
        )
