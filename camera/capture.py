"""
Camera capture module.

Wraps OpenCV's VideoCapture in a clean, type-annotated interface that
the application orchestrator calls once per frame.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraCapture:
    """
    Manages a single webcam device via OpenCV (REQ-001, EXT-001).

    Usage::

        cam = CameraCapture(device_id=0, width=1280, height=720)
        cam.open()
        frame = cam.read()   # ndarray (H, W, 3) BGR uint8, or None
        cam.close()
    """

    def __init__(
        self,
        device_id: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
    ) -> None:
        """
        Initialise the capture descriptor (does not open the device).

        Parameters:
            device_id (int): OS camera index passed to cv2.VideoCapture.
            width (int): Requested frame width in pixels.
            height (int): Requested frame height in pixels.
            fps (int): Requested capture frame rate.
        """
        self._device_id = device_id
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: Optional[cv2.VideoCapture] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        """Actual capture width reported by the device after open()."""
        if self._cap is not None and self._cap.isOpened():
            return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return self._width

    @property
    def height(self) -> int:
        """Actual capture height reported by the device after open()."""
        if self._cap is not None and self._cap.isOpened():
            return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return self._height

    @property
    def fps(self) -> float:
        """Actual FPS reported by the device after open()."""
        if self._cap is not None and self._cap.isOpened():
            return self._cap.get(cv2.CAP_PROP_FPS)
        return float(self._fps)

    @property
    def is_open(self) -> bool:
        """True when the capture device is currently open and ready."""
        return self._cap is not None and self._cap.isOpened()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """
        Open and configure the camera device.

        Raises:
            RuntimeError: If the device cannot be opened (e.g. in use or
                          not present).
        """
        # Use DSHOW backend on Windows for lower-latency capture
        self._cap = cv2.VideoCapture(self._device_id, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            # Fall back to default backend
            self._cap = cv2.VideoCapture(self._device_id)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera device {self._device_id}. "
                "Ensure a webcam is connected and not in use by another "
                "application."
            )

        # Request preferred resolution and frame rate; the driver may
        # round to the nearest supported mode
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "Camera %d opened: %dx%d @ %.1f FPS",
            self._device_id,
            actual_w,
            actual_h,
            actual_fps,
        )

    def read(self) -> Optional[np.ndarray]:
        """
        Grab and decode the next camera frame.

        Returns:
            Optional[np.ndarray]: A (H, W, 3) BGR uint8 array on
            success, or None if the device is not open or the grab
            failed (e.g. device removed mid-session).
        """
        # Edge case: device closed or not yet opened
        if self._cap is None or not self._cap.isOpened():
            logger.warning("read() called on unopened camera device.")
            return None

        ok, frame = self._cap.read()
        if not ok or frame is None:
            logger.warning(
                "Camera %d failed to provide a frame.", self._device_id
            )
            return None

        frame = cv2.rotate(frame, cv2.ROTATE_180)
        return frame  # dtype=uint8, shape=(H, W, 3), BGR

    def close(self) -> None:
        """
        Release the camera device and free driver resources.

        Safe to call even when the device is not open.
        """
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera %d released.", self._device_id)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "CameraCapture":
        """Open the device on context entry."""
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        """Release the device on context exit regardless of exceptions."""
        self.close()
