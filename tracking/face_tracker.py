"""
Face tracking module.

Wraps MediaPipe FaceLandmarker to produce per-frame face landmark
positions, head-pose estimates, and blendshape coefficients (INF-002,
REQ-005, REQ-006).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe solution types used for type annotations
_mp_tasks = mp.tasks
_mp_vision = mp.tasks.vision
_mp_components = mp.tasks.components


@dataclass
class Landmark:
    """
    Normalised 3-D face landmark from MediaPipe.

    Coordinates are normalised to [0, 1] relative to the frame
    dimensions; z represents depth relative to the face centre.
    """

    x: float
    """Normalised horizontal position (0 = left, 1 = right)."""
    y: float
    """Normalised vertical position (0 = top, 1 = bottom)."""
    z: float
    """Relative depth (negative = closer to camera)."""


@dataclass
class HeadPose:
    """
    Coarse head orientation estimated from face geometry.

    Angles follow the right-hand rule:
    - yaw   : rotation around vertical axis (left/right look)
    - pitch : rotation around lateral axis  (up/down tilt)
    - roll  : rotation around depth axis    (head tilt)
    """

    yaw: float = 0.0
    """Yaw angle in degrees."""
    pitch: float = 0.0
    """Pitch angle in degrees."""
    roll: float = 0.0
    """Roll angle in degrees."""
    translation: Tuple[float, float, float] = field(
        default_factory=lambda: (0.0, 0.0, 0.0)
    )
    """Approximate head translation (x, y, z) in normalised units."""


@dataclass
class FaceTrackResult:
    """
    Complete output of one face tracking inference pass (REQ-005).

    Attributes:
        landmarks     : List of 478 normalised 3-D face landmarks.
        head_pose     : Estimated head orientation and rough translation.
        blendshapes   : Dict mapping blendshape name → coefficient [0, 1].
        face_detected : True when at least one face was found in the frame.
    """

    landmarks: List[Landmark] = field(default_factory=list)
    head_pose: HeadPose = field(default_factory=HeadPose)
    blendshapes: dict = field(default_factory=dict)
    face_detected: bool = False


class FaceTracker:
    """
    Stateful MediaPipe face tracker (INF-002, REQ-005).

    Performs per-frame:
    * 478-point 3-D face landmark detection
    * Head-pose estimation via geometry from landmark positions
    * Blendshape coefficient extraction (mouth-open, eye-blink, etc.)

    Usage::

        tracker = FaceTracker()
        tracker.setup()
        result = tracker.process(bgr_frame)   # FaceTrackResult or None
        tracker.teardown()
    """

    # Indices for key landmarks used in head-pose estimation
    _NOSE_TIP = 1
    _CHIN = 199
    _LEFT_EYE = 33
    _RIGHT_EYE = 263
    _LEFT_MOUTH = 61
    _RIGHT_MOUTH = 291

    def __init__(self, num_faces: int = 1) -> None:
        """
        Initialise the tracker descriptor (does not load models).

        Parameters:
            num_faces (int): Maximum number of faces to track
                             simultaneously.
        """
        self._num_faces = num_faces
        self._landmarker: Optional[_mp_vision.FaceLandmarker] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """
        Load and initialise the MediaPipe FaceLandmarker model.

        Downloads the model bundle on first run (~5 MB).  Subsequent
        calls use the cached bundle from MediaPipe's model store.

        Raises:
            RuntimeError: If the MediaPipe model file cannot be loaded.
        """
        try:
            options = _mp_vision.FaceLandmarkerOptions(
                base_options=_mp_tasks.BaseOptions(
                    model_asset_path=self._resolve_model_path()
                ),
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=self._num_faces,
                running_mode=_mp_vision.RunningMode.IMAGE,
            )
            self._landmarker = _mp_vision.FaceLandmarker.create_from_options(
                options
            )
            logger.info("FaceTracker ready (max %d face(s)).", self._num_faces)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"Failed to initialise MediaPipe FaceLandmarker: {exc}"
            ) from exc

    def teardown(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            logger.info("FaceTracker released.")

    # ------------------------------------------------------------------
    # Per-frame inference
    # ------------------------------------------------------------------

    def process(self, bgr_frame: np.ndarray) -> Optional[FaceTrackResult]:
        """
        Run face tracking on a single BGR frame.

        Parameters:
            bgr_frame (np.ndarray): A (H, W, 3) uint8 BGR frame from
                                    OpenCV.

        Returns:
            Optional[FaceTrackResult]: Tracking result for the first
            detected face, or None if the landmarker is not set up or
            an inference error occurs.
        """
        if self._landmarker is None:
            logger.warning("process() called before setup().")
            return None

        # MediaPipe expects RGB
        rgb = cv2_bgr_to_rgb(bgr_frame)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=rgb
        )

        try:
            detection = self._landmarker.detect(mp_image)
        except Exception as exc:  # noqa: BLE001
            logger.error("FaceTracker inference error: %s", exc)
            return None

        result = FaceTrackResult()

        if not detection.face_landmarks:
            return result  # face_detected defaults to False

        result.face_detected = True

        # -- Landmarks -----------------------------------------------
        raw_landmarks = detection.face_landmarks[0]
        result.landmarks = [
            Landmark(lm.x, lm.y, lm.z) for lm in raw_landmarks
        ]

        # -- Head pose (derived from key landmark geometry) -----------
        result.head_pose = self._estimate_head_pose(result.landmarks)

        # -- Blendshapes ----------------------------------------------
        if detection.face_blendshapes:
            result.blendshapes = {
                bs.category_name: bs.score
                for bs in detection.face_blendshapes[0]
            }

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_model_path() -> str:
        """
        Return the path to the MediaPipe face landmarker task bundle.

        Looks for the task file in ``<project_root>/assets/``.  If it is
        missing the file is downloaded automatically from the official
        MediaPipe model storage bucket (~3.6 MB, float16 variant).

        Returns:
            str: Absolute path to face_landmarker.task.
        """
        _MODEL_URL = (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
        bundle: Path = (
            Path(__file__).parent.parent / "assets" / "face_landmarker.task"
        )
        if not bundle.exists():
            import urllib.request  # noqa: PLC0415

            bundle.parent.mkdir(parents=True, exist_ok=True)
            logger.info(
                "face_landmarker.task not found — downloading from %s",
                _MODEL_URL,
            )
            urllib.request.urlretrieve(_MODEL_URL, bundle)
            logger.info("Model saved to %s (%d bytes)", bundle, bundle.stat().st_size)

        return str(bundle)

    @staticmethod
    def _estimate_head_pose(landmarks: List[Landmark]) -> HeadPose:
        """
        Derive a coarse head pose from key facial landmark positions.

        Uses the eye-centre-to-nose-tip vector for yaw/pitch and the
        inter-eye vector for roll.  This is an approximation; full PnP
        solve is omitted to keep CPU cost minimal.

        Parameters:
            landmarks (List[Landmark]): 478 normalised 3-D landmarks.

        Returns:
            HeadPose: Approximate yaw, pitch, roll in degrees.
        """
        if len(landmarks) < 468:
            return HeadPose()

        import math

        nose = landmarks[FaceTracker._NOSE_TIP]
        left_eye = landmarks[FaceTracker._LEFT_EYE]
        right_eye = landmarks[FaceTracker._RIGHT_EYE]

        # Eye centre
        eye_cx = (left_eye.x + right_eye.x) * 0.5
        eye_cy = (left_eye.y + right_eye.y) * 0.5
        eye_cz = (left_eye.z + right_eye.z) * 0.5

        # Vector from eye centre to nose tip
        vx = nose.x - eye_cx
        vy = nose.y - eye_cy
        vz = nose.z - eye_cz

        # Yaw: horizontal deviation of nose from eye centre
        yaw = math.degrees(math.atan2(vx, abs(vz) + 1e-6))
        # Pitch: vertical deviation
        pitch = math.degrees(math.atan2(vy, abs(vz) + 1e-6))

        # Roll: angle of inter-eye line from horizontal
        dx = right_eye.x - left_eye.x
        dy = right_eye.y - left_eye.y
        roll = math.degrees(math.atan2(dy, dx + 1e-6))

        # Translation: use nose-tip position as a proxy
        tx = (nose.x - 0.5) * 2.0   # map [0,1] → [-1,1]
        ty = (nose.y - 0.5) * 2.0
        tz = nose.z

        return HeadPose(yaw=yaw, pitch=pitch, roll=roll, translation=(tx, ty, tz))


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def cv2_bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    """
    Convert a BGR uint8 ndarray to RGB.

    Parameters:
        bgr (np.ndarray): Input image in BGR channel order.

    Returns:
        np.ndarray: Output image in RGB channel order.
    """
    import cv2  # noqa: PLC0415 — local import avoids top-level cv2 dep

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
