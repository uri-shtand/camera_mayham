"""
Face tracking module.

Wraps MediaPipe FaceLandmarker to produce per-frame face landmark
positions, head-pose estimates, and blendshape coefficients.

Supports two operating modes (spec §3, §4):
  - VIDEO mode (default): frames processed as a continuous stream;
    MediaPipe uses temporal information across frames for improved
    accuracy and stability.
  - IMAGE mode: each frame is treated independently with no prior-frame
    context; suitable for still images or non-sequential inputs.

Settings (mode, sensitivity, number of faces) are persisted to
``config/tracker_config.json`` and restored automatically on next
launch (spec §4 Widget behaviour).

Public surface
--------------
* :class:`TrackerMode`      — VIDEO / IMAGE enum
* :class:`TrackerConfig`    — persisted settings with save / load
* :class:`Landmark`         — single normalised 3-D face point
* :class:`HeadPose`         — yaw / pitch / roll angles
* :class:`FaceTrackResult`  — per-face data (landmarks, pose, blendshapes)
* :class:`FrameTrackResult` — frame-level wrapper over 0-N faces
* :class:`FaceTracker`      — stateful tracker; call setup / process / teardown
* :func:`cv2_bgr_to_rgb`    — BGR → RGB channel helper
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe solution types used for type annotations
_mp_tasks = mp.tasks
_mp_vision = mp.tasks.vision
_mp_components = mp.tasks.components

# Absolute path to the persisted tracker configuration file.
# The ``config/`` directory is created automatically on first save.
_CONFIG_PATH: Path = (
    Path(__file__).parent.parent / "config" / "tracker_config.json"
)


# ---------------------------------------------------------------------------
# Operating mode
# ---------------------------------------------------------------------------

class TrackerMode(str, Enum):
    """
    Operating mode for the face tracker (spec §3 Operating modes).

    Inheriting from ``str`` lets the values be JSON-serialised and
    compared directly against plain strings.

    Attributes:
        VIDEO: Continuous stream mode.  MediaPipe exploits inter-frame
               continuity to improve stability and recover from brief
               occlusions.  Requires monotonically increasing timestamps.
        IMAGE: Frame-independent mode.  Each frame is analysed from
               scratch with no context from prior frames.  Suitable for
               still images or non-sequential input.
    """

    VIDEO = "video"
    IMAGE = "image"


# ---------------------------------------------------------------------------
# Tracker configuration (spec §4 Configurable parameters)
# ---------------------------------------------------------------------------

@dataclass
class TrackerConfig:
    """
    Persisted configuration for :class:`FaceTracker` (spec §4).

    All three parameters are exposed in the face tracking settings
    widget and saved between app launches.

    Attributes:
        mode       : VIDEO (default) or IMAGE operating mode.
        sensitivity: Detection confidence threshold in [0.0, 1.0].
                     Lower = more willing to detect a face (fewer misses,
                     more false positives).  Default 0.5.
        num_faces  : Maximum faces detected per frame (1–4).  Faces are
                     ordered by detection confidence, highest first.
                     Default 1.
    """

    mode: TrackerMode = TrackerMode.VIDEO
    sensitivity: float = 0.5
    num_faces: int = 1

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """
        Serialise this config to ``config/tracker_config.json``.

        Creates the ``config/`` directory if absent.  Silently logs and
        returns on any I/O error so a missing save never crashes the app.
        """
        try:
            _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "mode": self.mode.value,
                "sensitivity": self.sensitivity,
                "num_faces": self.num_faces,
            }
            _CONFIG_PATH.write_text(json.dumps(data, indent=2))
            logger.debug("TrackerConfig saved to %s.", _CONFIG_PATH)
        except OSError as exc:
            logger.warning("Could not save tracker config: %s", exc)

    @classmethod
    def load(cls) -> "TrackerConfig":
        """
        Deserialise a previously saved config from disk.

        Returns a default :class:`TrackerConfig` instance when the file
        is absent, unreadable, or contains unexpected values — never
        raises.

        Returns:
            TrackerConfig: The restored config, or defaults on any error.
        """
        try:
            data = json.loads(_CONFIG_PATH.read_text())
            mode = TrackerMode(data.get("mode", TrackerMode.VIDEO.value))
            sensitivity = float(data.get("sensitivity", 0.5))
            num_faces = int(data.get("num_faces", 1))
            # Clamp to valid ranges
            sensitivity = max(0.0, min(1.0, sensitivity))
            num_faces = max(1, min(4, num_faces))
            cfg = cls(mode=mode, sensitivity=sensitivity, num_faces=num_faces)
            logger.debug("TrackerConfig loaded from %s.", _CONFIG_PATH)
            return cfg
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            logger.debug(
                "Could not load tracker config (%s) — using defaults.", exc
            )
            return cls()


# ---------------------------------------------------------------------------
# Landmark and HeadPose
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Per-face result
# ---------------------------------------------------------------------------

@dataclass
class FaceTrackResult:
    """
    Tracking output for a single detected face.

    When multiple faces are tracked, one :class:`FaceTrackResult` is
    produced per face.  Consumers access all faces via
    :attr:`FrameTrackResult.faces`; single-face consumers can use the
    :attr:`FrameTrackResult.primary_face` property for backwards
    compatibility.

    Attributes:
        landmarks    : List of 478 normalised 3-D face landmarks.
        head_pose    : Estimated head orientation and rough translation.
        blendshapes  : Dict mapping blendshape name → coefficient [0, 1].
        face_detected: True for every result stored in FrameTrackResult.
    """

    landmarks: List[Landmark] = field(default_factory=list)
    head_pose: HeadPose = field(default_factory=HeadPose)
    blendshapes: dict = field(default_factory=dict)
    face_detected: bool = False


# ---------------------------------------------------------------------------
# Frame-level result (multi-face wrapper)
# ---------------------------------------------------------------------------

@dataclass
class FrameTrackResult:
    """
    Complete output of one face tracking inference pass.

    Wraps zero or more per-face :class:`FaceTrackResult` objects.
    Faces are ordered by detection confidence, highest first (MediaPipe
    convention).

    Attributes:
        face_detected: True when at least one face was found this frame.
        faces        : Per-face results; empty when no face is detected.
    """

    face_detected: bool = False
    faces: List[FaceTrackResult] = field(default_factory=list)

    @property
    def primary_face(self) -> FaceTrackResult:
        """
        Return the highest-confidence face result for this frame.

        When no face is detected, returns a zeroed :class:`FaceTrackResult`
        (``face_detected=False``) so single-face consumers can be used
        without ``None`` guards.

        Returns:
            FaceTrackResult: First (highest-confidence) face, or an
            empty default when no face is present.
        """
        if self.faces:
            return self.faces[0]
        return FaceTrackResult()


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class FaceTracker:
    """
    Stateful MediaPipe face tracker (spec §3, §6).

    Performs per-frame:
    * 478-point 3-D face landmark detection
    * Head-pose estimation via geometry from landmark positions
    * Blendshape coefficient extraction (mouth-open, eye-blink, etc.)

    Configuration is loaded from disk at construction time (or supplied
    explicitly) and can be updated live via :meth:`reconfigure`.

    Usage::

        tracker = FaceTracker()           # loads saved config from disk
        tracker.setup()
        frame_result = tracker.process(bgr_frame)  # FrameTrackResult
        tracker.teardown()
    """

    # Indices for key landmarks used in head-pose estimation
    _NOSE_TIP = 1
    _CHIN = 199
    _LEFT_EYE = 33
    _RIGHT_EYE = 263
    _LEFT_MOUTH = 61
    _RIGHT_MOUTH = 291

    def __init__(self, config: Optional[TrackerConfig] = None) -> None:
        """
        Initialise the tracker descriptor (does not load models yet).

        If no config is provided the previously saved configuration is
        restored from disk via :meth:`TrackerConfig.load`; defaults are
        used when no saved config exists.

        Parameters:
            config (Optional[TrackerConfig]): Explicit settings to use.
                Pass ``None`` (default) to restore from disk.
        """
        self._config: TrackerConfig = (
            config if config is not None else TrackerConfig.load()
        )
        self._landmarker: Optional[_mp_vision.FaceLandmarker] = None
        # Monotonically increasing timestamp for detect_for_video (ms).
        # Reset to 0 on mode change so VIDEO sessions always start fresh.
        self._frame_timestamp_ms: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> TrackerConfig:
        """Return the current tracker configuration (read-only view)."""
        return self._config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_options(self) -> _mp_vision.FaceLandmarkerOptions:
        """
        Build a :class:`FaceLandmarkerOptions` from the current config.

        Maps :attr:`TrackerConfig.sensitivity` to all three MediaPipe
        detection/tracking confidence thresholds and selects the correct
        :class:`RunningMode`.

        Returns:
            FaceLandmarkerOptions: Ready-to-use options object.
        """
        mode_map = {
            TrackerMode.VIDEO: _mp_vision.RunningMode.VIDEO,
            TrackerMode.IMAGE: _mp_vision.RunningMode.IMAGE,
        }
        running_mode = mode_map.get(
            self._config.mode, _mp_vision.RunningMode.VIDEO
        )
        return _mp_vision.FaceLandmarkerOptions(
            base_options=_mp_tasks.BaseOptions(
                model_asset_path=self._resolve_model_path()
            ),
            running_mode=running_mode,
            num_faces=self._config.num_faces,
            min_face_detection_confidence=self._config.sensitivity,
            min_face_presence_confidence=self._config.sensitivity,
            min_tracking_confidence=self._config.sensitivity,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """
        Load and initialise the MediaPipe FaceLandmarker model.

        Uses the configuration supplied at construction (or restored from
        disk).  Downloads the model bundle on first run (~3.6 MB,
        stored in ``assets/``).

        Raises:
            RuntimeError: If the MediaPipe model file cannot be loaded.
        """
        try:
            options = self._build_options()
            self._landmarker = _mp_vision.FaceLandmarker.create_from_options(
                options
            )
            logger.info(
                "FaceTracker ready — mode=%s, sensitivity=%.2f, "
                "num_faces=%d.",
                self._config.mode.value,
                self._config.sensitivity,
                self._config.num_faces,
            )
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

    def reconfigure(self, config: TrackerConfig) -> None:
        """
        Apply new settings and reinitialise the landmarker in-place.

        Persists the new config to disk, closes the existing landmarker,
        and recreates it with the updated options.  When the operating
        mode changes the timestamp counter is reset so the new VIDEO
        session starts with a clean monotonic sequence.

        Errors during reinitialisation are logged; the tracker is left
        in a non-functional state until the next successful call.

        Parameters:
            config (TrackerConfig): The new configuration to apply.
        """
        mode_changed = config.mode != self._config.mode
        self._config = config
        config.save()

        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

        if mode_changed:
            self._frame_timestamp_ms = 0
            logger.debug("Tracker mode changed — timestamp counter reset.")

        try:
            options = self._build_options()
            self._landmarker = _mp_vision.FaceLandmarker.create_from_options(
                options
            )
            logger.info(
                "FaceTracker reconfigured — mode=%s, sensitivity=%.2f, "
                "num_faces=%d.",
                self._config.mode.value,
                self._config.sensitivity,
                self._config.num_faces,
            )
        except Exception as exc:
            logger.error(
                "FaceTracker reconfigure failed: %s. "
                "Tracker inactive until next reconfigure() or setup().",
                exc,
            )

    # ------------------------------------------------------------------
    # Per-frame inference
    # ------------------------------------------------------------------

    def process(self, bgr_frame: np.ndarray) -> FrameTrackResult:
        """
        Run face tracking on a single BGR frame.

        Uses :meth:`detect_for_video` in VIDEO mode (temporal context,
        monotonic timestamps) or :meth:`detect` in IMAGE mode (frame-
        independent).  All errors are caught and logged; the method
        always returns a valid :class:`FrameTrackResult` so the app
        never crashes on a bad frame.

        Parameters:
            bgr_frame (np.ndarray): A (H, W, 3) uint8 BGR frame from
                                    OpenCV.

        Returns:
            FrameTrackResult: Per-frame result with zero or more faces.
        """
        if self._landmarker is None:
            logger.warning(
                "process() called before setup() or after a failed "
                "reconfigure()."
            )
            return FrameTrackResult()

        rgb = cv2_bgr_to_rgb(bgr_frame)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            if self._config.mode == TrackerMode.VIDEO:
                # Timestamp must increase monotonically within a session.
                self._frame_timestamp_ms = int(time.monotonic() * 1000)
                detection = self._landmarker.detect_for_video(
                    mp_image, self._frame_timestamp_ms
                )
            else:
                detection = self._landmarker.detect(mp_image)
        except Exception as exc:  # noqa: BLE001
            logger.error("FaceTracker inference error: %s", exc)
            return FrameTrackResult()

        if not detection.face_landmarks:
            return FrameTrackResult(face_detected=False)

        faces = [
            self._extract_face_result(detection, i)
            for i in range(len(detection.face_landmarks))
        ]
        return FrameTrackResult(face_detected=True, faces=faces)

    def _extract_face_result(
        self,
        detection: object,
        index: int,
    ) -> FaceTrackResult:
        """
        Build a :class:`FaceTrackResult` for the face at ``index``.

        Parameters:
            detection: MediaPipe FaceLandmarkerResult for the frame.
            index (int): Face index within the result (0-based;
                         0 = highest confidence).

        Returns:
            FaceTrackResult: Populated result for that single face.
        """
        result = FaceTrackResult(face_detected=True)

        # Landmarks
        raw_landmarks = detection.face_landmarks[index]
        result.landmarks = [
            Landmark(lm.x, lm.y, lm.z) for lm in raw_landmarks
        ]

        # Head pose derived from landmark geometry
        result.head_pose = self._estimate_head_pose(result.landmarks)

        # Blendshapes (only if present at this index)
        if (
            detection.face_blendshapes
            and index < len(detection.face_blendshapes)
        ):
            result.blendshapes = {
                bs.category_name: bs.score
                for bs in detection.face_blendshapes[index]
            }

        return result

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_model_path() -> str:
        """
        Return the path to the MediaPipe face landmarker task bundle.

        Looks for ``face_landmarker.task`` in ``<project_root>/assets/``.
        If missing it is downloaded from the official MediaPipe model
        storage bucket (~3.6 MB, float16 variant, one-time download).

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
            logger.info(
                "Model saved to %s (%d bytes)",
                bundle,
                bundle.stat().st_size,
            )

        return str(bundle)

    @staticmethod
    def _estimate_head_pose(landmarks: List[Landmark]) -> HeadPose:
        """
        Derive a coarse head pose from key facial landmark positions.

        Uses the eye-centre-to-nose-tip vector for yaw/pitch and the
        inter-eye vector for roll.  This is a geometric approximation;
        a full PnP solve is omitted to keep CPU cost minimal.

        Parameters:
            landmarks (List[Landmark]): 478 normalised 3-D landmarks.

        Returns:
            HeadPose: Approximate yaw, pitch, roll in degrees.
        """
        if len(landmarks) < 468:
            return HeadPose()

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

        # Translation: nose-tip as head position proxy (mapped to [-1, 1])
        tx = (nose.x - 0.5) * 2.0
        ty = (nose.y - 0.5) * 2.0
        tz = nose.z

        return HeadPose(
            yaw=yaw, pitch=pitch, roll=roll, translation=(tx, ty, tz)
        )


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
