"""
Tests for tracking/face_tracker.py.

Tests cover:
* Data class defaults
* FaceTracker._estimate_head_pose (pure maths, no MediaPipe required)
* cv2_bgr_to_rgb colour channel swap
* FrameTrackResult.primary_face backwards-compatibility property
* TrackerConfig defaults, clamp behaviour, and JSON round-trip
* TrackerMode enum values

FaceTracker.process and FaceTracker.setup are integration tests
requiring a real MediaPipe model; they are skipped unless MediaPipe is
available and CAMERA_MAYHAM_INTEGRATION_TESTS=1 is set.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from tracking.face_tracker import (
    FaceTrackResult,
    FaceTracker,
    FrameTrackResult,
    HeadPose,
    Landmark,
    TrackerConfig,
    TrackerMode,
    cv2_bgr_to_rgb,
)

# ---------------------------------------------------------------------------
# TrackerMode
# ---------------------------------------------------------------------------


class TestTrackerMode:
    """Tests for the TrackerMode enum."""

    def test_video_value(self):
        """
        Test that TrackerMode.VIDEO has the expected string value.

        Expected: TrackerMode.VIDEO == "video".
        """
        assert TrackerMode.VIDEO == "video"

    def test_image_value(self):
        """
        Test that TrackerMode.IMAGE has the expected string value.

        Expected: TrackerMode.IMAGE == "image".
        """
        assert TrackerMode.IMAGE == "image"

    def test_from_string_video(self):
        """
        Test that TrackerMode can be constructed from a plain string.

        Expected: TrackerMode("video") == TrackerMode.VIDEO.
        """
        assert TrackerMode("video") is TrackerMode.VIDEO

    def test_from_string_image(self):
        """
        Test that TrackerMode can be constructed from a plain string.

        Expected: TrackerMode("image") == TrackerMode.IMAGE.
        """
        assert TrackerMode("image") is TrackerMode.IMAGE


# ---------------------------------------------------------------------------
# TrackerConfig
# ---------------------------------------------------------------------------


class TestTrackerConfig:
    """Tests for TrackerConfig defaults and persistence helpers."""

    def test_default_mode_is_video(self):
        """
        Test that the default operating mode is VIDEO.

        Expected: TrackerConfig().mode == TrackerMode.VIDEO.
        """
        assert TrackerConfig().mode == TrackerMode.VIDEO

    def test_default_sensitivity(self):
        """
        Test that the default sensitivity is 0.5.

        Expected: TrackerConfig().sensitivity == 0.5.
        """
        assert TrackerConfig().sensitivity == pytest.approx(0.5)

    def test_default_num_faces(self):
        """
        Test that the default number of faces is 1.

        Expected: TrackerConfig().num_faces == 1.
        """
        assert TrackerConfig().num_faces == 1

    def test_load_returns_defaults_on_missing_file(self, tmp_path, monkeypatch):
        """
        Test that TrackerConfig.load() returns defaults when the config
        file does not exist.

        Expected: returned config has default values.
        """
        import tracking.face_tracker as ft
        monkeypatch.setattr(ft, "_CONFIG_PATH", tmp_path / "no_file.json")
        cfg = ft.TrackerConfig.load()
        assert cfg.mode == TrackerMode.VIDEO
        assert cfg.sensitivity == pytest.approx(0.5)
        assert cfg.num_faces == 1

    def test_save_and_load_round_trip(self, tmp_path, monkeypatch):
        """
        Test that save() followed by load() restores the same values.

        Expected: loaded config matches the saved config.
        """
        import tracking.face_tracker as ft
        config_path = tmp_path / "tracker_config.json"
        monkeypatch.setattr(ft, "_CONFIG_PATH", config_path)
        original = TrackerConfig(
            mode=TrackerMode.IMAGE, sensitivity=0.3, num_faces=2
        )
        original.save()
        loaded = ft.TrackerConfig.load()
        assert loaded.mode == TrackerMode.IMAGE
        assert loaded.sensitivity == pytest.approx(0.3)
        assert loaded.num_faces == 2

    def test_load_clamps_sensitivity_above_one(self, tmp_path, monkeypatch):
        """
        Test that load() clamps sensitivity values above 1.0 to 1.0.

        Expected: loaded sensitivity == 1.0 when stored value > 1.0.
        """
        import json
        import tracking.face_tracker as ft
        config_path = tmp_path / "tracker_config.json"
        monkeypatch.setattr(ft, "_CONFIG_PATH", config_path)
        config_path.write_text(json.dumps({"sensitivity": 5.0, "num_faces": 1}))
        loaded = ft.TrackerConfig.load()
        assert loaded.sensitivity == pytest.approx(1.0)

    def test_load_clamps_num_faces_above_four(self, tmp_path, monkeypatch):
        """
        Test that load() clamps num_faces values above 4 to 4.

        Expected: loaded num_faces == 4 when stored value > 4.
        """
        import json
        import tracking.face_tracker as ft
        config_path = tmp_path / "tracker_config.json"
        monkeypatch.setattr(ft, "_CONFIG_PATH", config_path)
        config_path.write_text(json.dumps({"num_faces": 99, "sensitivity": 0.5}))
        loaded = ft.TrackerConfig.load()
        assert loaded.num_faces == 4


# ---------------------------------------------------------------------------
# FrameTrackResult
# ---------------------------------------------------------------------------


class TestFrameTrackResult:
    """Tests for the FrameTrackResult multi-face wrapper."""

    def test_face_detected_defaults_false(self):
        """
        Test that face_detected is False when no arguments are given.

        Expected: FrameTrackResult().face_detected == False.
        """
        assert FrameTrackResult().face_detected is False

    def test_faces_default_empty(self):
        """
        Test that faces is an empty list by default.

        Expected: FrameTrackResult().faces == [].
        """
        assert FrameTrackResult().faces == []

    def test_primary_face_returns_default_when_no_faces(self):
        """
        Test that primary_face returns a zeroed FaceTrackResult when
        no faces are present.

        Expected: primary_face.face_detected == False and landmarks == [].
        """
        frame = FrameTrackResult()
        primary = frame.primary_face
        assert isinstance(primary, FaceTrackResult)
        assert primary.face_detected is False
        assert primary.landmarks == []

    def test_primary_face_returns_first_face(self):
        """
        Test that primary_face returns faces[0] when faces are present.

        Expected: primary_face is the same object as faces[0].
        """
        face_a = FaceTrackResult(face_detected=True)
        face_b = FaceTrackResult(face_detected=True)
        frame = FrameTrackResult(face_detected=True, faces=[face_a, face_b])
        assert frame.primary_face is face_a

    def test_primary_face_is_not_none(self):
        """
        Test that primary_face never returns None, even with no faces.

        Expected: primary_face is always a FaceTrackResult instance.
        """
        assert FrameTrackResult().primary_face is not None


# ---------------------------------------------------------------------------
# Data class defaults
# ---------------------------------------------------------------------------


class TestFaceTrackResult:
    """Tests for the FaceTrackResult dataclass."""

    def test_face_detected_defaults_false(self):
        """
        Test that face_detected is False when no arguments are given.

        Expected: FaceTrackResult().face_detected == False.
        """
        result = FaceTrackResult()
        assert result.face_detected is False

    def test_landmarks_default_empty(self):
        """
        Test that landmarks is an empty list by default.

        Expected: FaceTrackResult().landmarks == [].
        """
        result = FaceTrackResult()
        assert result.landmarks == []

    def test_blendshapes_default_empty_dict(self):
        """
        Test that blendshapes is an empty dict by default.

        Expected: FaceTrackResult().blendshapes == {}.
        """
        result = FaceTrackResult()
        assert result.blendshapes == {}

    def test_head_pose_default(self):
        """
        Test that head_pose is a zeroed HeadPose by default.

        Expected: yaw, pitch, roll all equal 0.0.
        """
        result = FaceTrackResult()
        assert result.head_pose.yaw == pytest.approx(0.0)
        assert result.head_pose.pitch == pytest.approx(0.0)
        assert result.head_pose.roll == pytest.approx(0.0)


class TestHeadPose:
    """Tests for the HeadPose dataclass."""

    def test_translation_default(self):
        """
        Test that the default translation is (0, 0, 0).

        Expected: HeadPose().translation == (0.0, 0.0, 0.0).
        """
        hp = HeadPose()
        assert hp.translation == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# HeadPose estimation (pure maths)
# ---------------------------------------------------------------------------


def _make_landmarks(nose_tip_x=0.5, nose_tip_y=0.5, nose_tip_z=0.0):
    """
    Build a minimal 478-landmark list with key landmarks placed at
    controllable positions and the rest at the origin.

    Parameters:
        nose_tip_x (float): Normalised x of the nose tip (index 1).
        nose_tip_y (float): Normalised y of the nose tip (index 1).
        nose_tip_z (float): Normalised z of the nose tip (index 1).

    Returns:
        list[Landmark]: 478 landmarks.
    """
    lms = [Landmark(0.5, 0.5, 0.0)] * 478

    # Index 1 — nose tip
    lms[1] = Landmark(nose_tip_x, nose_tip_y, nose_tip_z)
    # Index 33 — left eye
    lms[33] = Landmark(0.35, 0.40, 0.0)
    # Index 263 — right eye
    lms[263] = Landmark(0.65, 0.40, 0.0)

    return lms


class TestEstimateHeadPose:
    """Tests for FaceTracker._estimate_head_pose."""

    def test_returns_head_pose_instance(self):
        """
        Test that _estimate_head_pose returns a HeadPose.

        Expected: return type is HeadPose.
        """
        lms = _make_landmarks()
        result = FaceTracker._estimate_head_pose(lms)
        assert isinstance(result, HeadPose)

    def test_centered_head_small_yaw(self):
        """
        Test that a centred nose produces near-zero yaw.

        Expected: |yaw| < 5 degrees when nose is directly below the
        eye centre.
        """
        lms = _make_landmarks(nose_tip_x=0.5, nose_tip_y=0.5)
        pose = FaceTracker._estimate_head_pose(lms)
        assert abs(pose.yaw) < 5.0

    def test_head_turned_right_positive_yaw(self):
        """
        Test that a nose shifted right of the eye centre yields positive yaw.

        Expected: yaw > 0 when nose_x > eye_centre_x.
        """
        # Eye centre is at x=0.5 (midpoint of 0.35 and 0.65)
        # Move nose to the right of centre
        lms = _make_landmarks(nose_tip_x=0.62, nose_tip_y=0.5)
        pose = FaceTracker._estimate_head_pose(lms)
        assert pose.yaw > 0

    def test_head_turned_left_negative_yaw(self):
        """
        Test that a nose shifted left of the eye centre yields negative yaw.

        Expected: yaw < 0 when nose_x < eye_centre_x.
        """
        lms = _make_landmarks(nose_tip_x=0.38, nose_tip_y=0.5)
        pose = FaceTracker._estimate_head_pose(lms)
        assert pose.yaw < 0

    def test_insufficient_landmarks_returns_default(self):
        """
        Test that fewer than 468 landmarks returns a zeroed HeadPose.

        Expected: HeadPose with all-zero angles when landmark list is short.
        """
        few_lms = [Landmark(0.5, 0.5, 0.0)] * 10
        pose = FaceTracker._estimate_head_pose(few_lms)
        assert pose.yaw == pytest.approx(0.0)
        assert pose.pitch == pytest.approx(0.0)
        assert pose.roll == pytest.approx(0.0)

    def test_horizontal_eyes_small_roll(self):
        """
        Test that perfectly horizontal eyes produce near-zero roll.

        Expected: |roll| < 5 degrees when both eyes have the same y.
        """
        lms = _make_landmarks()
        # Left eye y == right eye y (already set equal in helper)
        pose = FaceTracker._estimate_head_pose(lms)
        assert abs(pose.roll) < 5.0

    def test_translation_nose_centred(self):
        """
        Test that a nose at (0.5, 0.5) maps to approximately (0, 0, z).

        Expected: tx ≈ 0, ty ≈ 0 when nose is exactly at image centre.
        """
        lms = _make_landmarks(nose_tip_x=0.5, nose_tip_y=0.5)
        pose = FaceTracker._estimate_head_pose(lms)
        assert abs(pose.translation[0]) < 0.01
        assert abs(pose.translation[1]) < 0.01


# ---------------------------------------------------------------------------
# cv2_bgr_to_rgb helper
# ---------------------------------------------------------------------------


class TestCv2BgrToRgb:
    """Tests for the cv2_bgr_to_rgb colour channel swap helper."""

    def test_channel_swap(self):
        """
        Test that cv2_bgr_to_rgb swaps B and R channels.

        Expected: output[:, :, 0] == input[:, :, 2] (R←B) and vice versa.
        """
        bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        bgr[:, :, 0] = 10  # B
        bgr[:, :, 1] = 20  # G
        bgr[:, :, 2] = 30  # R

        rgb = cv2_bgr_to_rgb(bgr)
        assert np.all(rgb[:, :, 0] == 30)  # R
        assert np.all(rgb[:, :, 1] == 20)  # G (unchanged)
        assert np.all(rgb[:, :, 2] == 10)  # B

    def test_output_shape_unchanged(self):
        """
        Test that output shape matches input shape.

        Expected: rgb.shape == bgr.shape.
        """
        bgr = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        rgb = cv2_bgr_to_rgb(bgr)
        assert rgb.shape == bgr.shape

    def test_output_dtype_uint8(self):
        """
        Test that output dtype is uint8.

        Expected: rgb.dtype == np.uint8.
        """
        bgr = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb = cv2_bgr_to_rgb(bgr)
        assert rgb.dtype == np.uint8
