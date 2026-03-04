"""
Tests for tracking/face_tracker.py.

Tests cover:
* Data class defaults
* FaceTracker._estimate_head_pose (pure maths, no MediaPipe required)
* cv2_bgr_to_rgb colour channel swap

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
    HeadPose,
    Landmark,
    cv2_bgr_to_rgb,
)

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
