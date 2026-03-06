"""
Tests for filters/moustache.py (spec v1.0).

All tests avoid importing wgpu (no GPU required) by exercising only the
Python-level logic: identity, param validation, face-result injection,
the ``_has_face`` guard, anchor computation, alpha masking, and the
CPU-side canvas drawing (``_draw_moustache``).

Follows the testing conventions established in
``tests/test_face_landmark_filter.py``.
"""

from __future__ import annotations

import math
import unittest.mock as mock
from pathlib import Path
from typing import List

import numpy as np
import pytest

from filters.moustache import (
    MoustacheFilter,
    _ALPHA_THRESHOLD,
    _LM_MOUTH_LEFT,
    _LM_MOUTH_RIGHT,
    _LM_NOSE_TIP,
    _LM_UPPER_LIP,
    _MIN_RENDER_WIDTH,
    _NUM_SPRITES,
    _WIDTH_SCALE,
)
from tracking.face_tracker import FaceTrackResult, HeadPose, Landmark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_landmark(x: float = 0.5, y: float = 0.5) -> Landmark:
    """
    Create a Landmark with the given normalised position.

    Parameters:
        x (float): Normalised horizontal position [0, 1].
        y (float): Normalised vertical position [0, 1].

    Returns:
        Landmark: Landmark instance with z=0.
    """
    return Landmark(x=x, y=y, z=0.0)


def _make_landmarks(count: int = 478) -> List[Landmark]:
    """
    Create a list of ``count`` default Landmarks at (0.5, 0.5).

    Parameters:
        count (int): Number of landmarks to create.

    Returns:
        List[Landmark]: List of default Landmark instances.
    """
    return [_make_landmark() for _ in range(count)]


def _make_result(
    landmarks: List[Landmark] | None = None,
    face_detected: bool = True,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
) -> FaceTrackResult:
    """
    Build a FaceTrackResult with the supplied settings.

    Parameters:
        landmarks (List[Landmark] | None): If None, 478 default
            landmarks are used.
        face_detected (bool): Whether a face was detected.
        yaw (float): Yaw angle in degrees.
        pitch (float): Pitch angle in degrees.
        roll (float): Roll angle in degrees.

    Returns:
        FaceTrackResult: Populated result.
    """
    if landmarks is None:
        landmarks = _make_landmarks(478)
    return FaceTrackResult(
        landmarks=landmarks,
        face_detected=face_detected,
        head_pose=HeadPose(yaw=yaw, pitch=pitch, roll=roll),
    )


def _make_filter_with_fake_sprites(
    num_sprites: int = _NUM_SPRITES,
    sprite_size: tuple[int, int] = (60, 20),
) -> MoustacheFilter:
    """
    Create a MoustacheFilter with synthetic BGRA sprites injected into
    ``_sprites``, bypassing file I/O and GPU setup.

    Parameters:
        num_sprites (int): Number of fake sprites to inject.
        sprite_size (tuple[int, int]): (width, height) of each sprite.

    Returns:
        MoustacheFilter: Filter with ``_sprites`` populated and no GPU
            state.
    """
    flt = MoustacheFilter()
    sw, sh = sprite_size
    for _ in range(num_sprites):
        # Dark grey sprite on a white background so alpha masking is
        # exercisable in unit tests without loading disk files.
        sprite = np.full((sh, sw, 4), 50, dtype=np.uint8)   # grey
        sprite[:, :, 3] = 255                                 # opaque
        flt._sprites.append(sprite)
    return flt


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

class TestMoustacheFilterIdentity:
    """Tests for filter name and initial state (REQ-MS-011, REQ-MS-004)."""

    def setup_method(self) -> None:
        """Create a fresh filter before each test."""
        self.flt = MoustacheFilter()

    def test_name(self) -> None:
        """
        Test that the filter name is 'Moustache' (REQ-MS-011).

        Expected: ``filter.name`` returns ``"Moustache"``.
        """
        assert self.flt.name == "Moustache"

    def test_enabled_by_default(self) -> None:
        """
        Test that the filter is enabled after construction (BaseFilter
        default; application sets it to False at registration time).

        Expected: ``filter.enabled`` is ``True`` after bare construction.
        """
        assert self.flt.enabled is True

    def test_default_moustache_index_param(self) -> None:
        """
        Test that ``moustache_index`` defaults to 0 in params
        (REQ-MS-004).

        Expected: ``params['moustache_index']`` is ``0``.
        """
        assert self.flt.params["moustache_index"] == 0

    def test_face_result_is_none_by_default(self) -> None:
        """
        Test that no face result is stored before any injection.

        Expected: internal ``_face_result`` is ``None`` on construction.
        """
        assert self.flt._face_result is None

    def test_sprites_empty_before_setup(self) -> None:
        """
        Test that no sprites are cached before ``setup()`` is called.

        Expected: ``_sprites`` is an empty list.
        """
        assert self.flt._sprites == []


# ---------------------------------------------------------------------------
# Param validation
# ---------------------------------------------------------------------------

class TestMoustacheFilterParamValidation:
    """
    Tests for ``moustache_index`` clamping (REQ-MS-005, AC-MS-004,
    AC-MS-005).
    """

    def setup_method(self) -> None:
        """Create a fresh filter before each test."""
        self.flt = MoustacheFilter()

    def test_valid_index_returned_unchanged(self) -> None:
        """
        Test that a valid index in [0, 5] is returned without clamping.

        Expected: ``_get_index()`` returns the set value.
        """
        for i in range(_NUM_SPRITES):
            self.flt.params["moustache_index"] = i
            assert self.flt._get_index() == i

    def test_negative_index_clamped_to_zero(self) -> None:
        """
        Test that a negative index is clamped to 0 (AC-MS-004).

        Expected: ``_get_index()`` returns 0 when set to -1.
        """
        self.flt.params["moustache_index"] = -1
        assert self.flt._get_index() == 0

    def test_large_index_clamped_to_max(self) -> None:
        """
        Test that an out-of-range large index clamps to ``_NUM_SPRITES - 1``
        (AC-MS-005).

        Expected: ``_get_index()`` returns 5 when set to 99.
        """
        self.flt.params["moustache_index"] = 99
        assert self.flt._get_index() == _NUM_SPRITES - 1

    def test_non_integer_value_defaults_to_zero(self) -> None:
        """
        Test that a non-integer value in params is coerced or defaults
        to 0.

        Expected: no exception; index 0 is returned.
        """
        self.flt.params["moustache_index"] = "bad"
        assert self.flt._get_index() == 0


# ---------------------------------------------------------------------------
# Face result injection
# ---------------------------------------------------------------------------

class TestMoustacheFilterFaceResult:
    """
    Tests for face tracking data injection via ``update_face_result``
    (REQ-MS-010, AC-MS-010).
    """

    def setup_method(self) -> None:
        """Create a fresh filter before each test."""
        self.flt = MoustacheFilter()

    def test_stores_face_result(self) -> None:
        """
        Test that update_face_result stores the supplied result.

        Expected: ``_face_result`` matches the supplied result.
        """
        result = _make_result()
        self.flt.update_face_result(result)
        assert self.flt._face_result is result

    def test_stores_none(self) -> None:
        """
        Test that update_face_result accepts None without error
        (AC-MS-010).

        Expected: ``_face_result`` is set to ``None`` and no exception
        is raised.
        """
        self.flt.update_face_result(None)
        assert self.flt._face_result is None

    def test_replaces_previous_result(self) -> None:
        """
        Test that a new result replaces the old one.

        Expected: after two calls, only the second result is stored.
        """
        first = _make_result(yaw=10.0)
        second = _make_result(yaw=20.0)
        self.flt.update_face_result(first)
        self.flt.update_face_result(second)
        assert self.flt._face_result is second


# ---------------------------------------------------------------------------
# _has_face guard
# ---------------------------------------------------------------------------

class TestMoustacheFilterHasFace:
    """
    Tests for the ``_has_face`` guard that controls whether the
    moustache overlay is drawn (REQ-MS-009, AC-MS-001).
    """

    def setup_method(self) -> None:
        """Create a fresh filter before each test."""
        self.flt = MoustacheFilter()

    def test_false_when_no_result(self) -> None:
        """
        Test that ``_has_face`` returns False when result is None.

        Expected: False before any ``update_face_result`` call.
        """
        assert self.flt._has_face() is False

    def test_false_when_face_not_detected(self) -> None:
        """
        Test that ``_has_face`` returns False when face_detected is False.

        Expected: False even when landmarks are present (AC-MS-001).
        """
        result = _make_result(face_detected=False)
        self.flt.update_face_result(result)
        assert self.flt._has_face() is False

    def test_false_when_landmarks_too_few(self) -> None:
        """
        Test that ``_has_face`` returns False when fewer landmarks than
        the maximum required index are present.

        Expected: False when only 1 landmark is provided.
        """
        result = _make_result(
            landmarks=[_make_landmark()], face_detected=True
        )
        self.flt.update_face_result(result)
        assert self.flt._has_face() is False

    def test_true_when_face_detected_with_full_landmarks(self) -> None:
        """
        Test that ``_has_face`` returns True when face is detected and
        sufficient landmarks are present.

        Expected: True with 478 landmarks and face_detected=True.
        """
        result = _make_result(face_detected=True)
        self.flt.update_face_result(result)
        assert self.flt._has_face() is True


# ---------------------------------------------------------------------------
# Anchor computation
# ---------------------------------------------------------------------------

class TestMoustacheFilterAnchorCalc:
    """
    Tests for ``_compute_anchor`` (spec §6.5, AC-MS-007).
    """

    def _landmarks_with(
        self,
        mouth_left_x: float = 0.4,
        mouth_right_x: float = 0.6,
        nose_tip_y: float = 0.5,
        upper_lip_y: float = 0.55,
    ) -> List[Landmark]:
        """
        Build a 478-point landmark list with specific positions for the
        four anchor landmarks; all others default to (0.5, 0.5).

        Parameters:
            mouth_left_x (float): x of landmark 61 (left mouth corner).
            mouth_right_x (float): x of landmark 291 (right mouth corner).
            nose_tip_y (float): y of landmark 1 (nose tip).
            upper_lip_y (float): y of landmark 13 (upper lip centre).

        Returns:
            List[Landmark]: 478-element landmark list.
        """
        lm = _make_landmarks(478)
        lm[_LM_MOUTH_LEFT] = _make_landmark(x=mouth_left_x)
        lm[_LM_MOUTH_RIGHT] = _make_landmark(x=mouth_right_x)
        lm[_LM_NOSE_TIP] = _make_landmark(y=nose_tip_y)
        lm[_LM_UPPER_LIP] = _make_landmark(y=upper_lip_y)
        return lm

    def test_centre_x(self) -> None:
        """
        Test that centre_x is the midpoint of the mouth corners scaled
        to frame width (AC-MS-007).

        Given mouth_left_x=0.3, mouth_right_x=0.7, frame_width=100,
        expected cx = int((0.3 + 0.7) / 2 * 100) = 50.
        """
        lm = self._landmarks_with(mouth_left_x=0.3, mouth_right_x=0.7)
        cx, _, _, _ = MoustacheFilter._compute_anchor(
            lm, frame_width=100, frame_height=100, roll=0.0
        )
        assert cx == 50

    def test_centre_y(self) -> None:
        """
        Test that centre_y is the midpoint of nose tip and upper lip
        scaled to frame height (AC-MS-007).

        Given nose_tip_y=0.4, upper_lip_y=0.6, frame_height=200,
        expected cy = int((0.4 + 0.6) / 2 * 200) = 100.
        """
        lm = self._landmarks_with(nose_tip_y=0.4, upper_lip_y=0.6)
        _, cy, _, _ = MoustacheFilter._compute_anchor(
            lm, frame_width=200, frame_height=200, roll=0.0
        )
        assert cy == 100

    def test_render_width(self) -> None:
        """
        Test that render_width equals mouth distance × scale.

        Given mouth_left_x=0.3, mouth_right_x=0.7, frame_width=1000,
        mouth distance = 0.4 * 1000 = 400,
        expected rw = int(400 * _WIDTH_SCALE).
        """
        lm = self._landmarks_with(mouth_left_x=0.3, mouth_right_x=0.7)
        _, _, rw, _ = MoustacheFilter._compute_anchor(
            lm, frame_width=1000, frame_height=720, roll=0.0
        )
        # Use the same floating-point path as the implementation to
        # avoid rounding discrepancies (e.g. 0.7 - 0.3 ≠ 0.4 exactly).
        expected = int(abs(0.7 - 0.3) * 1000 * _WIDTH_SCALE)
        assert rw == expected

    def test_render_width_minimum_enforced(self) -> None:
        """
        Test that render_width is never less than ``_MIN_RENDER_WIDTH``.

        Given zero mouth distance (both corners at same x), expected
        render_width == _MIN_RENDER_WIDTH.
        """
        lm = self._landmarks_with(mouth_left_x=0.5, mouth_right_x=0.5)
        _, _, rw, _ = MoustacheFilter._compute_anchor(
            lm, frame_width=1280, frame_height=720, roll=0.0
        )
        assert rw == _MIN_RENDER_WIDTH

    def test_roll_passed_through(self) -> None:
        """
        Test that the head roll angle is passed through unchanged.

        Expected: the fourth return value equals the supplied roll.
        """
        lm = _make_landmarks(478)
        _, _, _, returned_roll = MoustacheFilter._compute_anchor(
            lm, frame_width=1280, frame_height=720, roll=42.5
        )
        assert returned_roll == pytest.approx(42.5)


# ---------------------------------------------------------------------------
# Alpha mask
# ---------------------------------------------------------------------------

class TestMoustacheFilterAlphaMask:
    """
    Tests for ``_apply_alpha_mask`` (REQ-MS-002, AC-MS-009).
    """

    def test_white_pixels_become_transparent(self) -> None:
        """
        Test that pixels with luminance ≥ threshold become alpha=0.

        Given a row of pure-white BGRA pixels (255, 255, 255, 255),
        expected alpha channel = 0.
        """
        white = np.full((1, 4, 4), 255, dtype=np.uint8)
        result = MoustacheFilter._apply_alpha_mask(white.copy())
        assert np.all(result[:, :, 3] == 0)

    def test_dark_pixels_become_opaque(self) -> None:
        """
        Test that pixels with luminance < threshold become alpha=255.

        Given a row of black BGRA pixels (0, 0, 0, 255), expected alpha
        channel = 255.
        """
        black = np.zeros((1, 4, 4), dtype=np.uint8)
        black[:, :, 3] = 255
        result = MoustacheFilter._apply_alpha_mask(black.copy())
        assert np.all(result[:, :, 3] == 255)

    def test_near_threshold_boundary(self) -> None:
        """
        Test alpha assignment at the exact luminance threshold boundary.

        A pixel with all channels exactly at ``_ALPHA_THRESHOLD`` has
        luminance = threshold → alpha = 0.
        A pixel with value ``_ALPHA_THRESHOLD - 1`` has luminance just
        below threshold → alpha = 255.
        """
        t = _ALPHA_THRESHOLD

        # Pixel at threshold → transparent.
        at_threshold = np.full((1, 1, 4), t, dtype=np.uint8)
        result_at = MoustacheFilter._apply_alpha_mask(at_threshold.copy())
        assert result_at[0, 0, 3] == 0

        # Pixel just below threshold → opaque.
        below_threshold = np.full((1, 1, 4), t - 1, dtype=np.uint8)
        result_below = MoustacheFilter._apply_alpha_mask(
            below_threshold.copy()
        )
        assert result_below[0, 0, 3] == 255


# ---------------------------------------------------------------------------
# _draw_moustache — CPU canvas drawing
# ---------------------------------------------------------------------------

class TestMoustacheFilterDrawMoustache:
    """
    Tests for ``_draw_moustache`` CPU canvas drawing (AC-MS-001,
    AC-MS-002).
    """

    def test_blank_canvas_when_no_face(self) -> None:
        """
        Test that the RGBA canvas is fully transparent when no face is
        detected (AC-MS-001, REQ-MS-009).

        Expected: all alpha values are 0, no pixel modification.
        """
        flt = _make_filter_with_fake_sprites()
        flt.update_face_result(None)
        canvas = flt._draw_moustache(320, 240)
        assert canvas.shape == (240, 320, 4)
        assert np.all(canvas[:, :, 3] == 0)

    def test_blank_canvas_when_face_not_detected(self) -> None:
        """
        Test that the canvas is blank when face_detected is False
        (AC-MS-001).

        Expected: all alpha values are 0.
        """
        flt = _make_filter_with_fake_sprites()
        result = _make_result(face_detected=False)
        flt.update_face_result(result)
        canvas = flt._draw_moustache(320, 240)
        assert np.all(canvas[:, :, 3] == 0)

    def test_moustache_drawn_when_face_detected(self) -> None:
        """
        Test that at least one non-transparent pixel appears when a face
        is detected (AC-MS-002).

        Expected: at least one pixel with alpha > 0 in the output canvas.
        """
        flt = _make_filter_with_fake_sprites()
        result = _make_result(face_detected=True)
        flt.update_face_result(result)
        canvas = flt._draw_moustache(640, 480)
        # Some pixels should be non-transparent (moustache rendered).
        assert np.any(canvas[:, :, 3] > 0)

    def test_canvas_shape_matches_frame(self) -> None:
        """
        Test that the returned RGBA canvas always matches the supplied
        frame dimensions.

        Expected: canvas shape is (height, width, 4).
        """
        flt = _make_filter_with_fake_sprites()
        flt.update_face_result(None)
        for w, h in [(1280, 720), (640, 480), (320, 240)]:
            canvas = flt._draw_moustache(w, h)
            assert canvas.shape == (h, w, 4)

    def test_different_indices_produce_different_canvases(
        self,
    ) -> None:
        """
        Test that different moustache indices produce different overlays
        (REQ-MS-006, AC-MS-003).

        Given two sprites of different colours, canvases for index 0 and
        index 1 must differ.
        """
        flt = MoustacheFilter()
        sw, sh = 60, 20
        # Sprite 0: dark red
        s0 = np.zeros((sh, sw, 4), dtype=np.uint8)
        s0[:, :, 2] = 150  # red channel
        s0[:, :, 3] = 255
        # Sprite 1: dark blue
        s1 = np.zeros((sh, sw, 4), dtype=np.uint8)
        s1[:, :, 0] = 150  # blue channel
        s1[:, :, 3] = 255
        # Fill out remaining sprites with defaults.
        for _ in range(_NUM_SPRITES - 2):
            flt._sprites.append(np.zeros((sh, sw, 4), dtype=np.uint8))
        flt._sprites.insert(0, s0)
        flt._sprites.insert(1, s1)
        result = _make_result(face_detected=True)
        flt.update_face_result(result)

        flt.params["moustache_index"] = 0
        canvas0 = flt._draw_moustache(640, 480)
        flt.params["moustache_index"] = 1
        canvas1 = flt._draw_moustache(640, 480)

        assert not np.array_equal(canvas0, canvas1)
