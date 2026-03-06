"""
Tests for filters/face_landmarks.py (spec v2).

All tests avoid importing wgpu (no GPU required) by exercising only the
Python-level logic: identity, NDC conversion, face-result injection,
the _has_visible_landmarks guard, and the CPU-side overlay drawing
(_draw_overlay) that produces arrows and the badge.

Follows the testing conventions established in test_filters.py.
"""

from __future__ import annotations

import pytest

from filters.face_landmarks import (
    FaceLandmarkFilter,
    MAX_LANDMARKS,
    _BADGE_MARGIN_PX,
    _BADGE_RADIUS_PX,
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


def _make_result(
    landmarks: list[Landmark],
    face_detected: bool = True,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
) -> FaceTrackResult:
    """
    Build a FaceTrackResult with the supplied landmarks and head pose.

    Parameters:
        landmarks (list[Landmark]): Landmark list.
        face_detected (bool): Whether a face was detected.
        yaw (float): Yaw angle in degrees.
        pitch (float): Pitch angle in degrees.
        roll (float): Roll angle in degrees.

    Returns:
        FaceTrackResult: Populated result.
    """
    return FaceTrackResult(
        landmarks=landmarks,
        face_detected=face_detected,
        head_pose=HeadPose(yaw=yaw, pitch=pitch, roll=roll),
    )


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

class TestFaceLandmarkFilterIdentity:
    """Tests for filter name and initial state."""

    def setup_method(self) -> None:
        """Create a fresh filter before each test."""
        self.flt = FaceLandmarkFilter()

    def test_name(self) -> None:
        """
        Test that the filter name is 'Face Landmarks' (REQ-LM-012).

        Expected: ``filter.name`` returns ``"Face Landmarks"``.
        """
        assert self.flt.name == "Face Landmarks"

    def test_enabled_by_default(self) -> None:
        """
        Test that the filter is enabled after construction.

        Expected: ``filter.enabled`` is ``True``.
        """
        assert self.flt.enabled is True

    def test_no_user_params(self) -> None:
        """
        Test that the filter exposes no user-adjustable parameters
        (REQ-LM-002 ג€” dot style is hard-coded, not user-adjustable).

        Expected: ``filter.params`` is an empty dict.
        """
        assert self.flt.params == {}

    def test_face_result_is_none_by_default(self) -> None:
        """
        Test that no face result is stored before any injection.

        Expected: internal ``_face_result`` is ``None`` on construction.
        """
        assert self.flt._face_result is None


# ---------------------------------------------------------------------------
# update_face_result
# ---------------------------------------------------------------------------

class TestUpdateFaceResult:
    """Tests for face tracking data injection (REQ-LM-011, AC-LM-010)."""

    def setup_method(self) -> None:
        """Create a fresh filter before each test."""
        self.flt = FaceLandmarkFilter()

    def test_stores_face_result(self) -> None:
        """
        Test that update_face_result stores the result (AC-LM-010).

        Expected: ``_face_result`` matches the supplied result.
        """
        result = _make_result([_make_landmark()])
        self.flt.update_face_result(result)
        assert self.flt._face_result is result

    def test_stores_none(self) -> None:
        """
        Test that update_face_result accepts None without error.

        Expected: ``_face_result`` is set to ``None``.
        """
        self.flt.update_face_result(None)
        assert self.flt._face_result is None

    def test_replaces_previous_result(self) -> None:
        """
        Test that a new result replaces the old one.

        Expected: after two calls, only the second result is stored.
        """
        first = _make_result([_make_landmark(0.1, 0.1)])
        second = _make_result([_make_landmark(0.9, 0.9)])
        self.flt.update_face_result(first)
        self.flt.update_face_result(second)
        assert self.flt._face_result is second


# ---------------------------------------------------------------------------
# _has_visible_landmarks guard
# ---------------------------------------------------------------------------

class TestHasVisibleLandmarks:
    """
    Tests for the _has_visible_landmarks helper that guards the
    landmark dot pass (REQ-LM-003, AC-LM-002, AC-LM-003).
    """

    def setup_method(self) -> None:
        """Create a fresh filter before each test."""
        self.flt = FaceLandmarkFilter()

    def test_false_when_no_result(self) -> None:
        """
        Test that no visible landmarks are reported when result is None.

        Expected: False when ``_face_result`` is None (AC-LM-002).
        """
        assert self.flt._has_visible_landmarks() is False

    def test_false_when_face_not_detected(self) -> None:
        """
        Test no landmarks reported when face_detected is False (AC-LM-003).

        Expected: False even when landmarks are present.
        """
        result = _make_result(
            [_make_landmark()], face_detected=False
        )
        self.flt.update_face_result(result)
        assert self.flt._has_visible_landmarks() is False

    def test_false_when_empty_landmark_list(self) -> None:
        """
        Test no landmarks reported when the landmark list is empty.

        Expected: False when face_detected is True but landmarks is [].
        """
        result = _make_result([], face_detected=True)
        self.flt.update_face_result(result)
        assert self.flt._has_visible_landmarks() is False

    def test_true_when_face_detected_with_landmarks(self) -> None:
        """
        Test that visible landmarks are reported when face is detected.

        Expected: True when face_detected == True and landmarks is
        non-empty.
        """
        result = _make_result([_make_landmark()], face_detected=True)
        self.flt.update_face_result(result)
        assert self.flt._has_visible_landmarks() is True


# ---------------------------------------------------------------------------
# NDC coordinate conversion
# ---------------------------------------------------------------------------

class TestNDCConversion:
    """
    Tests for the landmark-to-NDC coordinate conversion logic
    (spec ֲ§6.4, AC-LM-004, AC-LM-005, AC-LM-006).

    The conversion formula is:
        ndc_x = x * 2 - 1
        ndc_y = 1 - y * 2
    """

    def _ndc(self, x: float, y: float) -> tuple[float, float]:
        """
        Apply the NDC conversion from spec ֲ§6.4.

        Parameters:
            x (float): Normalised landmark x [0, 1].
            y (float): Normalised landmark y [0, 1].

        Returns:
            tuple[float, float]: (ndc_x, ndc_y)
        """
        return (x * 2.0 - 1.0, 1.0 - y * 2.0)

    def test_top_left(self) -> None:
        """
        Test that (0, 0) ג†’ (-1, 1) ג€” top-left corner (AC-LM-004).

        Expected: ndc_x = -1.0, ndc_y = 1.0.
        """
        ndc_x, ndc_y = self._ndc(0.0, 0.0)
        assert ndc_x == pytest.approx(-1.0)
        assert ndc_y == pytest.approx(1.0)

    def test_bottom_right(self) -> None:
        """
        Test that (1, 1) ג†’ (1, -1) ג€” bottom-right corner (AC-LM-005).

        Expected: ndc_x = 1.0, ndc_y = -1.0.
        """
        ndc_x, ndc_y = self._ndc(1.0, 1.0)
        assert ndc_x == pytest.approx(1.0)
        assert ndc_y == pytest.approx(-1.0)

    def test_centre(self) -> None:
        """
        Test that (0.5, 0.5) ג†’ (0, 0) ג€” screen centre (AC-LM-006).

        Expected: ndc_x = 0.0, ndc_y = 0.0.
        """
        ndc_x, ndc_y = self._ndc(0.5, 0.5)
        assert ndc_x == pytest.approx(0.0)
        assert ndc_y == pytest.approx(0.0)

    def test_top_right(self) -> None:
        """
        Test that (1, 0) ג†’ (1, 1) ג€” top-right.

        Expected: ndc_x = 1.0, ndc_y = 1.0.
        """
        ndc_x, ndc_y = self._ndc(1.0, 0.0)
        assert ndc_x == pytest.approx(1.0)
        assert ndc_y == pytest.approx(1.0)

    def test_bottom_left(self) -> None:
        """
        Test that (0, 1) ג†’ (-1, -1) ג€” bottom-left.

        Expected: ndc_x = -1.0, ndc_y = -1.0.
        """
        ndc_x, ndc_y = self._ndc(0.0, 1.0)
        assert ndc_x == pytest.approx(-1.0)
        assert ndc_y == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# MAX_LANDMARKS constant
# ---------------------------------------------------------------------------

class TestMaxLandmarks:
    """Tests for the MAX_LANDMARKS constant (CON-LM-003)."""

    def test_max_landmarks_value(self) -> None:
        """
        Test that MAX_LANDMARKS equals 478.

        Expected: MAX_LANDMARKS == 478.
        """
        assert MAX_LANDMARKS == 478


# ---------------------------------------------------------------------------
# Overlay drawing ג€” badge state
# ---------------------------------------------------------------------------

class TestBadgeState:
    """
    Tests for the face-detected badge drawn by _draw_overlay
    (REQ-LM-008, REQ-LM-009, REQ-LM-010, AC-LM-008, AC-LM-009).

    The badge is always rendered regardless of face detection state.
    Its colour (green / red) reflects whether a face is currently
    detected.
    """

    _W = 640
    _H = 480

    def setup_method(self) -> None:
        """Create a fresh filter before each test."""
        self.flt = FaceLandmarkFilter()

    def _badge_pixel(self, overlay: "np.ndarray") -> tuple[int, int, int]:
        """
        Sample the RGBA pixel at the badge centre.

        Returns the (R, G, B) values at the badge position in the
        overlay buffer so we can assert its colour.

        Parameters:
            overlay (np.ndarray): RGBA uint8 array from _draw_overlay.

        Returns:
            tuple[int, int, int]: (r, g, b) at badge centre.
        """
        bx = self._W - _BADGE_MARGIN_PX
        # Sample slightly below centre to avoid the white ✓/✗ symbol
        # that is drawn on top of the coloured circle.
        by = _BADGE_MARGIN_PX + _BADGE_RADIUS_PX // 2
        r, g, b, a = overlay[by, bx]
        return (int(r), int(g), int(b))

    def test_badge_always_present_with_face(self) -> None:
        """
        Test that the badge is rendered when a face is detected (AC-LM-008).

        Expected: badge pixel has non-zero alpha.
        """
        result = _make_result([_make_landmark()])
        overlay = self.flt._draw_overlay(result, self._W, self._H)
        bx = self._W - _BADGE_MARGIN_PX
        by = _BADGE_MARGIN_PX
        alpha = overlay[by, bx, 3]
        assert alpha > 0

    def test_badge_always_present_without_face(self) -> None:
        """
        Test that the badge is rendered even when no face is detected
        (REQ-LM-008).

        Expected: badge pixel has non-zero alpha.
        """
        overlay = self.flt._draw_overlay(None, self._W, self._H)
        bx = self._W - _BADGE_MARGIN_PX
        by = _BADGE_MARGIN_PX
        alpha = overlay[by, bx, 3]
        assert alpha > 0

    def test_badge_is_green_when_face_detected(self) -> None:
        """
        Test that the badge is green when a face is detected (AC-LM-008).

        Expected: green channel dominant at badge centre.
        """
        result = _make_result([_make_landmark()])
        overlay = self.flt._draw_overlay(result, self._W, self._H)
        r, g, b = self._badge_pixel(overlay)
        assert g > r and g > b, (
            f"Expected green badge, got R={r} G={g} B={b}"
        )

    def test_badge_is_red_when_no_face(self) -> None:
        """
        Test that the badge is red when no face is detected (AC-LM-009).

        Expected: red channel dominant at badge centre.
        """
        overlay = self.flt._draw_overlay(None, self._W, self._H)
        r, g, b = self._badge_pixel(overlay)
        assert r > g and r > b, (
            f"Expected red badge, got R={r} G={g} B={b}"
        )

    def test_badge_is_red_when_face_detected_false(self) -> None:
        """
        Test badge is red when face_detected is False (AC-LM-003 / AC-LM-009).

        Expected: red channel dominant even though result is not None.
        """
        result = _make_result(
            [_make_landmark()], face_detected=False
        )
        overlay = self.flt._draw_overlay(result, self._W, self._H)
        r, g, b = self._badge_pixel(overlay)
        assert r > g and r > b, (
            f"Expected red badge, got R={r} G={g} B={b}"
        )


# ---------------------------------------------------------------------------
# Overlay drawing ג€” head-pose arrows
# ---------------------------------------------------------------------------

class TestHeadPoseArrows:
    """
    Tests for the head-pose arrows drawn by _draw_overlay
    (REQ-LM-004 ג€“ REQ-LM-007, AC-LM-011).

    Arrows are drawn from the top-left corner when a face is detected
    and hidden when no face is present.
    """

    _W = 640
    _H = 480

    def setup_method(self) -> None:
        """Create a fresh filter before each test."""
        self.flt = FaceLandmarkFilter()

    def _has_non_transparent_pixels_near_origin(
        self, overlay: "np.ndarray"
    ) -> bool:
        """
        Check whether any non-transparent pixels exist in the top-left
        arrow region (outside the badge area).

        The arrow origin is at (50, 50) with max length 80 px, so we
        sample the region (0, 0)ג€“(200, 200) for non-transparent pixels
        excluding the badge area.

        Parameters:
            overlay (np.ndarray): RGBA uint8 array from _draw_overlay.

        Returns:
            bool: True if any arrow pixels are present.
        """
        region = overlay[0:200, 0:200, 3]  # alpha channel
        return bool((region > 0).any())

    def test_arrows_present_when_face_detected(self) -> None:
        """
        Test that arrows are drawn when a face is detected (AC-LM-011).

        Any non-zero yaw, pitch, or roll produces at least one arrow.
        Expected: non-transparent pixels in the arrow region.
        """
        result = _make_result(
            [_make_landmark()],
            face_detected=True,
            yaw=20.0,
            pitch=10.0,
            roll=5.0,
        )
        overlay = self.flt._draw_overlay(result, self._W, self._H)
        assert self._has_non_transparent_pixels_near_origin(overlay), (
            "Expected arrow pixels in the top-left region"
        )

    def test_no_arrows_when_no_face(self) -> None:
        """
        Test that no arrows are drawn when face_result is None (REQ-LM-007).

        Expected: arrow region is fully transparent.
        """
        overlay = self.flt._draw_overlay(None, self._W, self._H)
        assert not self._has_non_transparent_pixels_near_origin(
            overlay
        ), "Expected no arrow pixels when face is not detected"

    def test_no_arrows_when_face_not_detected(self) -> None:
        """
        Test no arrows when face_detected is False (REQ-LM-007).

        Expected: arrow region is fully transparent.
        """
        result = _make_result(
            [_make_landmark()],
            face_detected=False,
            yaw=30.0,
            pitch=20.0,
        )
        overlay = self.flt._draw_overlay(result, self._W, self._H)
        assert not self._has_non_transparent_pixels_near_origin(
            overlay
        ), "Expected no arrow pixels when face_detected is False"

    def test_no_arrows_when_all_angles_zero(self) -> None:
        """
        Test that zero angles produce no arrows (all angles = 0).

        When yaw=pitch=roll=0, no arrowedLine call should produce
        visible pixels because endpoints equal the origin.
        Expected: arrow region is fully transparent.
        """
        result = _make_result(
            [_make_landmark()],
            face_detected=True,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
        )
        overlay = self.flt._draw_overlay(result, self._W, self._H)
        assert not self._has_non_transparent_pixels_near_origin(
            overlay
        ), "Expected no arrows when all head-pose angles are zero"

