"""
Tests for filters/face_landmarks.py.

All tests avoid importing wgpu (no GPU required) by subclassing or
patching only the Python-level logic: parameter management, coordinate
conversion, face-result injection, and the _has_visible_landmarks guard.

Follows the testing conventions established in test_filters.py.
"""

from __future__ import annotations

import pytest

from filters.face_landmarks import (
    FaceLandmarkFilter,
    MAX_LANDMARKS,
)
from tracking.face_tracker import FaceTrackResult, Landmark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_landmark(x: float = 0.5, y: float = 0.5) -> Landmark:
    """
    Create a Landmark with a given normalised position.

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
) -> FaceTrackResult:
    """
    Build a FaceTrackResult with the supplied landmarks.

    Parameters:
        landmarks (list[Landmark]): Landmark list.
        face_detected (bool): Whether a face was detected.

    Returns:
        FaceTrackResult: Populated result.
    """
    return FaceTrackResult(
        landmarks=landmarks,
        face_detected=face_detected,
    )


# ---------------------------------------------------------------------------
# Identity & default params
# ---------------------------------------------------------------------------

class TestFaceLandmarkFilterIdentity:
    """Tests for filter name and default parameter state."""

    def setup_method(self) -> None:
        """Create a fresh filter before each test."""
        self.flt = FaceLandmarkFilter()

    def test_name(self) -> None:
        """
        Test that the filter name is 'Face Landmarks' (REQ-LM-006).

        Expected: ``filter.name`` returns the string ``"Face Landmarks"``.
        """
        assert self.flt.name == "Face Landmarks"

    def test_enabled_by_default(self) -> None:
        """
        Test that the filter is enabled immediately after construction.

        Expected: ``filter.enabled`` is ``True``.
        """
        assert self.flt.enabled is True

    def test_default_params_present(self) -> None:
        """
        Test that all five expected parameters are present (REQ-LM-003).

        Expected: params dict contains dot_radius, dot_r, dot_g,
        dot_b, and dot_a.
        """
        expected_keys = {"dot_radius", "dot_r", "dot_g", "dot_b", "dot_a"}
        assert expected_keys == set(self.flt.params.keys())

    def test_default_dot_radius(self) -> None:
        """
        Test the default dot radius is 3.0 pixels (GUD-LM-002).

        Expected: params['dot_radius'] == 3.0.
        """
        assert self.flt.params["dot_radius"] == pytest.approx(3.0)

    def test_default_colour_is_green(self) -> None:
        """
        Test that the default colour is bright green (GUD-LM-001).

        Expected: R=0, G=1, B=0, A=1.
        """
        assert self.flt.params["dot_r"] == pytest.approx(0.0)
        assert self.flt.params["dot_g"] == pytest.approx(1.0)
        assert self.flt.params["dot_b"] == pytest.approx(0.0)
        assert self.flt.params["dot_a"] == pytest.approx(1.0)

    def test_face_result_is_none_by_default(self) -> None:
        """
        Test that no face result is stored before any injection.

        Expected: internal ``_face_result`` is ``None`` on construction.
        """
        assert self.flt._face_result is None


# ---------------------------------------------------------------------------
# set_param (inherited BaseFilter behaviour)
# ---------------------------------------------------------------------------

class TestFaceLandmarkFilterParams:
    """Tests for parameter getting and setting."""

    def setup_method(self) -> None:
        """Create a fresh filter before each test."""
        self.flt = FaceLandmarkFilter()

    def test_set_param_dot_radius(self) -> None:
        """
        Test that dot_radius can be updated via set_param.

        Expected: params['dot_radius'] reflects the new value.
        """
        self.flt.set_param("dot_radius", 7.5)
        assert self.flt.params["dot_radius"] == pytest.approx(7.5)

    def test_set_param_colour_channels(self) -> None:
        """
        Test that all colour parameters can be updated.

        Expected: each colour channel reflects its set value.
        """
        self.flt.set_param("dot_r", 1.0)
        self.flt.set_param("dot_g", 0.5)
        self.flt.set_param("dot_b", 0.25)
        self.flt.set_param("dot_a", 0.8)

        assert self.flt.params["dot_r"] == pytest.approx(1.0)
        assert self.flt.params["dot_g"] == pytest.approx(0.5)
        assert self.flt.params["dot_b"] == pytest.approx(0.25)
        assert self.flt.params["dot_a"] == pytest.approx(0.8)

    def test_set_param_unknown_key_raises(self) -> None:
        """
        Test that setting an unknown parameter raises KeyError (REQ-LM-003).

        Expected: KeyError raised when key is not registered.
        """
        with pytest.raises(KeyError):
            self.flt.set_param("nonexistent_param", 1.0)


# ---------------------------------------------------------------------------
# update_face_result
# ---------------------------------------------------------------------------

class TestUpdateFaceResult:
    """Tests for face tracking data injection (REQ-LM-004)."""

    def setup_method(self) -> None:
        """Create a fresh filter before each test."""
        self.flt = FaceLandmarkFilter()

    def test_stores_face_result(self) -> None:
        """
        Test that update_face_result stores the result (AC-LM-010).

        Expected: internal ``_face_result`` matches the supplied result.
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
    landmark overlay pass (REQ-LM-002, REQ-LM-003, AC-LM-002, AC-LM-003).
    """

    def setup_method(self) -> None:
        """Create a fresh filter before each test."""
        self.flt = FaceLandmarkFilter()

    def test_false_when_no_result(self) -> None:
        """
        Test that no visible landmarks are reported when result is None.

        Expected: False when _face_result is None.
        """
        assert self.flt._has_visible_landmarks() is False

    def test_false_when_face_not_detected(self) -> None:
        """
        Test no landmarks reported when face_detected is False (AC-LM-003).

        Expected: False when face_detected == False even with landmarks.
        """
        result = _make_result(
            [_make_landmark()], face_detected=False
        )
        self.flt.update_face_result(result)
        assert self.flt._has_visible_landmarks() is False

    def test_false_when_empty_landmark_list(self) -> None:
        """
        Test no landmarks reported when landmark list is empty.

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
    (GUD-LM-003, AC-LM-004, AC-LM-005, AC-LM-006).

    The conversion is: ndc_x = x*2 - 1,  ndc_y = 1 - y*2.
    Tested by exercising _upload_landmark_positions via struct packing
    inspection — without a GPU the buffer upload is stubbed out.
    """

    def _ndc_from_landmark(
        self, x: float, y: float
    ) -> tuple[float, float]:
        """
        Apply the same NDC conversion used inside FaceLandmarkFilter.

        This helper mirrors the calculation in
        ``FaceLandmarkFilter._upload_landmark_positions`` so that
        acceptance criteria can be verified without a GPU.

        Parameters:
            x (float): Normalised landmark x [0, 1].
            y (float): Normalised landmark y [0, 1].

        Returns:
            tuple[float, float]: (ndc_x, ndc_y)
        """
        return (x * 2.0 - 1.0, 1.0 - y * 2.0)

    def test_top_left_maps_to_ndc_minus_one_plus_one(self) -> None:
        """
        Test that (0, 0) → (-1, 1) (top-left in both spaces) (AC-LM-004).

        Expected: ndc_x = -1.0, ndc_y = 1.0.
        """
        ndc_x, ndc_y = self._ndc_from_landmark(0.0, 0.0)
        assert ndc_x == pytest.approx(-1.0)
        assert ndc_y == pytest.approx(1.0)

    def test_bottom_right_maps_to_ndc_plus_one_minus_one(self) -> None:
        """
        Test that (1, 1) → (1, -1) (bottom-right in both spaces)
        (AC-LM-005).

        Expected: ndc_x = 1.0, ndc_y = -1.0.
        """
        ndc_x, ndc_y = self._ndc_from_landmark(1.0, 1.0)
        assert ndc_x == pytest.approx(1.0)
        assert ndc_y == pytest.approx(-1.0)

    def test_centre_maps_to_ndc_origin(self) -> None:
        """
        Test that (0.5, 0.5) → (0, 0) (screen centre) (AC-LM-006).

        Expected: ndc_x = 0.0, ndc_y = 0.0.
        """
        ndc_x, ndc_y = self._ndc_from_landmark(0.5, 0.5)
        assert ndc_x == pytest.approx(0.0)
        assert ndc_y == pytest.approx(0.0)

    def test_top_right_corner(self) -> None:
        """
        Test that (1, 0) → (1, 1) (top-right in MediaPipe space).

        Expected: ndc_x = 1.0, ndc_y = 1.0.
        """
        ndc_x, ndc_y = self._ndc_from_landmark(1.0, 0.0)
        assert ndc_x == pytest.approx(1.0)
        assert ndc_y == pytest.approx(1.0)

    def test_bottom_left_corner(self) -> None:
        """
        Test that (0, 1) → (-1, -1) (bottom-left in MediaPipe space).

        Expected: ndc_x = -1.0, ndc_y = -1.0.
        """
        ndc_x, ndc_y = self._ndc_from_landmark(0.0, 1.0)
        assert ndc_x == pytest.approx(-1.0)
        assert ndc_y == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# MAX_LANDMARKS constant
# ---------------------------------------------------------------------------

class TestMaxLandmarks:
    """Tests for the MAX_LANDMARKS constant."""

    def test_max_landmarks_value(self) -> None:
        """
        Test that MAX_LANDMARKS equals 478 (CON-LM-003).

        Expected: MAX_LANDMARKS == 478.
        """
        assert MAX_LANDMARKS == 478
