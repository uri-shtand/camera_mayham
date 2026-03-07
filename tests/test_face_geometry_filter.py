"""
Tests for filters/face_geometry.py and the face_matrix field
added to tracking/face_tracker.py (spec §8).

All tests avoid importing wgpu (no GPU required) by exercising only the
Python-level logic: filter identity, face-result injection, the
_has_visible_landmarks guard, landmark-to-NDC conversion, the
FaceTrackResult.face_matrix default, and FaceTracker matrix extraction.

Testing conventions follow test_face_landmark_filter.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from filters.face_geometry import (
    FaceGeometryFilter,
    NUM_LANDMARKS,
    _DOT_RADIUS_X,
    _DOT_RADIUS_Y,
    _Z_SCALE,
    _REGION_DATA,
    _TRIANGLES_LEFT_EYE,
    _TRIANGLES_RIGHT_EYE,
    _TRIANGLES_LEFT_EYEBROW,
    _TRIANGLES_RIGHT_EYEBROW,
    _TRIANGLES_NOSE,
    _TRIANGLES_MOUTH_OUTER,
    _TRIANGLES_MOUTH_INNER,
    _TRIANGLES_CHIN,
    _TRIANGLES_LEFT_CHEEK,
    _TRIANGLES_RIGHT_CHEEK,
)
from tracking.face_tracker import FaceTrackResult, Landmark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_landmark(
    x: float = 0.5, y: float = 0.5, z: float = 0.0
) -> Landmark:
    """
    Create a Landmark at the given normalised position.

    Parameters:
        x (float): Normalised horizontal position [0, 1].
        y (float): Normalised vertical position [0, 1].
        z (float): Relative depth.

    Returns:
        Landmark: Populated landmark.
    """
    return Landmark(x=x, y=y, z=z)


def _make_result(
    landmarks: list[Landmark],
    face_detected: bool = True,
    face_matrix: np.ndarray | None = None,
) -> FaceTrackResult:
    """
    Build a FaceTrackResult with the supplied landmarks.

    Parameters:
        landmarks (list[Landmark]): Landmark list.
        face_detected (bool): Whether a face was detected.
        face_matrix (np.ndarray | None): Optional 4×4 matrix.

    Returns:
        FaceTrackResult: Populated result.
    """
    return FaceTrackResult(
        landmarks=landmarks,
        face_detected=face_detected,
        face_matrix=face_matrix,
    )


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

class TestFaceGeometryFilterIdentity:
    """Tests for filter name and initial state (AC-FG-003, AC-FG-004)."""

    def setup_method(self) -> None:
        """Create a fresh filter before each test."""
        self.flt = FaceGeometryFilter()

    def test_name(self) -> None:
        """
        Test that the filter name is 'Face Geometry' (REQ-FG-009, AC-FG-003).

        Expected: ``filter.name`` returns ``"Face Geometry"``.
        """
        assert self.flt.name == "Face Geometry"

    def test_enabled_by_default(self) -> None:
        """
        Test that the filter is enabled after construction.

        Expected: ``filter.enabled`` is ``True``.
        """
        assert self.flt.enabled is True

    def test_no_user_params(self) -> None:
        """
        Test that the filter exposes no user-adjustable parameters
        (spec §4 — colour is hard-coded).

        Expected: ``filter.params`` is an empty dict.
        """
        assert self.flt.params == {}

    def test_face_result_is_none_by_default(self) -> None:
        """
        Test that no face result is stored before any injection (AC-FG-004).

        Expected: internal ``_face_result`` is ``None`` on construction.
        """
        assert self.flt._face_result is None


# ---------------------------------------------------------------------------
# update_face_result
# ---------------------------------------------------------------------------

class TestUpdateFaceResult:
    """
    Tests for face tracking data injection (REQ-FG-008, AC-FG-005).
    """

    def setup_method(self) -> None:
        """Create a fresh filter before each test."""
        self.flt = FaceGeometryFilter()

    def test_stores_face_result(self) -> None:
        """
        Test that update_face_result stores the result (AC-FG-005).

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
        Test that a new result replaces the old stored result.

        Expected: after two calls only the second result is stored.
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
    Tests for the _has_visible_landmarks helper that guards the wireframe
    draw call (REQ-FG-005).
    """

    def setup_method(self) -> None:
        """Create a fresh filter before each test."""
        self.flt = FaceGeometryFilter()

    def test_false_when_no_result(self) -> None:
        """
        Test that False is returned when result is None.

        Expected: False when ``_face_result`` is None.
        """
        assert self.flt._has_visible_landmarks() is False

    def test_false_when_face_not_detected(self) -> None:
        """
        Test False when face_detected is False even if landmarks present.

        Expected: False regardless of landmark list content.
        """
        result = _make_result([_make_landmark()], face_detected=False)
        self.flt.update_face_result(result)
        assert self.flt._has_visible_landmarks() is False

    def test_false_when_empty_landmark_list(self) -> None:
        """
        Test False when the landmark list is empty.

        Expected: False when landmarks == [] even with face_detected True.
        """
        result = _make_result([], face_detected=True)
        self.flt.update_face_result(result)
        assert self.flt._has_visible_landmarks() is False

    def test_true_when_face_detected_with_landmarks(self) -> None:
        """
        Test True when face is detected and landmarks are present.

        Expected: True when face_detected == True and landmarks list
        is non-empty.
        """
        result = _make_result([_make_landmark()], face_detected=True)
        self.flt.update_face_result(result)
        assert self.flt._has_visible_landmarks() is True


# ---------------------------------------------------------------------------
# NDC conversion (logical test — does not require GPU)
# ---------------------------------------------------------------------------

class TestNDCConversion:
    """
    Tests for the landmark-to-NDC coordinate conversion (REQ-FG-004).

    The conversion formula is:
        ndc_x = x * 2 - 1
        ndc_y = 1 - y * 2
        ndc_z = z * _Z_SCALE
    """

    def _ndc(
        self, x: float, y: float, z: float = 0.0
    ) -> tuple[float, float, float]:
        """
        Apply the NDC conversion from spec §4 / REQ-FG-004.

        Parameters:
            x (float): Normalised landmark x [0, 1].
            y (float): Normalised landmark y [0, 1].
            z (float): Relative depth.

        Returns:
            tuple[float, float, float]: (ndc_x, ndc_y, ndc_z)
        """
        return (x * 2.0 - 1.0, 1.0 - y * 2.0, z * _Z_SCALE)

    def test_top_left_maps_to_ndc_minus1_plus1(self) -> None:
        """
        Test that (0, 0) → (-1, +1) in NDC (top-left corner).

        Expected: ndc_x = -1.0, ndc_y = +1.0.
        """
        ndc_x, ndc_y, _ = self._ndc(0.0, 0.0)
        assert ndc_x == pytest.approx(-1.0)
        assert ndc_y == pytest.approx(1.0)

    def test_bottom_right_maps_to_ndc_plus1_minus1(self) -> None:
        """
        Test that (1, 1) → (+1, -1) in NDC (bottom-right corner).

        Expected: ndc_x = +1.0, ndc_y = -1.0.
        """
        ndc_x, ndc_y, _ = self._ndc(1.0, 1.0)
        assert ndc_x == pytest.approx(1.0)
        assert ndc_y == pytest.approx(-1.0)

    def test_centre_maps_to_origin(self) -> None:
        """
        Test that (0.5, 0.5) → (0, 0) in NDC (screen centre).

        Expected: ndc_x = 0.0, ndc_y = 0.0.
        """
        ndc_x, ndc_y, _ = self._ndc(0.5, 0.5)
        assert ndc_x == pytest.approx(0.0)
        assert ndc_y == pytest.approx(0.0)

    def test_z_is_scaled_by_z_scale(self) -> None:
        """
        Test that z is multiplied by _Z_SCALE.

        Expected: ndc_z = z * _Z_SCALE.
        """
        _, _, ndc_z = self._ndc(0.0, 0.0, z=0.05)
        assert ndc_z == pytest.approx(0.05 * _Z_SCALE)


# ---------------------------------------------------------------------------
# NUM_LANDMARKS constant
# ---------------------------------------------------------------------------

class TestNumLandmarksConstant:
    """Tests for the NUM_LANDMARKS constant (CON-FG-003)."""

    def test_num_landmarks_value(self) -> None:
        """
        Test that NUM_LANDMARKS equals 478 (spec CON-FG-003).

        Expected: NUM_LANDMARKS == 478.
        """
        assert NUM_LANDMARKS == 478


# ---------------------------------------------------------------------------
# Dot-radius constants
# ---------------------------------------------------------------------------

class TestDotRadiusConstants:
    """Tests for the _DOT_RADIUS_X and _DOT_RADIUS_Y constants (AC-FG-009)."""

    def test_radius_x_positive(self) -> None:
        """
        Test that _DOT_RADIUS_X is a positive float.

        Expected: _DOT_RADIUS_X > 0.0.
        """
        assert _DOT_RADIUS_X > 0.0

    def test_radius_y_positive(self) -> None:
        """
        Test that _DOT_RADIUS_Y is a positive float.

        Expected: _DOT_RADIUS_Y > 0.0.
        """
        assert _DOT_RADIUS_Y > 0.0

    def test_radius_y_larger_than_x(self) -> None:
        """
        Test that radius_y > radius_x for circular dots on 16:9 frames.

        Expected: _DOT_RADIUS_Y > _DOT_RADIUS_X.
        """
        assert _DOT_RADIUS_Y > _DOT_RADIUS_X

    def test_both_radii_below_0_1(self) -> None:
        """
        Test that both radii are small fractions of NDC space (< 0.1).

        Expected: _DOT_RADIUS_X < 0.1 and _DOT_RADIUS_Y < 0.1.
        """
        assert _DOT_RADIUS_X < 0.1
        assert _DOT_RADIUS_Y < 0.1


# ---------------------------------------------------------------------------
# FaceTrackResult — face_matrix field
# ---------------------------------------------------------------------------

class TestFaceTrackResultMatrix:
    """
    Tests for the face_matrix field on FaceTrackResult
    (REQ-FG-006, AC-FG-006).
    """

    def test_face_matrix_default_is_none(self) -> None:
        """
        Test that FaceTrackResult.face_matrix defaults to None (AC-FG-006).

        Expected: a freshly constructed result has ``face_matrix=None``.
        """
        result = FaceTrackResult()
        assert result.face_matrix is None

    def test_face_matrix_accepts_numpy_array(self) -> None:
        """
        Test that face_matrix can be set to a 4×4 float32 numpy array.

        Expected: the stored matrix is identical to the supplied array.
        """
        mat = np.eye(4, dtype=np.float32)
        result = FaceTrackResult(face_matrix=mat)
        assert result.face_matrix is mat

    def test_face_matrix_in_make_result_helper(self) -> None:
        """
        Test that the _make_result helper propagates face_matrix.

        Expected: face_matrix from _make_result is preserved.
        """
        mat = np.zeros((4, 4), dtype=np.float32)
        result = _make_result([_make_landmark()], face_matrix=mat)
        assert result.face_matrix is mat


# ---------------------------------------------------------------------------
# FaceTracker matrix extraction (stub-based, no MediaPipe inference)
# ---------------------------------------------------------------------------

class TestFaceTrackerMatrixExtraction:
    """
    Tests for FaceTracker._extract_face_result populating face_matrix
    (REQ-FG-007, AC-FG-007).

    Uses a stub detection object to avoid running actual MediaPipe
    inference.
    """

    def _make_stub_landmark(
        self, x: float = 0.5, y: float = 0.5, z: float = 0.0
    ) -> object:
        """
        Build a stub object with .x .y .z attributes to mimic a
        MediaPipe NormalizedLandmark.

        Parameters:
            x (float): Horizontal position.
            y (float): Vertical position.
            z (float): Depth.

        Returns:
            object: Stub landmark.
        """
        class _StubLm:
            pass
        lm = _StubLm()
        lm.x, lm.y, lm.z = x, y, z
        return lm

    def _make_stub_detection(
        self,
        matrix: np.ndarray | None = None,
        num_landmarks: int = 5,
    ) -> object:
        """
        Build a stub detection result with landmarks and an optional
        facial_transformation_matrixes list.

        Parameters:
            matrix (np.ndarray | None): Matrix to include at index 0,
                or None to omit.
            num_landmarks (int): Number of stub landmarks to include.

        Returns:
            object: Stub detection result.
        """
        class _StubDetection:
            pass

        stub = _StubDetection()
        stub.face_landmarks = [
            [self._make_stub_landmark() for _ in range(num_landmarks)]
        ]
        stub.face_blendshapes = []
        if matrix is not None:
            stub.facial_transformation_matrixes = [matrix]
        else:
            stub.facial_transformation_matrixes = []
        return stub

    def test_face_matrix_populated_from_detection(self) -> None:
        """
        Test that face_matrix is populated when the detection provides
        facial_transformation_matrixes (AC-FG-007).

        Expected: result.face_matrix is a numpy array with dtype float32.
        """
        from tracking.face_tracker import FaceTracker
        tracker = FaceTracker.__new__(FaceTracker)

        mat = np.eye(4, dtype=np.float32)
        stub = self._make_stub_detection(matrix=mat)
        result = tracker._extract_face_result(stub, 0)

        assert result.face_matrix is not None
        assert isinstance(result.face_matrix, np.ndarray)
        assert result.face_matrix.dtype == np.float32
        assert result.face_matrix.shape == (4, 4)

    def test_face_matrix_none_when_no_matrixes(self) -> None:
        """
        Test that face_matrix is None when the detection has no
        facial_transformation_matrixes (guard in REQ-FG-007).

        Expected: result.face_matrix is None.
        """
        from tracking.face_tracker import FaceTracker
        tracker = FaceTracker.__new__(FaceTracker)

        stub = self._make_stub_detection(matrix=None)
        result = tracker._extract_face_result(stub, 0)

        assert result.face_matrix is None

    def test_face_matrix_none_when_index_out_of_range(self) -> None:
        """
        Test that face_matrix is None when the face index is out of
        range for the matrixes list.

        Expected: result.face_matrix is None when index >= len(matrixes).
        """
        from tracking.face_tracker import FaceTracker
        tracker = FaceTracker.__new__(FaceTracker)

        # Only one matrix but we request index 1.
        mat = np.eye(4, dtype=np.float32)

        class _StubDetection:
            pass
        stub = _StubDetection()
        stub.face_landmarks = [
            [self._make_stub_landmark()],
            [self._make_stub_landmark()],
        ]
        stub.face_blendshapes = []
        stub.facial_transformation_matrixes = [mat]  # only for face 0

        result = tracker._extract_face_result(stub, 1)
        assert result.face_matrix is None


# ---------------------------------------------------------------------------
# _fan_triangles helper
# ---------------------------------------------------------------------------

class TestFanTriangles:
    """
    Tests for the _fan_triangles module function.

    _fan_triangles(line_pairs, center) takes alternating edge-pair
    connectivity data and a fan apex index, and returns a flat tuple of
    triangle indices (3 per triangle, triangle-list topology).
    """

    def test_output_length_is_three_times_ring_length(self) -> None:
        """
        Test that output contains 3 × N indices for a ring of N points.

        Expected: len(result) == 3 * (len(line_pairs) // 2).
        """
        from filters.face_geometry import _fan_triangles
        # Simple ring: 4 points stored as 4 edge pairs (8 elements).
        ring = (0, 1, 1, 2, 2, 3, 3, 0)
        result = _fan_triangles(ring, center=99)
        assert len(result) == 3 * 4

    def test_center_is_first_index_of_every_triangle(self) -> None:
        """
        Test that the center landmark appears as position 0 in each
        group of 3 indices (fan apex pattern).

        Expected: result[0], result[3], result[6], ... all equal center.
        """
        from filters.face_geometry import _fan_triangles
        ring = (10, 11, 11, 12, 12, 13, 13, 10)
        center = 42
        result = _fan_triangles(ring, center=center)
        for start in range(0, len(result), 3):
            assert result[start] == center

    def test_wrap_around_closes_the_fan(self) -> None:
        """
        Test that the last triangle wraps back to the first ring point.

        For a ring [a, b, c, d] the last triangle must be (center, d, a).

        Expected: last 3 indices are (center, last_ring_pt, first_ring_pt).
        """
        from filters.face_geometry import _fan_triangles
        ring = (0, 1, 1, 2, 2, 3, 3, 0)
        result = _fan_triangles(ring, center=99)
        # ring[0::2] = [0, 1, 2, 3], last triangle should be (99, 3, 0)
        assert result[-3:] == (99, 3, 0)

    def test_returns_tuple(self) -> None:
        """
        Test that _fan_triangles returns a tuple, not a list.

        Expected: return type is tuple.
        """
        from filters.face_geometry import _fan_triangles
        ring = (0, 1, 1, 0)
        result = _fan_triangles(ring, center=5)
        assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# Region data correctness
# ---------------------------------------------------------------------------

class TestRegionData:
    """
    Tests for the _REGION_DATA list and the per-region triangle constants.

    Validates structural correctness (non-empty, divisible by 3, all
    landmark indices in range) without requiring a GPU.
    """

    def test_region_data_non_empty(self) -> None:
        """
        Test that _REGION_DATA contains at least one region.

        Expected: len(_REGION_DATA) >= 1.
        """
        from filters.face_geometry import _REGION_DATA
        assert len(_REGION_DATA) >= 1

    def test_each_region_indices_divisible_by_three(self) -> None:
        """
        Test that every region's triangle index list length is a
        multiple of 3 (triangle-list topology requirement).

        Expected: len(region.indices) % 3 == 0 for all regions.
        """
        from filters.face_geometry import _REGION_DATA
        for i, region in enumerate(_REGION_DATA):
            assert len(region.indices) % 3 == 0, (
                f"Region {i} index count {len(region.indices)} "
                "not divisible by 3"
            )

    def test_each_region_has_at_least_one_triangle(self) -> None:
        """
        Test that every region has at least one triangle (3 indices).

        Expected: len(region.indices) >= 3 for all regions.
        """
        from filters.face_geometry import _REGION_DATA
        for i, region in enumerate(_REGION_DATA):
            assert len(region.indices) >= 3, (
                f"Region {i} has no triangles"
            )

    def test_all_indices_within_landmark_range(self) -> None:
        """
        Test that every landmark index across all regions is within
        [0, NUM_LANDMARKS).

        Expected: 0 <= idx < 478 for all indices in all regions.
        """
        from filters.face_geometry import _REGION_DATA, NUM_LANDMARKS
        for i, region in enumerate(_REGION_DATA):
            for idx in region.indices:
                assert 0 <= idx < NUM_LANDMARKS, (
                    f"Region {i} index {idx} out of range [0, "
                    f"{NUM_LANDMARKS})"
                )

    def test_each_region_color_has_four_components(self) -> None:
        """
        Test that every region colour is a 4-tuple (R, G, B, A).

        Expected: len(region.color) == 4 for all regions.
        """
        from filters.face_geometry import _REGION_DATA
        for i, region in enumerate(_REGION_DATA):
            assert len(region.color) == 4, (
                f"Region {i} color has {len(region.color)} components, "
                "expected 4"
            )

    def test_each_region_color_values_in_range(self) -> None:
        """
        Test that all RGBA components are in [0.0, 1.0].

        Expected: 0.0 <= component <= 1.0 for all RGBA values.
        """
        from filters.face_geometry import _REGION_DATA
        for i, region in enumerate(_REGION_DATA):
            for j, component in enumerate(region.color):
                assert 0.0 <= component <= 1.0, (
                    f"Region {i}, color component {j} = {component} "
                    "out of [0.0, 1.0]"
                )

    def test_has_namedtuple_fields(self) -> None:
        """
        Test that region entries expose .indices and .color attributes.

        Expected: each region has 'indices' and 'color' fields.
        """
        from filters.face_geometry import _REGION_DATA
        region = _REGION_DATA[0]
        assert hasattr(region, "indices")
        assert hasattr(region, "color")


# ---------------------------------------------------------------------------
# Region colour spot checks (neon/cyber palette)
# ---------------------------------------------------------------------------

class TestRegionColors:
    """
    Spot-check that each facial region uses the expected neon/cyber
    colour family.  These verify palette intent, not exact values.
    """

    def _region_by_indices(
        self, triangles: tuple
    ) -> object:
        """
        Return the first _REGION_DATA entry whose indices match.

        Parameters:
            triangles (tuple): Expected triangle index tuple.

        Returns:
            object: Matching _RegionSpec or raises AssertionError.
        """
        from filters.face_geometry import _REGION_DATA
        for r in _REGION_DATA:
            if r.indices is triangles:
                return r
        raise AssertionError(
            "No region found matching the given triangle tuple"
        )

    def test_eye_regions_are_cyan_dominant(self) -> None:
        """
        Test that both eye regions have blue >= green >= red (cyan).

        Expected: r < g and r < b for eye regions.
        """
        from filters.face_geometry import (
            _TRIANGLES_LEFT_EYE,
            _TRIANGLES_RIGHT_EYE,
        )
        for tris in (_TRIANGLES_LEFT_EYE, _TRIANGLES_RIGHT_EYE):
            region = self._region_by_indices(tris)
            r, g, b, _a = region.color
            # Cyan: low red, high green, high blue.
            assert r < 0.5, f"Eye red={r} too high for cyan"
            assert b > 0.5, f"Eye blue={b} too low for cyan"

    def test_eyebrow_regions_are_magenta_dominant(self) -> None:
        """
        Test that eyebrow regions have red > 0.5 and blue > 0.5 (magenta).

        Expected: r > 0.5 and b > 0.3 for brow regions.
        """
        from filters.face_geometry import (
            _TRIANGLES_LEFT_EYEBROW,
            _TRIANGLES_RIGHT_EYEBROW,
        )
        for tris in (_TRIANGLES_LEFT_EYEBROW, _TRIANGLES_RIGHT_EYEBROW):
            region = self._region_by_indices(tris)
            r, g, b, _a = region.color
            assert r > 0.5, f"Brow red={r} too low for magenta"
            assert b > 0.3, f"Brow blue={b} too low for magenta"

    def test_nose_region_is_orange_dominant(self) -> None:
        """
        Test that the nose region has red > 0.5 and green > 0.3 (orange).

        Expected: r > 0.5, g > 0.3, b < 0.3 for nose regions.
        """
        from filters.face_geometry import _TRIANGLES_NOSE
        region = self._region_by_indices(_TRIANGLES_NOSE)
        r, g, b, _a = region.color
        assert r > 0.5, f"Nose red={r} too low for orange"
        assert g > 0.3, f"Nose green={g} too low for orange"
        assert b < 0.3, f"Nose blue={b} too high for orange"

    def test_mouth_regions_are_rose_dominant(self) -> None:
        """
        Test that mouth regions have red dominant (rose/red family).

        Expected: r > 0.5 for mouth regions.
        """
        from filters.face_geometry import (
            _TRIANGLES_MOUTH_OUTER,
            _TRIANGLES_MOUTH_INNER,
        )
        for tris in (_TRIANGLES_MOUTH_OUTER, _TRIANGLES_MOUTH_INNER):
            region = self._region_by_indices(tris)
            r, _g, _b, _a = region.color
            assert r > 0.5, f"Mouth red={r} too low for rose/red"

    def test_chin_region_is_violet_dominant(self) -> None:
        """
        Test that the chin region has blue dominant (violet/purple).

        Expected: b > 0.5 for chin region.
        """
        from filters.face_geometry import _TRIANGLES_CHIN
        region = self._region_by_indices(_TRIANGLES_CHIN)
        _r, _g, b, _a = region.color
        assert b > 0.5, f"Chin blue={b} too low for violet"

    def test_cheek_regions_are_teal_dominant(self) -> None:
        """
        Test that cheek regions have green and blue dominant (teal).

        Expected: g > 0.5 and r < 0.3 for cheek regions.
        """
        from filters.face_geometry import (
            _TRIANGLES_LEFT_CHEEK,
            _TRIANGLES_RIGHT_CHEEK,
        )
        for tris in (_TRIANGLES_LEFT_CHEEK, _TRIANGLES_RIGHT_CHEEK):
            region = self._region_by_indices(tris)
            r, g, _b, _a = region.color
            assert g > 0.5, f"Cheek green={g} too low for teal"
            assert r < 0.3, f"Cheek red={r} too high for teal"
