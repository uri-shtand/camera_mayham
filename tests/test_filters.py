"""
Tests for filters/base.py and the three concrete filter implementations.

These tests validate the filter interface contract (parameter management,
enable/disable) using a lightweight stub that avoids importing wgpu so
the tests run without a GPU.
"""

from __future__ import annotations

import pytest

from filters.base import BaseFilter


# ---------------------------------------------------------------------------
# Concrete stub — used to test the abstract base class contract
# ---------------------------------------------------------------------------

class _StubFilter(BaseFilter):
    """
    Minimal concrete filter for testing BaseFilter without GPU resources.

    Implements all abstract methods as no-ops so tests can exercise the
    base class logic in isolation.
    """

    @property
    def name(self) -> str:
        """
        Returns:
            str: 'Stub'
        """
        return "Stub"

    def _build_pipeline(self, device, texture_format):
        """No-op pipeline build for testing."""
        pass

    def apply(self, encoder, input_texture, output_texture):
        """No-op apply for testing."""
        pass


# ---------------------------------------------------------------------------
# BaseFilter tests
# ---------------------------------------------------------------------------

class TestBaseFilter:
    """Tests for the BaseFilter abstract base class."""

    def setup_method(self):
        """Create a fresh stub filter before each test."""
        self.flt = _StubFilter()

    def test_enabled_by_default(self):
        """
        Test that a new filter is enabled by default.

        Expected behaviour: ``filter.enabled`` is True immediately after
        construction.
        """
        assert self.flt.enabled is True

    def test_params_empty_by_default(self):
        """
        Test that a base filter has an empty params dict.

        Expected: stub filter inherits empty params from BaseFilter.__init__.
        """
        assert self.flt.params == {}

    def test_set_registered_param(self):
        """
        Test that ``set_param`` updates a registered parameter.

        Expected: after set_param('key', value) the params dict reflects
        the new value.
        """
        self.flt.params["key"] = 0.5
        self.flt.set_param("key", 0.8)
        assert self.flt.params["key"] == pytest.approx(0.8)

    def test_set_unregistered_param_raises(self):
        """
        Test that ``set_param`` raises KeyError for unknown parameters.

        Expected: KeyError with a message naming the unknown key.
        """
        with pytest.raises(KeyError, match="unknown_key"):
            self.flt.set_param("unknown_key", 1.0)

    def test_repr(self):
        """
        Test that the repr contains the filter name and enabled state.

        Expected: repr string includes class name, filter name, enabled flag.
        """
        r = repr(self.flt)
        assert "Stub" in r
        assert "enabled=True" in r

    def test_teardown_clears_device(self):
        """
        Test that teardown clears the device reference.

        Expected: _device is None after teardown even if it was set.
        """
        self.flt._device = object()  # simulate a device reference
        self.flt.teardown()
        assert self.flt._device is None


# ---------------------------------------------------------------------------
# GrayscaleFilter tests (parameters only — no GPU)
# ---------------------------------------------------------------------------

class TestGrayscaleFilter:
    """Tests for GrayscaleFilter parameter defaults and set_param."""

    def setup_method(self):
        """Import and instantiate GrayscaleFilter."""
        from filters.grayscale import GrayscaleFilter  # noqa: PLC0415

        self.flt = GrayscaleFilter()

    def test_name(self):
        """
        Test that the filter name is 'Grayscale'.

        Expected: name property returns the exact string 'Grayscale'.
        """
        assert self.flt.name == "Grayscale"

    def test_default_strength(self):
        """
        Test that default strength is 1.0 (full greyscale).

        Expected: params['strength'] == 1.0 after construction.
        """
        assert self.flt.params["strength"] == pytest.approx(1.0)

    def test_set_strength(self):
        """
        Test updating the strength parameter via set_param.

        Expected: params dict reflects the new value.
        """
        self.flt.set_param("strength", 0.4)
        assert self.flt.params["strength"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# EdgeDetectionFilter tests
# ---------------------------------------------------------------------------

class TestEdgeDetectionFilter:
    """Tests for EdgeDetectionFilter parameter defaults."""

    def setup_method(self):
        """Import and instantiate EdgeDetectionFilter."""
        from filters.edge_detection import EdgeDetectionFilter  # noqa: PLC0415

        self.flt = EdgeDetectionFilter()

    def test_name(self):
        """
        Test that the filter name is 'EdgeDetection'.

        Expected: name property returns 'EdgeDetection'.
        """
        assert self.flt.name == "EdgeDetection"

    def test_default_intensity(self):
        """
        Test that default intensity is greater than zero.

        Expected: params['intensity'] > 0.
        """
        assert self.flt.params["intensity"] > 0

    def test_default_edge_colour_is_tuple(self):
        """
        Test that edge_colour is a 3-element tuple.

        Expected: params['edge_colour'] is a tuple of length 3.
        """
        colour = self.flt.params["edge_colour"]
        assert isinstance(colour, tuple)
        assert len(colour) == 3


# ---------------------------------------------------------------------------
# ColourShiftFilter tests
# ---------------------------------------------------------------------------

class TestColourShiftFilter:
    """Tests for ColourShiftFilter parameter defaults."""

    def setup_method(self):
        """Import and instantiate ColourShiftFilter."""
        from filters.colour_shift import ColourShiftFilter  # noqa: PLC0415

        self.flt = ColourShiftFilter()

    def test_name(self):
        """
        Test that the filter name is 'ColourShift'.

        Expected: name property returns 'ColourShift'.
        """
        assert self.flt.name == "ColourShift"

    def test_default_hue_shift_in_range(self):
        """
        Test that default hue_shift is within [0, 360].

        Expected: 0 <= params['hue_shift'] <= 360.
        """
        shift = self.flt.params["hue_shift"]
        assert 0 <= shift <= 360

    def test_default_saturation_positive(self):
        """
        Test that default saturation is positive.

        Expected: params['saturation'] > 0.
        """
        assert self.flt.params["saturation"] > 0
