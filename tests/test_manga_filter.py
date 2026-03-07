"""
Tests for filters/manga.py — MangaFilter class.

All tests are GPU-free: they exercise the Python interface only
(construction, parameter defaults, set_param, name, enabled state
and teardown).  No wgpu device or shader compilation is required.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# MangaFilter tests
# ---------------------------------------------------------------------------

class TestMangaFilter:
    """Tests for MangaFilter parameter defaults and interface contract."""

    def setup_method(self):
        """Import and instantiate MangaFilter before each test."""
        from filters.manga import MangaFilter  # noqa: PLC0415

        self.flt = MangaFilter()

    def test_name(self):
        """
        Test that the filter name is 'Manga'.

        Expected: name property returns the exact string 'Manga'.
        """
        assert self.flt.name == "Manga"

    def test_enabled_by_default(self):
        """
        Test that a newly constructed MangaFilter is enabled.

        Expected: ``filter.enabled`` is True immediately after
        construction.
        """
        assert self.flt.enabled is True

    def test_default_params_keys(self):
        """
        Test that exactly the three documented parameters are present.

        Expected: params dict has keys 'edge_threshold',
        'posterize_levels', and 'dot_scale'.
        """
        assert set(self.flt.params.keys()) == {
            "edge_threshold",
            "posterize_levels",
            "dot_scale",
        }

    def test_default_edge_threshold(self):
        """
        Test that edge_threshold defaults to 0.15.

        Expected: params['edge_threshold'] == 0.15 after construction.
        """
        assert self.flt.params["edge_threshold"] == pytest.approx(0.15)

    def test_default_posterize_levels(self):
        """
        Test that posterize_levels defaults to 4.0.

        Expected: params['posterize_levels'] == 4.0 after construction.
        """
        assert self.flt.params["posterize_levels"] == pytest.approx(4.0)

    def test_default_dot_scale(self):
        """
        Test that dot_scale defaults to 4.0.

        Expected: params['dot_scale'] == 4.0 after construction.
        """
        assert self.flt.params["dot_scale"] == pytest.approx(4.0)

    def test_set_param_updates_edge_threshold(self):
        """
        Test that set_param updates edge_threshold correctly.

        Expected: after set_param('edge_threshold', 0.3) the params
        dict reflects the new value.
        """
        self.flt.set_param("edge_threshold", 0.3)
        assert self.flt.params["edge_threshold"] == pytest.approx(0.3)

    def test_set_param_updates_posterize_levels(self):
        """
        Test that set_param updates posterize_levels correctly.

        Expected: after set_param('posterize_levels', 6.0) the params
        dict reflects the new value.
        """
        self.flt.set_param("posterize_levels", 6.0)
        assert self.flt.params["posterize_levels"] == pytest.approx(6.0)

    def test_set_param_updates_dot_scale(self):
        """
        Test that set_param updates dot_scale correctly.

        Expected: after set_param('dot_scale', 8.0) the params dict
        reflects the new value.
        """
        self.flt.set_param("dot_scale", 8.0)
        assert self.flt.params["dot_scale"] == pytest.approx(8.0)

    def test_set_unknown_param_raises(self):
        """
        Test that set_param raises KeyError for unknown parameters.

        Expected: KeyError raised with the unknown key name in the
        message.
        """
        with pytest.raises(KeyError, match="ink_colour"):
            self.flt.set_param("ink_colour", 1.0)

    def test_disable_filter(self):
        """
        Test that the filter can be disabled.

        Expected: setting ``filter.enabled = False`` persists; the
        filter remains constructable and its name is unchanged.
        """
        self.flt.enabled = False
        assert self.flt.enabled is False
        assert self.flt.name == "Manga"

    def test_teardown_clears_resources(self):
        """
        Test that teardown clears device and buffer references.

        Expected: _device is None after teardown even if it was set
        externally to simulate a live device reference.
        """
        self.flt._device = object()  # simulate a device reference
        self.flt._param_buffer = object()  # simulate a buffer
        self.flt.teardown()
        assert self.flt._device is None
        assert self.flt._param_buffer is None
