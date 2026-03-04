"""
Tests for app/state.py — AppState shared application state.

All tests run without GPU/camera hardware.
"""

from __future__ import annotations

import pytest

from app.state import AppState
from filters.base import BaseFilter


# ---------------------------------------------------------------------------
# Stub fixtures
# ---------------------------------------------------------------------------

class _StubFilter(BaseFilter):
    """Minimal filter stub for state tests."""

    def __init__(self, name: str, enabled: bool = True) -> None:
        super().__init__()
        self._name = name
        self.enabled = enabled
        self.params = {}

    @property
    def name(self) -> str:
        return self._name

    def _build_pipeline(self, device, texture_format):
        pass

    def apply(self, encoder, input_texture, output_texture):
        pass


class _StubGame:
    """
    Minimal game stub that satisfies the stop() interface.

    Does not inherit BaseGame to avoid wgpu imports.
    """

    def __init__(self, name: str = "StubGame") -> None:
        self._name = name
        self.stopped = False
        self.started = False

    @property
    def name(self) -> str:
        return self._name

    def start(self) -> None:
        """Record that start() was called."""
        self.started = True

    def stop(self) -> None:
        """Record that stop() was called."""
        self.stopped = True


# ---------------------------------------------------------------------------
# AppState tests
# ---------------------------------------------------------------------------

class TestAppState:
    """Tests for AppState filter chain and game lifecycle management."""

    def setup_method(self):
        """Create a fresh AppState before each test."""
        self.state = AppState()

    # -- Initial state ---------------------------------------------------

    def test_running_defaults_to_true(self):
        """
        Test that the running flag is True on creation.

        Expected: state.running == True.
        """
        assert self.state.running is True

    def test_no_filters_on_creation(self):
        """
        Test that no filters are registered on a new AppState.

        Expected: state.filters is an empty list.
        """
        assert self.state.filters == []

    def test_no_active_game_on_creation(self):
        """
        Test that no game is active on a new AppState.

        Expected: state.active_game is None.
        """
        assert self.state.active_game is None

    # -- Filter registration ---------------------------------------------

    def test_register_filter_appends(self):
        """
        Test that register_filter appends to the filter list.

        Expected: len(state.filters) == 1 after one registration.
        """
        self.state.register_filter(_StubFilter("A"))
        assert len(self.state.filters) == 1

    def test_register_multiple_filters_preserves_order(self):
        """
        Test that filter order matches registration order.

        Expected: filters appear in insertion order.
        """
        self.state.register_filter(_StubFilter("A"))
        self.state.register_filter(_StubFilter("B"))
        self.state.register_filter(_StubFilter("C"))
        names = [f.name for f in self.state.filters]
        assert names == ["A", "B", "C"]

    def test_remove_filter_by_name(self):
        """
        Test that remove_filter removes the named filter.

        Expected: filter is absent after removal; list length decreases.
        """
        self.state.register_filter(_StubFilter("A"))
        self.state.register_filter(_StubFilter("B"))
        self.state.remove_filter("A")
        names = [f.name for f in self.state.filters]
        assert "A" not in names
        assert len(self.state.filters) == 1

    def test_remove_missing_filter_raises(self):
        """
        Test that removing a non-existent filter raises KeyError.

        Expected: KeyError mentioning the missing filter name.
        """
        with pytest.raises(KeyError, match="missing"):
            self.state.remove_filter("missing")

    def test_get_filter_returns_correct_instance(self):
        """
        Test that get_filter retrieves the correct filter by name.

        Expected: returned instance has the queried name.
        """
        flt = _StubFilter("Target")
        self.state.register_filter(flt)
        result = self.state.get_filter("Target")
        assert result is flt

    def test_get_filter_returns_none_for_unknown(self):
        """
        Test that get_filter returns None for unregistered names.

        Expected: None returned, no exception raised.
        """
        assert self.state.get_filter("Unknown") is None

    # -- Enabled filters -------------------------------------------------

    def test_enabled_filters_returns_only_enabled(self):
        """
        Test that enabled_filters excludes disabled filters.

        Expected: only filters with enabled=True appear in the result.
        """
        self.state.register_filter(_StubFilter("On", enabled=True))
        self.state.register_filter(_StubFilter("Off", enabled=False))
        self.state.register_filter(_StubFilter("OnAgain", enabled=True))
        enabled = self.state.enabled_filters()
        names = [f.name for f in enabled]
        assert names == ["On", "OnAgain"]

    def test_enabled_filters_empty_when_all_disabled(self):
        """
        Test that enabled_filters returns empty list when all disabled.

        Expected: empty list when every filter is disabled.
        """
        self.state.register_filter(_StubFilter("A", enabled=False))
        assert self.state.enabled_filters() == []

    # -- Game lifecycle --------------------------------------------------

    def test_launch_game_sets_active_game(self):
        """
        Test that launch_game sets state.active_game.

        Expected: state.active_game is the launched game instance.
        """
        game = _StubGame()
        self.state.launch_game(game)
        assert self.state.active_game is game

    def test_launch_game_calls_start(self):
        """
        Test that launch_game calls start() on the new game.

        Expected: game.started == True after launch.
        """
        game = _StubGame()
        self.state.launch_game(game)
        assert game.started is True

    def test_launch_game_stops_previous(self):
        """
        Test that launching a new game stops the previous one.

        Expected: the first game's stop() is called before the second
        game starts.
        """
        first = _StubGame("First")
        second = _StubGame("Second")
        self.state.launch_game(first)
        self.state.launch_game(second)
        assert first.stopped is True
        assert self.state.active_game is second

    def test_stop_game_clears_active_game(self):
        """
        Test that stop_game sets active_game to None.

        Expected: state.active_game is None after stop_game().
        """
        game = _StubGame()
        self.state.launch_game(game)
        self.state.stop_game()
        assert self.state.active_game is None

    def test_stop_game_calls_stop(self):
        """
        Test that stop_game calls stop() on the game.

        Expected: game.stopped == True.
        """
        game = _StubGame()
        self.state.launch_game(game)
        self.state.stop_game()
        assert game.stopped is True

    def test_stop_game_noop_when_no_game(self):
        """
        Test that stop_game is safe when no game is active.

        Expected: no exception raised; active_game remains None.
        """
        self.state.stop_game()  # should not raise
        assert self.state.active_game is None
