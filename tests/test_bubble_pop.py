"""
Tests for games/base.py and games/bubble_pop.py.

BubblePop game logic tests run without GPU hardware by calling update()
directly and inspecting state rather than exercising the render path.
"""

from __future__ import annotations

import pytest

from games.base import GameState
from games.bubble_pop import (
    _JAW_OPEN_THRESHOLD,
    _MAX_MISSES,
    BubblePopGame,
    Bubble,
)
from tracking.face_tracker import FaceTrackResult, Landmark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_face_result(jaw_open: float = 0.0, nose_x: float = 0.5) -> FaceTrackResult:
    """
    Create a minimal FaceTrackResult with controlled jawOpen and nose x.

    Parameters:
        jaw_open (float): Coefficient for the jawOpen blendshape.
        nose_x (float): Normalised x position of the nose-tip landmark.

    Returns:
        FaceTrackResult: Configured result.
    """
    result = FaceTrackResult(face_detected=True)
    # Build 478 landmarks; only index 1 (nose tip) matters for pop logic
    result.landmarks = [Landmark(0.5, 0.5, 0.0)] * 478
    result.landmarks[1] = Landmark(nose_x, 0.5, 0.0)
    result.blendshapes = {"jawOpen": jaw_open}
    return result


# ---------------------------------------------------------------------------
# GameState tests
# ---------------------------------------------------------------------------

class TestGameState:
    """Tests for the GameState enum."""

    def test_enum_values_exist(self):
        """
        Test that IDLE, RUNNING, and FINISHED states are defined.

        Expected: all three enum members exist on GameState.
        """
        assert GameState.IDLE
        assert GameState.RUNNING
        assert GameState.FINISHED


# ---------------------------------------------------------------------------
# BubblePopGame lifecycle tests (no GPU)
# ---------------------------------------------------------------------------

class TestBubblePopGameLifecycle:
    """Tests for BubblePopGame start/stop lifecycle without GPU."""

    def setup_method(self):
        """Create a BubblePopGame instance without calling setup()."""
        self.game = BubblePopGame()

    def test_initial_state_idle(self):
        """
        Test that the game starts in IDLE state.

        Expected: game.state == GameState.IDLE.
        """
        assert self.game.state == GameState.IDLE

    def test_initial_score_zero(self):
        """
        Test that the initial score is 0.

        Expected: game.score == 0.
        """
        assert self.game.score == 0

    def test_name_is_bubble_pop(self):
        """
        Test that the game name is 'BubblePop'.

        Expected: game.name == 'BubblePop'.
        """
        assert self.game.name == "BubblePop"

    def test_start_transitions_to_running(self):
        """
        Test that start() sets state to RUNNING.

        Expected: game.state == GameState.RUNNING after start().
        """
        self.game.start()
        assert self.game.state == GameState.RUNNING

    def test_start_resets_score(self):
        """
        Test that start() resets score to 0.

        Expected: game.score == 0 after start(), regardless of prior score.
        """
        self.game.score = 99
        self.game.start()
        assert self.game.score == 0

    def test_stop_transitions_to_idle(self):
        """
        Test that stop() sets state to IDLE.

        Expected: game.state == GameState.IDLE after stop().
        """
        self.game.start()
        self.game.stop()
        assert self.game.state == GameState.IDLE

    def test_stop_clears_bubbles(self):
        """
        Test that stop() clears the active bubble list.

        Expected: no bubbles remain after stop().
        """
        self.game.start()
        self.game._bubbles.append(Bubble(0.0, 0.0, 0.05, (1, 1, 1, 1), 0.1))
        self.game.stop()
        assert len(self.game._bubbles) == 0


# ---------------------------------------------------------------------------
# BubblePopGame update logic tests
# ---------------------------------------------------------------------------

class TestBubblePopGameUpdate:
    """Tests for BubblePopGame.update game logic (no GPU required)."""

    def setup_method(self):
        """Start a fresh game ready for update() calls."""
        self.game = BubblePopGame()
        self.game.start()

    def test_update_does_nothing_when_idle(self):
        """
        Test that update() is a no-op when the game is not RUNNING.

        Expected: no exception, score stays 0, state stays IDLE.
        """
        self.game.stop()  # transition to IDLE
        self.game.update(FaceTrackResult(), 0.016)
        assert self.game.score == 0

    def test_bubble_falls_over_time(self):
        """
        Test that a bubble's y position decreases after each update.

        Expected: bubble.y < initial_y after one update tick.
        """
        bubble = Bubble(0.0, 0.8, 0.05, (1, 1, 1, 1), 0.2)
        self.game._bubbles.append(bubble)
        initial_y = bubble.y
        self.game.update(FaceTrackResult(), 0.1)
        assert bubble.y < initial_y

    def test_escaped_bubble_increments_miss_count(self):
        """
        Test that a bubble leaving the bottom increments misses.

        Expected: _misses increases when bubble.y < -1.2.
        """
        bubble = Bubble(0.0, -1.3, 0.05, (1, 1, 1, 1), 0.0)
        self.game._bubbles.append(bubble)
        self.game.update(FaceTrackResult(), 0.016)
        assert self.game._misses == 1

    def test_max_misses_triggers_game_over(self):
        """
        Test that reaching _MAX_MISSES transitions to FINISHED.

        Expected: game.state == GameState.FINISHED after max misses.
        """
        self.game._misses = _MAX_MISSES - 1
        # Add a bubble that will escape immediately
        self.game._bubbles.append(Bubble(0.0, -2.0, 0.05, (1, 1, 1, 1), 0.0))
        self.game.update(FaceTrackResult(), 0.016)
        assert self.game.state == GameState.FINISHED

    def test_mouth_open_pops_nearby_bubble(self):
        """
        Test that opening the mouth pops a bubble near the nose position.

        Expected: score increments and bubble is removed when jawOpen
        exceeds threshold and nose is close to a bubble.
        """
        # Place a bubble at NDC (0, 0) — matches nose at x=0.5
        bubble = Bubble(0.0, 0.0, 0.1, (1, 1, 1, 1), 0.0)
        self.game._bubbles.append(bubble)

        # Simulate mouth closed → mouth open (edge detection)
        self.game._jaw_was_open = False
        face = _make_face_result(jaw_open=_JAW_OPEN_THRESHOLD, nose_x=0.5)
        self.game.update(face, 0.016)

        assert self.game.score == 1
        assert bubble not in self.game._bubbles

    def test_bubble_not_popped_when_mouth_stays_open(self):
        """
        Test that holding the mouth open does NOT repeatedly pop bubbles.

        Expected: score increments only once per open event (edge detect).
        """
        bubble1 = Bubble(0.0, 0.0, 0.1, (1, 1, 1, 1), 0.0)
        bubble2 = Bubble(0.0, 0.0, 0.1, (1, 1, 1, 1), 0.0)
        self.game._bubbles.extend([bubble1, bubble2])

        face = _make_face_result(jaw_open=_JAW_OPEN_THRESHOLD, nose_x=0.5)

        # First open — should pop one bubble
        self.game._jaw_was_open = False
        self.game.update(face, 0.016)
        score_after_first = self.game.score

        # Mouth stays open — should NOT pop again
        self.game._jaw_was_open = True
        self.game.update(face, 0.016)
        score_after_second = self.game.score

        assert score_after_second == score_after_first

    def test_far_bubble_not_popped(self):
        """
        Test that a mouth-open event does not pop a distant bubble.

        Expected: score stays 0 when no bubble is within pop range.
        """
        # Bubble far from nose position (NDC x=0 vs nose x=0.5→NDC 0)
        bubble = Bubble(0.9, 0.9, 0.05, (1, 1, 1, 1), 0.0)
        self.game._bubbles.append(bubble)

        face = _make_face_result(jaw_open=_JAW_OPEN_THRESHOLD, nose_x=0.5)
        self.game._jaw_was_open = False
        self.game.update(face, 0.016)

        assert self.game.score == 0
