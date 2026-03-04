"""Mini-games package — face-driven interactive games."""

from games.base import BaseGame, GameState  # noqa: F401
from games.bubble_pop import BubblePopGame  # noqa: F401

__all__ = ["BaseGame", "GameState", "BubblePopGame"]
