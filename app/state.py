"""
Application state module.

Holds all mutable state that is shared between the rendering pipeline,
face tracker, UI widget panel, and mini-game system.  State mutations
happen exclusively on the main thread; the render loop reads atomically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from filters.base import BaseFilter
    from games.base import BaseGame
    from overlays.base import BaseOverlay
    from tracking.face_tracker import FaceTrackResult


@dataclass
class AppState:
    """
    Central mutable state for the Camera Mayham application.

    All fields are read/written only from the main thread.  The render
    pipeline and UI panel share a single AppState instance passed at
    construction time.
    """

    # ------------------------------------------------------------------
    # Runtime control
    # ------------------------------------------------------------------
    running: bool = True
    """Set to False to request a clean application shutdown."""

    # ------------------------------------------------------------------
    # Camera frame — last captured BGR frame from OpenCV
    # ------------------------------------------------------------------
    camera_frame: Optional[np.ndarray] = None
    """Latest raw camera frame (H×W×3, dtype=uint8, BGR channel order)."""

    camera_width: int = 1280
    """Capture / render width in pixels (PER-001)."""

    camera_height: int = 720
    """Capture / render height in pixels (PER-001)."""

    # ------------------------------------------------------------------
    # Face tracking — result of the most recent MediaPipe inference pass
    # ------------------------------------------------------------------
    face_result: Optional[FaceTrackResult] = None
    """Latest face tracking result, or None when no face is detected."""

    # ------------------------------------------------------------------
    # Filter chain — ordered list of active GPU shader filters (REQ-002)
    # ------------------------------------------------------------------
    filters: List[BaseFilter] = field(default_factory=list)
    """
    Ordered list of registered filters.  Each filter exposes an
    ``enabled`` flag that the pipeline respects per frame (REQ-003).
    """

    # ------------------------------------------------------------------
    # 3D overlays — head-tracked model rendered above filters (REQ-007)
    # ------------------------------------------------------------------
    active_overlay: Optional[BaseOverlay] = None
    """Currently active 3D overlay model, or None for no overlay."""

    # ------------------------------------------------------------------
    # Mini-game — single active game instance at a time (REQ-008)
    # ------------------------------------------------------------------
    active_game: Optional[BaseGame] = None
    """Currently running mini-game, or None when no game is active."""

    # ------------------------------------------------------------------
    # Diagnostics — updated each frame by the render pipeline
    # ------------------------------------------------------------------
    last_frame_time_ms: float = 0.0
    """GPU frame time for the most recently completed frame (ms)."""

    frame_count: int = 0
    """Total number of frames rendered since application start."""

    def register_filter(self, flt: BaseFilter) -> None:
        """
        Add a filter to the end of the filter chain.

        Parameters:
            flt (BaseFilter): The filter instance to register.
        """
        self.filters.append(flt)

    def remove_filter(self, name: str) -> None:
        """
        Remove a filter from the chain by name.

        Parameters:
            name (str): The unique name of the filter to remove.

        Raises:
            KeyError: If no filter with the given name exists.
        """
        for i, f in enumerate(self.filters):
            if f.name == name:
                self.filters.pop(i)
                return
        raise KeyError(f"Filter '{name}' not found in the filter chain.")

    def get_filter(self, name: str) -> Optional[BaseFilter]:
        """
        Retrieve a registered filter by name.

        Parameters:
            name (str): The unique name of the filter.

        Returns:
            Optional[BaseFilter]: The filter instance, or None if not found.
        """
        for flt in self.filters:
            if flt.name == name:
                return flt
        return None

    def enabled_filters(self) -> List[BaseFilter]:
        """
        Return only the filters that are currently enabled.

        Returns:
            List[BaseFilter]: Filters with ``enabled == True`` in order.
        """
        return [f for f in self.filters if f.enabled]

    def launch_game(self, game: BaseGame) -> None:
        """
        Replace the active mini-game, stopping the previous one cleanly.

        Parameters:
            game (BaseGame): The new game instance to launch.
        """
        if self.active_game is not None:
            self.active_game.stop()
        self.active_game = game
        game.start()

    def stop_game(self) -> None:
        """Stop and clear the active mini-game if one is running."""
        if self.active_game is not None:
            self.active_game.stop()
            self.active_game = None
