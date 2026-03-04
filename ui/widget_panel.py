"""
Widget panel — dearpygui-based control panel (REQ-010, §4.3).

The panel runs in its own OS window (separate from the wgpu render
window).  It is driven via :py:meth:`WidgetPanel.render_frame`, which
must be called once per iteration of the main loop — this replaces the
``dpg.start_dearpygui()`` blocking call and keeps the main loop in
control (CON-005: panel must not block the render loop).

Panel sections
--------------
* **Filters**   — per-filter toggle + parameter sliders (REQ-003, REQ-004)
* **Overlays**  — 3D overlay selector (REQ-007)
* **Games**     — launch / stop mini-games (REQ-008)
* **Diagnostics** — FPS and frame time readout
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

import dearpygui.dearpygui as dpg

if TYPE_CHECKING:
    from app.state import AppState
    from games.base import BaseGame

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tag constants — stable DearPyGui item tags for reliable lookups
# ---------------------------------------------------------------------------
_TAG_WINDOW = "main_window"
_TAG_FPS_TEXT = "fps_text"
_TAG_FRAME_MS_TEXT = "frame_ms_text"
_TAG_GAME_STATUS = "game_status"


class WidgetPanel:
    """
    Interactive side-panel built with Dear PyGui (REQ-010).

    The panel mutates ``AppState`` directly: toggling a filter sets
    ``filter.enabled``; launching a game calls
    ``state.launch_game(game_instance)``.

    Parameters:
        state (AppState): Shared application state (passed at
            construction so callbacks can close over it).
        game_factory (Callable[[], BaseGame]): Zero-argument callable
            that produces fresh BaseGame instances for the launch button.
    """

    def __init__(
        self,
        state: AppState,
        game_factory: Optional[Callable[[], BaseGame]] = None,
    ) -> None:
        """
        Initialise the panel descriptor (does not create any DPG items).

        Parameters:
            state (AppState): Shared application state.
            game_factory (Optional[Callable[[], BaseGame]]): Factory
                function that creates mini-game instances.  If None the
                launch button is disabled.
        """
        self._state = state
        self._game_factory = game_factory
        self._ready = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """
        Create the Dear PyGui context and build all panel widgets.

        Must be called once from the main thread before the render loop.
        """
        dpg.create_context()
        dpg.create_viewport(
            title="Camera Mayham — Controls",
            width=340,
            height=640,
            resizable=True,
        )
        dpg.setup_dearpygui()

        with dpg.window(
            label="Controls",
            tag=_TAG_WINDOW,
            no_close=True,
            no_move=True,
            no_resize=True,
            width=320,
            height=620,
            pos=(0, 0),
        ):
            self._build_diagnostics_section()
            dpg.add_separator()
            self._build_filters_section()
            dpg.add_separator()
            self._build_games_section()

        dpg.show_viewport()
        self._ready = True
        logger.debug("WidgetPanel ready.")

    def teardown(self) -> None:
        """Destroy the Dear PyGui context and release resources."""
        if self._ready:
            dpg.destroy_context()
            self._ready = False
            logger.debug("WidgetPanel torn down.")

    # ------------------------------------------------------------------
    # Per-frame tick
    # ------------------------------------------------------------------

    def render_frame(self) -> None:
        """
        Render one Dear PyGui frame and update diagnostic readouts.

        Must be called every main-loop iteration.  Takes ~0.5 ms on a
        typical system; all widget callbacks are processed synchronously
        during this call (CON-005).
        """
        if not self._ready:
            return

        # Refresh diagnostic labels with latest state values
        fps = (
            1000.0 / self._state.last_frame_time_ms
            if self._state.last_frame_time_ms > 0
            else 0.0
        )
        dpg.set_value(_TAG_FPS_TEXT, f"FPS:        {fps:.1f}")
        dpg.set_value(
            _TAG_FRAME_MS_TEXT,
            f"Frame time: {self._state.last_frame_time_ms:.2f} ms",
        )

        # Update game status label
        if self._state.active_game is not None:
            game = self._state.active_game
            dpg.set_value(
                _TAG_GAME_STATUS,
                f"{game.name}  score={game.score}  "
                f"[{game.state.name}]",
            )
        else:
            dpg.set_value(_TAG_GAME_STATUS, "No game running")

        dpg.render_dearpygui_frame()

    def is_running(self) -> bool:
        """
        Check whether the Dear PyGui viewport is still open.

        Returns:
            bool: True while the panel window has not been closed.
        """
        return self._ready and dpg.is_dearpygui_running()

    # ------------------------------------------------------------------
    # Panel section builders (called once from setup)
    # ------------------------------------------------------------------

    def _build_diagnostics_section(self) -> None:
        """Add the diagnostics / FPS readout section."""
        dpg.add_text("Diagnostics", color=(200, 200, 100))
        dpg.add_text("FPS:        0.0", tag=_TAG_FPS_TEXT)
        dpg.add_text("Frame time: 0.00 ms", tag=_TAG_FRAME_MS_TEXT)

    def _build_filters_section(self) -> None:
        """
        Add a toggle and parameter sliders for every registered filter.
        """
        dpg.add_text("Filters", color=(100, 200, 100))

        if not self._state.filters:
            dpg.add_text("  (no filters registered)", color=(150, 150, 150))
            return

        for flt in self._state.filters:
            # Unique tag derived from filter name for reliable callbacks
            filter_name = flt.name

            with dpg.collapsing_header(label=filter_name, default_open=True):
                # Enable / disable toggle  (REQ-003)
                dpg.add_checkbox(
                    label="Enabled",
                    default_value=flt.enabled,
                    callback=self._make_filter_toggle_cb(filter_name),
                )

                # Parameter sliders (REQ-004)
                for param_key, param_val in flt.params.items():
                    if isinstance(param_val, float):
                        dpg.add_slider_float(
                            label=param_key,
                            default_value=param_val,
                            min_value=0.0,
                            max_value=self._param_max(param_key),
                            callback=self._make_param_float_cb(
                                filter_name, param_key
                            ),
                        )
                    elif isinstance(param_val, tuple) and len(param_val) == 3:
                        dpg.add_color_edit(
                            label=param_key,
                            default_value=[int(v * 255) for v in param_val],
                            no_alpha=True,
                            callback=self._make_param_colour_cb(
                                filter_name, param_key
                            ),
                        )

    def _build_games_section(self) -> None:
        """Add the mini-game launch/stop controls."""
        dpg.add_text("Mini-Games", color=(100, 150, 255))
        dpg.add_text("No game running", tag=_TAG_GAME_STATUS)

        enabled = self._game_factory is not None
        dpg.add_button(
            label="Launch BubblePop",
            enabled=enabled,
            callback=self._on_launch_game,
        )
        dpg.add_button(
            label="Stop Game",
            callback=self._on_stop_game,
        )

    # ------------------------------------------------------------------
    # Callback factories
    # ------------------------------------------------------------------

    def _make_filter_toggle_cb(
        self, filter_name: str
    ) -> Callable:
        """
        Return a DPG callback that toggles the named filter's enabled flag.

        Parameters:
            filter_name (str): Name of the filter to control.

        Returns:
            Callable: DPG-compatible callback.
        """
        def cb(sender: int, app_data: bool, user_data: object) -> None:
            flt = self._state.get_filter(filter_name)
            if flt is not None:
                flt.enabled = app_data
                logger.debug(
                    "Filter '%s' enabled=%s", filter_name, app_data
                )
        return cb

    def _make_param_float_cb(
        self, filter_name: str, param_key: str
    ) -> Callable:
        """
        Return a DPG callback that updates a float parameter on a filter.

        Parameters:
            filter_name (str): Name of the target filter.
            param_key (str): Parameter key within the filter's params dict.

        Returns:
            Callable: DPG-compatible callback.
        """
        def cb(sender: int, app_data: float, user_data: object) -> None:
            flt = self._state.get_filter(filter_name)
            if flt is not None:
                flt.params[param_key] = app_data
        return cb

    def _make_param_colour_cb(
        self, filter_name: str, param_key: str
    ) -> Callable:
        """
        Return a DPG callback that updates an RGB colour parameter.

        Parameters:
            filter_name (str): Name of the target filter.
            param_key (str): Parameter key (expected value is (r,g,b)).

        Returns:
            Callable: DPG-compatible callback.
        """
        def cb(sender: int, app_data: list, user_data: object) -> None:
            flt = self._state.get_filter(filter_name)
            if flt is not None:
                # DPG returns [0,255] ints; convert to [0,1] floats
                r, g, b = app_data[0], app_data[1], app_data[2]
                flt.params[param_key] = (r / 255.0, g / 255.0, b / 255.0)
        return cb

    def _on_launch_game(self, *_: object) -> None:
        """Launch a new mini-game instance when the button is pressed."""
        if self._game_factory is not None:
            game = self._game_factory()
            self._state.launch_game(game)
            logger.info("Launched game: %s", game.name)

    def _on_stop_game(self, *_: object) -> None:
        """Stop the active game when the Stop button is pressed."""
        self._state.stop_game()
        logger.info("Game stopped by user.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _param_max(param_key: str) -> float:
        """
        Return a sensible slider maximum for known parameter names.

        Parameters:
            param_key (str): Parameter key string.

        Returns:
            float: Maximum slider value for this parameter.
        """
        _maxes = {
            "strength": 1.0,
            "intensity": 10.0,
            "hue_shift": 360.0,
            "saturation": 4.0,
            "vignette_strength": 1.0,
            "vignette_radius": 1.0,
        }
        return _maxes.get(param_key, 1.0)
