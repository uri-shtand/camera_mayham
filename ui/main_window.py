"""
Main application window — single-window design.

Replaces the previous dual-window layout (wgpu ``GlfwRenderCanvas`` +
separate Dear PyGui control panel) with one Dear PyGui viewport that
contains both the camera feed area and the widget bar.

Layout (spec-design-main-screen §4.1)
--------------------------------------
::

    ┌─────────────────────────────────────────┐
    │                                         │
    │   Camera Area  (GPU-rendered feed)      │
    │                                         │
    │   [ expansion tray — overlaid, if open ]│
    ├─────────────────────────────────────────┤
    │  Widget Bar  (icon + label items)       │
    └─────────────────────────────────────────┘

The camera feed is displayed via a Dear PyGui dynamic ``rgba8unorm``
texture that is updated each frame with the pixel array produced by the
offscreen GPU pipeline.

Each widget item in the bar has:
  * An icon button (primary click: activate/deactivate)
  * A small expand chevron button (secondary click: open expansion tray)
  * A text label below the icon

Expansion trays slide up from the bar (spec REQ-031) and are implemented
as a separate floating DPG window anchored above the widget bar.

Selection semantics (spec §3, REQ-020, REQ-021):
  * Filters — single-select; activating one deactivates the previous.
  * Games   — single-select; launching one stops the previous.

All state mutations go through :py:class:`app.state.AppState`; this
class never mutates GPU pipeline state directly (spec GUD-003).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

import dearpygui.dearpygui as dpg
import numpy as np

if TYPE_CHECKING:
    from app.state import AppState
    from games.base import BaseGame

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layout constants (spec §4.1)
# ---------------------------------------------------------------------------
BAR_HEIGHT: int = 88          # fixed height of the widget bar in pixels
ITEM_WIDTH: int = 90          # width of one widget item cell
ICON_BTN_W: int = 56          # icon button width inside a cell
ICON_BTN_H: int = 44          # icon button height inside a cell
TRAY_HEIGHT: int = 230        # pixel height of the expansion tray overlay
SEPARATOR_W: int = 2          # pixel width of inter-group separator bar

# Extra pixels added to the viewport outer height to absorb the OS window
# title bar and any DPI-dependent chrome.  On Windows the title bar is
# typically ~32 px at 96 DPI; 50 px gives comfortable headroom.
_VIEWPORT_CHROME_HEIGHT: int = 50

# ---------------------------------------------------------------------------
# DPG tag constants — stable identifiers for reliable item lookup
# ---------------------------------------------------------------------------
_TAG_MAIN_WIN: str = "mw_main"
_TAG_CAMERA_TEX: str = "mw_camera_tex"
_TAG_CAMERA_IMAGE: str = "mw_camera_img"
_TAG_BAR_WIN: str = "mw_bar"
_TAG_TRAY_WIN: str = "mw_tray"
_TAG_FPS_TEXT: str = "mw_fps"
_TAG_FRAME_MS_TEXT: str = "mw_frame_ms"

# ---------------------------------------------------------------------------
# Colour tokens for active / inactive widget states
# ---------------------------------------------------------------------------
_COLOUR_ACTIVE_BG: Tuple[int, ...] = (70, 130, 200, 255)
_COLOUR_ACTIVE_HOVER: Tuple[int, ...] = (90, 150, 220, 255)
_COLOUR_INACTIVE_BG: Tuple[int, ...] = (50, 50, 55, 255)
_COLOUR_INACTIVE_HOVER: Tuple[int, ...] = (70, 70, 80, 255)
_COLOUR_ICON_TEXT: Tuple[int, ...] = (230, 230, 230, 255)
_COLOUR_LABEL_ACTIVE: Tuple[int, ...] = (120, 190, 255, 255)
_COLOUR_LABEL_INACTIVE: Tuple[int, ...] = (180, 180, 180, 255)
_COLOUR_EXPAND_BG: Tuple[int, ...] = (38, 38, 42, 255)


# ---------------------------------------------------------------------------
# Widget item descriptor
# ---------------------------------------------------------------------------

@dataclass
class WidgetItem:
    """
    Describes a single interactive item in the widget bar.

    Parameters:
        id (str): Unique stable identifier used to build DPG tag names.
        label (str): Short display text shown below the icon.
        icon (str): Short symbol / emoji rendered as the icon label.
        category (str): ``"filter"``, ``"game"``, or ``"action"``.
        is_expandable (bool): Whether this item has an expansion tray.
    """

    id: str
    label: str
    icon: str
    category: str
    is_expandable: bool = True

    # ------------------------------------------------------------------
    # Derived DPG tag helpers
    # ------------------------------------------------------------------

    @property
    def btn_tag(self) -> str:
        """DPG tag for the primary icon button."""
        return f"wbtn_{self.id}"

    @property
    def expand_tag(self) -> str:
        """DPG tag for the expand-chevron button."""
        return f"wexp_{self.id}"

    @property
    def lbl_tag(self) -> str:
        """DPG tag for the label text item."""
        return f"wlbl_{self.id}"

    @property
    def theme_tag(self) -> str:
        """DPG tag for the per-item button theme."""
        return f"wthm_{self.id}"


# ---------------------------------------------------------------------------
# GameConfig helper
# ---------------------------------------------------------------------------

@dataclass
class GameConfig:
    """
    Associates a mini-game widget item with its factory function.

    Parameters:
        item (WidgetItem): The widget bar item descriptor.
        factory (Callable[[], BaseGame]): Zero-argument callable that
            creates a fresh game instance.
    """

    item: WidgetItem
    factory: Callable[[], "BaseGame"]


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow:
    """
    Single-window host: camera area on top, widget bar at the bottom.

    The caller drives the window by calling :py:meth:`setup` once, then
    :py:meth:`render_frame` every main-loop iteration, and finally
    :py:meth:`teardown` on exit.

    Parameters:
        state (AppState): Shared application state; widget callbacks
            mutate this to activate filters / launch games.
        camera_width (int): Camera / render resolution width in pixels.
        camera_height (int): Camera / render resolution height in pixels.
        game_configs (Optional[List[GameConfig]]): Ordered list of game
            widget descriptors.  Each entry provides a :py:class:`WidgetItem`
            and a factory callable.  Pass ``None`` or an empty list for no
            game widgets.
    """

    def __init__(
        self,
        state: "AppState",
        camera_width: int = 1280,
        camera_height: int = 720,
        game_configs: Optional[List[GameConfig]] = None,
    ) -> None:
        """
        Initialise the window descriptor (does not create any DPG items).

        Parameters:
            state (AppState): Shared application state.
            camera_width (int): Width of the camera / render area.
            camera_height (int): Height of the camera / render area.
            game_configs (Optional[List[GameConfig]]): Game widget
                descriptors and factories.
        """
        self._state = state
        self._cam_w = camera_width
        self._cam_h = camera_height
        self._game_configs: List[GameConfig] = game_configs or []

        # Built widget items for each filter and game (populated in setup)
        self._filter_items: List[WidgetItem] = []
        self._game_items: List[WidgetItem] = []

        # Maps game item id → GameConfig for launch callbacks
        self._game_config_map: Dict[str, GameConfig] = {}

        # Maps filter item id → filter name (str) for activate lookups.
        # Built in _build_widget_bar; avoids fragile string reversal.
        self._filter_name_map: Dict[str, str] = {}

        # DPG theme that removes inner window padding so the camera image
        # and widget bar fill the primary window edge-to-edge.
        self._theme_zero_padding: int = 0

        # Currently expanded item id (or None)
        self._expanded_id: Optional[str] = None

        # DPG themes for active/inactive button states
        self._theme_active: int = 0
        self._theme_inactive: int = 0

        self._ready: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """
        Create the Dear PyGui context, build all UI items, and show the
        viewport.

        Must be called once from the main thread before the render loop.
        """
        dpg.create_context()

        # Add _VIEWPORT_CHROME_HEIGHT to absorb the OS title bar so the
        # drawable client area is at least cam_h + BAR_HEIGHT.
        vp_h = self._cam_h + BAR_HEIGHT + _VIEWPORT_CHROME_HEIGHT
        dpg.create_viewport(
            title="Camera Mayham",
            width=self._cam_w,
            height=vp_h,
            resizable=False,
        )
        dpg.setup_dearpygui()

        self._build_themes()
        self._build_texture_registry()
        self._build_primary_window()
        self._build_expansion_tray()

        dpg.show_viewport()
        # Make the primary DPG window fill the entire viewport client area
        # automatically, regardless of OS title-bar height or DPI scaling.
        dpg.set_primary_window(_TAG_MAIN_WIN, True)
        self._ready = True
        logger.debug(
            "MainWindow ready (%dx%d + bar, viewport outer h=%d).",
            self._cam_w, self._cam_h + BAR_HEIGHT, vp_h,
        )

    def teardown(self) -> None:
        """Destroy the Dear PyGui context and release all resources."""
        if self._ready:
            dpg.destroy_context()
            self._ready = False
            logger.debug("MainWindow torn down.")

    # ------------------------------------------------------------------
    # Per-frame tick
    # ------------------------------------------------------------------

    def render_frame(
        self, pixels: Optional[np.ndarray] = None
    ) -> None:
        """
        Update the camera texture and tick the Dear PyGui frame.

        Must be called every main-loop iteration.

        Parameters:
            pixels (Optional[np.ndarray]): Latest GPU-rendered frame as an
                ``(H, W, 4)`` uint8 RGBA array.  Pass ``None`` if no new
                frame is available this iteration.
        """
        if not self._ready:
            return

        if pixels is not None:
            self._update_camera_texture(pixels)

        # Refresh diagnostics readout.
        fps = (
            1000.0 / self._state.last_frame_time_ms
            if self._state.last_frame_time_ms > 0
            else 0.0
        )
        if dpg.does_item_exist(_TAG_FPS_TEXT):
            dpg.set_value(_TAG_FPS_TEXT, f"FPS: {fps:.1f}")
        if dpg.does_item_exist(_TAG_FRAME_MS_TEXT):
            dpg.set_value(
                _TAG_FRAME_MS_TEXT,
                f"{self._state.last_frame_time_ms:.2f} ms",
            )

        # Refresh expansion tray game status if a game tray is open.
        if self._expanded_id is not None:
            self._refresh_tray_status()

        dpg.render_dearpygui_frame()

    def is_running(self) -> bool:
        """
        Check whether the viewport is still open.

        Returns:
            bool: True while the DPG viewport has not been closed.
        """
        return self._ready and dpg.is_dearpygui_running()

    # ------------------------------------------------------------------
    # Build helpers — called once from setup()
    # ------------------------------------------------------------------

    def _build_themes(self) -> None:
        """
        Create DPG themes for active and inactive widget button states,
        plus a zero-padding theme applied to the outer primary window so
        the camera image and bar fill it edge-to-edge.
        """
        # ── Zero-padding theme for the primary window ─────────────────
        with dpg.theme() as self._theme_zero_padding:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding,
                    0, 0,
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_ItemSpacing,
                    0, 0,
                    category=dpg.mvThemeCat_Core,
                )

        with dpg.theme() as self._theme_active:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(
                    dpg.mvThemeCol_Button,
                    _COLOUR_ACTIVE_BG,
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonHovered,
                    _COLOUR_ACTIVE_HOVER,
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_Text,
                    _COLOUR_ICON_TEXT,
                    category=dpg.mvThemeCat_Core,
                )

        with dpg.theme() as self._theme_inactive:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(
                    dpg.mvThemeCol_Button,
                    _COLOUR_INACTIVE_BG,
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonHovered,
                    _COLOUR_INACTIVE_HOVER,
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_Text,
                    _COLOUR_ICON_TEXT,
                    category=dpg.mvThemeCat_Core,
                )

    def _build_texture_registry(self) -> None:
        """
        Register the dynamic camera texture with Dear PyGui.

        The texture is initialised with zeros (solid black) and updated
        each frame in :py:meth:`_update_camera_texture`.
        """
        # DPG's default_value requires a Python list; initialise once at
        # startup as all zeros (solid black).  The runtime cost is acceptable
        # since this only runs during setup.
        zeroes: List[float] = [0.0] * (
            self._cam_w * self._cam_h * 4
        )
        with dpg.texture_registry():
            dpg.add_dynamic_texture(
                width=self._cam_w,
                height=self._cam_h,
                default_value=zeroes,
                tag=_TAG_CAMERA_TEX,
            )

    def _build_primary_window(self) -> None:
        """
        Build the full-viewport primary window containing the camera
        image and the widget bar.

        Width and height are left at the DPG default (-1 = auto) because
        ``set_primary_window`` is called after ``show_viewport()`` and
        resizes this window to fill the full viewport client area.
        The zero-padding theme removes all internal DPG window padding so
        the camera image starts at pixel (0, 0).
        """
        with dpg.window(
            tag=_TAG_MAIN_WIN,
            no_title_bar=True,
            no_move=True,
            no_resize=True,
            no_scrollbar=True,
            no_bring_to_front_on_focus=True,
        ):
            # ── Camera area ──────────────────────────────────────────
            dpg.add_image(
                _TAG_CAMERA_TEX,
                tag=_TAG_CAMERA_IMAGE,
                width=self._cam_w,
                height=self._cam_h,
            )

            # ── Widget bar ───────────────────────────────────────────
            self._build_widget_bar()

        dpg.bind_item_theme(_TAG_MAIN_WIN, self._theme_zero_padding)

    def _build_widget_bar(self) -> None:
        """
        Build the widget bar child window and populate it with items.

        Items are organised into two groups separated by a visual
        divider: **Filters** and **Games**.
        """
        # Build filter widget items from registered filters in state.
        for flt in self._state.filters:
            icon = _filter_icon(flt.name)
            # Normalise id: lowercase, strip non-alphanumeric to underscores.
            item_id = (
                "flt_"
                + "".join(
                    c if c.isalnum() else "_"
                    for c in flt.name.lower()
                ).strip("_")
            )
            item = WidgetItem(
                id=item_id,
                label=_display_name(flt.name),
                icon=icon,
                category="filter",
                is_expandable=bool(flt.params),
            )
            self._filter_items.append(item)
            # Record the authoritative name for later lookup.
            self._filter_name_map[item_id] = flt.name

        # Build game widget items from provided GameConfig list.
        for cfg in self._game_configs:
            self._game_items.append(cfg.item)
            self._game_config_map[cfg.item.id] = cfg

        with dpg.child_window(
            tag=_TAG_BAR_WIN,
            width=0,     # 0 = fill available width automatically
            height=BAR_HEIGHT,
            no_scrollbar=True,
            border=False,
        ):
            # Divider line at the top of the bar.
            dpg.add_separator()

            with dpg.group(horizontal=True):
                # ── Filter group ─────────────────────────────────────
                if self._filter_items:
                    for item in self._filter_items:
                        self._build_widget_item(item)
                    self._build_group_separator()

                # ── Game group ───────────────────────────────────────
                if self._game_items:
                    for item in self._game_items:
                        self._build_widget_item(item)
                    self._build_group_separator()

                # ── Diagnostics (right-flush) ─────────────────────────
                dpg.add_spacer(width=8)
                with dpg.group():
                    dpg.add_spacer(height=10)
                    dpg.add_text(
                        "FPS: 0.0",
                        tag=_TAG_FPS_TEXT,
                        color=(160, 160, 160, 200),
                    )
                    dpg.add_text(
                        "0.00 ms",
                        tag=_TAG_FRAME_MS_TEXT,
                        color=(130, 130, 130, 200),
                    )

    def _build_widget_item(self, item: WidgetItem) -> None:
        """
        Build the DPG items for one widget entry in the bar.

        Layout per item (inside a fixed-width group)::

            ┌──────────────┐
            │    [icon]    │  ← icon button (primary action)
            │ [label text] │  ← label below
            │     [∨]      │  ← expand chevron (if expandable)
            └──────────────┘

        Parameters:
            item (WidgetItem): The widget item descriptor.
        """
        with dpg.group(
            tag=item.btn_tag + "_outer",
            width=ITEM_WIDTH,
        ):
            dpg.add_spacer(height=6)

            # ── Icon button (primary click) ───────────────────────────
            icon_indent = max(0, (ITEM_WIDTH - ICON_BTN_W) // 2)
            dpg.add_button(
                label=item.icon,
                tag=item.btn_tag,
                width=ICON_BTN_W,
                height=ICON_BTN_H,
                indent=icon_indent,
                callback=self._make_activate_cb(item),
            )
            dpg.bind_item_theme(item.btn_tag, self._theme_inactive)

            # ── Label text ───────────────────────────────────────────
            lbl_indent = max(
                0, (ITEM_WIDTH - len(item.label) * 7) // 2
            )
            dpg.add_text(
                item.label,
                tag=item.lbl_tag,
                indent=lbl_indent,
                color=list(_COLOUR_LABEL_INACTIVE),
            )

            # ── Expand chevron (secondary action, if expandable) ──────
            if item.is_expandable:
                exp_indent = max(0, (ITEM_WIDTH - 20) // 2)
                dpg.add_button(
                    label="v",
                    tag=item.expand_tag,
                    width=20,
                    height=14,
                    indent=exp_indent,
                    callback=self._make_expand_cb(item),
                )

    def _build_group_separator(self) -> None:
        """
        Add a thin vertical separator bar between widget item groups.
        """
        with dpg.group(width=SEPARATOR_W):
            dpg.add_spacer(height=8)
            dpg.add_separator()

    def _build_expansion_tray(self) -> None:
        """
        Create the expansion tray floating window (initially hidden).

        The tray is a Dear PyGui window positioned just above the widget
        bar.  Its contents are rebuilt dynamically each time it is shown
        for a different widget item.
        """
        tray_y = self._cam_h - TRAY_HEIGHT
        with dpg.window(
            tag=_TAG_TRAY_WIN,
            label="Options",
            no_title_bar=False,
            no_move=True,
            no_resize=True,
            no_collapse=True,
            no_scrollbar=True,
            no_close=True,
            pos=(0, tray_y),
            width=self._cam_w,
            height=TRAY_HEIGHT,
            show=False,
        ):
            pass  # Contents rebuilt dynamically via _rebuild_tray_contents

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _make_activate_cb(self, item: WidgetItem) -> Callable:
        """
        Return a DPG callback that activates or deactivates a widget item.

        Parameters:
            item (WidgetItem): The target widget item.

        Returns:
            Callable: DPG-compatible callback.
        """
        def cb(sender: int, app_data: object, user_data: object) -> None:
            self._on_activate(item)
        return cb

    def _make_expand_cb(self, item: WidgetItem) -> Callable:
        """
        Return a DPG callback that opens or closes the expansion tray.

        Parameters:
            item (WidgetItem): The target widget item.

        Returns:
            Callable: DPG-compatible callback.
        """
        def cb(sender: int, app_data: object, user_data: object) -> None:
            self._on_expand(item)
        return cb

    def _make_param_float_cb(
        self, filter_name: str, param_key: str
    ) -> Callable:
        """
        Return a DPG callback that updates a float parameter on a filter.

        Parameters:
            filter_name (str): Name of the target filter.
            param_key (str): Parameter key within ``filter.params``.

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
        Return a DPG callback that updates an RGB colour parameter on a
        filter.  Dear PyGui returns [0, 255] ints; the filter stores
        normalised (r, g, b) floats.

        Parameters:
            filter_name (str): Name of the target filter.
            param_key (str): Parameter key (expected value is (r, g, b)).

        Returns:
            Callable: DPG-compatible callback.
        """
        def cb(sender: int, app_data: list, user_data: object) -> None:
            flt = self._state.get_filter(filter_name)
            if flt is not None:
                r, g, b = app_data[0], app_data[1], app_data[2]
                flt.params[param_key] = (
                    r / 255.0, g / 255.0, b / 255.0
                )
        return cb

    # ------------------------------------------------------------------
    # Interaction handlers
    # ------------------------------------------------------------------

    def _on_activate(self, item: WidgetItem) -> None:
        """
        Primary click handler: activate or toggle-off a widget item.

        Filter semantics (REQ-020, REQ-023):
            Activating an inactive filter deactivates any current filter
            and activates the clicked one.  Clicking an active filter
            item deactivates it (toggle-off).

        Game semantics (REQ-021, REQ-023):
            Same logic applied to mini-games.

        Parameters:
            item (WidgetItem): The clicked widget item.
        """
        if item.category == "filter":
            filter_name = self._filter_name_map.get(item.id, "")
            if not filter_name:
                logger.warning(
                    "No filter name mapped for item '%s'.", item.id
                )
                return
            if self._state.active_filter_name == filter_name:
                # Toggle off.
                self._state.deactivate_filter()
                logger.debug("Filter '%s' deactivated.", filter_name)
            else:
                self._state.activate_filter(filter_name)
                logger.debug("Filter '%s' activated.", filter_name)

        elif item.category == "game":
            cfg = self._game_config_map.get(item.id)
            if cfg is None:
                return
            game_running = (
                self._state.active_game is not None
                and self._state.active_game.name == cfg.item.label
            )
            if game_running:
                self._state.stop_game()
                logger.info("Game '%s' stopped.", cfg.item.label)
            else:
                game = cfg.factory()
                self._state.launch_game(game)
                logger.info("Game '%s' launched.", cfg.item.label)

        self._refresh_item_visuals()

    def _on_expand(self, item: WidgetItem) -> None:
        """
        Expand-chevron click handler: open or close the expansion tray
        for this item (REQ-030 - REQ-033).

        Parameters:
            item (WidgetItem): The widget item whose tray was triggered.
        """
        if self._expanded_id == item.id:
            # Already open for this item — close it.
            self._close_tray()
        else:
            self._open_tray(item)

    def _open_tray(self, item: WidgetItem) -> None:
        """
        Open the expansion tray for ``item``, replacing any currently
        open tray (spec REQ-032).

        Parameters:
            item (WidgetItem): The item to expand.
        """
        self._close_tray()
        self._expanded_id = item.id
        self._rebuild_tray_contents(item)
        dpg.show_item(_TAG_TRAY_WIN)
        logger.debug("Tray opened for item '%s'.", item.id)

    def _close_tray(self) -> None:
        """
        Hide the expansion tray without changing any active/inactive
        state (spec REQ-036).
        """
        if dpg.does_item_exist(_TAG_TRAY_WIN):
            dpg.hide_item(_TAG_TRAY_WIN)
        self._expanded_id = None

    def _rebuild_tray_contents(self, item: WidgetItem) -> None:
        """
        Clear the expansion tray window and populate it with controls
        appropriate for ``item``'s category and current state.

        Parameters:
            item (WidgetItem): The item being expanded.
        """
        # Delete all children of the tray window.
        children = dpg.get_item_children(_TAG_TRAY_WIN, slot=1) or []
        for child in children:
            dpg.delete_item(child)

        if item.category == "filter":
            self._build_filter_tray(item)
        elif item.category == "game":
            self._build_game_tray(item)

    def _build_filter_tray(self, item: WidgetItem) -> None:
        """
        Populate the tray with the filter's adjustable parameter
        controls (spec REQ-034).

        Parameters:
            item (WidgetItem): The filter widget item being expanded.
        """
        filter_name = self._filter_name_map.get(item.id, "")
        flt = self._state.get_filter(filter_name)
        if flt is None:
            return

        dpg.add_text(
            f"{flt.name} — parameters",
            parent=_TAG_TRAY_WIN,
            color=(200, 200, 100, 255),
        )
        dpg.add_separator(parent=_TAG_TRAY_WIN)

        if not flt.params:
            dpg.add_text(
                "(no adjustable parameters)",
                parent=_TAG_TRAY_WIN,
                color=(150, 150, 150, 255),
            )
            return

        for param_key, param_val in flt.params.items():
            if isinstance(param_val, float):
                dpg.add_slider_float(
                    label=param_key,
                    default_value=param_val,
                    min_value=0.0,
                    max_value=_param_max(param_key),
                    callback=self._make_param_float_cb(
                        filter_name, param_key
                    ),
                    parent=_TAG_TRAY_WIN,
                    width=self._cam_w - 160,
                )
            elif isinstance(param_val, tuple) and len(param_val) == 3:
                dpg.add_color_edit(
                    label=param_key,
                    default_value=[int(v * 255) for v in param_val],
                    no_alpha=True,
                    callback=self._make_param_colour_cb(
                        filter_name, param_key
                    ),
                    parent=_TAG_TRAY_WIN,
                )

    def _build_game_tray(self, item: WidgetItem) -> None:
        """
        Populate the tray with the game's status display (spec REQ-035).

        Parameters:
            item (WidgetItem): The game widget item being expanded.
        """
        dpg.add_text(
            item.label,
            parent=_TAG_TRAY_WIN,
            color=(100, 150, 255, 255),
        )
        dpg.add_separator(parent=_TAG_TRAY_WIN)
        dpg.add_text(
            self._game_status_text(),
            tag="tray_game_status",
            parent=_TAG_TRAY_WIN,
        )

    # ------------------------------------------------------------------
    # Per-frame update helpers
    # ------------------------------------------------------------------

    def _update_camera_texture(self, pixels: np.ndarray) -> None:
        """
        Convert the RGBA uint8 frame to float32 and push it to the DPG
        dynamic texture (spec §4.1 Camera Area).

        The conversion is: ``float_value = uint8_value / 255.0``.

        Parameters:
            pixels (np.ndarray): ``(H, W, 4)`` uint8 RGBA array from the
                offscreen GPU pipeline.
        """
        frame_f32: np.ndarray = pixels.astype(np.float32)
        frame_f32 *= 1.0 / 255.0
        dpg.set_value(_TAG_CAMERA_TEX, frame_f32.flatten())

    def _refresh_item_visuals(self) -> None:
        """
        Update button themes and label colours for all widget items to
        reflect the current active / inactive state in ``AppState``
        (spec REQ-022).
        """
        for item in self._filter_items:
            flt_name = self._filter_name_map.get(item.id, "")
            active = (self._state.active_filter_name == flt_name)
            self._set_item_active(item, active)

        for item in self._game_items:
            cfg = self._game_config_map.get(item.id)
            if cfg is None:
                continue
            active = (
                self._state.active_game is not None
                and self._state.active_game.name == cfg.item.label
            )
            self._set_item_active(item, active)

    def _set_item_active(self, item: WidgetItem, active: bool) -> None:
        """
        Apply the active or inactive visual theme to a single widget item.

        Parameters:
            item (WidgetItem): The widget item to update.
            active (bool): Whether the item should display as active.
        """
        if not dpg.does_item_exist(item.btn_tag):
            return
        theme = self._theme_active if active else self._theme_inactive
        dpg.bind_item_theme(item.btn_tag, theme)

        lbl_colour = (
            list(_COLOUR_LABEL_ACTIVE)
            if active
            else list(_COLOUR_LABEL_INACTIVE)
        )
        if dpg.does_item_exist(item.lbl_tag):
            dpg.configure_item(item.lbl_tag, color=lbl_colour)

    def _refresh_tray_status(self) -> None:
        """
        Refresh the game status text inside an open game expansion tray.
        Called each frame while a game tray is open.
        """
        if not dpg.does_item_exist("tray_game_status"):
            return
        dpg.set_value("tray_game_status", self._game_status_text())

    def _game_status_text(self) -> str:
        """
        Build a one-line game status string for display in the tray.

        Returns:
            str: Status text, e.g. ``"BubblePop  score=5  [RUNNING]"``.
        """
        game = self._state.active_game
        if game is None:
            return "No game running"
        return (
            f"{game.name}  score={game.score}  [{game.state.name}]"
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _filter_icon(filter_name: str) -> str:
    """
    Return a short icon label for a filter based on its name.

    Falls back to the first three uppercase characters of the name if
    no specific mapping is defined.

    Parameters:
        filter_name (str): The filter's ``name`` property.

    Returns:
        str: A short label (1–4 chars) used as the widget icon.
    """
    _icons: Dict[str, str] = {
        "Grayscale": "B/W",
        "EdgeDetection": "EDG",
        "ColourShift": "CLR",
    }
    return _icons.get(filter_name, filter_name[:3].upper())


def _display_name(filter_name: str) -> str:
    """
    Convert a CamelCase filter name to a human-readable display string.

    Examples::

        "EdgeDetection" → "Edge Detection"
        "ColourShift"   → "Colour Shift"
        "Grayscale"     → "Grayscale"

    Parameters:
        filter_name (str): The filter's internal ``name`` property.

    Returns:
        str: Space-separated display name.
    """
    import re
    # Insert a space before each uppercase letter that follows a lowercase one.
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", filter_name)
    return spaced


def _param_max(param_key: str) -> float:
    """
    Return a sensible slider maximum for well-known filter parameter keys.

    Parameters:
        param_key (str): The parameter key string from ``filter.params``.

    Returns:
        float: Maximum value for the slider widget.
    """
    _maxes: Dict[str, float] = {
        "strength": 1.0,
        "intensity": 10.0,
        "hue_shift": 360.0,
        "saturation": 4.0,
        "vignette_strength": 1.0,
        "vignette_radius": 1.0,
    }
    return _maxes.get(param_key, 1.0)
