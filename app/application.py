"""
Application orchestrator.

The Application class owns and coordinates all top-level subsystems:

* Camera (OpenCV VideoCapture)
* Face tracker (MediaPipe)
* Rendering pipeline (wgpu — offscreen mode, no separate canvas window)
* Main window (single Dear PyGui viewport: camera area + widget bar)

It drives the main loop and is the single point of startup/shutdown for
the entire application (GUD-001, GUD-004).

Single-window architecture (spec-design-main-screen §4.1)
----------------------------------------------------------

The GPU pipeline renders into an offscreen ``rgba8unorm`` texture; after
each frame the pixels are read back to CPU via ``queue.read_texture`` and
forwarded to the :py:class:`ui.main_window.MainWindow` for display.  This
keeps all UI elements — camera feed, widget bar, expansion trays — inside
one operating-system window.

Main loop::

    while running:
        frame   = camera.read()
        state.camera_frame = frame
        state.face_result  = tracker.process(frame)
        pixels  = pipeline.render_frame(state)   # GPU work + pixel readback
        window.render_frame(pixels)              # DPG tick + display
"""

from __future__ import annotations

import logging

from app.state import AppState
from camera.capture import CameraCapture
from filters.colour_shift import ColourShiftFilter
from filters.edge_detection import EdgeDetectionFilter
from filters.face_landmarks import FaceLandmarkFilter
from filters.grayscale import GrayscaleFilter
from filters.moustache import MoustacheFilter
from games.bubble_pop import BubblePopGame
from rendering.pipeline import RenderPipeline
from tracking.face_tracker import FaceTracker, TrackerConfig
from ui.main_window import GameConfig, MainWindow, WidgetItem

logger = logging.getLogger(__name__)


class Application:
    """
    Top-level application class that owns all subsystems and drives the
    main loop (GUD-001, GUD-004).

    A single :py:class:`ui.main_window.MainWindow` Dear PyGui viewport
    serves as the sole OS window; the GPU pipeline runs in offscreen mode
    and forwards rendered pixel arrays to the window each frame
    (spec-design-main-screen §4.1).

    Usage::

        app = Application()
        app.run()

    Parameters:
        camera_device_id (int): Index of the webcam device to open.
        width (int): Render / capture width in pixels.
        height (int): Render / capture height in pixels.
    """

    def __init__(
        self,
        camera_device_id: int = 0,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        """
        Initialise the application state and subsystem descriptors.

        Parameters:
            camera_device_id (int): OS camera index.
            width (int): Render resolution width.
            height (int): Render resolution height.
        """
        # Shared state — mutated from UI callbacks and read by the
        # render loop.
        self._state = AppState(camera_width=width, camera_height=height)

        # Subsystems (configured in _setup)
        self._camera = CameraCapture(
            device_id=camera_device_id,
            width=width,
            height=height,
        )
        # Config is loaded from disk automatically inside FaceTracker.__init__.
        self._tracker = FaceTracker()
        # Offscreen pipeline — canvas=None (single-window mode)
        self._pipeline: RenderPipeline | None = None
        self._window: MainWindow | None = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Start the application: open subsystems, enter the main loop, and
        clean up on exit.

        This is the single entry point called from ``main.py``.
        """
        try:
            self._setup()
            self._main_loop()
        except Exception:  # pragma: no cover
            logger.exception("Unhandled exception — shutting down.")
            raise
        finally:
            self._teardown()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        """
        Open the camera, initialise face tracking, register built-in
        filters, build the GPU pipeline in offscreen mode, and set up
        the single main window.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        logger.info("Starting Camera Mayham…")

        # -- Camera ---------------------------------------------------
        self._camera.open()
        self._state.camera_width = self._camera.width
        self._state.camera_height = self._camera.height

        # -- Face tracker ---------------------------------------------
        self._tracker.setup()
        # Sync saved config into shared state so the UI can read it.
        self._state.tracker_config = self._tracker.config

        # -- Built-in filters (REQ-002); all off by default -----------
        for flt in [
            GrayscaleFilter(),
            EdgeDetectionFilter(),
            ColourShiftFilter(),
            FaceLandmarkFilter(),
            MoustacheFilter(),
        ]:
            flt.enabled = False
            self._state.register_filter(flt)

        # -- GPU pipeline (offscreen — no canvas window) --------------
        self._pipeline = RenderPipeline(canvas=None)
        self._pipeline.setup(self._state)

        # -- Game configurations for the widget bar -------------------
        game_configs = [
            GameConfig(
                item=WidgetItem(
                    id="game_bubblepop",
                    label="BubblePop",
                    icon="●",
                    category="game",
                    is_expandable=True,
                ),
                factory=BubblePopGame,
            ),
        ]

        # -- Single main window (camera area + widget bar) ------------
        self._window = MainWindow(
            state=self._state,
            camera_width=self._state.camera_width,
            camera_height=self._state.camera_height,
            game_configs=game_configs,
        )
        # Register the tracker settings widget before building the UI.
        self._window.set_tracker_widget(
            config=self._tracker.config,
            on_change=self._on_tracker_config_changed,
        )
        self._window.setup()

        logger.info("Setup complete.")

    # ------------------------------------------------------------------
    # Tracker reconfiguration callback
    # ------------------------------------------------------------------

    def _on_tracker_config_changed(
        self, new_config: "TrackerConfig"
    ) -> None:
        """
        Callback invoked by the UI when the user changes tracker settings.

        Applies the new configuration to the tracker (which also persists
        it to disk) and syncs AppState so the UI can always read the
        current values.

        Parameters:
            new_config (TrackerConfig): The updated tracker configuration.
        """
        self._tracker.reconfigure(new_config)
        self._state.tracker_config = new_config

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _main_loop(self) -> None:
        """
        Run the application main loop until the user closes the window
        or ``state.running`` is set to False.

        Each iteration:

        1. Check whether the main window is still open.
        2. Capture a camera frame.
        3. Run face tracking.
        4. Render the GPU frame (offscreen); receive pixel array.
        5. Forward pixels to the main window for display.
        """
        logger.info("Entering main loop.")

        while self._state.running:
            # Check if the main window was closed
            if not self._window.is_running():
                self._state.running = False
                break

            # -- Camera capture --------------------------------------
            frame = self._camera.read()
            if frame is not None:
                self._state.camera_frame = frame

                # -- Face tracking -----------------------------------
                self._state.face_result = self._tracker.process(frame)

            # -- GPU frame (offscreen) → pixel array -----------------
            pixels = self._pipeline.render_frame(self._state)

            # -- Main window tick + camera display -------------------
            self._window.render_frame(pixels)

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def _teardown(self) -> None:
        """
        Release all subsystem resources in reverse setup order.

        Safe to call even if setup was only partially completed.
        """
        logger.info("Shutting down…")

        if self._window is not None:
            self._window.teardown()

        if self._pipeline is not None:
            self._pipeline.teardown(self._state)

        self._tracker.teardown()
        self._camera.close()

        logger.info("Camera Mayham exited cleanly.")
