"""
Application orchestrator.

The Application class owns and coordinates all top-level subsystems:

* Camera (OpenCV VideoCapture)
* Face tracker (MediaPipe)
* Rendering pipeline (wgpu)
* Widget panel (Dear PyGui)

It drives the main loop and is the single point of startup/shutdown for
the entire application (GUD-001, GUD-004).

Main loop
---------

::

    while running:
        frame = camera.read()
        state.camera_frame = frame
        state.face_result  = tracker.process(frame)
        pipeline.render_frame(state)     # GPU work
        panel.render_frame()             # UI tick
"""

from __future__ import annotations

import logging
import sys

from rendercanvas.glfw import GlfwRenderCanvas

from app.state import AppState
from camera.capture import CameraCapture
from filters.colour_shift import ColourShiftFilter
from filters.edge_detection import EdgeDetectionFilter
from filters.grayscale import GrayscaleFilter
from games.bubble_pop import BubblePopGame
from rendering.pipeline import RenderPipeline
from tracking.face_tracker import FaceTracker
from ui.widget_panel import WidgetPanel

logger = logging.getLogger(__name__)


class Application:
    """
    Top-level application class that owns all subsystems and drives the
    main loop (GUD-001, GUD-004).

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
        # Shared state — mutated from the UI callbacks and render loop
        self._state = AppState(camera_width=width, camera_height=height)

        # Subsystems (configured in setup)
        self._camera = CameraCapture(
            device_id=camera_device_id,
            width=width,
            height=height,
        )
        self._tracker = FaceTracker(num_faces=1)
        self._canvas: GlfwRenderCanvas | None = None
        self._pipeline: RenderPipeline | None = None
        self._panel: WidgetPanel | None = None

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
        filters, create the wgpu canvas, and set up the widget panel.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        logger.info("Starting Camera Mayham…")

        # -- Camera ---------------------------------------------------
        self._camera.open()
        self._state.camera_width = self._camera.width
        self._state.camera_height = self._camera.height

        # -- Face tracker --------------------------------------------
        self._tracker.setup()

        # -- Built-in filters (REQ-002) --------------------------------
        for flt in [
            GrayscaleFilter(),
            EdgeDetectionFilter(),
            ColourShiftFilter(),
        ]:
            flt.enabled = False           # all off by default
            self._state.register_filter(flt)

        # -- wgpu canvas (GLFW native window) -------------------------
        self._canvas = GlfwRenderCanvas(
            title="Camera Mayham",
            size=(self._state.camera_width, self._state.camera_height),
        )

        # -- Rendering pipeline ---------------------------------------
        self._pipeline = RenderPipeline(self._canvas)
        self._pipeline.setup(self._state)

        # -- Widget panel (Dear PyGui in a separate OS window) --------
        self._panel = WidgetPanel(
            state=self._state,
            game_factory=BubblePopGame,
        )
        self._panel.setup()

        logger.info("Setup complete.")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _main_loop(self) -> None:
        """
        Run the application main loop until the user closes the window
        or ``state.running`` is set to False.

        Each iteration:
        1. Poll GLFW events (handles window close).
        2. Capture a camera frame.
        3. Run face tracking.
        4. Render the GPU frame.
        5. Tick the widget panel.
        """
        logger.info("Entering main loop.")

        while self._state.running:
            # Check if the render window was closed
            if self._canvas.is_closed():
                self._state.running = False
                break

            # Check if the control panel was closed
            if not self._panel.is_running():
                self._state.running = False
                break

            # -- Camera capture --------------------------------------
            frame = self._camera.read()
            if frame is not None:
                self._state.camera_frame = frame

                # -- Face tracking -----------------------------------
                self._state.face_result = self._tracker.process(frame)

            # -- GPU frame -------------------------------------------
            self._pipeline.render_frame(self._state)

            # -- UI tick ---------------------------------------------
            self._panel.render_frame()

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def _teardown(self) -> None:
        """
        Release all subsystem resources in reverse setup order.

        Safe to call even if setup was only partially completed.
        """
        logger.info("Shutting down…")

        if self._panel is not None:
            self._panel.teardown()

        if self._pipeline is not None:
            self._pipeline.teardown(self._state)

        self._tracker.teardown()
        self._camera.close()

        logger.info("Camera Mayham exited cleanly.")
