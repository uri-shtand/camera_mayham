"""
Bubble Pop mini-game.

Bubbles drift down the screen from random horizontal positions.  The
player pops them by opening their mouth (jawOpen blendshape) while
centering their head over a bubble.  Each popped bubble scores one
point.  The game ends when 10 bubbles escape off the bottom (REQ-008,
REQ-009, AC-005).

Implementation notes
--------------------
* Bubbles are rendered as coloured quads using a simple vertex shader.
* The game state (positions, velocities) lives on the CPU; geometry
  is uploaded to a dynamic vertex buffer each frame (no allocation
  churn — the buffer is pre-allocated at max capacity).
* Face input binding: ``jawOpen`` blendshape threshold triggers "pop".
"""

from __future__ import annotations

import logging
import random
import struct
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import wgpu

from games.base import BaseGame, GameState
from tracking.face_tracker import FaceTrackResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WGSL shader — renders axis-aligned coloured quads
# ---------------------------------------------------------------------------
_WGSL = """
struct Bubble {
    // Bubble centre in NDC, radius in NDC units, and RGBA colour
    @location(0) position : vec2f,
    @location(1) radius   : f32,
    @location(2) colour   : vec4f,
}

struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) colour : vec4f,
    @location(1) uv     : vec2f,  // local coords relative to bubble centre
}

@vertex
fn vs_main(
    @builtin(vertex_index) vi : u32,
    @builtin(instance_index) ii : u32,
    bubble: Bubble,
) -> VertexOutput {
    // Two-triangle quad per instance
    var offsets = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
        vec2f(-1.0,  1.0), vec2f(1.0, -1.0), vec2f( 1.0, 1.0),
    );
    let off = offsets[vi];
    let pos = bubble.position + off * bubble.radius;

    var out: VertexOutput;
    out.position = vec4f(pos, 0.0, 1.0);
    out.colour   = bubble.colour;
    out.uv       = off;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    // Discard pixels outside the circle
    let dist = length(in.uv);
    if dist > 1.0 { discard; }

    // Soft edge
    let alpha = in.colour.a * (1.0 - smoothstep(0.85, 1.0, dist));
    return vec4f(in.colour.rgb, alpha);
}
"""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAX_BUBBLES = 20
_BYTES_PER_BUBBLE = (
    2 * 4       # position  (2× f32)
    + 1 * 4     # radius    (1× f32)
    + 1 * 4     # padding   (1× f32 to align colour)
    + 4 * 4     # colour    (4× f32)
)  # 40 bytes

_JAW_OPEN_THRESHOLD = 0.65  # blendshape coefficient to trigger a pop
_SPAWN_INTERVAL = 1.2       # seconds between new bubbles
_BUBBLE_SPEED = 0.18        # NDC units per second
_MAX_MISSES = 10            # bubbles that can escape before game over


@dataclass
class Bubble:
    """Represents a single on-screen bubble."""

    x: float          # NDC horizontal centre [-1, 1]
    y: float          # NDC vertical centre   [-1, 1]
    radius: float     # NDC radius
    colour: tuple     # (r, g, b, a)
    speed: float      # NDC units per second downward


class BubblePopGame(BaseGame):
    """
    Bubble Pop mini-game driven by the jawOpen blendshape (REQ-008,
    AC-005).

    Parameters are hard-coded for now; future work can expose them via
    ``params`` for the widget panel to adjust (REQ-004 extension point).
    """

    def __init__(self) -> None:
        """Initialise bubble pop game state."""
        super().__init__()
        self._bubbles: List[Bubble] = []
        self._misses: int = 0
        self._time_since_spawn: float = 0.0
        self._pipeline: Any = None
        self._vertex_buffer: Optional[wgpu.GPUBuffer] = None
        self._bind_group_layout: Any = None
        self._jaw_was_open: bool = False  # edge-detection for jaw press

    @property
    def name(self) -> str:
        """
        Unique game name.

        Returns:
            str: 'BubblePop'
        """
        return "BubblePop"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _setup_pipeline(
        self,
        device: wgpu.GPUDevice,
        texture_format: wgpu.TextureFormat,
    ) -> None:
        """
        Compile the bubble render shader and pre-allocate the vertex
        buffer at maximum capacity to avoid per-frame allocation
        (CON-002).

        Parameters:
            device (wgpu.GPUDevice): The active WebGPU device.
            texture_format (wgpu.TextureFormat): Target texture format.
        """
        shader = device.create_shader_module(code=_WGSL)

        # Per-instance vertex buffer layout (one entry per bubble)
        vertex_layout: List[Any] = [
            {
                "array_stride": _BYTES_PER_BUBBLE,
                "step_mode": "instance",
                "attributes": [
                    {"format": "float32x2", "offset": 0,  "shader_location": 0},
                    {"format": "float32",   "offset": 8,  "shader_location": 1},
                    # offset 12 = padding float
                    {"format": "float32x4", "offset": 16, "shader_location": 2},
                ],
            }
        ]

        self._pipeline = device.create_render_pipeline(
            layout="auto",
            vertex={
                "module": shader,
                "entry_point": "vs_main",
                "buffers": vertex_layout,
            },
            fragment={
                "module": shader,
                "entry_point": "fs_main",
                "targets": [
                    {
                        "format": texture_format,
                        "blend": {
                            "color": {
                                "src_factor": "src-alpha",
                                "dst_factor": "one-minus-src-alpha",
                                "operation": "add",
                            },
                            "alpha": {
                                "src_factor": "one",
                                "dst_factor": "one-minus-src-alpha",
                                "operation": "add",
                            },
                        },
                    }
                ],
            },
            primitive={"topology": "triangle-list"},
            depth_stencil=None,
            multisample=None,
        )

        # Pre-allocate vertex buffer for max bubbles
        self._vertex_buffer = device.create_buffer(
            size=_MAX_BUBBLES * _BYTES_PER_BUBBLE,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
        )

    def _on_start(self) -> None:
        """Reset game variables when the game starts or restarts."""
        self._bubbles.clear()
        self._misses = 0
        self._time_since_spawn = 0.0
        self._jaw_was_open = False

    def _on_stop(self) -> None:
        """Clear active bubbles on stop."""
        self._bubbles.clear()

    def teardown(self) -> None:
        """Release GPU resources."""
        self._pipeline = None
        self._vertex_buffer = None
        super().teardown()

    # ------------------------------------------------------------------
    # Per-frame logic
    # ------------------------------------------------------------------

    def update(self, face_result: FaceTrackResult, dt: float) -> None:
        """
        Advance bubble positions, handle mouth-open pop events, and
        manage game-over condition.

        Parameters:
            face_result (FaceTrackResult): Latest face tracking result.
            dt (float): Delta time in seconds since the last frame.
        """
        if self.state != GameState.RUNNING:
            return

        # -- Spawn new bubbles ----------------------------------------
        self._time_since_spawn += dt
        if (
            self._time_since_spawn >= _SPAWN_INTERVAL
            and len(self._bubbles) < _MAX_BUBBLES
        ):
            self._spawn_bubble()
            self._time_since_spawn = 0.0

        # -- Move bubbles downward ------------------------------------
        escaped: List[Bubble] = []
        for bubble in self._bubbles:
            bubble.y -= bubble.speed * dt
            if bubble.y < -1.2:  # off-screen
                escaped.append(bubble)

        for b in escaped:
            self._bubbles.remove(b)
            self._misses += 1

        # -- Pop detection via jaw blendshape -------------------------
        jaw_open = False
        if face_result.face_detected:
            jaw_open = (
                face_result.blendshapes.get("jawOpen", 0.0)
                >= _JAW_OPEN_THRESHOLD
            )

        # Edge-detect rising edge (mouth just opened)
        just_opened = jaw_open and not self._jaw_was_open
        self._jaw_was_open = jaw_open

        if just_opened and face_result.face_detected:
            self._try_pop(face_result)

        # -- Check game-over ------------------------------------------
        if self._misses >= _MAX_MISSES:
            logger.info(
                "BubblePop game over — score: %d, misses: %d",
                self.score,
                self._misses,
            )
            self.state = GameState.FINISHED

    def _try_pop(self, face_result: FaceTrackResult) -> None:
        """
        Attempt to pop the bubble nearest to the nose-tip landmark.

        Uses the nose-tip normalised x-position to determine which
        bubble the player is targeting.

        Parameters:
            face_result (FaceTrackResult): Used to get nose-tip position.
        """
        if not face_result.landmarks:
            return

        # Nose-tip landmark index 1, map from [0,1] to NDC [-1,1]
        nose = face_result.landmarks[1]
        nose_ndc_x = (nose.x - 0.5) * 2.0
        nose_ndc_y = -(nose.y - 0.5) * 2.0  # flip Y

        popped: Optional[Bubble] = None
        best_dist = 0.25  # max NDC distance to register a pop

        for bubble in self._bubbles:
            dx = bubble.x - nose_ndc_x
            dy = bubble.y - nose_ndc_y
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_dist:
                best_dist = dist
                popped = bubble

        if popped is not None:
            self._bubbles.remove(popped)
            self.score += 1
            logger.debug("Bubble popped! Score: %d", self.score)

    # ------------------------------------------------------------------
    # Per-frame rendering
    # ------------------------------------------------------------------

    def render(
        self,
        pass_encoder: wgpu.GPURenderPassEncoder,
        viewport_width: int,
        viewport_height: int,
    ) -> None:
        """
        Upload bubble geometry and draw all active bubbles.

        Parameters:
            pass_encoder (wgpu.GPURenderPassEncoder): Active render pass
                encoder (game pass).
            viewport_width (int): Render target width in pixels.
            viewport_height (int): Render target height in pixels.
        """
        if not self._bubbles or self._device is None:
            return

        # Build vertex data for all active bubbles
        data = bytearray()
        for bubble in self._bubbles:
            # position (2× f32), radius (f32), pad (f32), colour (4× f32)
            data += struct.pack(
                "ffff ffff",
                bubble.x, bubble.y, bubble.radius, 0.0,   # pos + pad
                *bubble.colour,
            )

        if not data:
            return

        self._device.queue.write_buffer(self._vertex_buffer, 0, bytes(data))

        pass_encoder.set_pipeline(self._pipeline)
        pass_encoder.set_vertex_buffer(0, self._vertex_buffer)
        # 6 vertices per quad × num_bubbles instances
        pass_encoder.draw(6, len(self._bubbles), 0, 0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _random_colour() -> tuple:
        """
        Generate a random vivid RGBA colour for a bubble.

        Returns:
            tuple: (r, g, b, a) all floats in [0, 1].
        """
        colours = [
            (1.0, 0.35, 0.35, 0.85),  # red
            (0.35, 0.75, 1.0, 0.85),  # blue
            (0.5, 1.0, 0.5, 0.85),    # green
            (1.0, 0.85, 0.2, 0.85),   # yellow
            (0.9, 0.4, 1.0, 0.85),    # purple
        ]
        return random.choice(colours)

    def _spawn_bubble(self) -> None:
        """Create a new bubble at the top of the screen."""
        x = random.uniform(-0.85, 0.85)
        radius = random.uniform(0.05, 0.11)
        speed = _BUBBLE_SPEED * random.uniform(0.7, 1.4)
        self._bubbles.append(
            Bubble(
                x=x,
                y=1.1 + radius,  # just above the visible area
                radius=radius,
                colour=self._random_colour(),
                speed=speed,
            )
        )
