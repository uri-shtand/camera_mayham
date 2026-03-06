# Camera Mayham — Copilot Instructions

## Project Overview

**Camera Mayham** is a GPU-accelerated desktop application that transforms a live webcam feed into an interactive visual playground. The camera feed is the hero of the UI. The user's face acts as a game controller via real-time facial landmark tracking.

Key capabilities:
- Live webcam display at 60 FPS (1280×720 target)
- GPU shader-based visual filters (grayscale, edge detection, colour shift, face landmarks)
- Real-time face tracking: landmarks, head pose, blendshapes (expressions)
- 3D model overlays anchored to the tracked face
- Interactive mini-games driven by facial input (e.g. BubblePop — open mouth to pop bubbles)
- Single-window UI: camera area on top, fixed widget bar docked at the bottom

## Tech Stack

| Component       | Technology                                      |
|-----------------|-------------------------------------------------|
| Language        | Python 3.11+                                    |
| GPU Rendering   | `wgpu` (WebGPU) + WGSL shaders                  |
| Camera          | OpenCV (`VideoCapture`)                         |
| Face Tracking   | MediaPipe (landmarks, head pose, blendshapes)   |
| UI Framework    | Dear PyGui (single-window)                      |
| 3D Math         | PyRR (vectors, matrices, quaternions)           |
| 3D Models       | Trimesh (glTF/OBJ/STL loading)                  |
| Numerics        | NumPy (frame buffers, GPU data)                 |
| Testing         | pytest + Pillow (visual regression)             |

GPU rendering targets cross-vendor support (AMD/Nvidia) via WebGPU. No vendor-specific extensions.

## Module Map

| Module          | Purpose                                                          |
|-----------------|------------------------------------------------------------------|
| `app/`          | Application orchestration; owns subsystems; drives the main loop |
| `camera/`       | Webcam frame capture via OpenCV                                  |
| `tracking/`     | Face detection and data extraction via MediaPipe                 |
| `filters/`      | GPU shader-based visual effects (one class per filter)           |
| `games/`        | Interactive mini-games driven by facial input                    |
| `overlays/`     | 3D models anchored to the tracked face                           |
| `rendering/`    | GPU pipeline orchestration and render passes                     |
| `ui/`           | Dear PyGui main window and widget panel                          |
| `config/`       | Persisted tracker settings (`tracker_config.json`)               |
| `assets/`       | MediaPipe model file (`.task`)                                   |
| `3dModels/`     | 3D asset files (glTF)                                            |
| `spec/`         | Product and design specifications                                |
| `tests/`        | Unit and integration tests (pytest)                              |

## Architecture

### Render Pipeline

Each frame flows through the following passes in order:

```
BackgroundPass  →  camera frame  →  tex[0]
FilterPass      →  tex[0]  →  shader filters  →  tex[n]
OverlayPass     →  composite 3D models
GamePass        →  composite game elements
PostPass        →  final blit  →  readback to CPU
```

The final frame is read back to a NumPy array and displayed via a Dear PyGui texture. The GPU pipeline runs **offscreen** — there is no native GPU window. Everything renders inside a single Dear PyGui OS window.

### Key Design Patterns

**Plugin ABC pattern** — Extend these base classes to add new functionality, no core changes needed:
- `BaseFilter` (`filters/base.py`): `setup()` → `apply(encoder, in_tex, out_tex)` → `teardown()`
- `BaseGame` (`games/base.py`): `setup()` → `start()` → `update(state, face_result)` + `render(pass_encoder)` → `stop()`
- `BaseOverlay` (`overlays/base.py`): `setup()` → `render(pass_encoder, head_pose)`

**Ping-pong textures** — `FilterPass` alternates between two RGBA8 textures each frame to avoid feedback loops and eliminate per-frame GPU allocation.

**AppState dataclass** (`app/state.py`) — Single mutable shared state passed through the system:
- Written only from the main thread
- `activate_filter(name)` / `activate_game(name)` enforce single-select semantics (enabling one disables all others)
- Access filters/games by name: `get_filter()`, `get_game()`

## Coding Conventions

Follow the rules in `.github/instructions/python.instructions.md` for all Python files. Key points:
- PEP 8 style; lines ≤ 79 characters
- Type hints on all function signatures (`typing` module)
- PEP 257 docstrings on all public classes and functions
- Descriptive function names; break complex logic into smaller helpers

## Performance Targets

- **60 FPS** at 1280×720
- **GPU frame time < 16 ms**
- No CPU/GPU synchronisation (`readback`) inside the hot render loop except at the final `PostPass`
- No blocking calls on the main thread (camera capture and face tracking run on worker threads)

## Testing Conventions

- Framework: `pytest`
- Tests must be run **inside the project's virtual environment (venv)**
- Unit tests must **not** depend on a real GPU, camera, or MediaPipe model — use stub/mock fixtures
- The `_Stub*` pattern is used for fake implementations of `BaseFilter`, `BaseGame`, etc.
- Visual regression tests use Pillow to compare rendered frames against pre-recorded expected outputs
- Target: **> 80% coverage** for core pipeline and plugin interfaces

## Constraints

- **Offline only** — no network calls; all models are bundled in `assets/`
- **Windows target** — primary development and test platform
- **No vendor-specific GPU extensions** — WebGPU cross-vendor support must be maintained
- **Non-blocking UI** — the Dear PyGui event loop must never stall; heavy work runs on threads
