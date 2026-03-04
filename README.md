# Realtime FaceFX

A modern GPU-accelerated desktop application that captures camera input, processes it through real-time filters, and renders the result using WebGPU.

Built with **Python + wgpu**, designed for high performance on both AMD and Nvidia GPUs.

* https://github.com/pygfx/wgpu-py?tab=readme-ov-file

---

## Features

- Live camera capture
- Face tracking with 3D landmarks
- GPU-based comic & stylized filters
- 3D object overlays (hats, masks, face wraps)
- Real-time particle effects (e.g. fire from mouth)
- Blendshape-driven interaction logic
- Cross-platform GPU rendering (Vulkan / DX12 / Metal via WebGPU)

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ |
| Camera | OpenCV |
| Face Tracking | MediaPipe |
| GPU Rendering | wgpu (WebGPU) |
| Shaders | WGSL |
| Windowing | GLFW |
| Math | NumPy + PyGLM |
| 3D Assets | glTF |

---

## Architecture Overview

```
Camera (OpenCV)
       ↓
Face Tracking (MediaPipe)
       ↓
CPU Sync Layer
       ↓
GPU Pipeline (wgpu)
├── Vertex Shaders
├── Fragment Shaders
└── Compute Shaders
       ↓
Screen Output
```

Python orchestrates the system. The GPU performs all heavy rendering, warping, compositing, and particle simulation.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/realtime-facefx.git
cd realtime-facefx
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
.venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
python main.py
```

**Requirements:**

- Modern GPU with Vulkan / DX12 / Metal support
- Updated GPU drivers
- Python 3.11+

---

## Filter Types

### Comic / Stylized Filters

- Edge detection
- Posterization
- Color quantization
- Shader-based tone mapping

### Face Wrap Filters

- MediaPipe face mesh
- Vertex displacement in GPU
- UV texture projection

### 3D Object Filters

- glTF model loading
- Head pose transformation
- Depth-aware compositing

### Interactive Effects

- Mouth-open detection
- Particle system (compute shader)
- Real-time collision logic

---

## Mini Game Framework (Planned)

The engine is designed to support lightweight GPU-driven mini games.

**Example:**

1. Open mouth → emit fire particles
2. Particles collide with enemies
3. Score and health logic handled in Python

---

## Project Structure

```
realtime-facefx/
├── main.py
├── camera/
│   └── capture.py
├── tracking/
│   └── face_tracker.py
├── render/
│   ├── renderer.py
│   └── shaders/
│       ├── vertex.wgsl
│       ├── fragment.wgsl
│       └── compute_particles.wgsl
├── assets/
│   ├── models/
│   └── textures/
├── filters/
│   ├── comic.py
│   └── face_wrap.py
└── README.md
```

---

## Development Roadmap

| Phase | Goals |
|-------|-------|
| Phase 1 | Camera → GPU texture → screen, basic fullscreen shader |
| Phase 2 | Face mesh rendering, landmark debugging overlay |
| Phase 3 | 3D object attachment, face wrap distortion |
| Phase 4 | Compute-based particle engine, mini-game prototype |

---

## Performance Goals

- 60 FPS at 720p
- All visual effects handled on GPU
- Minimal CPU-GPU synchronization
- Cross-vendor compatibility

🛠 Debugging Tips

Validate Vulkan support using system tools

Use validation layers when available

Log GPU adapter info on startup

Keep textures and buffers reused between frames

📜 License

MIT License

👤 Author

Built as a modern GPU playground for real-time interactive effects.
