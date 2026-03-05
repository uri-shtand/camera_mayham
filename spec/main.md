---
title: Camera Mayham — Product Overview & High-Level Specification
version: 1.1
date_created: 2026-03-04
last_updated: 2026-03-05
tags: [architecture, design, app]
---

# Introduction

Camera Mayham is a GPU-accelerated desktop application that turns a live webcam feed into an interactive visual experience. Users can apply real-time visual filters, attach 3D objects to their face, and play lightweight mini-games — all controlled through facial expressions and head movement.

The application is built in Python, with all rendering offloaded to the GPU via `wgpu` (a cross-platform WebGPU implementation). Face tracking runs locally on-device with no network connection required.

This document defines what the product does, how its major parts relate to each other, and what "done" looks like. Detailed behaviour for individual subsystems is covered in separate linked specifications.

---

## 1. Purpose & Scope

**What this document covers:** The core product requirements, system boundaries, performance expectations, and architectural principles for Camera Mayham.

**What it does not cover:** Low-level rendering pipeline design, individual filter architecture, particle systems, mini-game internals, or UI component structure — each of those has its own dedicated spec.

**Who it is for:** Engineers building or extending the system, designers shaping the user experience, and AI systems used for code generation or analysis.

**Key assumptions:**
- Windows desktop is the primary and required target platform.
- All GPU work goes through `wgpu` — no direct OpenGL or Vulkan calls.
- Everything runs locally; no server-side processing or cloud services are needed.

---

## 2. Definitions

| Term | Meaning |
|------|---------|
| **GPU** | Graphics Processing Unit — the hardware chip responsible for rendering |
| **wgpu** | A Rust-based WebGPU library with Python bindings, used as the GPU layer |
| **WebGPU** | A modern GPU API that works across AMD, Nvidia, and Intel hardware |
| **FPS** | Frames Per Second — how many frames the app renders each second |
| **Face Landmark** | A specific detected point on a face (e.g., tip of the nose, corner of an eye) |
| **Blendshape** | A named facial expression score (e.g., `mouthOpen = 0.85`) used as a game input |
| **Head Pose** | The estimated 3D rotation and position of the user's head |
| **Filter** | A GPU shader that transforms the look of the camera frame (e.g., grayscale, edge detection) |
| **Mini-game** | A small interactive game embedded in the render pipeline, controlled by the user's face |
| **3D Overlay** | A 3D model rendered over the camera feed, anchored to the user's face |
| **Widget Panel** | The side panel UI where users control filters, overlays, and mini-games |
| **Pipeline Pass** | One stage in the rendering sequence (e.g., draw the camera frame, then draw filters on top) |

---

## 3. What the Product Must Do

### Core Features

Camera Mayham has six primary feature areas:

1. **Live camera display** — The webcam feed is captured and rendered to the screen every frame at the target frame rate. This is the foundation everything else builds on.

2. **Visual filters** — Users can apply one or more GPU-powered filters to the camera feed (e.g., grayscale, colour shift, edge detection). Filters can be toggled on or off instantly, and their parameters (like intensity or colour tint) can be adjusted while the app is running.

3. **Face tracking** — The app continuously detects the user's face and extracts:
   - Landmark positions (eyes, nose, mouth, jawline, etc.)
   - Head pose (rotation and translation in 3D)
   - Blendshape scores (quantified expressions like "raised eyebrow" or "mouth open")

4. **3D overlays** — 3D models can be attached to the tracked face and rendered over the camera feed with correct depth, transparency, and blending. The models follow head movement in real time.

5. **Mini-games** — Lightweight games run within the render pipeline and use facial input as the controller (e.g., open your mouth to shoot a bubble). Games render as an overlay on top of the camera feed.

6. **Widget panel** — A side panel gives users control over all of the above: toggling filters, adjusting settings, choosing 3D overlays, and launching mini-games.

### Constraints

These are hard rules the implementation must respect:

- **GPU-first rendering:** Filters, 3D overlays, and mini-game visuals should run on the GPU wherever practical. CPU fallback is permitted only when a feature cannot reasonably be GPU-implemented.
- **Minimal CPU-GPU sync:** Synchronisation between the CPU and GPU must be kept to a minimum to avoid frame stalls or unnecessary buffer reallocations.
- **GPU vendor-neutral:** The app must work on AMD and Nvidia hardware using the same code path — no vendor-specific extensions.
- **Fully offline:** No internet connection is required at any point. All processing runs locally.
- **Non-blocking UI:** The widget panel runs in parallel with the render loop and must never stall frame production.
- **Windows platform:** The application targets Windows as its only supported OS.
- **Beutiful and fun UX:** It should be engaging and fun to use.

### Performance Targets

| Target | Value |
|--------|-------|
| Rendering throughput | 60 FPS at 1280×720 |
| GPU frame time | Under 16 ms per frame |
| Frame pacing | Stable — no visible jitter |
| Memory churn | Minimal during steady-state rendering |

### Architectural Principles

These guidelines shape how the codebase should be structured:

- **Decoupled subsystems:** Rendering, face tracking, filters, and mini-games are independent. None of them should depend on the internal details of another.
- **Pluggable by design:** Adding a new filter or mini-game should not require touching the core rendering pipeline.
- **CPU for logic, GPU for pixels:** Python orchestrates state and timing; shaders and draw calls handle everything visual.
- **Immediate UI feedback:** Any change made in the widget panel should be visible in the very next rendered frame.
- **Filters and mini-games are autowired** Any new filter or mini game is automatically added to the widget in the right place.
7 **Filter configuration** Each filter has a configuration file that defines the default definitions for it.

---

## 4. How It Fits Together

### Rendering Pipeline

Every frame flows through the same sequence of steps:

```
Webcam input
    ↓
Face tracking  (landmarks, head pose, blendshapes)
    ↓
Application state  (which filters/overlays/games are active)
    ↓
GPU render pipeline
    ├─ Background pass       → draws the raw camera frame
    ├─ Filter pass           → applies enabled shader filters
    ├─ 3D overlay pass       → renders face-attached 3D models
    ├─ Game pass             → draws mini-game visuals and runs game logic
    └─ Post-processing pass  → final compositing and output
    ↓
Display
```

### Functional Modules

| Module | What it does |
|--------|-------------|
| Camera capture | Grabs frames from the webcam and uploads them to the GPU |
| Filter system | Runs GPU shaders and manages filter state and parameters |
| Face tracker | Detects the face and produces per-frame landmark, pose, and expression data |
| 3D overlay renderer | Loads and renders 3D models anchored to the tracked face |
| Mini-game framework | Manages game lifecycle, maps face input to game events, issues draw calls |
| Widget panel | Presents UI controls and writes user choices into application state |

### Screen Layout

| Area | Description |
|------|-------------|
| Primary render area | Fills the window — shows the live camera with all layers composited on top |
| Widget panel | A collapsible side panel for controlling filters, overlays, and games |

### Plugin Contracts

**Every filter exposes:**
- An on/off toggle that takes effect without reloading the pipeline
- A set of named parameters (floats, ints, colours) adjustable at runtime
- A `wgpu`-compatible GPU shader entry point

**Every mini-game exposes:**
- Start and stop lifecycle methods
- A declaration of which face inputs map to which game actions
- GPU draw calls to issue during the game pass

---

## 5. Acceptance Criteria

The following conditions must all be true for the product to be considered complete and correct:

| ID | Scenario | Expected outcome |
|----|----------|-----------------|
| AC-001 | App starts and loads | Live camera feed appears in the render area at the target frame rate |
| AC-002 | Filter is enabled | Effect is visible from the next rendered frame onward |
| AC-003 | Filter is disabled | Effect is gone from the next rendered frame onward |
| AC-004 | 3D overlay is selected while a face is detected | Model tracks head rotation and position in real time |
| AC-005 | Mini-game is launched and user opens their mouth | Game responds to the blendshape input |
| AC-006 | App running at 1280×720 on either AMD or Nvidia GPU | Stable 60 FPS with no unhandled GPU errors |
| AC-007 | Multiple filters are active simultaneously | Each can be toggled independently without restarting the pipeline |
| AC-008 | App is closed | Shuts down cleanly with no GPU errors or crashes |

---

## 6. Testing Strategy

Testing covers three levels:

- **Unit tests** check individual component logic (filter state, game events, tracking data handling) in isolation using `pytest`.
- **Integration tests** verify that pipeline stages interact correctly — for example, that a filter enabled in application state actually runs in the filter pass each frame.
- **Manual / visual tests** confirm rendered output looks correct, since visual quality cannot be fully automated.

For test reproducibility, pre-recorded webcam clips stand in for live camera input. Where a real GPU is unavailable (e.g., in CI), a software rasterizer or pre-recorded pipeline output is used as a substitute.

**Coverage requirement:** Core pipeline orchestration and all filter/mini-game plugin interfaces must have greater than 80% unit test coverage.

**Performance tests:** Frame timing benchmarks run on reference hardware to confirm the 60 FPS and 16 ms GPU frame time targets are met.

---

## 7. Design Rationale

Camera Mayham is designed as a real-time GPU canvas, not just a filter app. The central idea is that **the face is a game controller** — landmark positions and expression scores become input events that drive visuals, games, and effects. The camera feed is the world; shaders and 3D models are the art layer on top.

**Why `wgpu` instead of OpenGL or Vulkan directly?**
WebGPU is cross-vendor by design, meaning the same code runs on AMD and Nvidia without conditional rendering paths. It also exposes modern GPU compute capabilities that older APIs like OpenGL lack.

**Why Python?**
Python's computer vision ecosystem (MediaPipe for face tracking, OpenCV for capture) makes it the natural choice for orchestration. All performance-sensitive work — shaders, draw calls, texture operations — is delegated to the GPU, so Python's speed is not a bottleneck.

**Why pluggable modules?**
Keeping filters and mini-games as plugins that slot into a fixed pipeline means the project can grow (more filters, more games, new effect types) without every new feature requiring changes to the core rendering loop.

---

## 8. Dependencies

### Hardware & OS

| Component | Role |
|-----------|------|
| Webcam + OS camera API | Source of the live video feed |
| AMD or Nvidia GPU + driver | Hardware execution of all rendering and compute |
| Windows OS | Required host platform |

### Software Libraries

| Library | Role |
|---------|------|
| `wgpu` (WebGPU via Rust/Python) | Cross-vendor GPU rendering and compute |
| MediaPipe (or equivalent) | On-device face landmark detection, head pose, and blendshapes |
| Python ≥ 3.10 | Application orchestration, state management, and plugin loading |

No third-party cloud services or network APIs are required.

---

## 9. Future Extensibility

The architecture is designed to accommodate the following without modifying the core pipeline:

- New filter packs (additional shaders)
- Compute-heavy GPU effects (particles, fluid simulation)
- Multi-face tracking
- Hand or body tracking as an additional input source
- Audio-reactive visual effects
- More complex mini-games with persistent state

---

## 10. Related Specifications

| Document | What it covers |
|----------|----------------|
| [spec-design-main-screen.md](spec-design-main-screen.md) | Window layout, widget bar, expansion tray, and active-state model |
| `spec-design-filter-system.md` | Filter architecture, shader interface, and compositing |
| `spec-architecture-rendering-pipeline.md` | GPU pipeline technical design |
| `spec-design-particle-engine.md` | Particle system design |
| `spec-design-minigame-framework.md` | Mini-game framework and face input model |
| `spec-design-ui-architecture.md` | Widget bar implementation architecture |