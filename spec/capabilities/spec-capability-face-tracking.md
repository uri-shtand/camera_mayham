---
title: Camera Mayham — Face Tracking Capability
version: 1.2
date_created: 2026-03-06
last_updated: 2026-03-06
tags: [capability, face-tracking, mediapipe, tracking]
---

# Introduction

Face tracking is the part of Camera Mayham that watches the webcam feed and
figures out where the user's face is, which way their head is pointing, and what
expression they are making — all in real time, entirely on-device.

Every frame, it produces three things: the positions of 478 points on the face,
the orientation of the head (turning, tilting, nodding), and a set of named
expression scores (e.g. "mouth open" or "left eye blink").  Filters, 3D
overlays, and mini-games all consume this data to respond to the user's face.


## Documentation and examples

* https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
* https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python
* https://pylessons.com/face-detection
---

## 1. Purpose & Scope

**Purpose:** Describe what the face tracking capability does, what data it
produces, and what "working correctly" looks like.

**Scope:** Covers the tracker itself — how it starts up, what it outputs each
frame, and how it shuts down.  Does not cover how individual features (filters,
games, overlays) use the tracking output; those are defined in their own
specifications.

**Audience:** Engineers, designers, and AI systems building or extending Camera
Mayham.

**Assumptions:**

- The app runs on a standard Windows desktop; the CPU is fast enough to run
  face tracking at interactive frame rates alongside the rest of the app.
- The number of faces tracked simultaneously is configurable (default: 1).
- No internet connection is needed during normal use; the model file is bundled
  with the app or downloaded once on first launch.

---

## 2. Definitions

| Term | Definition |
|------|------------|
| **Landmark** | One of 478 points on the user's face (e.g. corner of the eye, tip of the nose). Each point has an X/Y position within the frame and a rough depth value. |
| **Head pose** | The direction the head is facing — described as three angles: left/right turn (yaw), up/down tilt (pitch), and side-to-side tilt (roll). |
| **Blendshape** | A named score between 0 and 1 that measures one specific facial expression, such as `mouthOpen` or `eyeBlinkLeft`. |
| **Tracking result** | The bundle of data the tracker produces for a single frame: landmarks, head pose, blendshapes, and a flag indicating whether a face was found at all. |
| **Model file** | The pre-trained AI model (`face_landmarker.task`) that the tracker loads at startup to perform detection. |
| **Video mode** | The tracker processes a continuous stream of frames from the webcam, using temporal information across frames to improve accuracy and stability. |
| **Image mode** | The tracker processes each frame independently, with no temporal context from prior frames.  Suitable for still images or non-sequential inputs. |
| **Sensitivity** | A configurable parameter that controls how readily the tracker reports a detection — higher sensitivity may detect partial or distant faces but can increase false positives. |

---

## 3. What Face Tracking Must Do

### Core behaviour

- **Detect the face.** Every frame, determine whether a face is visible in the
  camera image.  If no face is found, report that clearly so consumers can
  respond gracefully (e.g. pause a game, skip rendering overlays).

- **Report landmark positions.** When a face is detected, provide the position
  of all 478 face points within the frame.  Positions are expressed as
  fractions of the frame size (0 = left/top edge, 1 = right/bottom edge) so
  they work regardless of camera resolution.

- **Report head orientation.** Estimate which way the head is pointing:
  - **Yaw** — turning left or right
  - **Pitch** — nodding up or down
  - **Roll** — tilting side to side

- **Report expression scores.** Provide a score for each recognised facial
  expression so that games and overlays can react to what the user is doing with
  their face.

- **Run silently.** Tracking errors (e.g. a corrupted frame) must not crash the
  app.  The tracker logs the problem and skips that frame.

### Operating modes

The tracker supports two operating modes that must be selected at initialisation:

- **Video mode** (default) — frames are processed as a continuous stream.  The
  tracker exploits inter-frame continuity to smooth results and recover quickly
  when the face is briefly occluded.
- **Image mode** — each frame is treated as a standalone image with no
  relationship to previous frames.  Use this mode when processing still images
  or when the input is not temporally ordered.

The active mode is exposed in the tracker's configuration and can be changed
through the face tracking settings widget before a session starts.

### Startup & shutdown

- The tracker loads its AI model when the app starts.  If the model file is
  missing it is downloaded automatically (~3.6 MB, one-time).
- The tracker releases its resources cleanly when the app closes.

### What it does not do

- It does not use the GPU — tracking runs on the CPU so the GPU is fully
  available for rendering.
- It does not send any data off-device.

---

## 4. Configuration & Settings Widget

Face tracking exposes a dedicated settings widget in the application UI.  Users
can open this widget at any time to inspect and adjust the tracker's behaviour
without restarting the app.

### Configurable parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| **Operating mode** | `video` or `image` — controls temporal processing (see §3 Operating modes) | `video` |
| **Sensitivity** | Detection confidence threshold (0.0–1.0).  Lower values make the tracker more willing to report a face; higher values require stronger evidence. | `0.5` |
| **Number of faces** | Maximum number of faces the tracker will detect and report per frame (1–4). | `1` |

### Widget behaviour

- Changes to sensitivity and number of faces take effect on the next processed
  frame; no restart is required.
- Operating mode changes require the tracker to reinitialise; the widget
  indicates this with a brief loading state.
- All settings persist between app launches.

---

## 5. Output Data

Each frame the tracker produces a **tracking result** containing:

| Field | When face found | When no face found |
|-------|----------------|-------------------|
| Face detected | `true` | `false` |
| Landmarks | 478 points with X, Y, depth (per face) | Empty |
| Head pose | Yaw, pitch, roll in degrees (per face) | All zeros |
| Blendshapes | Named expression scores (0–1) (per face) | Empty |

When **number of faces** is greater than 1, the result contains a list of
per-face entries ordered by detection confidence (highest first).  Consumers
should always check the "face detected" flag before using the other fields.

---

## 6. Lifecycle

The tracker goes through three stages:

1. **Start up** — Load the AI model and apply the saved configuration
   (operating mode, sensitivity, number of faces).  Fails loudly if the model
   cannot be loaded (file corrupt, disk full, etc.) so the problem is obvious.
2. **Run** — Process frames one at a time and return a tracking result for each.
3. **Shut down** — Release the model and all associated resources.

The tracker is created once when the app launches and lives for the entire
session.  It reinitialises automatically when the operating mode is changed via
the settings widget.

---

## 7. Acceptance Criteria

| Scenario | Expected result |
|----------|----------------|
| App launches with model file present | Tracker starts without errors |
| App launches without model file | Model is downloaded automatically; tracker starts normally |
| Frame arrives with a visible face (video mode) | Result contains 478 landmarks, a head pose, and expression scores; "face detected" is true |
| Frame arrives with a visible face (image mode) | Same as video mode; result is computed independently with no prior frame context |
| Frame arrives with no face visible | Result has empty landmarks and scores; "face detected" is false |
| A corrupted or unexpected frame is processed | Error is logged; that frame is skipped; the app keeps running |
| User opens the settings widget | Widget displays current operating mode, sensitivity, and number of faces |
| User changes sensitivity in widget | New threshold applies from the next frame onwards without restarting |
| User changes number of faces to 2 | Tracker reports up to 2 faces per frame, ordered by confidence |
| User switches operating mode via widget | Tracker reinitialises; widget shows a loading state during reinitialisation |
| Settings are saved and app is relaunched | Previously saved configuration is restored on startup |
| App closes normally | Tracker shuts down and releases all resources cleanly |

---

## 8. Related Specifications

| Specification | Relationship |
|---------------|-------------|
| `spec/main.md` | Product overview that establishes face tracking as a core capability |
| `spec/filters/spec-design-filter-face-landmarks.md` | Filter that visualises the landmark positions produced by this capability |
