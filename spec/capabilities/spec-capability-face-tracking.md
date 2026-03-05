---
title: Camera Mayham — Face Tracking Capability
version: 1.1
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
- Only one face is tracked at a time.
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

### Startup & shutdown

- The tracker loads its AI model when the app starts.  If the model file is
  missing it is downloaded automatically (~3.6 MB, one-time).
- The tracker releases its resources cleanly when the app closes.

### What it does not do

- It does not use the GPU — tracking runs on the CPU so the GPU is fully
  available for rendering.
- It does not send any data off-device.
- It does not track more than one face at a time.

---

## 4. Output Data

Each frame the tracker produces a **tracking result** containing:

| Field | When face found | When no face found |
|-------|----------------|-------------------|
| Face detected | `true` | `false` |
| Landmarks | 478 points with X, Y, depth | Empty |
| Head pose | Yaw, pitch, roll in degrees | All zeros |
| Blendshapes | Named expression scores (0–1) | Empty |

Consumers should always check the "face detected" flag before using the other
fields.

---

## 5. Lifecycle

The tracker goes through three stages:

1. **Start up** — Load the AI model. Fails loudly if the model cannot be
   loaded (file corrupt, disk full, etc.) so the problem is obvious.
2. **Run** — Process frames one at a time and return a tracking result for each.
3. **Shut down** — Release the model and all associated resources.

The tracker is created once when the app launches and lives for the entire
session.  It is not designed to be started and stopped repeatedly.

---

## 6. Acceptance Criteria

| Scenario | Expected result |
|----------|----------------|
| App launches with model file present | Tracker starts without errors |
| App launches without model file | Model is downloaded automatically; tracker starts normally |
| Frame arrives with a visible face | Result contains 478 landmarks, a head pose, and expression scores; "face detected" is true |
| Frame arrives with no face visible | Result has empty landmarks and scores; "face detected" is false |
| A corrupted or unexpected frame is processed | Error is logged; that frame is skipped; the app keeps running |
| App closes normally | Tracker shuts down and releases all resources cleanly |

---

## 7. Related Specifications

| Specification | Relationship |
|---------------|-------------|
| `spec/main.md` | Product overview that establishes face tracking as a core capability |
| `spec/filters/spec-design-filter-face-landmarks.md` | Filter that visualises the landmark positions produced by this capability |
