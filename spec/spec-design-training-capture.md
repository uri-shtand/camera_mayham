---
title: Camera Mayham - Training Data Capture Feature
version: 1.0
date_created: 2026-03-04
last_updated: 2026-03-04
tags: [design, ui, app, data]
---

# Introduction

This specification defines the behaviour of the **Training Data Capture** feature
for Camera Mayham. The feature adds a dedicated widget bar Action item that, when
triggered, prompts the user for a class label and then captures 20 raw camera
frames at 100 ms intervals, saving them to a local folder structured for use as a
machine-learning training dataset.

---

## 1. Purpose & Scope

**Purpose:** Allow users to quickly build labelled image datasets from the live
camera feed directly within the Camera Mayham UI, without leaving the application
or requiring external tooling.

**Scope:** Covers the UI entry point (widget bar Action item), the label-input
dialog, the frame capture sequence, the file-system output structure, the
AppState additions required, and the non-blocking execution model. Does not cover
model training, dataset management, import/export tooling, or GPU post-processing
of captured images.

**Audience:** Engineers and AI systems implementing or extending Camera Mayham.

**Assumptions:**
- The host machine has a writeable local filesystem at the working directory of
  the process.
- The camera device is open and delivering frames when the capture is triggered.
- Captured images are intended for offline model training; real-time feedback on
  dataset quality is out of scope.

---

## 2. Definitions

| Term | Definition |
|------|------------|
| **Capture Session** | A single triggered sequence of 20 raw frame saves associated with one user-supplied class label |
| **Class Label** | A short, user-supplied string that identifies the semantic category of the captured images (e.g. `"smile"`, `"open_mouth"`)  |
| **Raw Frame** | The camera frame as returned by `CameraCapture.read()` — an `(H, W, 3)` BGR uint8 `np.ndarray` — before any filter, GPU processing, or rendering pass |
| **Session Timestamp** | A `YYYYMMDD_HHMMSS` wall-clock string recorded at the moment capture begins; used to namespace filenames within a class directory |
| **Output Root** | The `./training_data/` directory relative to the process working directory |
| **Class Directory** | A subdirectory of the Output Root named after the sanitised Class Label, e.g. `./training_data/smile/` |
| **Action Widget** | A `WidgetItem` with `category="action"` — a reserved category in the widget bar for non-filter, non-game operations |
| **Capture Button** | The Action widget item in the widget bar that triggers a Capture Session |
| **DPG** | Dear PyGui — the UI library used for the widget bar and all dialogs |

---

## 3. Requirements, Constraints & Guidelines

### Functional Requirements

- **REQ-001**: The widget bar shall contain a **Capture Button** Action widget
  item, displayed alongside the filter and game items, separated by a group
  divider in accordance with `spec-design-main-screen` REQ-012.
- **REQ-002**: The Capture Button shall use the category `"action"` on its
  `WidgetItem` descriptor and shall not have an expansion tray
  (`is_expandable=False`).
- **REQ-003**: When the Capture Button is clicked and no capture is already in
  progress, the system shall present a **label-input dialog** requesting the user
  to enter a Class Label string before any frames are captured.
- **REQ-004**: The label-input dialog shall be a Dear PyGui modal window so that
  no secondary OS window is created (consistent with `spec-design-main-screen`
  REQ-001).
- **REQ-005**: If the user cancels the dialog or submits an empty string, the
  Capture Session shall be **aborted**; no images shall be written and the Capture
  Button shall return to its inactive state immediately.
- **REQ-006**: After the user confirms a valid Class Label, the system shall
  begin a Capture Session: exactly **20 raw frames** shall be captured at
  intervals of **100 ms**, giving a nominal session duration of approximately
  1.9 seconds.
- **REQ-007**: Captured frames shall be the **raw camera frames** returned by
  `CameraCapture.read()` prior to any GPU upload, filter application, or
  rendering pipeline processing.
- **REQ-008**: During a Capture Session the Capture Button shall enter a
  **capturing visual state** — distinct from both the active and inactive states
  — to indicate that capture is in progress (e.g. pulsing colour or alternate
  icon label).
- **REQ-009**: If the Capture Button is clicked while a Capture Session is
  already in progress, the click shall be **ignored**; no nested or concurrent
  sessions are permitted.
- **REQ-010**: Each captured frame shall be saved as a **PNG file** (lossless)
  to preserve pixel fidelity required for model training.
- **REQ-011**: Captured images shall be saved at:
  ```
  ./training_data/<sanitised_label>/<session_timestamp>_<frame_index>.png
  ```
  where `<frame_index>` is a zero-padded two-digit integer from `00` to `19`.
  Example path: `./training_data/smile/20260304_143052_00.png`.
- **REQ-012**: The **Output Root** (`./training_data/`) shall be created
  automatically if it does not already exist.
- **REQ-013**: The **Class Directory** (`./training_data/<label>/`) shall be
  created automatically if it does not already exist.
- **REQ-014**: If a frame cannot be captured (i.e. `CameraCapture.read()`
  returns `None`) the system shall **skip** that frame index and continue the
  sequence without failing the session; the remaining frames shall still be
  attempted on their scheduled intervals.
- **REQ-015**: After all 20 frame attempts are complete, the Capture Button
  shall return to its **inactive visual state** regardless of how many frames
  were successfully saved.
- **REQ-016**: On completion, a brief status indicator shall be shown (e.g. a
  DPG tooltip or temporary text overlay) reporting how many images were
  successfully saved and the path they were written to.

### Non-Blocking Execution Constraint

- **CON-001**: The frame capture loop (20 iterations × 100 ms sleep) shall
  execute on a **background thread** so that the main render loop is not
  blocked during the capture session. The render loop shall continue to process
  and display frames at the target rate during capture. (This is an extension of
  the existing system-level `CON-005`.)

### File-System Constraints

- **CON-002**: The Class Label shall be **sanitised** before use as a directory
  name: whitespace shall be replaced with underscores; any character that is not
  an ASCII alphanumeric, hyphen, or underscore shall be removed; the result
  shall be lower-cased. Labels that reduce to an empty string after sanitisation
  shall be treated as invalid (see REQ-005).
- **CON-003**: The Session Timestamp shall be formatted as
  `YYYYMMDD_HHMMSS` using the system's local clock at the moment the first frame
  capture attempt begins.
- **CON-004**: If two sessions share the same Class Label and Session Timestamp
  (sub-second race), existing files shall **not** be overwritten; the second
  session shall be treated as a new session — callers must ensure this by
  checking the timestamp at session start.

### Guidelines

- **GUD-001**: The background capture thread shall acquire the latest raw frame
  reference from `AppState.camera_frame` rather than calling
  `CameraCapture.read()` directly, to avoid contention with the camera capture
  thread already running in the main loop.
- **GUD-002**: `AppState` shall be extended with a boolean flag
  `is_capturing: bool = False`; the background thread sets it to `True` at
  start and `False` on completion, allowing the UI to poll it each frame for
  button state updates.
- **GUD-003**: Python's `threading.Thread` shall be used for the capture
  background thread; no async frameworks or subprocess spawning are required.
- **GUD-004**: Image write failures (e.g. `OSError` from disk full) shall be
  caught per-frame, logged at `WARNING` level, and counted as skipped frames;
  the session shall not raise an unhandled exception.

---

## 4. Interfaces & Data Contracts

### 4.1 AppState Extension

The following attribute shall be added to `AppState`
(`app/state.py`):

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `is_capturing` | `bool` | `False` | `True` while a Capture Session is in progress |

No methods are added to `AppState`; the capture logic is self-contained in the
capture worker thread launched by the UI callback.

### 4.2 Widget Item Descriptor

```
WidgetItem(
    id        = "capture",
    label     = "Capture",
    icon      = "📷",
    category  = "action",
    is_expandable = False,
)
```

The item shall be appended after the games group in the widget bar, separated by
a group divider.

### 4.3 Capture Session Sequence

```
User clicks Capture Button
    │
    ▼
is_capturing == True?  ─── YES ──▶  ignore click
    │ NO
    ▼
Show label-input DPG modal
    │
    ├── User cancels / empty input ──▶ abort, button stays inactive
    │
    └── User confirms label
            │
            ▼
        Sanitise label → create directories
            │
            ▼
        Set AppState.is_capturing = True
        Set Capture Button to "capturing" visual state
            │
            ▼
        Launch background thread:
            for i in 0..19:
                sleep(100 ms)       ← skips first sleep to capture index 0 immediately
                frame ← AppState.camera_frame  (latest raw BGR ndarray)
                if frame is None → skip, log warning
                else → save as PNG at ./training_data/<label>/<timestamp>_<NN>.png
            │
            ▼
        Set AppState.is_capturing = False
        Set Capture Button back to inactive state
        Show completion status (saved_count / 20, directory path)
```

### 4.4 File-System Output Structure

```
./training_data/
└── <sanitised_label>/
    ├── <YYYYMMDD_HHMMSS>_00.png
    ├── <YYYYMMDD_HHMMSS>_01.png
    ├── …
    └── <YYYYMMDD_HHMMSS>_19.png
```

Each PNG is a full-resolution BGR image as captured by the camera (e.g.
1280 × 720 at default settings). OpenCV's `cv2.imwrite()` shall be used for
encoding.

### 4.5 Label Sanitisation Algorithm

```
1. Strip leading/trailing whitespace.
2. Replace internal whitespace runs with a single underscore.
3. Remove any character that is not in [a-zA-Z0-9_-].
4. Lower-case the result.
5. If the result is empty → treat as cancelled (REQ-005).
```

---

## 5. Acceptance Criteria

- **AC-001**: Given the application is running and a camera is open, when the
  user clicks the Capture Button and enters the label `"smile"`, then exactly
  20 PNG files are created under `./training_data/smile/` within approximately
  2.5 seconds.
- **AC-002**: Given the Capture Button is clicked and the user cancels the
  label dialog, then no files are written and the button returns to its inactive
  state.
- **AC-003**: Given the Capture Button is clicked and the user submits an empty
  label, then no files are written, no directory is created, and the button
  returns to its inactive state.
- **AC-004**: Given a capture is in progress, when the user clicks the Capture
  Button again, then the click is ignored and no second session starts.
- **AC-005**: Given the Capture Button is clicked, when capture is in progress,
  then the render loop continues to update the camera display at the target frame
  rate without stalling.
- **AC-006**: Given `./training_data/` does not exist, when a capture session
  completes, then the directory has been created automatically.
- **AC-007**: Given 3 out of 20 frames are `None` (camera drops), when the
  session completes, then 17 PNG files are saved and a warning logs the 3
  skipped frames; no exception is raised.
- **AC-008**: Given the label `"Open Mouth!"` is entered, when the session
  completes, the images are saved under `./training_data/open_mouth/`.
- **AC-009**: Given a capture session completes successfully, then the Capture
  Button returns to its inactive visual state and a completion message reports
  the count of saved images and the output path.
- **AC-010**: Given two sessions with the same label run back-to-back in the
  same second, then the second session's files do not overwrite the first
  session's files.

---

## 6. Test Automation Strategy

- **Test Levels**: Unit (sanitisation logic, file-naming, thread lifecycle),
  Integration (AppState flag, widget state changes), Manual (end-to-end UI flow)
- **Frameworks**: `pytest` for unit and integration tests; `unittest.mock` to
  patch `AppState.camera_frame` and `cv2.imwrite` for deterministic testing
- **Test Data Management**: Mock `np.ndarray` frames used as synthetic camera
  output; temporary directories (`tmp_path` fixture) used for all file writes
- **CI/CD Integration**: Automated unit tests run on every pull request; no
  live camera or GPU required for unit/integration test execution
- **Coverage Requirements**: Sanitisation function, session worker thread logic,
  and AppState flag mutations must have >90% unit test coverage
- **Performance Testing**: Verify that a 20-frame capture session does not
  reduce average render FPS by more than 5% on reference hardware

---

## 7. Rationale & Context

The training data capture feature turns Camera Mayham into a lightweight dataset
collection tool — particularly useful for capturing labelled facial expression
images that can train a custom gesture/expression classifier to feed back into
the game or filter pipeline.

Raw frames (pre-filter) are captured to ensure training images are not
contaminated by application-specific visual effects, keeping the dataset
applicable to models that will run on unprocessed camera input.

The 100 ms interval over 20 frames provides a 1.9-second burst that captures
natural variation in expression or pose without requiring the user to hold a pose
for an extended period.

Background-thread execution ensures the interactive experience is not degraded —
the user can watch their own camera feed throughout the capture sequence, which
also helps them hold the correct expression.

---

## 8. Dependencies & External Integrations

### External Systems

| ID | Component | Purpose |
|----|-----------|---------|
| **EXT-001** | Webcam / OS camera API | Source of raw frames via `CameraCapture.read()` |
| **EXT-002** | Local filesystem | Output target for captured PNG files |

### Third-Party Services

_None. All capture and save operations are local._

### Infrastructure Dependencies

| ID | Component | Requirement |
|----|-----------|-------------|
| **INF-001** | `opencv-python` (`cv2`) | PNG encoding via `cv2.imwrite()` |
| **INF-002** | `dearpygui` (DPG) | Label-input modal dialog and button state updates |
| **INF-003** | Python `threading` stdlib | Background capture thread (no additional packages required) |
| **INF-004** | Python `pathlib` stdlib | Directory creation (`Path.mkdir(parents=True, exist_ok=True)`) |

---

## 9. Related Specifications

| Document | Scope |
|----------|-------|
| `main.md` | High-level product requirements and architecture |
| `spec-design-main-screen.md` | Widget bar layout, Action item category, expansion tray model |
