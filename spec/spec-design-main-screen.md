---
title: Camera Mayham — Main Screen Design
version: 2.0
date_created: 2026-03-04
last_updated: 2026-03-06
tags: [design, ui, ux, app]
---

# Introduction

Camera Mayham is a creative playground that turns a live webcam feed into an
interactive visual experience. The main screen is the product — it should feel
alive, immediate, and fun the moment it opens. This specification defines the
layout, visual character, interaction model, and behavioural requirements for
that screen.

The central design principle is **camera-first**: the live feed fills the window
and is the hero. All controls are secondary to it — they frame the experience
without competing with it. When you're not interacting, the UI steps back and
your face is all you see. When you do reach for a control, it responds with
warmth and clarity.

---

## 1. Purpose & Scope

**Purpose:** Define the visual layout, UX character, structural regions, widget
bar behaviour, and expansion tray model for the Camera Mayham main screen.

**Scope:** Covers the main window layout, visual language, the Widget Bar, widget
item types and states, the Expansion Tray, status overlays, and acceptance
criteria. Does not cover the GPU rendering pipeline, individual filter or
mini-game internals, or icon artwork — those are addressed in dedicated
specifications.

**Audience:** Engineers, designers, and AI systems implementing or extending the
Camera Mayham UI.

**Assumptions:**

- The main screen is a single OS window; no secondary windows are created during
  normal use.
- The Widget Bar and Camera Area are both contained within that one window.
- Exact icon artwork and font choices are governed by a future design-system
  specification; this document specifies intent and principle, not pixel values.

---

## 2. Definitions

| Term | Definition |
|------|------------|
| **Main Screen** | The single top-level application window containing all UI regions |
| **Camera Area** | The primary region — fills the window above the Widget Bar — showing the live GPU-composited camera output |
| **Widget Bar** | The persistent toolbar docked to the bottom of the window; contains Widget Items and has its own dark background |
| **Widget Item** | One interactive element in the Widget Bar representing a single activatable feature (a filter, mini-game, or reserved action) |
| **Widget Icon** | The graphical symbol inside a Widget Item that communicates the feature's identity |
| **Widget Label** | The short text beneath the icon naming the feature |
| **Expansion Tray** | A panel that slides up from the Widget Bar when a Widget Item is expanded, revealing parameter controls; overlays the lower part of the Camera Area |
| **Active State** | The visual and functional state of a Widget Item whose associated feature is currently running |
| **Inactive State** | The resting visual state of a Widget Item whose feature is off |
| **Single-select** | A selection mode where activating one item in a group automatically deactivates the previously active item in that group |
| **Status HUD** | A subtle overlay positioned in the corner of the Camera Area showing live diagnostics (FPS, face tracking state) |
| **Filter** | A GPU shader-based visual transformation applied to the camera feed |
| **Mini-game** | A lightweight interactive game rendered over the camera feed and controlled by the user's face |

---

## 3. Requirements, Constraints & Guidelines

### 3.1 UX Vision

Camera Mayham should feel like a polished creative tool — part magic mirror, part
arcade cabinet. These UX principles are non-negotiable and take precedence over
convenience shortcuts in implementation:

- **Camera is the hero.** The live feed occupies all available space. The Widget
  Bar is a purposeful frame, not a panel bolted onto a canvas.
- **Engagement through feedback.** Every interaction should produce an immediate,
  satisfying response — visual or otherwise. Dead clicks feel broken; the UI
  must always acknowledge the user.
- **Calm when idle, alive when active.** In the resting state the application is
  quiet; active features should draw the eye and signal that something is
  happening.
- **Delight lives in the details.** Smooth transitions, glowing active states,
  responsive hover effects, and animated tray entry all contribute to the
  feeling of a quality product. Polish is not optional.

### 3.2 Layout Requirements

- **REQ-001**: The application shall present a single OS-level window; no
  secondary windows shall be created for normal operation.
- **REQ-002**: The window is divided into two vertical regions: the **Camera
  Area** filling the top, and the **Widget Bar** docked to the bottom at a fixed
  height.
- **REQ-003**: The Widget Bar shall have an opaque, distinctly styled background
  — dark, rich, and clearly separated from the camera feed above it. The area
  must remain legible regardless of what the camera is showing.
- **REQ-004**: The Camera Area fills all remaining vertical space above the Widget
  Bar and resizes with the window. The Widget Bar height is fixed.
- **REQ-005**: When an Expansion Tray is open it overlays the lower portion of
  the Camera Area, sitting above the Widget Bar. Opening a tray does not
  reposition or resize the Camera Area or the Widget Bar.

### 3.3 Widget Bar Requirements

- **REQ-010**: The Widget Bar shall display a horizontal row of Widget Items.
- **REQ-011**: Each Widget Item shall display a **Widget Icon** and a short
  **Widget Label** below it.
- **REQ-012**: Widget Items shall be grouped by category — **Filters**, **Games**,
  and **Actions** (reserved). A visual separator shall mark the boundary between
  groups.
- **REQ-013**: The Widget Bar shall remain fully visible and interactive at all
  times, including while an Expansion Tray is open.
- **REQ-014**: Widget Items shall share a consistent fixed size within the bar.

### 3.4 Visual & Animation Requirements

These requirements define the *feel* of the interface:

- **REQ-015**: Active Widget Items shall be visually arresting. They must stand
  out from inactive items through a combination of accent colour, a glowing or
  illuminated treatment, and a clearly styled label. The user should be able to
  tell at a glance which features are on.
- **REQ-016**: Inactive Widget Items shall have a subtle, recessed appearance —
  present but not competing for attention. Hover states shall provide visible
  but understated feedback.
- **REQ-017**: Activating or deactivating a Widget Item shall produce a brief
  visual transition (e.g., a colour flash, icon scale pulse, or glow fade) that
  confirms the state change. This transition must complete within 150 ms.
- **REQ-018**: The Expansion Tray shall enter and exit with a smooth slide
  animation. Tray entry must complete within 200 ms; tray exit must complete
  within 150 ms.
- **REQ-019**: The Widget Bar background shall use depth cues (e.g., a subtle
  top-edge highlight, inner shadow, or elevated look) to visually separate it
  from the camera feed and reinforce that it is a distinct interactive surface.

### 3.5 Status HUD Requirements

- **REQ-025**: A **Status HUD** shall be displayed as a lightweight overlay in a
  corner of the Camera Area (default: top-right). It shall show, at minimum:
  - Current rendering frame rate (FPS)
  - Current frame time in milliseconds
  - Face tracking state (detected / not detected), conveyed as an icon or
    colour-coded indicator rather than raw text
- **REQ-026**: The Status HUD shall be visually unobtrusive — small, low-opacity
  text or icon elements that do not distract from the camera feed. It must not
  use a solid opaque background panel.
- **REQ-027**: Face detection state shall be communicated clearly and at a
  glance: a distinct visual treatment for "face detected" vs "no face in
  frame" (e.g., a green dot vs a grey dot, a tracking icon that's lit vs dim).

### 3.6 Activation & Selection Requirements

- **REQ-030**: Filters use single-select: activating a filter widget deactivates
  and stops any previously active filter. Only one filter may be active at a
  time.
- **REQ-031**: Mini-games use single-select: launching a game widget stops any
  previously active game. Only one game may be active at a time.
- **REQ-032**: Activating an already-active Widget Item shall deactivate it
  (toggle-off); the feature stops and the item returns to its inactive state.

### 3.7 Expansion Tray Requirements

- **REQ-040**: Widget Items that support additional configuration shall expose a
  secondary interaction to open the Expansion Tray. Acceptable triggers: a
  dedicated expand affordance on the item (e.g., a small chevron), or a
  secondary click gesture.
- **REQ-041**: The Expansion Tray shall appear above the Widget Bar, overlaying
  the lower Camera Area.
- **REQ-042**: Only one Expansion Tray may be open at a time. Opening a second
  tray shall close the first.
- **REQ-043**: Clicking the Camera Area or a different Widget Item while a tray
  is open shall dismiss the tray.
- **REQ-044**: The Expansion Tray for a **Filter** shall expose the filter's
  adjustable parameters (sliders, colour pickers). Controls shall be clearly
  styled and easy to use at a glance.
- **REQ-045**: The Expansion Tray for a **Game** shall expose game-specific
  settings and a live status display (score, game state).
- **REQ-046**: Dismissing the Expansion Tray shall not alter the active or
  inactive state of the associated Widget Item.
- **REQ-047**: The Expansion Tray background shall be dark and semi-opaque,
  allowing the camera feed to show through subtly while keeping controls legible.
  It must feel like a floating layer, not a hard-edged panel insertion.

### 3.8 Performance Constraints

- **CON-001**: Widget Bar rendering and event handling must not block the GPU
  render loop or introduce measurable frame time jitter (60 FPS target per
  `spec/main.md PER-001`).
- **CON-002**: Widget state changes must be reflected in the next rendered camera
  frame with no latency beyond the normal frame pipeline.
- **CON-003**: Expansion Tray animation must complete within 200 ms on target
  hardware. The slide animation must be smooth with no visible judder.

### 3.9 Decoupling Guideline

- **GUD-001**: The Widget Bar must communicate with the rest of the application
  exclusively through the shared `AppState` object. It must not reference or
  mutate GPU pipeline internals directly.

---

## 4. Interfaces & Data Contracts

### 4.1 Main Screen Layout Regions

| Region | Position | Size Behaviour | Description |
|--------|----------|----------------|-------------|
| **Camera Area** | Top of window | Fills all height above the Widget Bar; resizes with window | Primary display: composited GPU render output |
| **Widget Bar** | Bottom of window | Fixed height, full window width | Persistent interactive toolbar; always visible |
| **Expansion Tray** | Above Widget Bar (overlays Camera Area) | Variable height, full window width | Context-sensitive parameter panel; conditionally visible, animated |
| **Status HUD** | Corner of Camera Area (top-right default) | Small, fixed size | Live diagnostics overlay; always visible, minimally intrusive |

### 4.2 Widget Item Structure

| Property | Type | Description |
|----------|------|-------------|
| `id` | `string` | Unique stable identifier |
| `label` | `string` | Short display text shown below the icon |
| `icon` | image / asset reference | Symbol rendered above the label |
| `category` | `enum: filter \| game \| action` | Governs grouping and single-select rules |
| `is_active` | `bool` | Whether the associated feature is currently running |
| `is_expandable` | `bool` | Whether this item supports an Expansion Tray |
| `expanded` | `bool` | Whether the Expansion Tray is currently open for this item |

### 4.3 Widget Bar → Application State Interface

The Widget Bar communicates with the rest of the application exclusively through
`AppState`:

| User Action | State Mutation |
|-------------|----------------|
| Activate a filter widget | Deactivate current filter (if any); set active filter to selected |
| Deactivate an active filter widget | Clear active filter |
| Launch a game widget | Stop current game (if any); launch selected game |
| Stop an active game widget | Stop current game |
| Open Expansion Tray | No state mutation; tray reads current configuration from `AppState` |
| Adjust a parameter in Expansion Tray | Update the parameter on the relevant filter or game in `AppState` |

### 4.4 Expansion Tray Content by Widget Category

| Widget Category | Expansion Tray Contents |
|-----------------|------------------------|
| Filter | Parameter controls (sliders, colour pickers) sourced from `filter.params` |
| Game | Live game status (score, state) and optional game-specific settings |
| Action | To be defined in future action-specific specifications |

### 4.5 Active Filter Tracking in AppState

To enforce single-select filter semantics, `AppState` exposes:

| Field / Method | Type | Description |
|----------------|------|-------------|
| `active_filter_name` | `Optional[str]` | Name of the currently active filter, or `None` |
| `activate_filter(name: str)` | method | Disable all filters, enable the named one, update `active_filter_name` |
| `deactivate_filter()` | method | Disable all filters, clear `active_filter_name` |

The existing `active_game` / `launch_game()` / `stop_game()` interface on
`AppState` already provides equivalent single-select semantics for games and
requires no structural change.

---

## 5. Acceptance Criteria

### Layout & Structure

- **AC-001** — *Given* the application starts, *when* the main screen loads,
  *then* a single window is visible containing a Camera Area and a Widget Bar;
  no secondary windows are opened.
- **AC-002** — *Given* the main screen is open and the camera is active, *then*
  the Camera Area displays the live camera feed filling all available space above
  the Widget Bar with no visual bleed between feed and bar.

### Feature Activation

- **AC-003** — *Given* multiple filter widget items exist, *when* the user
  activates Filter A while Filter B is active, *then* Filter B is deactivated
  and only Filter A is active.
- **AC-004** — *Given* an active filter widget item, *when* the user clicks it a
  second time, *then* the filter is deactivated and the item returns to its
  inactive visual state.
- **AC-005** — *Given* any active Widget Item, *then* that item displays a
  visually distinct active state that is clearly distinguishable from inactive
  items at a glance.

### Expansion Tray

- **AC-006** — *Given* a Widget Item is expandable, *when* the expand interaction
  is triggered, *then* the Expansion Tray appears above the Widget Bar within
  200 ms and displays appropriate controls for that item.
- **AC-007** — *Given* an Expansion Tray is open, *when* the user clicks the
  Camera Area or a different Widget Item, *then* the tray is dismissed and the
  associated item's active or inactive state is unchanged.
- **AC-008** — *Given* Tray A is open, *when* the user expands Widget B, *then*
  Tray A closes and Tray B opens.
- **AC-009** — *Given* a parameter control in an open Expansion Tray, *when* the
  user adjusts it, *then* the change is reflected in the camera output within
  the next rendered frame.

### Visual & Animation

- **AC-010** — *Given* a Widget Item is activated or deactivated, *then* a brief
  visual transition is visible on the item confirming the state change; the
  transition completes within 150 ms.
- **AC-011** — *Given* the Expansion Tray opens or closes, *then* the animation
  is visibly smooth with no judder; entry completes within 200 ms, exit within
  150 ms.
- **AC-012** — *Given* the Status HUD is visible, *then* FPS, frame time, and
  face tracking state are all readable without a solid background panel, and the
  HUD does not distract from the camera feed during normal use.
- **AC-013** — *Given* a face is actively tracked, *then* the Status HUD face
  indicator shows a distinct "detected" state; when no face is visible, it shows
  a distinct "not detected" state.

### Performance

- **AC-014** — *Given* continuous widget interaction, *when* frame timing is
  measured, *then* the GPU render loop maintains 60 FPS with no measurable
  degradation attributable to Widget Bar processing.

---

## 6. Validation Criteria

- Widget Bar rendering is confirmed non-blocking by frame-timing benchmarks run
  with simulated widget interaction load (validates `CON-001`).
- `activate_filter` / `deactivate_filter` and game launch/stop paths have >80%
  unit test coverage.
- Visual acceptance criteria (AC-005, AC-010 through AC-013) are validated by
  manual screenshot review during development.
- Expansion Tray timing (AC-006, AC-011) is confirmed by recorded interaction
  walkthroughs or programmatic timer measurements.

---

## 7. Related Specifications

| Document | Scope |
|----------|-------|
| [spec/main.md](spec/main.md) | High-level product architecture, performance targets, and rendering pipeline overview |
| [spec/filters/spec-design-filter-face-landmarks.md](spec/filters/spec-design-filter-face-landmarks.md) | Face landmark filter design |
| `spec-design-filter-system.md` | Full filter architecture and parameter interface (to be created) |
| `spec-design-minigame-framework.md` | Mini-game lifecycle and facial input model (to be created) |
