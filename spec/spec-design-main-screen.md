---
title: Camera Mayham - Main Screen UI Design
version: 1.0
date_created: 2026-03-04
last_updated: 2026-03-04
tags: [design, ui, app]
---

# Introduction

This specification defines the layout, interaction model, and behavioural
requirements for the Camera Mayham main screen. The main screen is a single
application window that presents the live camera output alongside a persistent
bottom widget bar, replacing the previous dual-window arrangement of a separate
render canvas and a separate control panel.

## 1. Purpose & Scope

**Purpose:** Define the visual layout, structural regions, widget bar behaviour,
and expansion tray model for the Camera Mayham main screen.

**Scope:** Covers the main window layout, the bottom widget bar, widget item
types, the expansion tray, active/inactive states, and acceptance criteria.
Does not cover internal rendering pipeline implementation, GPU compositing
strategy, or individual filter/game implementation details — those are addressed
in dedicated specifications.

**Audience:** Engineers, designers, and AI systems implementing or extending the
Camera Mayham UI.

**Assumptions:**
- The main screen is a single OS window.
- The widget bar and camera feed area are both contained within that one window.
- No secondary OS windows are created during normal use.
- Exact visual styling (colours, typography, icon artwork) is out of scope for
  this document and governed by a future design-system specification.

---

## 2. Definitions

| Term | Definition |
|------|------------|
| **Main Screen** | The single top-level application window containing all primary UI regions |
| **Camera Area** | The rectangular region occupying the upper portion of the main screen that displays the live camera output with all active rendering passes composited on top |
| **Widget Bar** | The persistent horizontal toolbar docked to the bottom of the main screen, containing widget items; has its own opaque background |
| **Widget Item** | A discrete interactive element within the widget bar representing one activatable feature (a filter, a mini-game, or a future action) |
| **Widget Icon** | A graphical symbol rendered inside the widget item that communicates its function at a glance |
| **Widget Label** | Short text rendered below the widget icon that names the feature associated with the item |
| **Expansion Tray** | A panel that slides up from the widget bar when a widget item is expanded, revealing additional options for that item; overlays the lower portion of the Camera Area |
| **Active State** | The visual and functional state of a widget item when its associated feature is currently running or applied |
| **Inactive State** | The default visual state of a widget item when its feature is not running |
| **Single-select** | A selection mode where activating one item in a group automatically deactivates any previously active item in the same group |
| **Filter** | A GPU shader-based visual transformation applied to the camera feed |
| **Mini-game** | A lightweight interactive game rendered on top of the camera feed |

---

## 3. Requirements, Constraints & Guidelines

### Layout Requirements

- **REQ-001**: The application shall present a single OS-level window as the
  main screen; no secondary windows shall be created for normal UI operation.
- **REQ-002**: The main screen shall be divided into two vertical regions: a
  **Camera Area** occupying the top portion, and a **Widget Bar** docked to the
  bottom with a fixed height.
- **REQ-003**: The Widget Bar shall have an opaque background so that the camera
  output does not bleed through it; the Camera Area and Widget Bar are visually
  distinct regions.
- **REQ-004**: The Camera Area shall fill all available vertical space above the
  Widget Bar. The Widget Bar height is fixed and does not resize with the window.
- **REQ-005**: When an Expansion Tray is open, it shall appear above the Widget
  Bar by overlaying the lower portion of the Camera Area. Opening a tray shall
  not resize or reposition the Camera Area or the Widget Bar.

### Widget Bar Requirements

- **REQ-010**: The Widget Bar shall display a horizontal row of Widget Items.
- **REQ-011**: Each Widget Item shall display a **Widget Icon** and a
  **Widget Label** below the icon.
- **REQ-012**: Widget Items shall be grouped by category: **Filters**, **Games**,
  and **Actions** (reserved for future use). Visual or structural separators
  shall distinguish groups.
- **REQ-013**: The Widget Bar shall remain fully visible and interactive at all
  times, including when an Expansion Tray is open.
- **REQ-014**: Widget Items shall have a consistent fixed width and height within
  the bar.

### Activation & Selection Requirements

- **REQ-020**: Filters shall operate under single-select rules: activating a
  filter widget item shall deactivate and stop any previously active filter.
  Only one filter may be active at a time.
- **REQ-021**: Mini-games shall operate under single-select rules: launching a
  game widget item shall stop any previously active game. Only one game may
  be active at a time.
- **REQ-022**: An active widget item shall display a distinct active visual
  state — for example, a highlighted background, accent border, or icon tint —
  to communicate its status to the user.
- **REQ-023**: Activating an already-active widget item shall deactivate it
  (toggle-off behaviour); the feature is stopped and the item returns to its
  inactive visual state.

### Expansion Tray Requirements

- **REQ-030**: Widget items that support additional configuration shall provide
  a secondary interaction to open the Expansion Tray. Acceptable triggers
  include a dedicated affordance (e.g., a small expand chevron) or a long press
  on the item.
- **REQ-031**: The Expansion Tray shall appear above the Widget Bar, overlaying
  the lower portion of the Camera Area.
- **REQ-032**: Only one Expansion Tray may be visible at a time. Opening a tray
  for a second widget item shall close any currently open tray first.
- **REQ-033**: Clicking or tapping outside the currently open Expansion Tray
  (on the Camera Area or on a different widget item) shall dismiss the tray.
- **REQ-034**: The Expansion Tray for a **Filter** widget item shall expose at
  minimum the filter's adjustable parameters (e.g., float sliders, colour
  pickers).
- **REQ-035**: The Expansion Tray for a **Game** widget item may expose
  game-specific settings and a live status display (score, game state).
- **REQ-036**: Dismissing the Expansion Tray shall not change the active or
  inactive state of the associated widget item.

### Performance Constraints

- **CON-001**: Widget Bar rendering and event handling must not block the GPU
  render loop or introduce measurable frame time jitter (target: 60 FPS per
  `PER-001` in `spec/main.md`).
- **CON-002**: Expansion Tray appearance must complete within 200 ms on target
  hardware.
- **CON-003**: Widget state changes (activate, deactivate) must be reflected in
  application state and consequently in the next rendered camera frame with no
  additional latency beyond the normal frame pipeline.

### Guidelines

- **GUD-001**: The Widget Bar should follow the visual conventions of modern
  desktop and mobile toolbars (comparable to a bottom tab bar or application
  dock) to minimise the learning curve.
- **GUD-002**: The Camera Area must remain visible above the bar at all times;
  controls must not obscure the live video feed mid-interaction.
- **GUD-003**: The Widget Bar must not couple directly to the GPU rendering
  pipeline; it shall communicate exclusively through the shared application
  state (`AppState`).
- **GUD-004**: A diagnostics display (FPS, frame time) should be accessible
  from the UI, either as a persistent indicator or via a dedicated action widget.

---

## 4. Interfaces & Data Contracts

### 4.1 Main Screen Layout Regions

| Region | Position | Size Behaviour | Description |
|--------|----------|----------------|-------------|
| **Camera Area** | Top of window | Fills remaining height above the Widget Bar | Displays the composited GPU render output |
| **Widget Bar** | Bottom of window | Fixed height, full window width | Persistent interactive toolbar |
| **Expansion Tray** | Above Widget Bar (overlays Camera Area) | Variable height, full window width | Context-sensitive option panel; conditionally visible |

### 4.2 Widget Item Structure

Each Widget Item exposes the following logical properties:

| Property | Type | Description |
|----------|------|-------------|
| `id` | string | Unique identifier for this widget item |
| `label` | string | Short display text rendered below the icon |
| `icon` | image/asset reference | Icon rendered above the label |
| `category` | enum: `filter \| game \| action` | Governs grouping and single-select rules |
| `is_active` | bool | Whether the associated feature is currently running |
| `is_expandable` | bool | Whether this item supports an Expansion Tray |
| `expanded` | bool | Whether the Expansion Tray is currently open for this item |

### 4.3 Widget Bar → Application State Interface

The Widget Bar communicates with the rest of the application exclusively through
the shared `AppState` object:

| User Action | State Mutation |
|-------------|----------------|
| Activate a filter widget | Deactivate current filter (if any); set active filter to selected |
| Deactivate an active filter widget | Clear active filter; disable current filter |
| Launch a game widget | Stop current game (if any); launch selected game |
| Stop an active game widget | Stop current game |
| Open Expansion Tray | No state mutation; tray reads current item configuration from `AppState` |
| Adjust a parameter in Expansion Tray | Update the parameter on the relevant filter or game in `AppState` |

### 4.4 Expansion Tray Content by Widget Category

| Widget Category | Expansion Tray Contents |
|-----------------|------------------------|
| Filter | Parameter controls (float sliders, colour pickers) sourced from `filter.params` |
| Game | Live game status (score, game state) and optional game-specific settings |
| Action | To be defined in future action-specific specifications |

### 4.5 Active Filter Tracking in AppState

To enforce single-select filter semantics, `AppState` shall expose:

| Field / Method | Type | Description |
|----------------|------|-------------|
| `active_filter_name` | `Optional[str]` | Name of the currently active filter, or `None` |
| `activate_filter(name: str)` | method | Disable all filters, enable the named filter, set `active_filter_name` |
| `deactivate_filter()` | method | Disable all filters, clear `active_filter_name` |

The existing `active_game` / `launch_game()` / `stop_game()` interface on
`AppState` already provides equivalent single-select semantics for games and
requires no structural change.

---

## 5. Acceptance Criteria

- **AC-001**: Given the application starts, when the main screen loads, then a
  single window is visible containing a Camera Area and a Widget Bar; no
  secondary windows are opened.
- **AC-002**: Given the main screen is open and the camera is active, then the
  Camera Area displays the live camera feed occupying the full area above the
  Widget Bar with no visual overlap between the feed and the bar background.
- **AC-003**: Given multiple filter widget items exist, when the user activates
  Filter A while Filter B is already active, then Filter B is deactivated and
  only Filter A is active.
- **AC-004**: Given an active filter widget item, when the user clicks it a
  second time, then the filter is deactivated and the item returns to its
  inactive visual state.
- **AC-005**: Given a widget item is expandable, when the user triggers the
  expand interaction, then the Expansion Tray appears above the Widget Bar
  within 200 ms and displays appropriate controls for that item.
- **AC-006**: Given an Expansion Tray is open, when the user clicks outside it,
  then the tray is dismissed and the associated widget item's active or inactive
  state is unchanged.
- **AC-007**: Given Expansion Tray A is open, when the user expands Widget B,
  then Tray A closes and Tray B opens.
- **AC-008**: Given a parameter control in an open Expansion Tray, when the user
  adjusts it, then the change is reflected in the camera output in the next
  rendered frame.
- **AC-009**: Given continuous widget interaction, when measured, then the GPU
  render loop maintains 60 FPS with no measurable frame time degradation
  attributable to Widget Bar processing.
- **AC-010**: Given any widget item is active, then the item displays a
  visually distinct active state (e.g., highlighted background or accent border)
  distinguishing it from inactive items.

---

## 6. Test Automation Strategy

- **Test Levels**: Unit (widget state logic, `AppState` single-select mutations),
  Integration (Widget Bar ↔ `AppState` ↔ pipeline), Manual/Visual (layout,
  tray appearance, active state indicators)
- **Frameworks**: `pytest` for unit and integration tests; manual visual
  inspection for layout and interactive behaviour
- **Test Data Management**: Widget interactions simulated programmatically via
  direct `AppState` mutations to avoid dependency on live camera input
- **Coverage Requirements**: `activate_filter` / `deactivate_filter` and game
  launch/stop paths must have >80% unit test coverage
- **Performance Testing**: Frame timing benchmark under simulated widget
  interaction load to validate `CON-001`

---

## 7. Rationale & Context

The previous dual-window arrangement (a GPU render canvas plus a separate
Dear PyGui control panel) is functional but unfamiliar to users: modern
applications present a single integrated window. A bottom widget bar follows
well-established UX conventions seen in mobile OS docks, media players, and
creative tools, making the application approachable without documentation.

The Widget Bar's opaque background is intentional: a transparent or translucent
bar would become illegible against bright or high-contrast camera scenes. An
opaque bar guarantees consistent readability under all conditions.

Expansion Trays are preferred over a permanently visible parameter panel because
the primary value of the application is the camera feed itself. Controls should
appear on demand and recede when not needed, maximising the visible camera area.

Single-select semantics for both filters and games reflect the current rendering
pipeline, which processes one active filter pass and one active game pass per
frame. This constraint may be relaxed in a future revision if multi-filter
composition is introduced.

---

## 8. Dependencies & External Integrations

### Infrastructure Dependencies

| ID | Component | Requirement |
|----|-----------|-------------|
| **INF-001** | `AppState` | Widget Bar reads feature state and writes activation/parameter mutations exclusively through this shared object |
| **INF-002** | GPU Rendering Pipeline | Consumes `AppState` to produce the composited camera output displayed in the Camera Area |
| **INF-003** | UI rendering toolkit (implementation-defined) | Must support a single-window model with a docked bottom bar, icon + label items, and conditionally visible overlay panels |

---

## 9. Related Specifications

| Document | Scope |
|----------|-------|
| [spec/main.md](spec/main.md) | High-level product architecture, performance targets, and pipeline overview |
| `spec-design-filter-system.md` | Filter architecture and parameter interface |
| `spec-design-minigame-framework.md` | Mini-game lifecycle and input model |
| `spec-design-ui-architecture.md` | Widget Bar implementation architecture (to be created) |
