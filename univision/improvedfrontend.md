# Frontend Implementation Plan
## OpenCV Visual Pipeline Workbench

This document refines the original concept in `frontend_concept.md` into a practical, frontend-only build plan. It keeps the product direction intact while narrowing the initial implementation to the editor experience we can ship confidently without backend dependencies.

---

## 1. Frontend-Only Scope

The first implementation should focus on the browser application only:

- Workbench layout and navigation
- Block palette and search
- Canvas-based node editing
- Port and connection validation
- Block inspector and config editing
- Graph serialization, save, and load
- Mocked code generation panel
- Keyboard shortcuts and editor usability

The following should be deferred from the initial build:

- Real API-based AI code generation
- Running Python code from the UI
- Live execution previews
- Collaboration features
- Marketplace and community sharing
- Full custom block builder

This keeps the first version focused on proving the core editor workflow.

---

## 2. Recommended Product Changes

### 2.1 Current Block Catalogue

The following blocks are currently implemented in the `blockRegistry.ts`:

- **Input**: `Image Input`, `RTSP Stream`
- **Detection**: `YOLO Detector`
- **Preprocessing**: `Grayscale`
- **OCR**: `EasyOCR`
- **Post-Processing**: `Regex Validator`
- **Output**: `Annotator`, `Console Logger`

*Planned for future release*: `Video File`, `License Plate Detector`, `Resize`, `Threshold`, `Comment Node`, `Merge`.

This is enough to validate the UX without introducing excessive schema and state complexity.

### 2.2 Treat code generation as an interface, not a built feature

For the frontend phase, the app should not depend on a live AI provider. Instead:

- Define a `generateCode(graph)` adapter interface
- Mock the response locally
- Render loading, success, and validation error states in the code panel

This keeps the frontend integration-ready while avoiding backend coupling.

### 2.3 Use schema-driven block configuration

Avoid building one custom React component per block unless the block truly needs unique UI behavior.

Each block definition should include:

- metadata
- input and output ports
- default config
- config field schema
- validation rules

This allows the inspector UI to render dynamically and keeps the system maintainable.

### 2.4 Make the right panel tab-based for MVP

Instead of splitting the right panel vertically between inspector and code viewer, use tabs:

- `Inspector`
- `Code`

This improves readability on laptop screens and reduces layout pressure.

### 2.5 Add onboarding to reduce editor friction

The workbench needs a better first-run experience:

- starter template cards
- empty canvas hint
- quick-add modal on double-click
- short shortcut hint overlay

This will make the tool feel much more approachable.

---

## 3. Frontend Architecture

### 3.1 Core stack

- **React 18 + TypeScript**: Main application framework.
- **Vite**: Build tool and dev server.
- **React Flow**: Canvas-based node editing library.
- **Zustand**: Lightweight state management.
- **Tailwind CSS**: Utility-first styling.
- **Lucide React**: Iconography.
- **Monaco Editor**: Code viewer and editor.
- **Framer Motion**: Smooth animations.
- **Dagre**: Graph auto-layout library.
- **Vitest + React Testing Library**: Testing suite.

### 3.2 Architecture principles

- The graph JSON is the source of truth
- UI state is separate from graph state
- Block definitions live in a registry
- Validation happens before code generation
- Save/load format must remain stable from the start

---

## 4. Current UI Components & File Structure

The frontend application consists of several key UI areas organized into a cohesive workbench.

### App Shell & Layout (`src/app/`, `src/components/layout/`)
- **AppShell**: The main application container that provides context providers and the overall layout.
- **Topbar**: The top navigation bar containing the application title, project actions, and status indicators.
- **LeftSidebar**: A responsive sidebar containing the tool palette and templates.
- **RightPanel**: A tabbed panel that switches between the configurations inspector and generated code.

### Node Canvas (`src/components/canvas/`)
Powered by React Flow, this is the main workspace.
- **WorkbenchCanvas**: The core editable canvas area where users drag, drop, and connect vision nodes. It supports zoom, pan, and mini-map navigation.
- **CanvasToolbar**: A floating toolbar providing quick actions like zooming and layout arrangement.
- **WorkbenchMiniMap**: A small preview of the entire graph for quick navigation.

### Block Definitions (`src/components/blocks/`)
Nodes that represent individual pipeline operations.
- **BlockNode**: The visual representation of a vision operation (e.g., YOLO Detector, Resize). It dynamically renders based on the registry.
- **BlockPort**: Input/Output connection points enforcing data typing.
- **BlockStatusBadge**: Status indicators (e.g. error, valid) overlaid on blocks.
- **BlockCategoryBar**: Visual grouping markers for identifying block types.

### Component Palette (`src/components/palette/`)
The sidebar library of available vision concepts.
- **BlockPalette**: A searchable catalogue of available blocks that can be dragged directly onto the canvas.
- **BlockCategorySection**: Organizes the palette items conceptually (Input, Detection, Post-Processing, etc.).
- **BlockPaletteItem**: Individual draggable items in the catalogue.
- **TemplateLibrary**: Pre-configured graph templates to help users get started quickly.

### Inspector (`src/components/inspector/`)
Dynamic forms to edit node properties.
- **BlockInspector**: Renders when a node is selected, looking up the node's schema in the `blockRegistry`.
- **ConfigFieldRenderer**: Analyzes schema types and mounts the appropriate form fields.
- **fields/**: Specific input controls (TextField, NumberField, SelectField, ToggleField) tailored to configuration rules.

### Code Panel (`src/components/code/`)
The final output region.
- **CodePanel**: A Monaco-editor powered view showing the generated Python script. It dynamically reflects the graph's validated state.
- **CodeToolbar**: Tools to copy or download the generated code.
- **GenerateCodeButton**: Triggers the translation from visual graph to Python code.

### File Structure Map

```text
src/
  app/
    AppShell.tsx
  components/
    blocks/
      BlockCategoryBar.tsx
      BlockNode.tsx
      BlockPort.tsx
      BlockStatusBadge.tsx
    canvas/
      CanvasToolbar.tsx
      WorkbenchCanvas.tsx
      WorkbenchMiniMap.tsx
    code/
      CodePanel.tsx
      CodeToolbar.tsx
      GenerateCodeButton.tsx
    inspector/
      BlockInspector.tsx
      ConfigFieldRenderer.tsx
      fields/
        TextField.tsx
        NumberField.tsx
        SelectField.tsx
        ToggleField.tsx
    layout/
      LeftSidebar.tsx
      RightPanel.tsx
      Topbar.tsx
    palette/
      BlockCategorySection.tsx
      BlockPalette.tsx
      BlockPaletteItem.tsx
      TemplateLibrary.tsx
  constants/
    categories.ts
    keyboardShortcuts.ts
    portTypes.ts
    templates.ts
  lib/
    autoLayout.ts
    blockRegistry.ts
    cycleDetection.ts
    graphSerializer.ts
    graphValidator.ts
    mockCodeGenerator.ts
  store/
    codeStore.ts
    graphStore.ts
    historyStore.ts
    uiStore.ts
  types/
    block.ts
    configSchema.ts
    connection.ts
    graph.ts
    port.ts
```

---

## 5. Implementation Phases

### Phase 1: Foundation

Goal: establish the shell, types, and state model.

Tasks:

- initialize Vite React TypeScript app
- set up Tailwind, base theme tokens, and layout primitives
- define all TypeScript domain types
- create the block registry format
- build Zustand stores for graph, UI, and code state
- define sample starter templates

Deliverable:

- static shell with left panel, canvas area, and right panel tabs

### Phase 2: Canvas Editor MVP

Goal: make the workbench interactive.

Tasks:

- integrate React Flow
- support drag-drop block creation from palette
- support node move, select, multi-select, and delete
- support edge creation and removal
- add minimap, zoom, fit view, and grid background
- build reusable `BlockNode` and `BlockPort`

Deliverable:

- users can visually build a graph on the canvas

### Phase 3: Validation and Inspector

Goal: make the graph semantically correct and editable.

Tasks:

- implement same-type port validation
- enforce one connection per input
- block self-loops
- detect DAG cycles
- build schema-driven inspector form renderer
- update node status from validation state

Deliverable:

- users can configure blocks and receive real-time graph validation

### Phase 4: Save, Load, and Templates

Goal: make the editor persistent and reusable.

Tasks:

- serialize graph to JSON
- load graph from JSON
- autosave to local storage
- add import/export project actions
- add starter pipeline templates

Deliverable:

- users can save, restore, and start from templates

### Phase 5: Code Panel and UX Polish

Goal: make the frontend feel complete before backend integration.

Tasks:

- integrate Monaco Editor
- create mocked code generation adapter
- show validation errors before generation
- add loading and success states
- implement keyboard shortcuts
- add context menus and duplicate action
- add auto-layout using Dagre
- add helpful empty and onboarding states

Deliverable:

- polished frontend-only prototype ready for backend hookup

---

## 6. Suggested MVP Feature Set

The MVP should include only the following:

- desktop-first three-panel layout
- searchable block palette
- draggable block placement
- typed connections
- validation feedback
- inspector-driven config editing
- graph save/load
- templates
- mocked code generation
- undo/redo
- keyboard shortcuts

The MVP should exclude:

- execution mode
- authentication
- cloud save
- collaboration
- per-block previews
- full library coverage

---

## 7. State Model

Use separate stores for clear responsibility boundaries.

### Graph store (`graphStore.ts`)

Holds:
- `projectName`: the name of the current project
- `blocks`: list of `GraphBlock` objects on the canvas
- `connections`: list of `GraphConnection` (edge) objects
- `selectedBlockId`: ID of the currently focused block

Actions:
- `addBlock(type, position)`: instantiates a new block with defaults from the registry
- `updateBlockConfig(blockId, key, value)`: updates a specific config field and saves to disk
- `setSelectedBlockId(id)`: handles node selection state
- `setGraph(graph)`: replaces the entire graph state (used for loading/templates)
- `addConnection(connection)`: adds a new edge between ports
- `removeConnection(connectionId)`: deletes an existing edge

### UI store (`uiStore.ts`)

Holds:
- `rightPanelTab`: active right-panel tab (`inspector` | `code`)
- `paletteQuery`: current search query in the block palette

Actions:
- `setRightPanelTab`
- `setPaletteQuery`

### Code store (`codeStore.ts`)

Holds:
- `code`: generated Python code string
- `status`: generation status (`idle` | `loading` | `ready` | `error`)
- `issues`: validation issues encountered during generation

Actions:
- `setLoading`
- `setCode`
- `setIssues`

### History store (`historyStore.ts`)

Holds:
- `undoDepth`: number of available undo steps
- `redoDepth`: number of available redo steps

Actions:
- `setDepths`

---

## 8. Validation Rules

The frontend validator (`graphValidator.ts`) implements the following rules:

- **Self-loops**: A block cannot connect to itself.
- **Type Safety**: Source and target port types must match (e.g., `frame` to `frame`).
- **Fan-in Constraint**: Each input port accepts only one incoming connection.
- **Cycle Detection**: The graph must be a Directed Acyclic Graph (DAG).
- **Configuration**: Required config fields (defined in the `blockRegistry`) must be present.

The validator returns structured `ValidationIssue` objects:
- `id`: unique identifier for the issue
- `level`: severity (`error` | `warning`)
- `message`: human-readable description
- `blockId` / `portId` / `connectionId`: optional references to the offending elements

---

## 9. UX Recommendations

### Layout

- Keep the left palette collapsible
- Use tabs in the right panel instead of a stacked split
- Keep the canvas visually dominant

### Onboarding

- show an empty-state hint on first load
- offer 2 to 3 templates
- support double-click quick-add

### Performance

- keep node components lightweight
- avoid unnecessary block-specific components
- derive expensive graph computations outside render paths

### Accessibility

- keyboard access for primary actions
- visible focus states
- readable contrast in the node cards and ports

---

## 10. Testing Plan

Prioritize tests around behavior rather than visuals.

### Unit tests

- graph serialization
- graph validation
- cycle detection
- block registry config parsing
- store actions

### Integration tests

- dragging a block from palette to canvas
- connecting compatible ports
- rejecting invalid connections
- editing config in inspector updates node state
- save/load restores graph correctly

---

## 11. Final Recommendation

The best path is to build this as a frontend graph editor first, not as a full platform. The strongest first version is a polished node-based workbench with a smaller block library, schema-driven forms, stable graph JSON, and a mocked code-generation experience.

That approach will:

- reduce delivery risk
- make the UI testable early
- avoid premature backend coupling
- give a clean base for later AI and execution integration

---

## 12. Current Status & Next Steps

The frontend build baseline is **complete**. The core editor, registry, validation, and serialization layers are fully operational.

### Completed Foundations
1. **App Shell**: Responsive three-panel layout with tabbed right panel.
2. **Registry**: Schema-driven block definitions for extensible pipeline creation.
3. **Canvas**: React Flow integration with custom nodes, ports, and auto-layout.
4. **Validation**: Real-time DAG validation, type checking, and config auditing.
5. **Persistence**: Graph state is automatically serialized to `localStorage`.

### Immediate Next Steps
1. **Extend Block Library**: Implement `Video File`, `License Plate Detector`, and `Annotator` logic.
2. **AI Integration**: Replace the `mockCodeGenerator` with a real LLM-backed service.
3. **Execution Layer**: Hook up the generated code to a Python runner for live previews.
4. **Enhanced Onboarding**: Add interactive tutorials and more starter templates.
