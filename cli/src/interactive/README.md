# Interactive Mode Module

This directory contains the implementation of TRAE Agent's interactive mode, providing a real-time terminal-based user interface for agent interactions.

## Directory Structure

```
interactive/
├── README.md              # This documentation file
├── mod.rs                 # Module declarations and exports
├── app.rs                 # Main application component and entry point
├── animation.rs           # Animation system and easing functions
├── message_handler.rs     # Message processing and conversion logic
├── task_executor.rs       # Agent task execution with UI integration
├── terminal_output.rs     # Terminal output abstraction and formatting
├── text_utils.rs          # Text processing utilities (Unicode-aware)
├── state.rs              # Interactive state management
├── file_search/          # High-performance file search system
│   ├── README.md         # File search system documentation
│   ├── mod.rs            # Main search interface
│   ├── engine.rs         # Core search engine with absolute path support
│   ├── input_parser.rs   # Input parsing logic for @ syntax
│   ├── fuzzy.rs          # Intelligent fuzzy matching algorithm
│   ├── cache.rs          # High-performance file caching system
│   ├── config.rs         # Search configuration management
│   ├── git_integration.rs # Git integration and .gitignore support
│   ├── provider.rs       # Search provider abstraction
│   └── tests.rs          # Comprehensive test suite
└── components/           # UI component modules
    ├── mod.rs            # Component module declarations
    ├── logo.rs           # TRAE ASCII art logo component
    ├── status_line.rs    # Dynamic status line component
    └── input_section.rs  # Input area and status bar component
```

## Core Modules

### `app.rs` - Main Application Component

The main entry point for the interactive mode, containing the `TraeApp` component and application context.

**Key Components:**

- `TraeApp`: Main iocraft component that orchestrates the UI
- `AppContext`: Immutable application configuration and state
- `run_interactive_mode()`: Entry point function for interactive mode

**Responsibilities:**

- Manages the overall UI layout and component hierarchy using the Router system
- Handles message broadcasting between components
- Coordinates header output and message display
- Integrates with the terminal output system
- Processes file references from user input using `@` syntax

### `animation.rs` - Animation System

Provides animation utilities for smooth UI transitions and visual feedback.

**Key Types:**

- `Easing`: Enum for different easing functions (Linear, EaseOutCubic, EaseInOutCubic)
- `UiAnimationConfig`: Configuration for UI animations with environment variable support
- `TokenAnimation`: Animation state for token counting with smooth transitions
- `SpinnerAnimation`: Rotating spinner for status indication

**Key Functions:**

- `apply_easing(easing, t)`: Applies easing function to normalized time value
- `UiAnimationConfig::from_env()`: Creates config with environment variable overrides

**Environment Variables:**

- `TRAE_UI_EASING`: Animation easing type
- `TRAE_UI_FRAME_MS`: Frame interval in milliseconds
- `TRAE_UI_DURATION_MS`: Animation duration in milliseconds

### `file_search/` - High-Performance File Search System

A comprehensive file search system that enables users to quickly find and reference files using the `@` syntax.

**Key Components:**

- `FileSearchSystem`: Main search interface with caching and Git integration
- `FileSearchEngine`: Core search engine with absolute path support
- `FuzzyMatcher`: Intelligent fuzzy matching with multiple match types
- `FileCache`: High-performance caching system with smart invalidation
- `InputParser`: Handles `@` syntax parsing and query extraction

**Search Features:**

- **Intelligent Fuzzy Search**: Exact, continuous, prefix, word boundary, and fuzzy matching
- **Absolute Path Support**: Automatically converts absolute paths to relative paths for searching
- **Real-time Search**: Sub-100ms response time with live result updates
- **Git Integration**: Respects .gitignore rules and excludes build artifacts
- **Smart Caching**: 5-second cache validity with automatic invalidation

**User Interface:**

- **Trigger**: Type `@` to show all files, `@query` to search
- **Display**: Search results show relative paths (e.g., `cli/src/main.rs`) for better context
- **Smart Exclusion**: Already referenced files are automatically excluded from search results
- **Navigation**: Arrow keys or Ctrl+P/N to navigate results
- **Selection**: Tab or Enter to insert file path
- **Cancellation**: Esc to close search results

**Supported Syntax:**

```
@                           → Show all files
@main                       → Search for files containing "main"
@src/                       → Search for files in src directory
@/absolute/path/to/file     → Search using absolute path (auto-converted)
hello @file.txt             → File reference in mixed content
```

**Architecture Benefits:**

- Modular design with clear separation of concerns
- Extensible provider pattern for different search backends
- Comprehensive test coverage with unit and integration tests
- Performance optimized for large codebases

### `message_handler.rs` - Message Processing

Handles message types, conversion, and processing logic for the interactive UI.

**Key Types:**

- `AppMessage`: Enum for different message types in the interactive app
- `ContentBlock`: Enum for different content block types (UserInput, AgentText, ToolStatus, ToolResult)

**Key Functions:**

- `get_random_status_word()`: Returns a random status word for initial display
- `generate_message_id()`: Creates unique message identifiers
- `identify_content_block(content, role)`: Determines content block type
- `is_bash_output_content(content)`: Checks if content is bash output
- `app_message_to_ui_message(app_message)`: Converts AppMessage to UI message tuple

**Message Types:**

- `SystemMessage`: System notifications
- `UserMessage`: User input messages
- `InteractiveUpdate`: Real-time updates from agent execution
- `AgentTaskStarted`: Agent task initiation with operation name
- `AgentExecutionCompleted`: Task completion notification
- `TokenUpdate`: Token usage updates for animation

### `task_executor.rs` - Task Execution

Manages agent task execution with UI integration and token tracking.

**Key Components:**

- `TokenTrackingOutputHandler`: Custom output handler that forwards events and tracks tokens
- `execute_agent_task()`: Executes agent tasks asynchronously with UI updates

**Features:**

- Real-time token usage tracking and animation
- Status updates during task execution
- Integration with interactive output handler
- Custom tool registry for interactive mode tools

### `terminal_output.rs` - Terminal Output Abstraction

#### Cancellation (Interrupt)

- UI sends `AppMessage::AgentExecutionInterrupted` (e.g., on Ctrl+C)
- Task executor forwards this to core by calling the agent's global AbortController
- Core emits `AgentEvent::ExecutionInterrupted { context, reason }` and stops gracefully

Programmatic example (non-UI):

```rust
let (abort_controller, _reg) = coro_core::agent::AbortController::new();
let mut agent = coro_core::agent::AgentCore::new_with_llm_config(
    agent_config,
    llm_config,
    Box::new(coro_core::output::events::NullOutput),
    Some(abort_controller.clone()),
).await?;
// In another task/thread:
abort_controller.cancel();
```

Provides terminal output utilities and formatting functions that work with the AgentOutput system.

**Key Traits:**

- `OutputHandle`: Abstraction over different output handles (StdoutHandle, StderrHandle)

**Key Functions:**

- `output_content_block()`: Formats and outputs content blocks with appropriate spacing
- `overwrite_previous_lines()`: Overwrites previous terminal lines using ANSI escape sequences
- `update_status_line_at_position()`: Updates status line at specific terminal position
- `apply_color()` / `apply_rgb_color()`: Applies ANSI color formatting

**Features:**

- Unicode-aware text formatting
- ANSI escape sequence support for terminal manipulation
- Block-based content formatting with proper spacing
- Color support for different content types

### `text_utils.rs` - Text Processing Utilities

Provides text processing utilities for the interactive UI with Unicode support.

**Key Functions:**

- `wrap_text(text, max_width)`: Unicode-aware text wrapping with word and character breaking
- `get_terminal_width()`: Gets terminal width with fallback
- `text_width(text)`: Calculates display width considering Unicode characters
- `char_width(ch)`: Calculates display width of a single character

**Features:**

- Proper handling of CJK (Chinese, Japanese, Korean) characters
- Word-boundary aware text wrapping
- Character-level wrapping for very long words
- Terminal width detection with fallbacks

## UI Components (`components/`)

### `logo.rs` - TRAE Logo Component

Displays the TRAE ASCII art logo with gradient colors.

**Key Components:**

- `TraeLogo`: iocraft component for rendering the logo
- `TRAE_LOGO_LINES`: Static logo lines
- `LOGO_COLORS`: RGB color gradient for logo lines
- `output_logo_to_terminal()`: Function to output logo to terminal with colors

### `status_line.rs` - Dynamic Status Line

Shows real-time agent execution status, progress, and token usage.

**Key Components:**

- `DynamicStatusLine`: iocraft component for status display
- `StatusLineContext`: Context containing UI sender and animation config
- `DynamicStatusLineProps`: Component properties

**Features:**

- Real-time status updates during agent execution
- Animated token counting with configurable easing
- Spinner animation for visual feedback
- Elapsed time tracking
- Interrupt instruction display
- Status updates via `status_report` no longer reset elapsed time or token counters during a running task (only the first start initializes them).

### `router.rs` - Router Component

Provides page routing and navigation functionality using iocraft's conditional rendering.

**Key Components:**

- `Router`: iocraft component that renders different pages based on current route
- `RouterProps`: Component properties containing router state and page renderers
- `RouterBuilder`: Builder pattern for creating router configurations
- `PageRenderer`: Type alias for page render functions

**Features:**

- Conditional rendering based on route state
- Support for multiple pages with different components
- Fallback page for unknown routes
- Builder pattern for easy configuration
- Integration with core router types from `trae_agent_rs_core::interactive::router`

### `input_section.rs` - Input Section

Handles user input and displays the status bar at the bottom of the interface.

**Key Components:**

- `InputSection`: iocraft component for input handling
- `InputSectionContext`: Context containing config, project path, and UI sender
- `spawn_ui_agent_task()`: Spawns agent task execution with UI events

**Features:**

- Real-time keyboard input handling
- Input validation and processing
- Task execution triggering
- Status bar display with project information
- Placeholder text for empty input
- Robust cursor rendering with soft-wrapping and wide-character support (fixed bug where cursor disappeared or shifted on overlong lines)

## Architecture Design

### Data Flow

1. **User Input** → `InputSection` → `spawn_ui_agent_task()`
2. **Task Execution** → `TokenTrackingOutputHandler` → `AppMessage` broadcast
3. **UI Updates** → Components subscribe to `AppMessage` → Real-time display updates
4. **Output** → `terminal_output` utilities → Formatted terminal display

### Component Communication

- **Broadcast Channel**: `AppMessage` events are broadcast to all components
- **Context Passing**: Configuration and state passed through component contexts
- **Terminal Output**: Direct terminal manipulation for header and message display

### Design Principles

- **Separation of Concerns**: Each module has a single, well-defined responsibility
- **Clean Architecture**: Core business logic separated from UI presentation
- **AgentOutput Integration**: All output goes through the AgentOutput system
- **Real-time Updates**: Components react to events for smooth user experience

## Usage Guide

### Basic Usage

```rust
use crate::interactive::app::run_interactive_mode;
use trae_agent_rs_core::Config;
use std::path::PathBuf;

// Run interactive mode
run_interactive_mode(config, project_path).await?;
```

### Using Individual Components

```rust
use crate::interactive::components::logo::output_logo_to_terminal;
use crate::interactive::text_utils::wrap_text;
use crate::interactive::animation::UiAnimationConfig;

// Output logo to terminal
output_logo_to_terminal(&stdout);

// Wrap text with Unicode support
let wrapped = wrap_text("Long text here", 80);

// Create animation config
let anim_config = UiAnimationConfig::from_env();
```

## Router (`router/`)

### Router System for Page Navigation

A comprehensive routing system designed specifically for iocraft-based applications.

**Key Components:**

- `Route`: Route definition with metadata and configuration
- `RouteId`: Unique identifier for routes
- `Router`: Core router with state management and navigation
- `RouterState`: Current navigation state and history
- `RouterConfig`: Router configuration and route registry
- `IocraftRouter`: iocraft component for rendering pages based on routes
- `IocraftRouterBuilder`: Builder pattern for creating router configurations

**Features:**

- Type-safe route definitions and navigation
- Navigation history with configurable limits
- Default route support
- Conditional rendering based on current route
- Builder pattern for easy configuration
- Fallback page support for unknown routes

## Pages (`pages/`)

### `main_page.rs` - Main Page Component

The primary page component that contains the status line and input section.

**Key Components:**

- `MainPage`: iocraft component that renders the main interface
- `MainPageProps`: Component properties containing status and input contexts

**Features:**

- Displays `DynamicStatusLine` for real-time status updates
- Displays `InputSection` for user input handling
- Maintains the same layout as the original TraeApp
- Proper context passing to child components

### Extending the System

To add new UI components:

1. Create component file in `components/`
2. Define component props and context structures
3. Implement iocraft component with proper event handling
4. Add component to `components/mod.rs`
5. Integrate with main app in `app.rs` or add as a new page

To add new pages:

1. Create page component file in `pages/`
2. Define page props and implement iocraft component
3. Add page to `pages/mod.rs`
4. Register page with router in `app.rs` using `RouterBuilder`

### Environment Configuration

Configure animations and behavior through environment variables:

```bash
export TRAE_UI_EASING=ease_out_cubic
export TRAE_UI_FRAME_MS=10
export TRAE_UI_DURATION_MS=3000
```

## Testing

Each module includes comprehensive unit tests. Run tests with:

```bash
cargo test interactive
```

## Dependencies

- `iocraft`: Terminal UI framework
- `tokio`: Async runtime and synchronization
- `unicode-width`: Unicode-aware text width calculation
- `crossterm`: Terminal manipulation
- `rand`: Random status word selection
- `anyhow`: Error handling
