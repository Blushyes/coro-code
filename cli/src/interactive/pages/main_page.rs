//! Main page component for the interactive application
//!
//! This page contains the primary interface with status line and input section.

use crate::interactive::components::input_section::{InputSection, InputSectionContext};
use crate::interactive::components::status_line::{DynamicStatusLine, StatusLineContext};
use iocraft::prelude::*;

/// Properties for the main page component
#[derive(Props)]
pub struct MainPageProps {
    /// Context for the status line component
    pub status_context: StatusLineContext,
    /// Context for the input section component
    pub input_context: InputSectionContext,
}

impl Default for MainPageProps {
    fn default() -> Self {
        Self {
            status_context: StatusLineContext {
                ui_sender: tokio::sync::broadcast::channel(1).0,
                ui_anim: crate::interactive::animation::UiAnimationConfig::default(),
            },
            input_context: InputSectionContext {
                llm_config: coro_core::ResolvedLlmConfig::new(
                    coro_core::Protocol::OpenAICompat,
                    "https://api.openai.com/v1".to_string(),
                    "test-key".to_string(),
                    "gpt-4o".to_string(),
                ),
                project_path: std::path::PathBuf::from("."),
                ui_sender: tokio::sync::broadcast::channel(1).0,
                agent: std::sync::Arc::new(tokio::sync::Mutex::new(None)),
            },
        }
    }
}

/// Main page component that displays the status line and input section
#[component]
pub fn MainPage(hooks: Hooks, props: &MainPageProps) -> impl Into<AnyElement<'static>> {
    let _ = hooks; // Suppress unused variable warning for now

    element! {
        View(
            key: "main-page-container",
            flex_direction: FlexDirection::Column,
            height: 100pct,
            width: 100pct,
            padding: 1,
            position: Position::Relative,
            justify_content: JustifyContent::End, // Push content to bottom
        ) {
            // Dynamic status line (isolated component to prevent parent re-rendering)
            DynamicStatusLine(key: "dynamic-status-line", context: props.status_context.clone())

            // Fixed bottom area for input and status - this should never move
            InputSection(key: "input-section-component", context: props.input_context.clone())
        }
    }
}
