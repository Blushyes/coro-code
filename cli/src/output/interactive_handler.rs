//! Interactive output handler implementation for UI integration
//! Delegates all output behavior to CliOutputHandler while maintaining UI integration

use super::cli_handler::{CliOutputConfig, CliOutputHandler};
use super::formatters::{DiffFormatter, ToolFormatter};
use async_trait::async_trait;
use coro_core::output::{AgentEvent, AgentOutput, MessageLevel};
use coro_core::tools::output_formatter::{GRAY, RESET};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::Mutex;

/// Tools that should not display status indicators
static SILENT_TOOLS: &[&str] = &["sequentialthinking", "status_report"];

/// Check if a tool should be silent (no status display)
fn is_silent_tool(tool_name: &str) -> bool {
    SILENT_TOOLS.contains(&tool_name)
}

/// Message types for interactive UI updates
#[derive(Debug, Clone)]
pub enum InteractiveMessage {
    /// Agent thinking/reasoning output
    AgentThinking(String),
    /// Tool execution status update with execution ID for replacement
    ToolStatus {
        execution_id: String,
        status: String,
    },
    /// Tool execution result
    ToolResult(String),
    /// System message
    SystemMessage(String),
    /// Task completion
    TaskCompleted { success: bool, summary: String },
    /// Execution statistics
    ExecutionStats {
        steps: usize,
        duration: f64,
        tokens: Option<String>,
    },
    /// Debugging information
    Debug(String),
}

/// Interactive output configuration
#[derive(Debug, Clone)]
pub struct InteractiveOutputConfig {
    /// Whether to support real-time updates
    pub realtime_updates: bool,
    /// Whether to show detailed tool output
    pub show_tool_details: bool,
}

impl Default for InteractiveOutputConfig {
    fn default() -> Self {
        Self {
            realtime_updates: true,
            show_tool_details: true,
        }
    }
}

/// Interactive output handler that delegates all behavior to CliOutputHandler
/// while maintaining UI integration capabilities
pub struct InteractiveOutputHandler {
    /// The underlying CLI output handler that does the actual work
    cli_handler: CliOutputHandler,
    /// Channel sender for UI updates (optional for future use)
    _ui_sender: Option<mpsc::UnboundedSender<InteractiveMessage>>,
    /// Tool formatter for consistent formatting
    tool_formatter: ToolFormatter,
    /// Diff formatter for edit results
    diff_formatter: DiffFormatter,
    /// Track active tool executions
    active_tools: Arc<Mutex<HashMap<String, coro_core::output::ToolExecutionInfo>>>,
}

impl InteractiveOutputHandler {
    /// Create a new interactive output handler
    pub fn new(
        config: InteractiveOutputConfig,
        ui_sender: mpsc::UnboundedSender<InteractiveMessage>,
    ) -> Self {
        // Create CLI output handler with the same realtime_updates setting
        let cli_config = CliOutputConfig {
            realtime_updates: config.realtime_updates,
        };
        let cli_handler = CliOutputHandler::new(cli_config);

        Self {
            cli_handler,
            _ui_sender: Some(ui_sender),
            tool_formatter: ToolFormatter::new(),
            diff_formatter: DiffFormatter::new(),
            active_tools: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create with default configuration
    pub fn with_sender(ui_sender: mpsc::UnboundedSender<InteractiveMessage>) -> Self {
        Self::new(InteractiveOutputConfig::default(), ui_sender)
    }
}

#[async_trait]
impl AgentOutput for InteractiveOutputHandler {
    async fn emit_event(
        &self,
        event: AgentEvent,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // In interactive mode, only use UI messages to avoid duplicate output
        // Only delegate to CLI handler if we don't have a UI sender
        if let Some(ui_sender) = &self._ui_sender {
            match event {
                AgentEvent::ExecutionStarted { context: _ } => {
                    // Don't show task execution header in interactive mode
                    // The task execution will be shown through tool outputs
                }

                AgentEvent::ExecutionCompleted {
                    context: _,
                    success: _,
                    summary: _,
                } => {
                    // ...
                }

                AgentEvent::ToolExecutionStarted { tool_info } => {
                    // Skip status display for silent tools like sequentialthinking
                    if !is_silent_tool(&tool_info.tool_name) {
                        // Use same format as CLI mode
                        let status_msg = self.tool_formatter.format_tool_status(&tool_info);
                        if !status_msg.is_empty() {
                            let _ = ui_sender.send(InteractiveMessage::ToolStatus {
                                execution_id: tool_info.execution_id.clone(),
                                status: status_msg,
                            });
                        }
                    }
                    // Track tool for potential updates
                    let mut active_tools = self.active_tools.lock().await;
                    active_tools.insert(tool_info.execution_id.clone(), tool_info);
                }

                AgentEvent::ToolExecutionCompleted { tool_info } => {
                    // !!! IMPORTANT
                    // These tools, such as “thinking,” are already handled in `Agent::Thinking`.
                    // Here, it’s actually a status display that starts with a dot.

                    // Skip all output for silent tools
                    if is_silent_tool(&tool_info.tool_name) {
                        return Ok(());
                    }

                    // Remove from active tools tracking
                    let mut active_tools = self.active_tools.lock().await;
                    active_tools.remove(&tool_info.execution_id);

                    // Use same format as CLI mode
                    let status_msg = self.tool_formatter.format_tool_status(&tool_info);
                    if !status_msg.is_empty() {
                        let _ = ui_sender.send(InteractiveMessage::ToolStatus {
                            execution_id: tool_info.execution_id.clone(),
                            status: status_msg,
                        });
                    }

                    // Show result content if available
                    if let Some(result_display) = self.tool_formatter.format_tool_result(&tool_info)
                    {
                        let _ = ui_sender.send(InteractiveMessage::ToolResult(result_display));
                    }

                    // Show diff for edit tools
                    if tool_info.tool_name == "str_replace_based_edit_tool" {
                        if let Some(diff_display) =
                            self.diff_formatter.format_edit_result(&tool_info)
                        {
                            let _ = ui_sender.send(InteractiveMessage::ToolResult(diff_display));
                        }
                    }
                }

                AgentEvent::AgentThinking { thinking, .. } => {
                    // Use same format as CLI mode - gray color without prefix
                    let _ = ui_sender.send(InteractiveMessage::AgentThinking(format!(
                        "{}{}{}",
                        GRAY, thinking, RESET
                    )));
                }

                AgentEvent::Message { level, content, .. } => {
                    match level {
                        MessageLevel::Debug => {
                            // Debug messages are usually not shown in interactive mode
                        }
                        MessageLevel::Info => {
                            let msg = format!("Info  {}", content);
                            let _ = ui_sender.send(InteractiveMessage::SystemMessage(msg));
                        }
                        MessageLevel::Normal => {
                            // Normal text output - just the content without prefix
                            let _ = ui_sender.send(InteractiveMessage::SystemMessage(content));
                        }
                        MessageLevel::Warning => {
                            let msg = format!("Warning: {}", content);
                            let _ = ui_sender.send(InteractiveMessage::SystemMessage(msg));
                        }
                        MessageLevel::Error => {
                            let msg = format!("Error: {}", content);
                            let _ = ui_sender.send(InteractiveMessage::SystemMessage(msg));
                        }
                    }
                }

                AgentEvent::CompressionStarted {
                    level: _level,
                    current_tokens: _current_tokens,
                    target_tokens: _target_tokens,
                    reason: _reason,
                } => {
                    // let msg = format!("Starting {} compression: {} → {} tokens", level, current_tokens, target_tokens);
                    // let _ = ui_sender.send(InteractiveMessage::SystemMessage(msg));
                    // TODO
                }

                AgentEvent::CompressionCompleted {
                    tokens_saved,
                    messages_before,
                    messages_after,
                    summary: _,
                } => {
                    let msg = format!(
                        "Compression completed: {} → {} messages, saved {} tokens",
                        messages_before, messages_after, tokens_saved
                    );
                    let _ = ui_sender.send(InteractiveMessage::SystemMessage(msg));
                }

                AgentEvent::CompressionFailed {
                    error,
                    fallback_action,
                } => {
                    let msg = format!(
                        "Compression failed: {}. Fallback: {}",
                        error, fallback_action
                    );
                    let _ = ui_sender.send(InteractiveMessage::SystemMessage(msg));
                }

                _ => {
                    // Other events are handled by CLI output
                }
            }
        } else {
            // Fallback to CLI handler if no UI sender is available
            self.cli_handler.emit_event(event).await?;
        }

        Ok(())
    }

    fn supports_realtime_updates(&self) -> bool {
        // Delegate to CLI output handler
        self.cli_handler.supports_realtime_updates()
    }

    async fn flush(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Delegate to CLI output handler
        self.cli_handler.flush().await
    }
}
