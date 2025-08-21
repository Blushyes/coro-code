//! CLI output handler implementation

use super::formatters::{DiffFormatter, ToolFormatter};
use async_trait::async_trait;
use coro_core::output::{AgentEvent, AgentOutput, MessageLevel};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

/// Tools that should not display status indicators
static SILENT_TOOLS: &[&str] = &["sequentialthinking"];

/// Check if a tool should be silent (no status display)
fn is_silent_tool(tool_name: &str) -> bool {
    SILENT_TOOLS.contains(&tool_name)
}

/// CLI output configuration
#[derive(Debug, Clone)]
pub struct CliOutputConfig {
    /// Whether to support real-time updates
    pub realtime_updates: bool,
}

impl Default for CliOutputConfig {
    fn default() -> Self {
        Self {
            realtime_updates: true,
        }
    }
}

/// CLI output handler that formats events for terminal display
pub struct CliOutputHandler {
    config: CliOutputConfig,
    tool_formatter: ToolFormatter,
    diff_formatter: DiffFormatter,
    /// Track active tool executions for real-time updates
    active_tools: Arc<Mutex<HashMap<String, coro_core::output::ToolExecutionInfo>>>,
}

impl CliOutputHandler {
    /// Create a new CLI output handler
    pub fn new(config: CliOutputConfig) -> Self {
        Self {
            config,
            tool_formatter: ToolFormatter::new(),
            diff_formatter: DiffFormatter::new(),
            active_tools: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(CliOutputConfig::default())
    }
}

#[async_trait]
impl AgentOutput for CliOutputHandler {
    async fn emit_event(
        &self,
        event: AgentEvent,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        match event {
            AgentEvent::ExecutionStarted { context } => {
                debug!("Starting task execution...");
                debug!("Original goal: {}", context.original_goal);
                debug!("Current task: {}", context.current_task);
                debug!("Project path: {}", context.project_path);

                // Don't show task execution header in normal mode
                // The task execution will be shown through tool outputs
            }

            AgentEvent::ExecutionCompleted {
                context,
                success,
                summary,
            } => {
                if success {
                    debug!("Task Completed!");
                    debug!("Summary: {}", summary);
                } else {
                    debug!("Task Failed!");
                    debug!("Error: {}", summary);
                }

                // Always show execution statistics
                debug!("Executed {} steps", context.current_step);
                debug!("Duration: {:.2}s", context.execution_time.as_secs_f64());

                // Show token usage if available
                let token_usage = &context.token_usage;
                if token_usage.total_tokens > 0 {
                    debug!(
                        "Tokens: {} input + {} output = {} total",
                        token_usage.input_tokens,
                        token_usage.output_tokens,
                        token_usage.total_tokens
                    );
                }
            }

            AgentEvent::StepStarted { step_info } => {
                debug!("Step {}: {}", step_info.step_number, step_info.task);
            }

            AgentEvent::StepCompleted { step_info: _ } => {
                // Usually handled by individual tool completions
            }

            AgentEvent::ToolExecutionStarted { tool_info } => {
                // Skip status display for silent tools
                if !is_silent_tool(&tool_info.tool_name) {
                    // Show executing status (white dot)
                    let status = self.tool_formatter.format_tool_status(&tool_info);
                    if !status.is_empty() {
                        println!("{}", status);
                    }
                }
                // Always track tools for potential updates
                let mut active_tools = self.active_tools.lock().await;
                active_tools.insert(tool_info.execution_id.clone(), tool_info);
            }

            AgentEvent::ToolExecutionUpdated { tool_info: _ } => {
                // ...
            }

            AgentEvent::ToolExecutionCompleted { tool_info } => {
                // Skip all output for silent tools - their content is handled separately
                if is_silent_tool(&tool_info.tool_name) {
                    return Ok(());
                }

                let mut active_tools = self.active_tools.lock().await;

                if active_tools.contains_key(&tool_info.execution_id) {
                    // Tool was tracked, try to update the existing line
                    use std::io::Write;
                    // Try a different approach: move up and clear
                    print!("\x1b[1A\x1b[2K\r");
                    std::io::stdout().flush().unwrap_or(());

                    active_tools.remove(&tool_info.execution_id);
                } else {
                    // Tool wasn't tracked, this shouldn't happen but handle gracefully
                    // Don't print anything to avoid duplicates
                    return Ok(());
                }

                // Always show the final status (green/red dot)
                let final_status = self.tool_formatter.format_tool_status(&tool_info);
                if !final_status.is_empty() {
                    println!("{}", final_status);
                }

                // Show result content if available
                if let Some(result_display) = self.tool_formatter.format_tool_result(&tool_info) {
                    println!("{}", result_display);
                }

                // Show diff for edit tools
                if tool_info.tool_name == "str_replace_based_edit_tool" {
                    if let Some(diff_display) = self.diff_formatter.format_edit_result(&tool_info) {
                        println!("{}", diff_display);
                    }
                }
            }

            AgentEvent::AgentThinking {
                step_number: _,
                thinking,
            } => {
                // In normal mode, show thinking in gray color without prefix
                println!("\x1b[90m{}\x1b[0m", thinking);
            }

            AgentEvent::TokenUsageUpdated { token_usage: _ } => {
                // Token updates are handled by the UI layer, CLI doesn't need to show them
                // This is mainly for interactive mode
            }

            AgentEvent::StatusUpdate {
                status: _,
                metadata: _,
            } => {
                // Status updates are handled by the UI layer, CLI doesn't need to show them
                // This is mainly for interactive mode
            }

            AgentEvent::Message {
                level,
                content,
                metadata: _,
            } => {
                match level {
                    MessageLevel::Debug => {
                        debug!("Debug: {}", content);
                    }
                    MessageLevel::Info => {
                        info!("Info: {}", content);
                    }
                    MessageLevel::Normal => {
                        // Normal text output - just print without any prefix or emoji
                        println!("{}", content);
                    }
                    MessageLevel::Warning => {
                        warn!("Warning: {}", content);
                    }
                    MessageLevel::Error => {
                        error!("Error: {}", content);
                    }
                }
            }

            AgentEvent::CompressionStarted {
                level,
                current_tokens,
                target_tokens,
                reason,
            } => {
                info!(
                    "🗜️  Starting {} compression: {} → {} tokens ({})",
                    level, current_tokens, target_tokens, reason
                );
            }

            AgentEvent::CompressionCompleted {
                summary,
                tokens_saved,
                messages_before,
                messages_after,
            } => {
                info!(
                    "✅ Compression completed: {} → {} messages, saved {} tokens",
                    messages_before, messages_after, tokens_saved
                );
                debug!("Compression summary: {}", summary);
            }

            AgentEvent::CompressionFailed {
                error,
                fallback_action,
            } => {
                warn!("⚠️  Compression failed: {}", error);
                info!("Fallback: {}", fallback_action);
            }
        }

        Ok(())
    }

    fn supports_realtime_updates(&self) -> bool {
        self.config.realtime_updates
    }

    async fn flush(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use std::io::Write;
        std::io::stdout().flush().map_err(|e| e.into())
    }
}
