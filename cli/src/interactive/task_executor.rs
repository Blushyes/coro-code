//! Task execution module for interactive mode
//!
//! This module handles agent task execution with UI integration,
//! including token tracking and status updates.

use crate::interactive::message_handler::AppMessage;
use crate::output::interactive_handler::{InteractiveMessage, InteractiveOutputConfig};
use anyhow::Result;
use coro_core::ResolvedLlmConfig;
use std::path::PathBuf;
use tokio::sync::{broadcast, mpsc};

/// Custom output handler that forwards events and tracks tokens
pub struct TokenTrackingOutputHandler {
    interactive_handler: crate::output::interactive_handler::InteractiveOutputHandler,
    ui_sender: broadcast::Sender<AppMessage>,
}

impl TokenTrackingOutputHandler {
    pub fn new(
        interactive_config: InteractiveOutputConfig,
        interactive_sender: mpsc::UnboundedSender<InteractiveMessage>,
        ui_sender: broadcast::Sender<AppMessage>,
    ) -> Self {
        Self {
            interactive_handler: crate::output::interactive_handler::InteractiveOutputHandler::new(
                interactive_config,
                interactive_sender,
            ),
            ui_sender,
        }
    }
}

#[async_trait::async_trait]
impl coro_core::output::AgentOutput for TokenTrackingOutputHandler {
    async fn emit_event(
        &self,
        event: coro_core::output::AgentEvent,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Check for token updates and status updates in various events
        match &event {
            coro_core::output::AgentEvent::ExecutionCompleted { context, .. } => {
                if context.token_usage.total_tokens > 0 {
                    let _ = self.ui_sender.send(AppMessage::TokenUpdate {
                        tokens: context.token_usage.total_tokens,
                    });
                }
            }
            coro_core::output::AgentEvent::TokenUsageUpdated { token_usage } => {
                // Send immediate token update for smooth animation
                let _ = self.ui_sender.send(AppMessage::TokenUpdate {
                    tokens: token_usage.total_tokens,
                });
            }
            coro_core::output::AgentEvent::StatusUpdate { status, .. } => {
                // Send status update to UI
                let _ = self.ui_sender.send(AppMessage::AgentTaskStarted {
                    operation: status.clone(),
                });
            }
            _ => {}
        }

        // Forward to the interactive handler
        self.interactive_handler.emit_event(event).await
    }

    fn supports_realtime_updates(&self) -> bool {
        self.interactive_handler.supports_realtime_updates()
    }

    async fn flush(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.interactive_handler.flush().await
    }
}

/// Execute agent task with persistent agent to maintain conversation context
pub async fn execute_agent_task_with_context(
    task: String,
    llm_config: ResolvedLlmConfig,
    project_path: PathBuf,
    ui_sender: broadcast::Sender<AppMessage>,
    agent: std::sync::Arc<tokio::sync::Mutex<Option<coro_core::agent::AgentCore>>>,
) -> Result<()> {
    // Create a receiver to listen for interruption signals
    let mut interrupt_receiver = ui_sender.subscribe();
    use crate::tools::StatusReportToolFactory;

    // Create channel for InteractiveMessage and forward to AppMessage
    let (interactive_sender, mut interactive_receiver) = mpsc::unbounded_channel();
    let ui_sender_clone = ui_sender.clone();

    // Forward InteractiveMessage to AppMessage
    tokio::spawn(async move {
        while let Some(interactive_msg) = interactive_receiver.recv().await {
            let _ = ui_sender_clone.send(AppMessage::InteractiveUpdate(interactive_msg));
        }
    });

    // Lock the agent for the duration of this task
    let mut agent_guard = agent.lock().await;

    // If no agent exists, create one
    if agent_guard.is_none() {
        // Create agent configuration with CLI tools and status_report tool for interactive mode
        let mut agent_config = coro_core::AgentConfig {
            tools: crate::tools::get_default_cli_tools(),
            ..Default::default()
        };
        if !agent_config.tools.contains(&"status_report".to_string()) {
            agent_config.tools.push("status_report".to_string());
        }

        // Create TokenTrackingOutputHandler with UI integration
        let interactive_config = InteractiveOutputConfig {
            realtime_updates: true,
            show_tool_details: true,
        };
        let token_tracking_output = Box::new(TokenTrackingOutputHandler::new(
            interactive_config,
            interactive_sender,
            ui_sender.clone(),
        ));

        // Create CLI tool registry with status_report tool for interactive mode
        let mut tool_registry = crate::tools::create_cli_tool_registry();
        tool_registry.register_factory(Box::new(StatusReportToolFactory::with_ui_sender(
            ui_sender.clone(),
        )));

        // Create new agent
        let new_agent = coro_core::agent::AgentCore::new_with_output_and_registry(
            agent_config,
            llm_config,
            token_tracking_output,
            tool_registry,
        )
        .await?;

        *agent_guard = Some(new_agent);
    }

    // Get mutable reference to the agent
    let agent_ref = agent_guard.as_mut().unwrap();

    // Execute task with conversation continuation
    let task_future = agent_ref.execute_task_with_context(&task, &project_path);

    // Listen for interruption signals
    let interrupt_future = async {
        loop {
            match interrupt_receiver.recv().await {
                Ok(AppMessage::AgentExecutionInterrupted { .. }) => {
                    tracing::warn!("Task interrupted by user");
                    return Err(anyhow::anyhow!("Task interrupted by user"));
                }
                Ok(_) => continue, // Ignore other messages
                Err(_) => break,   // Channel closed
            }
        }
        Ok(())
    };

    // Add timeout to prevent hanging
    let timeout_future = tokio::time::sleep(tokio::time::Duration::from_secs(300)); // 5 minutes timeout

    // Race between task execution, interruption, and timeout
    tokio::select! {
        result = task_future => {
            result?;
        }
        interrupt_result = interrupt_future => {
            interrupt_result?;
        }
        _ = timeout_future => {
            tracing::error!("Task execution timed out after 5 minutes");
            return Err(anyhow::anyhow!("Task execution timed out"));
        }
    }

    Ok(())
}

/// Execute agent task asynchronously and send updates to UI
pub async fn execute_agent_task(
    task: String,
    llm_config: ResolvedLlmConfig,
    project_path: PathBuf,
    ui_sender: broadcast::Sender<AppMessage>,
) -> Result<()> {
    // Create a receiver to listen for interruption signals
    let mut interrupt_receiver = ui_sender.subscribe();
    use crate::tools::StatusReportToolFactory;

    // Create agent configuration with CLI tools and status_report tool for interactive mode
    let mut agent_config = coro_core::AgentConfig {
        tools: crate::tools::get_default_cli_tools(),
        ..Default::default()
    };
    if !agent_config.tools.contains(&"status_report".to_string()) {
        agent_config.tools.push("status_report".to_string());
    }

    // Create channel for InteractiveMessage and forward to AppMessage
    let (interactive_sender, mut interactive_receiver) = mpsc::unbounded_channel();
    let ui_sender_clone = ui_sender.clone();

    // Forward InteractiveMessage to AppMessage
    tokio::spawn(async move {
        while let Some(interactive_msg) = interactive_receiver.recv().await {
            let _ = ui_sender_clone.send(AppMessage::InteractiveUpdate(interactive_msg));
        }
    });

    // Create TokenTrackingOutputHandler with UI integration
    let interactive_config = InteractiveOutputConfig {
        realtime_updates: true,
        show_tool_details: true,
    };
    let token_tracking_output = Box::new(TokenTrackingOutputHandler::new(
        interactive_config,
        interactive_sender,
        ui_sender.clone(),
    ));

    // Create CLI tool registry with status_report tool for interactive mode
    let mut tool_registry = crate::tools::create_cli_tool_registry();
    tool_registry.register_factory(Box::new(StatusReportToolFactory::with_ui_sender(
        ui_sender.clone(),
    )));

    // Create and execute agent task
    let mut agent = coro_core::agent::AgentCore::new_with_output_and_registry(
        agent_config,
        llm_config,
        token_tracking_output,
        tool_registry,
    )
    .await?;

    // Execute task with interruption support
    let task_future = agent.execute_task_with_context(&task, &project_path);

    // Listen for interruption signals
    let interrupt_future = async {
        loop {
            match interrupt_receiver.recv().await {
                Ok(AppMessage::AgentExecutionInterrupted { .. }) => {
                    tracing::warn!("Task interrupted by user");
                    return Err(anyhow::anyhow!("Task interrupted by user"));
                }
                Ok(_) => continue, // Ignore other messages
                Err(_) => break,   // Channel closed
            }
        }
        Ok(())
    };

    // Race between task execution and interruption
    tokio::select! {
        result = task_future => {
            result?;
        }
        interrupt_result = interrupt_future => {
            interrupt_result?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use coro_core::output::AgentOutput;
    use tokio::sync::broadcast;

    #[test]
    fn test_token_tracking_output_handler_creation() {
        let (ui_sender, _) = broadcast::channel::<AppMessage>(10);
        let (interactive_sender, _) = mpsc::unbounded_channel();
        let config = InteractiveOutputConfig {
            realtime_updates: true,
            show_tool_details: true,
        };

        let handler = TokenTrackingOutputHandler::new(config, interactive_sender, ui_sender);
        assert!(handler.supports_realtime_updates());
    }
}
