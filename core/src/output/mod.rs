//! Output abstraction layer for the Coro Code core
//!
//! This module provides an abstract interface for outputting agent execution information,
//! allowing different implementations for CLI, API, logging, etc.

use crate::tools::{ToolCall, ToolResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Core only provides abstractions - implementations are in calling modules

/// Null output handler that discards all events (useful for testing and backward compatibility)
pub struct NullOutput;

#[async_trait]
impl AgentOutput for NullOutput {
    async fn emit_event(
        &self,
        _event: AgentEvent,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }
}

/// Convenience module for backward compatibility
pub mod events {
    pub use super::NullOutput;
}

/// Status of tool execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolExecutionStatus {
    /// Tool is currently executing
    Executing,
    /// Tool completed successfully
    Success,
    /// Tool failed with an error
    Error,
}

/// Rich tool execution information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecutionInfo {
    /// Unique identifier for this tool execution
    pub execution_id: String,
    /// Tool name (e.g., "bash", "str_replace_based_edit_tool")
    pub tool_name: String,
    /// Tool parameters/arguments
    pub parameters: HashMap<String, serde_json::Value>,
    /// Current execution status
    pub status: ToolExecutionStatus,
    /// Tool result (if completed)
    pub result: Option<ToolResult>,
    /// Timestamp of status change
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Additional metadata for tool-specific information
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Agent execution step information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStepInfo {
    /// Step number in the execution sequence
    pub step_number: usize,
    /// Current task description
    pub task: String,
    /// LLM thinking/reasoning (if available)
    pub thinking: Option<String>,
    /// Tool executions in this step
    pub tool_executions: Vec<ToolExecutionInfo>,
    /// Step completion status
    pub completed: bool,
}

/// Token usage statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenUsage {
    /// Total input tokens consumed
    pub input_tokens: u32,
    /// Total output tokens generated
    pub output_tokens: u32,
    /// Total tokens (input + output)
    pub total_tokens: u32,
}

/// Agent execution context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentExecutionContext {
    /// Agent configuration name or identifier
    pub agent_id: String,
    /// Original goal from the first user request (never overwritten)
    pub original_goal: String,
    /// Current task being executed (can be updated with new queries)
    pub current_task: String,
    /// Project path or working directory
    pub project_path: String,
    /// Maximum allowed steps
    pub max_steps: usize,
    /// Current step number
    pub current_step: usize,
    /// Total execution time so far
    pub execution_time: std::time::Duration,
    /// Token usage statistics
    pub token_usage: TokenUsage,
}

/// Events that can be emitted during agent execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentEvent {
    /// Agent execution started
    ExecutionStarted { context: AgentExecutionContext },
    /// Agent execution completed
    ExecutionCompleted {
        context: AgentExecutionContext,
        success: bool,
        summary: String,
    },
    /// New step started
    StepStarted { step_info: AgentStepInfo },
    /// Step completed
    StepCompleted { step_info: AgentStepInfo },
    /// Tool execution started
    ToolExecutionStarted { tool_info: ToolExecutionInfo },
    /// Tool execution status updated
    ToolExecutionUpdated { tool_info: ToolExecutionInfo },
    /// Tool execution completed
    ToolExecutionCompleted { tool_info: ToolExecutionInfo },
    /// Agent thinking/reasoning
    AgentThinking {
        step_number: usize,
        thinking: String,
    },
    /// Token usage updated (emitted after each LLM call)
    TokenUsageUpdated { token_usage: TokenUsage },
    /// Agent status update (for interactive mode status reporting)
    StatusUpdate {
        status: String,
        metadata: HashMap<String, serde_json::Value>,
    },
    /// General message or log
    Message {
        level: MessageLevel,
        content: String,
        metadata: HashMap<String, serde_json::Value>,
    },
    /// Conversation compression started
    CompressionStarted {
        level: String,
        current_tokens: u32,
        target_tokens: u32,
        reason: String,
    },
    /// Conversation compression completed
    CompressionCompleted {
        summary: String,
        tokens_saved: u32,
        messages_before: u32,
        messages_after: u32,
    },
    /// Conversation compression failed
    CompressionFailed {
        error: String,
        fallback_action: String,
    },
}

/// Message severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageLevel {
    Debug,
    Info,
    Normal,
    Warning,
    Error,
}

/// Abstract output interface for agent execution
#[async_trait]
pub trait AgentOutput: Send + Sync {
    /// Emit an agent event
    async fn emit_event(
        &self,
        event: AgentEvent,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Emit a message with specified level
    async fn emit_message(
        &self,
        level: MessageLevel,
        content: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.emit_event(AgentEvent::Message {
            level,
            content: content.to_string(),
            metadata: HashMap::new(),
        })
        .await
    }

    /// Emit debug message
    async fn debug(&self, content: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.emit_message(MessageLevel::Debug, content).await
    }

    /// Emit token usage update
    async fn emit_token_update(
        &self,
        token_usage: TokenUsage,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.emit_event(AgentEvent::TokenUsageUpdated { token_usage })
            .await
    }

    /// Emit info message
    async fn info(&self, content: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.emit_message(MessageLevel::Info, content).await
    }

    /// Emit warning message
    async fn warning(&self, content: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.emit_message(MessageLevel::Warning, content).await
    }

    /// Emit error message
    async fn error(&self, content: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.emit_message(MessageLevel::Error, content).await
    }

    /// Emit normal text message
    async fn normal(&self, content: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.emit_message(MessageLevel::Normal, content).await
    }

    /// Emit status update (for interactive mode status reporting)
    async fn emit_status_update(
        &self,
        status: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.emit_event(AgentEvent::StatusUpdate {
            status: status.to_string(),
            metadata: HashMap::new(),
        })
        .await
    }

    /// Check if this output handler supports real-time updates
    fn supports_realtime_updates(&self) -> bool {
        false
    }

    /// Flush any buffered output (for implementations that buffer)
    async fn flush(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }
}

/// Helper trait for creating tool execution info
pub trait ToolExecutionInfoBuilder {
    fn create_tool_execution_info(
        tool_call: &ToolCall,
        status: ToolExecutionStatus,
        result: Option<&ToolResult>,
    ) -> ToolExecutionInfo;
}

impl ToolExecutionInfoBuilder for ToolExecutionInfo {
    fn create_tool_execution_info(
        tool_call: &ToolCall,
        status: ToolExecutionStatus,
        result: Option<&ToolResult>,
    ) -> ToolExecutionInfo {
        let parameters = if let serde_json::Value::Object(map) = &tool_call.parameters {
            map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
        } else {
            let mut map = HashMap::new();
            map.insert("raw_parameters".to_string(), tool_call.parameters.clone());
            map
        };

        ToolExecutionInfo {
            execution_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            parameters,
            status,
            result: result.cloned(),
            timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
        }
    }
}
