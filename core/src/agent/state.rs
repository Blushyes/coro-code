//! Persisted agent context snapshot for export/import
//!
//! This module defines a lightweight snapshot structure for exporting the
//! essential conversation state and execution context of an agent, and helpers
//! to serialize/deserialize it to/from JSON or files. It is intentionally
//! independent of any live resources (LLM client, tool registry, etc.).

use crate::agent::config::AgentConfig;
use crate::llm::LlmMessage;
use crate::output::AgentExecutionContext;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A versioned, serializable snapshot of an agent's context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedAgentContext {
    /// Snapshot version for forward compatibility
    pub version: u32,
    /// Implementation identifier (e.g., "coro_agent")
    pub agent_type: String,
    /// When this snapshot was saved
    pub saved_at: DateTime<Utc>,
    /// Optional agent configuration captured at the time of saving
    pub config: Option<AgentConfig>,
    /// Full conversation message history (including system/user/assistant/tool)
    pub conversation_history: Vec<LlmMessage>,
    /// Execution context: goal, current task, token usage, etc.
    pub execution_context: Option<AgentExecutionContext>,
}

impl PersistedAgentContext {
    /// Create a new snapshot
    pub fn new(
        agent_type: String,
        config: Option<AgentConfig>,
        conversation_history: Vec<LlmMessage>,
        execution_context: Option<AgentExecutionContext>,
    ) -> Self {
        Self {
            version: 1,
            agent_type,
            saved_at: Utc::now(),
            config,
            conversation_history,
            execution_context,
        }
    }

    /// Serialize the snapshot to a JSON string
    pub fn to_json(&self) -> crate::error::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Deserialize a snapshot from a JSON string
    pub fn from_json(s: &str) -> crate::error::Result<Self> {
        Ok(serde_json::from_str::<Self>(s)?)
    }

    /// Save the snapshot to a file (creates parent directories if needed)
    pub fn to_file(&self, path: &Path) -> crate::error::Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let json = self.to_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load a snapshot from a file
    pub fn from_file(path: &Path) -> crate::error::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Self::from_json(&data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_json() {
        let snapshot = PersistedAgentContext::new(
            "coro_agent".to_string(),
            Some(AgentConfig::default()),
            vec![LlmMessage::system("hello"), LlmMessage::assistant("world")],
            Some(AgentExecutionContext {
                agent_id: "coro_agent".to_string(),
                original_goal: "OG".to_string(),
                current_task: "CT".to_string(),
                project_path: ".".to_string(),
                max_steps: 5,
                current_step: 1,
                execution_time: std::time::Duration::from_secs(0),
                token_usage: Default::default(),
            }),
        );

        let json = snapshot.to_json().expect("serialize");
        let restored = PersistedAgentContext::from_json(&json).expect("deserialize");

        assert_eq!(restored.version, 1);
        assert_eq!(restored.agent_type, "coro_agent");
        assert_eq!(restored.conversation_history.len(), 2);
        assert!(restored.execution_context.is_some());
        assert!(restored.config.is_some());
    }
}
