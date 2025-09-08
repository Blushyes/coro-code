//! Agent configuration structures

use serde::{Deserialize, Serialize};

/// Output mode for the agent
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OutputMode {
    /// Debug mode with detailed logging and verbose output
    Debug,
    /// Normal mode with clean, user-friendly output
    Normal,
}

impl Default for OutputMode {
    fn default() -> Self {
        Self::Normal
    }
}

/// Configuration for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Maximum number of execution steps
    pub max_steps: usize,

    /// Whether to enable lakeview integration
    pub enable_lakeview: bool,

    /// List of tools available to this agent
    pub tools: Vec<String>,

    /// Output mode for the agent (debug or normal)
    #[serde(default)]
    pub output_mode: OutputMode,

    /// Custom system prompt for the agent (optional)
    /// If not provided, the default system prompt will be used
    #[serde(default)]
    pub system_prompt: Option<String>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_steps: 200,
            enable_lakeview: true,
            tools: vec![
                "bash".to_string(),
                "str_replace_based_edit_tool".to_string(),
                "sequentialthinking".to_string(),
                "task_done".to_string(),
            ],
            output_mode: OutputMode::default(),
            system_prompt: None,
        }
    }
}

/// Builder for creating agents with resolved LLM configuration
pub struct AgentBuilder {
    llm_config: crate::config::ResolvedLlmConfig,
    agent_config: AgentConfig,
    abort_controller: Option<super::AbortController>,
}

impl AgentBuilder {
    /// Create a new agent builder with LLM configuration
    pub fn new(llm_config: crate::config::ResolvedLlmConfig) -> Self {
        Self {
            llm_config,
            agent_config: AgentConfig::default(),
            abort_controller: None,
        }
    }

    /// Set agent configuration
    pub fn with_agent_config(mut self, agent_config: AgentConfig) -> Self {
        self.agent_config = agent_config;
        self
    }

    /// Set maximum steps
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.agent_config.max_steps = max_steps;
        self
    }

    /// Set tools
    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.agent_config.tools = tools;
        self
    }

    /// Set output mode
    pub fn with_output_mode(mut self, output_mode: OutputMode) -> Self {
        self.agent_config.output_mode = output_mode;
        self
    }

    /// Set system prompt
    pub fn with_system_prompt(mut self, system_prompt: Option<String>) -> Self {
        self.agent_config.system_prompt = system_prompt;
        self
    }

    /// Inject a global AbortController for cancellation support
    pub fn with_cancellation(mut self, controller: super::AbortController) -> Self {
        self.abort_controller = Some(controller);
        self
    }

    /// Build the agent with the given output handler
    pub async fn build_with_output(
        self,
        output: Box<dyn crate::output::AgentOutput>,
    ) -> crate::error::Result<super::AgentCore> {
        super::AgentCore::new_with_llm_config(
            self.agent_config,
            self.llm_config,
            output,
            self.abort_controller,
        )
        .await
    }

    /// Build the agent with custom output handler and tool registry
    pub async fn build_with_output_and_registry(
        self,
        output: Box<dyn crate::output::AgentOutput>,
        tool_registry: crate::tools::ToolRegistry,
    ) -> crate::error::Result<super::AgentCore> {
        super::AgentCore::new_with_output_and_registry(
            self.agent_config,
            self.llm_config,
            output,
            tool_registry,
            self.abort_controller,
        )
        .await
    }

    /// Build the agent with null output (for testing)
    pub async fn build(self) -> crate::error::Result<super::AgentCore> {
        use crate::output::events::NullOutput;
        self.build_with_output(Box::new(NullOutput)).await
    }
}
