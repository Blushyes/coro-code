//! AgentCore implementation

use super::config::AgentConfig;
use crate::agent::prompt::{build_system_prompt_with_context, build_user_message};
use crate::agent::tokens::ConversationManager;
use crate::agent::{Agent, AgentExecution, AgentResult};
use crate::error::{AgentError, Result};
use crate::llm::{ChatOptions, LlmClient, LlmMessage};
use crate::output::{
    AgentEvent, AgentExecutionContext, AgentOutput, TokenUsage, ToolExecutionInfo,
    ToolExecutionInfoBuilder, ToolExecutionStatus,
};
use crate::tools::{ToolExecutor, ToolRegistry};
use crate::trajectory::{TrajectoryEntry, TrajectoryRecorder};
use async_trait::async_trait;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// TraeAgent - the main agent implementation
pub struct AgentCore {
    config: AgentConfig,
    llm_client: Arc<dyn LlmClient>,
    tool_executor: ToolExecutor,
    trajectory_recorder: Option<TrajectoryRecorder>,
    conversation_history: Vec<LlmMessage>,
    output: Box<dyn AgentOutput>,
    #[allow(dead_code)]
    current_task_displayed: bool,
    execution_context: Option<AgentExecutionContext>,
    conversation_manager: ConversationManager,
}

impl AgentCore {
    /// Create a new AgentCore with resolved LLM configuration
    pub async fn new_with_llm_config(
        agent_config: AgentConfig,
        llm_config: crate::config::ResolvedLlmConfig,
        output: Box<dyn AgentOutput>,
    ) -> Result<Self> {
        // Create LLM client based on protocol
        let llm_client: Arc<dyn LlmClient> = match llm_config.protocol {
            crate::config::Protocol::OpenAICompat => {
                Arc::new(crate::llm::OpenAiClient::new(&llm_config)?)
            }
            crate::config::Protocol::Anthropic => {
                Arc::new(crate::llm::AnthropicClient::new(&llm_config)?)
            }
            crate::config::Protocol::GoogleAI => {
                return Err(AgentError::NotInitialized.into()); // TODO: Implement GoogleAI client
            }
            crate::config::Protocol::AzureOpenAI => {
                // Azure OpenAI uses the same client as OpenAI
                Arc::new(crate::llm::OpenAiClient::new(&llm_config)?)
            }
            crate::config::Protocol::Custom(_) => {
                return Err(AgentError::NotInitialized.into()); // TODO: Implement custom protocol support
            }
        };

        // Create tool executor
        let tool_registry = crate::tools::ToolRegistry::default();
        let tool_executor = tool_registry.create_executor(&agent_config.tools);

        // Create unified conversation manager (simplified single component)
        let max_tokens = llm_config.params.max_tokens.unwrap_or(8192);
        let conversation_manager = ConversationManager::new(max_tokens, llm_client.clone());

        Ok(Self {
            config: agent_config,
            llm_client,
            tool_executor,
            trajectory_recorder: None,
            conversation_history: Vec::new(),
            output,
            current_task_displayed: false,
            execution_context: None,
            conversation_manager,
        })
    }

    /// Get agent configuration
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    /// Create a new TraeAgent with custom tool registry and output handler
    pub async fn new_with_output_and_registry(
        agent_config: AgentConfig,
        llm_config: crate::config::ResolvedLlmConfig,
        output: Box<dyn AgentOutput>,
        tool_registry: ToolRegistry,
    ) -> Result<Self> {
        // Create LLM client based on protocol
        let llm_client: Arc<dyn LlmClient> = match llm_config.protocol {
            crate::config::Protocol::OpenAICompat => {
                Arc::new(crate::llm::OpenAiClient::new(&llm_config)?)
            }
            crate::config::Protocol::Anthropic => {
                Arc::new(crate::llm::AnthropicClient::new(&llm_config)?)
            }
            crate::config::Protocol::GoogleAI => {
                return Err(AgentError::NotInitialized.into()); // TODO: Implement GoogleAI client
            }
            crate::config::Protocol::AzureOpenAI => {
                // Azure OpenAI uses the same client as OpenAI
                Arc::new(crate::llm::OpenAiClient::new(&llm_config)?)
            }
            crate::config::Protocol::Custom(_) => {
                return Err(AgentError::NotInitialized.into()); // TODO: Implement custom protocol support
            }
        };

        // Create tool executor with custom registry
        let tool_executor = tool_registry.create_executor(&agent_config.tools);

        // Create unified conversation manager (simplified single component)
        let max_tokens = llm_config.params.max_tokens.unwrap_or(8192);
        let conversation_manager = ConversationManager::new(max_tokens, llm_client.clone());

        Ok(Self {
            config: agent_config,
            llm_client,
            tool_executor,
            trajectory_recorder: None,
            conversation_history: Vec::new(),
            output,
            current_task_displayed: false,
            execution_context: None,
            conversation_manager,
        })
    }

    /// Create a new TraeAgent with default null output (for testing)
    pub async fn new(
        agent_config: AgentConfig,
        llm_config: crate::config::ResolvedLlmConfig,
    ) -> Result<Self> {
        use crate::output::events::NullOutput;
        Self::new_with_llm_config(agent_config, llm_config, Box::new(NullOutput)).await
    }

    /// Set a custom system prompt for the agent
    /// This will override any system prompt set in the configuration
    pub fn set_system_prompt(&mut self, system_prompt: Option<String>) {
        self.config.system_prompt = system_prompt;
    }

    /// Get the current system prompt from configuration
    pub fn get_configured_system_prompt(&self) -> Option<&String> {
        self.config.system_prompt.as_ref()
    }

    /// Get the system prompt for the agent with project context
    fn get_system_prompt(&self, project_path: &Path) -> String {
        // Use custom system prompt if provided, otherwise use default
        let base_prompt = if let Some(custom_prompt) = &self.config.system_prompt {
            // If custom prompt is provided, use it as-is with minimal generic context
            let system_context = crate::agent::prompt::build_system_context();

            format!(
                "{}\n\n\
                     [System Context]:\n{}",
                custom_prompt, system_context
            )
        } else {
            // Use default system prompt with full environment context from prompt.rs
            build_system_prompt_with_context(project_path)
        };

        format!(
            "{}\n\nAvailable tools: {}",
            base_prompt,
            self.tool_executor.list_tools().join(", ")
        )
    }

    /// Execute a single step of the agent
    async fn execute_step(&mut self, step: usize, project_path: &Path) -> Result<bool> {
        // Prepare messages - only add system prompt if conversation history doesn't start with one
        let mut messages = Vec::new();
        let needs_system_prompt = self.conversation_history.is_empty()
            || !matches!(
                self.conversation_history[0].role,
                crate::llm::MessageRole::System
            );

        if needs_system_prompt {
            messages.push(LlmMessage::system(self.get_system_prompt(project_path)));
        }
        messages.extend(self.conversation_history.clone());

        // Record LLM request
        if let Some(recorder) = &self.trajectory_recorder {
            recorder
                .record(TrajectoryEntry::llm_request(
                    messages.clone(),
                    self.llm_client.model_name().to_string(),
                    self.llm_client.provider_name().to_string(),
                    step,
                ))
                .await?;
        }

        // Get tool definitions
        let tool_definitions = self.tool_executor.get_tool_definitions();

        // Log agent thinking in debug mode
        let _ = self.output.debug("🤖 Agent thinking...").await;

        // Set up options
        let options = Some(ChatOptions {
            ..Default::default()
        });

        // Make LLM request (non-streaming) with detailed error handling
        let response = match self
            .llm_client
            .chat_completion(messages, Some(tool_definitions), options)
            .await
        {
            Ok(response) => response,
            Err(e) => {
                tracing::error!("❌ LLM request failed for step {}: {}", step, e);
                let _ = self
                    .output
                    .error(&format!("LLM request failed: {}", e))
                    .await;
                return Err(e);
            }
        };

        // Update token usage
        if let Some(usage) = &response.usage {
            if let Some(context) = &mut self.execution_context {
                context.token_usage.input_tokens += usage.prompt_tokens;
                context.token_usage.output_tokens += usage.completion_tokens;
                context.token_usage.total_tokens += usage.total_tokens;

                // Emit token update event immediately after LLM call
                self.output
                    .emit_token_update(context.token_usage.clone())
                    .await
                    .unwrap_or_else(|e| {
                        let _ = futures::executor::block_on(
                            self.output
                                .debug(&format!("Failed to emit token update event: {}", e)),
                        );
                    });
            }
        }

        // Record LLM response
        if let Some(recorder) = &self.trajectory_recorder {
            recorder
                .record(TrajectoryEntry::llm_response(
                    response.message.clone(),
                    response.usage.clone(),
                    response.finish_reason.as_ref().map(|r| format!("{:?}", r)),
                    step,
                ))
                .await?;
        }

        // Add response to conversation history
        self.conversation_history.push(response.message.clone());

        // Check if there are tool calls to execute
        if response.message.has_tool_use() {
            let tool_uses = response.message.get_tool_uses();

            for tool_use in tool_uses {
                if let crate::llm::ContentBlock::ToolUse { id, name, input } = tool_use {
                    // Display tool execution based on output mode
                    let tool_call = crate::tools::ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        parameters: input.clone(),
                        metadata: None,
                    };

                    // Create tool execution info and emit started event
                    let tool_info = ToolExecutionInfo::create_tool_execution_info(
                        &tool_call,
                        ToolExecutionStatus::Executing,
                        None,
                    );

                    self.output
                        .emit_event(AgentEvent::ToolExecutionStarted {
                            tool_info: tool_info.clone(),
                        })
                        .await
                        .unwrap_or_else(|e| {
                            let _ = futures::executor::block_on(self.output.debug(&format!(
                                "Failed to emit tool execution started event: {}",
                                e
                            )));
                        });

                    // Record tool call
                    if let Some(recorder) = &self.trajectory_recorder {
                        recorder
                            .record(TrajectoryEntry::tool_call(tool_call.clone(), step))
                            .await?;
                    }

                    // Execute tool
                    let tool_result = self.tool_executor.execute(tool_call.clone()).await?;

                    // Create completed tool execution info and emit completed event
                    let completed_tool_info = ToolExecutionInfo::create_tool_execution_info(
                        &tool_call,
                        if tool_result.success {
                            ToolExecutionStatus::Success
                        } else {
                            ToolExecutionStatus::Error
                        },
                        Some(&tool_result),
                    );

                    self.output
                        .emit_event(AgentEvent::ToolExecutionCompleted {
                            tool_info: completed_tool_info,
                        })
                        .await
                        .unwrap_or_else(|e| {
                            let _ = futures::executor::block_on(self.output.debug(&format!(
                                "Failed to emit tool execution completed event: {}",
                                e
                            )));
                        });

                    // Handle special tool behaviors
                    if name == "sequentialthinking" {
                        // For thinking tool, emit thinking event
                        if let Some(data) = &tool_result.data {
                            if let Some(thought) = data.get("thought") {
                                if let Some(thought_str) = thought.as_str() {
                                    self.output
                                        .emit_event(AgentEvent::AgentThinking {
                                            step_number: step,
                                            thinking: thought_str.to_string(),
                                        })
                                        .await
                                        .unwrap_or_else(|e| {
                                            let _ = futures::executor::block_on(self.output.debug(
                                                &format!("Failed to emit thinking event: {}", e),
                                            ));
                                        });
                                }
                            }
                        } else {
                            // Fallback: try to extract from content
                            if let Some(start) = tool_result.content.find("Thought: ") {
                                let thought_start = start + "Thought: ".len();
                                if let Some(end) = tool_result.content[thought_start..].find("\n\n")
                                {
                                    let thought =
                                        &tool_result.content[thought_start..thought_start + end];
                                    self.output
                                        .emit_event(AgentEvent::AgentThinking {
                                            step_number: step,
                                            thinking: thought.to_string(),
                                        })
                                        .await
                                        .unwrap_or_else(|e| {
                                            let _ = futures::executor::block_on(self.output.debug(
                                                &format!("Failed to emit thinking event: {}", e),
                                            ));
                                        });
                                }
                            }
                        }
                    }

                    // Record tool result
                    if let Some(recorder) = &self.trajectory_recorder {
                        recorder
                            .record(TrajectoryEntry::tool_result(tool_result.clone(), step))
                            .await?;
                    }

                    // Check if this is a task completion
                    if name == "task_done" && tool_result.success {
                        return Ok(true); // Task completed
                    }

                    // Add tool result to conversation
                    let result_message = LlmMessage {
                        role: crate::llm::MessageRole::Tool,
                        content: crate::llm::MessageContent::MultiModal(vec![
                            crate::llm::ContentBlock::ToolResult {
                                tool_use_id: id.clone(),
                                is_error: Some(!tool_result.success),
                                content: tool_result.content,
                            },
                        ]),
                        metadata: None,
                    };

                    self.conversation_history.push(result_message);
                }
            }

            // After executing tools, proceed to the next step.
            // Align with Python scheduler: one LLM call per step; tool results are appended,
            // and the next step will let the LLM process those results.
            return Ok(false);
        }

        // If no tool calls, handle text response
        if let Some(text_content) = response.message.get_text() {
            if !text_content.trim().is_empty() {
                // Emit the agent's text response as a normal message
                self.output.normal(&text_content).await.unwrap_or_else(|e| {
                    let _ = futures::executor::block_on(
                        self.output
                            .debug(&format!("Failed to emit agent response message: {}", e)),
                    );
                });
            }
        }

        // If no tool calls, we're done for this step
        Ok(false)
    }
}

#[async_trait]
impl Agent for AgentCore {
    async fn execute_task(&mut self, task: &str) -> AgentResult<AgentExecution> {
        // Use execute_task_with_context with current directory as default
        let current_dir = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        self.execute_task_with_context(task, &current_dir).await
    }

    fn config(&self) -> &AgentConfig {
        &self.config
    }

    fn agent_type(&self) -> &str {
        "coro_agent"
    }

    fn set_trajectory_recorder(&mut self, recorder: TrajectoryRecorder) {
        self.trajectory_recorder = Some(recorder);
    }

    fn trajectory_recorder(&self) -> Option<&TrajectoryRecorder> {
        self.trajectory_recorder.as_ref()
    }
}

impl AgentCore {
    /// Trim conversation history to prevent token overflow
    /// Keeps the system prompt and recent messages, removes older conversation
    /// TODO
    /// Apply intelligent compression to conversation history based on token usage
    async fn apply_intelligent_compression(&mut self) -> Result<()> {
        // Use the unified conversation manager - single method call!
        match self
            .conversation_manager
            .maybe_compress(
                self.conversation_history.clone(),
                self.execution_context.as_ref(),
            )
            .await
        {
            Ok(result) => {
                // Update conversation history
                self.conversation_history = result.messages;

                // If compression was applied, emit events
                if let Some(summary) = result.compression_applied {
                    // Emit compression started event
                    let _ = self
                        .output
                        .emit_event(AgentEvent::CompressionStarted {
                            level: summary.level.as_str().to_string(),
                            current_tokens: summary.tokens_before,
                            target_tokens: summary.tokens_after,
                            reason: format!(
                                "Token usage requires {} compression",
                                summary.level.as_str()
                            ),
                        })
                        .await;

                    // Emit completion event
                    let _ = self
                        .output
                        .emit_event(AgentEvent::CompressionCompleted {
                            summary: summary.summary.clone(),
                            tokens_saved: summary.tokens_saved,
                            messages_before: summary.messages_before,
                            messages_after: summary.messages_after,
                        })
                        .await;

                    tracing::info!("Compression completed: {}", summary.summary);
                }
            }
            Err(e) => {
                tracing::warn!(
                    "Compression failed: {}. Falling back to simple trimming.",
                    e
                );
                self.fallback_trim_conversation_history(50);

                let _ = self
                    .output
                    .emit_event(AgentEvent::CompressionFailed {
                        error: e.to_string(),
                        fallback_action: "Simple message trimming applied".to_string(),
                    })
                    .await;
            }
        }

        Ok(())
    }

    /// Fallback simple trim for when intelligent compression fails
    fn fallback_trim_conversation_history(&mut self, max_messages: usize) {
        if self.conversation_history.len() <= max_messages {
            return;
        }

        // Always keep the system prompt (first message)
        let mut new_history = Vec::new();
        if let Some(first_msg) = self.conversation_history.first() {
            if matches!(first_msg.role, crate::llm::MessageRole::System) {
                new_history.push(first_msg.clone());
            }
        }

        // Keep the most recent messages
        let keep_count = max_messages.saturating_sub(1); // -1 for system prompt
        let start_index = self.conversation_history.len().saturating_sub(keep_count);

        // Skip system prompt if it's already added
        let skip_first = !new_history.is_empty();
        let iter_start = if skip_first {
            std::cmp::max(1, start_index)
        } else {
            start_index
        };

        new_history.extend(self.conversation_history[iter_start..].iter().cloned());

        self.conversation_history = new_history;
    }

    /// Continue conversation with a new task without clearing history
    pub async fn execute_task_with_context(
        &mut self,
        task: &str,
        project_path: &Path,
    ) -> AgentResult<AgentExecution> {
        let start_time = Instant::now();

        // Create execution context or update existing one
        if self.execution_context.is_none() {
            self.execution_context = Some(AgentExecutionContext {
                agent_id: "coro_agent".to_string(),
                original_goal: task.to_string(),
                current_task: task.to_string(),
                project_path: project_path.to_string_lossy().to_string(),
                max_steps: self.config.max_steps,
                current_step: 0,
                execution_time: std::time::Duration::from_secs(0),
                token_usage: TokenUsage::default(),
            });
        } else {
            // Update only the current task, preserving the original goal
            if let Some(context) = &mut self.execution_context {
                context.current_task = task.to_string();
                context.current_step = 0;
            }
        }

        // Emit execution started event
        if let Some(context) = &self.execution_context {
            self.output
                .emit_event(AgentEvent::ExecutionStarted {
                    context: context.clone(),
                })
                .await
                .unwrap_or_else(|e| {
                    let _ = futures::executor::block_on(
                        self.output
                            .debug(&format!("Failed to emit execution started event: {}", e)),
                    );
                });
        }

        // Record task start
        if let Some(recorder) = &self.trajectory_recorder {
            recorder
                .record(TrajectoryEntry::task_start(
                    task.to_string(),
                    serde_json::to_value(&self.config).unwrap_or_default(),
                ))
                .await?;
        }

        // If conversation history is empty, add system prompt
        if self.conversation_history.is_empty() {
            self.conversation_history
                .push(LlmMessage::system(self.get_system_prompt(project_path)));
        }

        // Add user message with task
        let user_message = build_user_message(task);
        self.conversation_history
            .push(LlmMessage::user(&user_message));

        let mut step = 0;
        let mut task_completed = false;

        // Execute steps until completion or max steps reached
        while step < self.config.max_steps && !task_completed {
            step += 1;

            // Apply intelligent compression before each step to manage token usage
            self.apply_intelligent_compression().await?;

            match self.execute_step(step, project_path).await {
                Ok(completed) => {
                    task_completed = completed;

                    // Record step completion
                    if let Some(recorder) = &self.trajectory_recorder {
                        recorder
                            .record(TrajectoryEntry::step_complete(
                                format!("Step {} completed", step),
                                true,
                                step,
                            ))
                            .await?;
                    }
                }
                Err(e) => {
                    // Record error
                    if let Some(recorder) = &self.trajectory_recorder {
                        recorder
                            .record(TrajectoryEntry::error(
                                e.to_string(),
                                Some(format!("Step {}", step)),
                                step,
                            ))
                            .await?;
                    }

                    let duration = start_time.elapsed().as_millis() as u64;
                    return Ok(AgentExecution::failure(
                        format!("Error in step {}: {}", step, e),
                        step,
                        duration,
                    ));
                }
            }
        }

        let duration = start_time.elapsed();

        // Update execution context
        if let Some(context) = &mut self.execution_context {
            context.current_step = step;
            context.execution_time = duration;
        }

        // Record task completion
        if let Some(recorder) = &self.trajectory_recorder {
            recorder
                .record(TrajectoryEntry::task_complete(
                    task_completed,
                    if task_completed {
                        "Task completed successfully".to_string()
                    } else {
                        format!("Task incomplete after {} steps", step)
                    },
                    step,
                    duration.as_millis() as u64,
                ))
                .await?;
        }

        // Emit execution completed event
        if let Some(context) = &self.execution_context {
            let summary = if task_completed {
                "Task completed successfully".to_string()
            } else {
                format!("Task incomplete after {} steps", step)
            };

            self.output
                .emit_event(AgentEvent::ExecutionCompleted {
                    context: context.clone(),
                    success: task_completed,
                    summary: summary.clone(),
                })
                .await
                .unwrap_or_else(|e| {
                    let _ = futures::executor::block_on(
                        self.output
                            .debug(&format!("Failed to emit execution completed event: {}", e)),
                    );
                });
        }

        let duration_ms = duration.as_millis() as u64;

        if task_completed {
            Ok(AgentExecution::success(
                "Task completed successfully".to_string(),
                step,
                duration_ms,
            ))
        } else {
            Ok(AgentExecution::failure(
                format!("Task incomplete after {} steps", step),
                step,
                duration_ms,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;
    use crate::llm::{
        ChatOptions, LlmClient, LlmMessage, LlmResponse, MessageContent, MessageRole,
        ToolDefinition,
    };
    use crate::AgentConfig;
    use async_trait::async_trait;

    // Mock LLM client for testing
    struct MockLlmClient;

    impl MockLlmClient {
        fn new() -> Self {
            Self
        }
    }

    #[async_trait]
    impl LlmClient for MockLlmClient {
        async fn chat_completion(
            &self,
            _messages: Vec<LlmMessage>,
            _tools: Option<Vec<ToolDefinition>>,
            _options: Option<ChatOptions>,
        ) -> Result<LlmResponse> {
            Ok(LlmResponse {
                message: LlmMessage {
                    role: MessageRole::Assistant,
                    content: MessageContent::Text("Mock response".to_string()),
                    metadata: None,
                },
                usage: None,
                model: "mock-model".to_string(),
                finish_reason: None,
                metadata: None,
            })
        }

        fn model_name(&self) -> &str {
            "mock-model"
        }

        fn provider_name(&self) -> &str {
            "mock"
        }
    }

    #[test]
    fn test_system_prompt_configuration() {
        // Test AgentConfig with custom system prompt
        let agent_config = AgentConfig {
            system_prompt: Some("Custom system prompt for testing".to_string()),
            ..Default::default()
        };

        assert_eq!(
            agent_config.system_prompt,
            Some("Custom system prompt for testing".to_string())
        );

        // Test default AgentConfig has no system prompt
        let default_config = AgentConfig::default();
        assert_eq!(default_config.system_prompt, None);
    }

    #[test]
    fn test_system_prompt_serialization() {
        // Test that AgentConfig with system_prompt can be serialized/deserialized
        let config = AgentConfig {
            system_prompt: Some("Custom prompt".to_string()),
            ..Default::default()
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: AgentConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(
            deserialized.system_prompt,
            Some("Custom prompt".to_string())
        );
    }

    #[test]
    fn test_system_prompt_default_none() {
        // Test that default AgentConfig has None for system_prompt
        let config = AgentConfig::default();
        assert_eq!(config.system_prompt, None);
    }

    #[test]
    fn test_custom_system_prompt_excludes_project_context() {
        // Test that custom system prompt doesn't include project-specific information
        use crate::output::events::NullOutput;
        use crate::tools::ToolRegistry;
        use std::path::PathBuf;

        // Create a mock agent with custom system prompt
        let agent_config = AgentConfig {
            system_prompt: Some("You are a general purpose AI assistant.".to_string()),
            ..Default::default()
        };

        // Create minimal components for testing
        let tool_registry = ToolRegistry::default();
        let tool_executor = tool_registry.create_executor(&agent_config.tools);

        // Create unified conversation manager for testing
        let conversation_manager = ConversationManager::new(
            8192, // max_tokens
            std::sync::Arc::new(MockLlmClient::new()),
        );

        let agent = AgentCore {
            config: agent_config,
            llm_client: std::sync::Arc::new(MockLlmClient::new()),
            tool_executor,
            trajectory_recorder: None,
            conversation_history: Vec::new(),
            output: Box::new(NullOutput),
            current_task_displayed: false,
            execution_context: None,
            conversation_manager,
        };

        let project_path = PathBuf::from("/some/project/path");
        let system_prompt = agent.get_system_prompt(&project_path);

        // Should contain the custom prompt
        assert!(system_prompt.contains("You are a general purpose AI assistant."));

        // Should contain system context (OS, architecture, etc.)
        assert!(system_prompt.contains("System Information:"));
        assert!(system_prompt.contains("Operating System:"));

        // Should contain available tools
        assert!(system_prompt.contains("Available tools:"));

        // Should NOT contain project-specific information
        assert!(!system_prompt.contains("Project root path"));
        assert!(!system_prompt.contains("/some/project/path"));
        assert!(!system_prompt.contains("IMPORTANT: When using tools that require file paths"));
        assert!(!system_prompt.contains("You are an expert AI software engineering agent"));
    }
}
