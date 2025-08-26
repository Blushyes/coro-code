//! Unified conversation management with automatic compression
//!
//! Provides a single, simple interface for managing conversation token limits
//! and applying intelligent compression when needed.

use super::calculator::TokenCalculator;
use crate::error::Result;
use crate::llm::{ChatOptions, ContentBlock, LlmClient, LlmMessage, MessageContent, MessageRole};
use crate::output::AgentExecutionContext;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Compression level for different strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionLevel {
    /// Light compression: compress tool outputs and redundant content
    Light,
    /// Medium compression: summarize older conversation segments  
    Medium,
    /// Heavy compression: keep only essential recent context
    Heavy,
}

impl CompressionLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            CompressionLevel::Light => "light",
            CompressionLevel::Medium => "medium",
            CompressionLevel::Heavy => "heavy",
        }
    }
}

/// Summary of compression that was applied
#[derive(Debug, Clone)]
pub struct CompressionSummary {
    pub level: CompressionLevel,
    pub tokens_before: u32,
    pub tokens_after: u32,
    pub tokens_saved: u32,
    pub messages_before: u32,
    pub messages_after: u32,
    pub summary: String,
}

/// Result of maybe applying compression
#[derive(Debug)]
pub struct MaybeCompressedResult {
    /// The (possibly compressed) conversation messages
    pub messages: Vec<LlmMessage>,
    /// Compression details if compression was applied
    pub compression_applied: Option<CompressionSummary>,
}

/// Unified conversation manager with automatic compression
///
/// Handles all aspects of conversation token management:
/// - Token counting and limit monitoring
/// - Automatic compression when needed
/// - Tool call pairing preservation
/// - Context-aware summaries
pub struct ConversationManager {
    /// Maximum tokens allowed before compression
    max_tokens: u32,
    /// LLM client for generating summaries
    llm_client: Arc<dyn LlmClient>,
    /// Current estimated token count
    current_tokens: u32,
    /// Compression thresholds: [light, medium, heavy]
    compression_thresholds: [f64; 3],
    /// Compression targets: [light, medium, heavy]  
    compression_targets: [f64; 3],
    /// Recent user-assistant pairs to preserve
    preserve_recent_pairs: u32,
    /// Token budget for tool outputs before compression
    tool_output_budget: u32,
    /// Maximum tokens for generated summaries
    max_summary_tokens: u32,
}

impl ConversationManager {
    /// Create a new conversation manager
    pub fn new(max_tokens: u32, llm_client: Arc<dyn LlmClient>) -> Self {
        Self {
            max_tokens,
            llm_client,
            current_tokens: 0,
            compression_thresholds: [0.7, 0.8, 0.9], // 70%, 80%, 90%
            compression_targets: [0.6, 0.5, 0.3],    // 60%, 50%, 30%
            preserve_recent_pairs: 3,
            tool_output_budget: 2000,
            max_summary_tokens: 500,
        }
    }

    /// Maybe apply compression to conversation based on token usage
    ///
    /// This is the main public interface - automatically determines if compression
    /// is needed and applies the appropriate level.
    pub async fn maybe_compress(
        &mut self,
        messages: Vec<LlmMessage>,
        context: Option<&AgentExecutionContext>,
    ) -> Result<MaybeCompressedResult> {
        // Update current token count
        self.current_tokens = TokenCalculator::estimate_conversation_tokens(&messages);

        // Check if compression is needed
        let usage_ratio = self.current_tokens as f64 / self.max_tokens as f64;

        let compression_level = if usage_ratio >= self.compression_thresholds[2] {
            Some((
                CompressionLevel::Heavy,
                "Token usage exceeding heavy threshold",
            ))
        } else if usage_ratio >= self.compression_thresholds[1] {
            Some((
                CompressionLevel::Medium,
                "Token usage exceeding medium threshold",
            ))
        } else if usage_ratio >= self.compression_thresholds[0] {
            Some((
                CompressionLevel::Light,
                "Token usage exceeding light threshold",
            ))
        } else {
            None
        };

        match compression_level {
            Some((level, _reason)) => {
                let target_tokens =
                    (self.max_tokens as f64 * self.get_compression_target(level)) as u32;
                let messages_before_count = messages.len() as u32;
                let compressed_messages = self
                    .apply_compression(messages, level, target_tokens, context)
                    .await?;
                let tokens_after =
                    TokenCalculator::estimate_conversation_tokens(&compressed_messages);

                let summary = CompressionSummary {
                    level,
                    tokens_before: self.current_tokens,
                    tokens_after,
                    tokens_saved: self.current_tokens.saturating_sub(tokens_after),
                    messages_before: messages_before_count,
                    messages_after: compressed_messages.len() as u32,
                    summary: format!(
                        "{} compression: {} -> {} messages ({} -> {} tokens, {:.1}% reduction)",
                        level.as_str(),
                        messages_before_count,
                        compressed_messages.len(),
                        self.current_tokens,
                        tokens_after,
                        if self.current_tokens > 0 {
                            ((self.current_tokens - tokens_after) as f64
                                / self.current_tokens as f64)
                                * 100.0
                        } else {
                            0.0
                        }
                    ),
                };

                // Update current token count
                self.current_tokens = tokens_after;

                Ok(MaybeCompressedResult {
                    messages: compressed_messages,
                    compression_applied: Some(summary),
                })
            }
            None => {
                // No compression needed
                Ok(MaybeCompressedResult {
                    messages,
                    compression_applied: None,
                })
            }
        }
    }

    /// Get current token count estimate
    pub fn current_tokens(&self) -> u32 {
        self.current_tokens
    }

    /// Get maximum token limit
    pub fn max_tokens(&self) -> u32 {
        self.max_tokens
    }

    /// Get current usage ratio (0.0 to 1.0+)
    pub fn usage_ratio(&self) -> f64 {
        self.current_tokens as f64 / self.max_tokens as f64
    }

    // --- Internal Implementation ---

    fn get_compression_target(&self, level: CompressionLevel) -> f64 {
        match level {
            CompressionLevel::Light => self.compression_targets[0],
            CompressionLevel::Medium => self.compression_targets[1],
            CompressionLevel::Heavy => self.compression_targets[2],
        }
    }

    async fn apply_compression(
        &self,
        messages: Vec<LlmMessage>,
        level: CompressionLevel,
        _target_tokens: u32,
        context: Option<&AgentExecutionContext>,
    ) -> Result<Vec<LlmMessage>> {
        match level {
            CompressionLevel::Light => self.light_compression(messages).await,
            CompressionLevel::Medium => self.medium_compression(messages, context).await,
            CompressionLevel::Heavy => self.heavy_compression(messages).await,
        }
    }

    async fn light_compression(&self, mut messages: Vec<LlmMessage>) -> Result<Vec<LlmMessage>> {
        for message in &mut messages {
            if let MessageContent::MultiModal(blocks) = &mut message.content {
                for block in blocks {
                    if let ContentBlock::ToolResult { content, .. } = block {
                        if content.len() > self.tool_output_budget as usize {
                            *content = self.compress_tool_output(content).await?;
                        }
                    }
                }
            }
        }

        Ok(messages)
    }

    async fn medium_compression(
        &self,
        messages: Vec<LlmMessage>,
        context: Option<&AgentExecutionContext>,
    ) -> Result<Vec<LlmMessage>> {
        if messages.is_empty() {
            return Ok(messages);
        }

        // Separate system messages from conversation
        let (system_messages, conversation_messages): (Vec<_>, Vec<_>) = messages
            .into_iter()
            .partition(|msg| matches!(msg.role, MessageRole::System));

        if conversation_messages.is_empty() {
            return Ok(system_messages);
        }

        // Calculate how much to preserve
        let preserve_count = std::cmp::max(
            (conversation_messages.len() as f64 * 0.3).ceil() as usize,
            self.preserve_recent_pairs as usize * 2,
        );

        // Use tool-call-aware splitting
        let (to_compress, to_preserve) =
            self.split_preserving_tool_pairs(&conversation_messages, preserve_count);

        if to_compress.is_empty() {
            let mut result = system_messages;
            result.extend(to_preserve);
            return self.light_compression(result).await;
        }

        // Generate summary
        let summary = self.generate_summary(&to_compress, context).await?;
        let summary_msg =
            LlmMessage::system(format!("[Previous conversation summary]: {}", summary));

        // Reconstruct conversation
        let mut result = system_messages;
        result.push(summary_msg);
        result.extend(to_preserve);

        // Apply light compression to preserved messages
        self.light_compression(result).await
    }

    async fn heavy_compression(&self, messages: Vec<LlmMessage>) -> Result<Vec<LlmMessage>> {
        if messages.is_empty() {
            return Ok(messages);
        }

        let mut result = Vec::new();

        // Preserve all system messages
        for message in &messages {
            if matches!(message.role, MessageRole::System) {
                result.push(message.clone());
            }
        }

        // Keep only recent conversation with tool call preservation
        let conversation_messages: Vec<_> = messages
            .iter()
            .filter(|msg| !matches!(msg.role, MessageRole::System))
            .cloned()
            .collect();

        let keep_count = std::cmp::min(
            self.preserve_recent_pairs as usize * 2,
            conversation_messages.len(),
        );
        let (_, to_keep) = self.split_preserving_tool_pairs(&conversation_messages, keep_count);

        result.extend(to_keep);

        // Apply light compression to remaining content
        self.light_compression(result).await
    }

    async fn compress_tool_output(&self, output: &str) -> Result<String> {
        if output.len() <= self.tool_output_budget as usize {
            return Ok(output.to_string());
        }

        // Take sample from beginning and end
        let sample_size = (self.tool_output_budget as usize / 2).min(2000);
        let half_sample = sample_size / 2;

        // Safe UTF-8 character boundary slicing
        let beginning = if output.len() > half_sample {
            let mut boundary = half_sample;
            while boundary < output.len() && !output.is_char_boundary(boundary) {
                boundary += 1;
            }
            &output[..boundary]
        } else {
            output
        };

        let ending = if output.len() > sample_size {
            let start_pos = output.len().saturating_sub(half_sample);
            let mut boundary = start_pos;
            while boundary > 0 && !output.is_char_boundary(boundary) {
                boundary -= 1;
            }
            &output[boundary..]
        } else {
            ""
        };

        let sample_text = if ending.is_empty() {
            beginning.to_string()
        } else {
            format!("{}\\n[...]\\n{}", beginning, ending)
        };

        // Generate summary using LLM
        let summary_prompt = format!(
            "Please summarize the following tool output in 2-3 sentences, preserving any error messages and key results:\\n\\n{}",
            sample_text
        );

        let response = self
            .llm_client
            .chat_completion(
                vec![LlmMessage::user(summary_prompt)],
                None,
                Some(ChatOptions {
                    max_tokens: Some(200),
                    temperature: Some(0.3),
                    ..Default::default()
                }),
            )
            .await;

        match response {
            Ok(response) => {
                if let Some(summary) = response.message.get_text() {
                    Ok(format!("[Compressed output]: {}", summary))
                } else {
                    // Fallback to truncation
                    Ok(format!(
                        "{}...[truncated {} chars]",
                        &output[..sample_size.min(output.len())],
                        output.len().saturating_sub(sample_size)
                    ))
                }
            }
            Err(_e) => {
                // Fallback to truncation
                Ok(format!(
                    "{}...[truncated {} chars]",
                    &output[..sample_size.min(output.len())],
                    output.len().saturating_sub(sample_size)
                ))
            }
        }
    }

    async fn generate_summary(
        &self,
        messages: &[LlmMessage],
        context: Option<&AgentExecutionContext>,
    ) -> Result<String> {
        if messages.is_empty() {
            let goal = context
                .map(|c| c.original_goal.as_str())
                .unwrap_or("Empty conversation");
            return Ok(format!(
                "<state_snapshot><overall_goal>{}</overall_goal><key_knowledge>No prior knowledge</key_knowledge><file_system_state>No file operations</file_system_state><recent_actions>No actions taken</recent_actions><current_plan>No active plan</current_plan></state_snapshot>",
                goal
            ));
        }

        // Create conversation text representation
        let conversation_text = messages
            .iter()
            .filter_map(|msg| {
                let role_label = match msg.role {
                    MessageRole::User => "User",
                    MessageRole::Assistant => "Assistant",
                    MessageRole::Tool => "Tool",
                    MessageRole::System => "System",
                };

                msg.get_text().map(|content| {
                    let truncated = if matches!(msg.role, MessageRole::Tool) {
                        if content.len() > 2000 {
                            format!("{}...[truncated]", &content[..2000])
                        } else {
                            content
                        }
                    } else if content.len() > 800 {
                        format!("{}...", &content[..800])
                    } else {
                        content
                    };
                    format!("{}: {}", role_label, truncated)
                })
            })
            .collect::<Vec<_>>()
            .join("\\n");

        let limited_text = if conversation_text.len() > 15000 {
            format!("{}...[conversation continues]", &conversation_text[..15000])
        } else {
            conversation_text
        };

        // Create user request with context if available
        let user_request = if let Some(ctx) = context {
            format!(
                "ORIGINAL GOAL: {}\\nCURRENT TASK: {}\\nPROJECT PATH: {}\\n\\nFirst, reason in your scratchpad. Then, generate the <state_snapshot>. Use the ORIGINAL GOAL for the <overall_goal> field.\\n\\nConversation history:\\n{}",
                ctx.original_goal,
                ctx.current_task,
                ctx.project_path,
                limited_text
            )
        } else {
            format!(
                "First, reason in your scratchpad. Then, generate the <state_snapshot>.\\n\\nConversation history:\\n{}",
                limited_text
            )
        };

        let system_prompt = r#"You are the component that summarizes internal chat history into a given structure.

When the conversation history grows too large, you will be invoked to distill the entire history into a concise, structured XML snapshot. This snapshot is CRITICAL, as it will become the agent's *only* memory of the past. The agent will resume its work based solely on this snapshot. All crucial details, plans, errors, and user directives MUST be preserved.

First, you will think through the entire history in a private <scratchpad>. Review the user's overall goal, the agent's actions, tool outputs, file modifications, and any unresolved questions. Identify every piece of information that is essential for future actions.

After your reasoning is complete, generate the final <state_snapshot> XML object. Be incredibly dense with information. Omit any irrelevant conversational filler.

The structure MUST be as follows:

<state_snapshot>
    <overall_goal>
        <!-- A single, concise sentence describing the user's high-level objective. -->
        <!-- Example: "Refactor the authentication service to use a new JWT library." -->
    </overall_goal>

    <key_knowledge>
        <!-- Crucial facts, conventions, and constraints the agent must remember based on the conversation history and interaction with the user. Use bullet points. -->
        <!-- Example:
         - Build Command: `npm run build`
         - Testing: Tests are run with `npm test`. Test files must end in `.test.ts`.
         - API Endpoint: The primary API endpoint is `https://api.example.com/v2`.
         -->
    </key_knowledge>

    <file_system_state>
        <!-- List files that have been created, read, modified, or deleted. Note their status and critical learnings. -->
        <!-- Example:
         - CWD: `/home/user/project/src`
         - READ: `package.json` - Confirmed 'axios' is a dependency.
         - MODIFIED: `services/auth.ts` - Replaced 'jsonwebtoken' with 'jose'.
         - CREATED: `tests/new-feature.test.ts` - Initial test structure for the new feature.
        -->
    </file_system_state>

    <recent_actions>
        <!-- A summary of the last few significant agent actions and their outcomes. Focus on facts. -->
        <!-- Example:
         - Ran `grep 'old_function'` which returned 3 results in 2 files.
         - Ran `npm run test`, which failed due to a snapshot mismatch in `UserProfile.test.ts`.
         - Ran `ls -F static/` and discovered image assets are stored as `.webp`.
        -->
    </recent_actions>

    <current_plan>
        <!-- The agent's step-by-step plan. Mark completed steps. -->
        <!-- Example:
         1. [DONE] Identify all files using the deprecated 'UserAPI'.
         2. [IN PROGRESS] Refactor `src/components/UserProfile.tsx` to use the new 'ProfileAPI'.
         3. [TODO] Refactor the remaining files.
         4. [TODO] Update tests to reflect the API change.
        -->
    </current_plan>
</state_snapshot>"#;

        let messages_for_llm = vec![
            LlmMessage::system(system_prompt),
            LlmMessage::user(user_request),
        ];

        let response = self
            .llm_client
            .chat_completion(
                messages_for_llm,
                None,
                Some(ChatOptions {
                    max_tokens: Some(std::cmp::max(self.max_summary_tokens, 1000)),
                    temperature: Some(0.1),
                    ..Default::default()
                }),
            )
            .await;

        match response {
            Ok(response) => {
                if let Some(summary) = response.message.get_text() {
                    // Extract XML state_snapshot if present
                    if let Some(start) = summary.find("<state_snapshot>") {
                        if let Some(end) = summary.find("</state_snapshot>") {
                            let xml_summary = &summary[start..end + "</state_snapshot>".len()];
                            Ok(xml_summary.to_string())
                        } else {
                            Ok(summary)
                        }
                    } else {
                        // No XML found, wrap in basic structure
                        let fallback_goal = context
                            .map(|c| c.original_goal.as_str())
                            .unwrap_or("Conversation summary");
                        Ok(format!(
                            "<state_snapshot><overall_goal>{}</overall_goal><key_knowledge>{}</key_knowledge><file_system_state>Unable to extract file system state</file_system_state><recent_actions>Unable to extract recent actions</recent_actions><current_plan>Unable to extract plan</current_plan></state_snapshot>",
                            fallback_goal,
                            summary.chars().take(500).collect::<String>()
                        ))
                    }
                } else {
                    let fallback_goal = context
                        .map(|c| c.original_goal.as_str())
                        .unwrap_or("Unable to generate summary");
                    Ok(format!(
                        "<state_snapshot><overall_goal>{}</overall_goal><key_knowledge>Summary generation failed</key_knowledge><file_system_state>Unknown</file_system_state><recent_actions>Unknown</recent_actions><current_plan>Unknown</current_plan></state_snapshot>",
                        fallback_goal
                    ))
                }
            }
            Err(e) => {
                let fallback_goal = context
                    .map(|c| c.original_goal.as_str())
                    .unwrap_or("Summary generation error");
                Ok(format!(
                    "<state_snapshot><overall_goal>{}</overall_goal><key_knowledge>Error: {}</key_knowledge><file_system_state>Unknown</file_system_state><recent_actions>Unknown</recent_actions><current_plan>Unknown</current_plan></state_snapshot>",
                    fallback_goal, e
                ))
            }
        }
    }

    fn split_preserving_tool_pairs(
        &self,
        messages: &[LlmMessage],
        preserve_count: usize,
    ) -> (Vec<LlmMessage>, Vec<LlmMessage>) {
        if messages.len() <= preserve_count {
            return (Vec::new(), messages.to_vec());
        }

        // Find safe split point that doesn't break tool call pairs
        let mut split_index = messages.len().saturating_sub(preserve_count);

        while split_index > 0 && split_index < messages.len() {
            // Check if we're splitting right after an assistant message with tool calls
            if split_index > 0 {
                let prev_message = &messages[split_index - 1];
                if matches!(prev_message.role, MessageRole::Assistant)
                    && prev_message.has_tool_use()
                {
                    let current_message = &messages[split_index];
                    if matches!(current_message.role, MessageRole::Tool) {
                        // Move split point earlier to avoid breaking the pair
                        split_index = split_index.saturating_sub(1);
                        continue;
                    }
                }
            }
            break;
        }

        let (to_compress, to_preserve) = messages.split_at(split_index);
        (to_compress.to_vec(), to_preserve.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{FinishReason, LlmResponse, Usage};
    use async_trait::async_trait;
    use std::sync::Mutex;

    struct MockLlmClient {
        responses: Mutex<Vec<String>>,
    }

    impl MockLlmClient {
        fn new(responses: Vec<String>) -> Self {
            Self {
                responses: Mutex::new(responses),
            }
        }
    }

    #[async_trait]
    impl LlmClient for MockLlmClient {
        async fn chat_completion(
            &self,
            _messages: Vec<LlmMessage>,
            _tools: Option<Vec<crate::llm::ToolDefinition>>,
            _options: Option<ChatOptions>,
        ) -> Result<LlmResponse> {
            let mut responses = self.responses.lock().unwrap();
            let response_text = responses
                .pop()
                .unwrap_or_else(|| "Mock response".to_string());

            Ok(LlmResponse {
                message: LlmMessage::assistant(response_text),
                usage: Some(Usage {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                    total_tokens: 15,
                }),
                model: "mock-model".to_string(),
                finish_reason: Some(FinishReason::Stop),
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

    #[tokio::test]
    async fn test_no_compression_needed() {
        let mock_client = Arc::new(MockLlmClient::new(vec![]));
        let mut manager = ConversationManager::new(10000, mock_client);

        let messages = vec![
            LlmMessage::system("System message"),
            LlmMessage::user("Hello"),
            LlmMessage::assistant("Hi there!"),
        ];

        let result = manager
            .maybe_compress(messages.clone(), None)
            .await
            .unwrap();

        assert!(result.compression_applied.is_none());
        assert_eq!(result.messages.len(), messages.len());
    }

    #[tokio::test]
    async fn test_compression_applied() {
        let mock_client = Arc::new(MockLlmClient::new(vec![
            "Test conversation summary".to_string()
        ]));
        let mut manager = ConversationManager::new(100, mock_client); // Very low limit to force compression

        // Create many messages to trigger compression
        let mut messages = vec![LlmMessage::system("System message")];
        for i in 0..50 {
            messages.push(LlmMessage::user(format!("Message {}", i)));
            messages.push(LlmMessage::assistant(format!("Response {}", i)));
        }

        let result = manager.maybe_compress(messages, None).await.unwrap();

        assert!(result.compression_applied.is_some());
        let summary = result.compression_applied.unwrap();
        assert!(summary.tokens_saved > 0);
        assert!(result.messages.len() < 100); // Should be compressed
    }

    #[test]
    fn test_usage_ratio() {
        let mock_client = Arc::new(MockLlmClient::new(vec![]));
        let mut manager = ConversationManager::new(1000, mock_client);

        manager.current_tokens = 500;
        assert_eq!(manager.usage_ratio(), 0.5);

        manager.current_tokens = 750;
        assert_eq!(manager.usage_ratio(), 0.75);
    }
}
