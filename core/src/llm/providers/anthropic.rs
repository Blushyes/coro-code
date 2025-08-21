//! Anthropic Claude client implementation

use crate::config::ResolvedLlmConfig;
use crate::error::{LlmError, Result};
use crate::llm::{
    ChatOptions, FinishReason, LlmClient, LlmMessage, LlmResponse, LlmStreamChunk, MessageRole,
    ToolDefinition, Usage,
};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Anthropic Claude client
pub struct AnthropicClient {
    client: Client,
    api_key: String,
    base_url: String,
    model: String,
    #[allow(dead_code)]
    headers: std::collections::HashMap<String, String>,
}

impl AnthropicClient {
    /// Create a new Anthropic client from resolved LLM config
    pub fn new(config: &ResolvedLlmConfig) -> Result<Self> {
        if config.api_key.is_empty() {
            return Err(crate::error::Error::Llm(LlmError::Authentication {
                message: "No API key found for Anthropic".to_string(),
            }));
        }

        let client = Client::new();

        Ok(Self {
            client,
            api_key: config.api_key.clone(),
            base_url: config.base_url.clone(),
            model: config.model.clone(),
            headers: config.headers.clone(),
        })
    }
}

#[async_trait]
impl LlmClient for AnthropicClient {
    async fn chat_completion(
        &self,
        messages: Vec<LlmMessage>,
        tools: Option<Vec<ToolDefinition>>,
        options: Option<ChatOptions>,
    ) -> Result<LlmResponse> {
        let request = self.build_request(messages, tools, options)?;

        let response = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::Network {
                message: e.to_string(),
            })?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            return Err((LlmError::ApiError {
                status,
                message: error_text,
            })
            .into());
        }

        let anthropic_response: AnthropicResponse =
            response.json().await.map_err(|e| LlmError::Network {
                message: format!("Failed to parse response: {}", e),
            })?;

        Ok(self.convert_response(anthropic_response))
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn provider_name(&self) -> &str {
        "anthropic"
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    async fn chat_completion_stream(
        &self,
        _messages: Vec<LlmMessage>,
        _tools: Option<Vec<ToolDefinition>>,
        _options: Option<ChatOptions>,
    ) -> Result<Box<dyn futures::Stream<Item = Result<LlmStreamChunk>> + Send + Unpin + '_>> {
        // TODO: Implement streaming support
        Err((LlmError::InvalidRequest {
            message: "Streaming not yet implemented for Anthropic".to_string(),
        })
        .into())
    }
}

impl AnthropicClient {
    fn build_request(
        &self,
        messages: Vec<LlmMessage>,
        tools: Option<Vec<ToolDefinition>>,
        options: Option<ChatOptions>,
    ) -> Result<AnthropicRequest> {
        let options = options.unwrap_or_default();

        // Separate system messages from conversation messages
        let mut system_message = None;
        let mut conversation_messages = Vec::new();

        for message in messages {
            match message.role {
                MessageRole::System => {
                    if let Some(text) = message.get_text() {
                        system_message = Some(text);
                    }
                }
                _ => conversation_messages.push(message),
            }
        }

        let max_tokens = options.max_tokens.unwrap_or(4096);

        let temperature = options.temperature.unwrap_or(0.5);

        Ok(AnthropicRequest {
            model: self.model.clone(),
            max_tokens,
            temperature,
            system: system_message,
            messages: conversation_messages,
            tools: tools.map(|t| t.into_iter().map(|tool| tool.function).collect()),
            stop_sequences: options.stop,
        })
    }

    fn convert_response(&self, response: AnthropicResponse) -> LlmResponse {
        let message = LlmMessage::assistant(
            response
                .content
                .first()
                .map(|c| c.text.clone())
                .unwrap_or_default(),
        );

        let usage = response.usage.map(|u| Usage {
            prompt_tokens: u.input_tokens,
            completion_tokens: u.output_tokens,
            total_tokens: u.input_tokens + u.output_tokens,
        });

        let finish_reason = match response.stop_reason.as_str() {
            "end_turn" => Some(FinishReason::Stop),
            "max_tokens" => Some(FinishReason::Length),
            "tool_use" => Some(FinishReason::ToolCalls),
            _ => Some(FinishReason::Other(response.stop_reason)),
        };

        LlmResponse {
            message,
            usage,
            model: response.model,
            finish_reason,
            metadata: None,
        }
    }
}

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<LlmMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<crate::llm::FunctionDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    #[allow(dead_code)]
    id: String,
    model: String,
    #[serde(rename = "type")]
    #[allow(dead_code)]
    response_type: String,
    #[allow(dead_code)]
    role: String,
    content: Vec<AnthropicContent>,
    stop_reason: String,
    #[allow(dead_code)]
    stop_sequence: Option<String>,
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContent {
    #[serde(rename = "type")]
    #[allow(dead_code)]
    content_type: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}
