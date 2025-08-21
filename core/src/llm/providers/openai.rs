//! OpenAI client implementation using async-openai library

use crate::config::ResolvedLlmConfig;
use crate::error::{LlmError, Result};
use crate::llm::{
    ChatOptions, ContentBlock, FinishReason, LlmClient, LlmMessage, LlmResponse, LlmStreamChunk,
    MessageContent, MessageRole, ToolDefinition, Usage,
};
use crate::tools::ToolCall;
use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionMessageToolCall, ChatCompletionRequestAssistantMessage,
        ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessage, ChatCompletionRequestToolMessage,
        ChatCompletionRequestToolMessageContent, ChatCompletionRequestUserMessage,
        ChatCompletionTool, ChatCompletionToolType, CreateChatCompletionRequestArgs,
        FunctionObject,
    },
    Client,
};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::Value;

/// OpenAI client using async-openai library
pub struct OpenAiClient {
    client: Client<OpenAIConfig>,
    model: String,
    // Store base URL to determine streaming compatibility at runtime
    #[allow(dead_code)]
    base_url: String,
    #[allow(dead_code)]
    headers: std::collections::HashMap<String, String>,
}

impl OpenAiClient {
    /// Create a new OpenAI client from resolved LLM config
    pub fn new(config: &ResolvedLlmConfig) -> Result<Self> {
        if config.api_key.is_empty() {
            return Err(crate::error::Error::Llm(LlmError::Authentication {
                message: "No API key found for OpenAI".to_string(),
            }));
        }

        let mut openai_config = OpenAIConfig::new().with_api_key(&config.api_key);

        // Set custom base URL if provided
        let base_url = &config.base_url;
        if base_url != "https://api.openai.com" {
            openai_config = openai_config.with_api_base(base_url);
        }

        let client = Client::with_config(openai_config);

        Ok(Self {
            client,
            model: config.model.clone(),
            base_url: base_url.clone(),
            headers: config.headers.clone(),
        })
    }

    /// Convert our internal message format to async-openai format
    fn convert_messages(
        &self,
        messages: Vec<LlmMessage>,
    ) -> Result<Vec<ChatCompletionRequestMessage>> {
        let mut converted = Vec::new();

        for message in messages {
            match message.role {
                MessageRole::System => {
                    let content = self.extract_text_content(&message.content)?;
                    converted.push(ChatCompletionRequestMessage::System(
                        ChatCompletionRequestSystemMessage {
                            content: content.into(),
                            name: None,
                        },
                    ));
                }
                MessageRole::User => {
                    let content = self.extract_text_content(&message.content)?;
                    converted.push(ChatCompletionRequestMessage::User(
                        ChatCompletionRequestUserMessage {
                            content: content.into(),
                            name: None,
                        },
                    ));
                }
                MessageRole::Assistant => {
                    match &message.content {
                        MessageContent::Text(text) => {
                            converted.push(ChatCompletionRequestMessage::Assistant(
                                ChatCompletionRequestAssistantMessage {
                                    content: Some(
                                        ChatCompletionRequestAssistantMessageContent::Text(
                                            text.clone(),
                                        ),
                                    ),
                                    name: None,
                                    tool_calls: None,
                                    audio: None,
                                    refusal: None,
                                    ..Default::default()
                                },
                            ));
                        }
                        MessageContent::MultiModal(blocks) => {
                            let mut content = String::new();
                            let mut tool_calls = Vec::new();

                            for block in blocks {
                                match block {
                                    ContentBlock::Text { text } => {
                                        if !content.is_empty() {
                                            content.push('\n');
                                        }
                                        content.push_str(text);
                                    }
                                    ContentBlock::ToolUse { id, name, input } => {
                                        tool_calls.push(ChatCompletionMessageToolCall {
                                            id: id.clone(),
                                            r#type: ChatCompletionToolType::Function,
                                            function: async_openai::types::FunctionCall {
                                                name: name.clone(),
                                                arguments: input.to_string(),
                                            },
                                        });
                                    }
                                    _ => {} // Skip other types for now
                                }
                            }

                            converted.push(ChatCompletionRequestMessage::Assistant(
                                ChatCompletionRequestAssistantMessage {
                                    content: if content.is_empty() {
                                        None
                                    } else {
                                        Some(ChatCompletionRequestAssistantMessageContent::Text(
                                            content,
                                        ))
                                    },
                                    name: None,
                                    tool_calls: if tool_calls.is_empty() {
                                        None
                                    } else {
                                        Some(tool_calls)
                                    },
                                    audio: None,
                                    refusal: None,
                                    ..Default::default()
                                },
                            ));
                        }
                    }
                }
                MessageRole::Tool => {
                    // Push tool result message(s) without dropping other context
                    let mut pushed_any = false;
                    if let MessageContent::MultiModal(blocks) = &message.content {
                        for block in blocks {
                            if let ContentBlock::ToolResult {
                                tool_use_id,
                                content,
                                ..
                            } = block
                            {
                                converted.push(ChatCompletionRequestMessage::Tool(
                                    ChatCompletionRequestToolMessage {
                                        content: ChatCompletionRequestToolMessageContent::Text(
                                            content.clone(),
                                        ),
                                        tool_call_id: tool_use_id.clone(),
                                    },
                                ));
                                pushed_any = true;
                            }
                        }
                    }
                    if !pushed_any {
                        return Err((LlmError::InvalidRequest {
                            message: "Tool message must contain ToolResult".to_string(),
                        })
                        .into());
                    }
                }
            }
        }

        Ok(converted)
    }

    /// Extract text content from MessageContent
    fn extract_text_content(&self, content: &MessageContent) -> Result<String> {
        match content {
            MessageContent::Text(text) => Ok(text.clone()),
            MessageContent::MultiModal(blocks) => {
                let mut text_parts = Vec::new();
                for block in blocks {
                    if let ContentBlock::Text { text } = block {
                        text_parts.push(text.clone());
                    }
                }
                Ok(text_parts.join("\n"))
            }
        }
    }

    /// Convert our tool definitions to async-openai format
    fn convert_tools(&self, tools: Vec<ToolDefinition>) -> Vec<ChatCompletionTool> {
        tools
            .into_iter()
            .map(|tool| ChatCompletionTool {
                r#type: ChatCompletionToolType::Function,
                function: FunctionObject {
                    name: tool.function.name,
                    description: Some(tool.function.description),
                    parameters: Some(tool.function.parameters),
                    strict: None,
                },
            })
            .collect()
    }
}

#[async_trait]
impl LlmClient for OpenAiClient {
    async fn chat_completion(
        &self,
        messages: Vec<LlmMessage>,
        tools: Option<Vec<ToolDefinition>>,
        options: Option<ChatOptions>,
    ) -> Result<LlmResponse> {
        let converted_messages = self.convert_messages(messages)?;
        let converted_tools = tools.map(|t| self.convert_tools(t));

        // Log tool usage - important for debugging tool calls
        if let Some(ref tools) = converted_tools {
            tracing::debug!("OpenAI request with {} tools enabled", tools.len());
        }

        let mut request_builder = CreateChatCompletionRequestArgs::default();
        request_builder.model(&self.model);
        request_builder.messages(converted_messages);

        if let Some(tools) = converted_tools {
            request_builder.tools(tools);
        }

        if let Some(opts) = options {
            if let Some(max_tokens) = opts.max_tokens {
                request_builder.max_tokens(max_tokens);
            }
            if let Some(temperature) = opts.temperature {
                request_builder.temperature(temperature);
            }
            if let Some(top_p) = opts.top_p {
                request_builder.top_p(top_p);
            }
        }

        let request = request_builder.build().map_err(|e| {
            tracing::error!("Failed to build OpenAI request: {}", e);
            LlmError::InvalidRequest {
                message: format!("Failed to build request: {}", e),
            }
        })?;

        let response = self.client.chat().create(request).await.map_err(|e| {
            tracing::error!("OpenAI API call failed: {}", e);
            LlmError::ApiError {
                status: 500, // async-openai doesn't expose status codes directly
                message: e.to_string(),
            }
        })?;

        let result = self.convert_response(response);
        match &result {
            Ok(response) => {
                // Log tool usage in response - critical for debugging tool calls
                if let MessageContent::MultiModal(blocks) = &response.message.content {
                    let tool_use_count = blocks
                        .iter()
                        .filter(|block: &&ContentBlock| {
                            matches!(block, ContentBlock::ToolUse { .. })
                        })
                        .count();
                    if tool_use_count > 0 {
                        tracing::debug!("OpenAI response contains {} tool calls", tool_use_count);
                        // Log tool call details
                        for block in blocks {
                            if let ContentBlock::ToolUse { id, name, .. } = block {
                                tracing::debug!("Tool call: {} (id: {})", name, id);
                            }
                        }
                    }
                }

                // Log finish reason if it's tool-related
                if let Some(FinishReason::ToolCalls) = response.finish_reason {
                    tracing::debug!("OpenAI response finished due to tool calls");
                }
            }
            Err(e) => {
                tracing::error!("Failed to convert OpenAI response: {}", e);
            }
        }

        result
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn provider_name(&self) -> &str {
        "openai"
    }

    fn supports_streaming(&self) -> bool {
        // Streaming is disabled for simplicity
        false
    }

    async fn chat_completion_stream(
        &self,
        messages: Vec<LlmMessage>,
        tools: Option<Vec<ToolDefinition>>,
        options: Option<ChatOptions>,
    ) -> Result<Box<dyn futures::Stream<Item = Result<LlmStreamChunk>> + Send + Unpin + '_>> {
        let converted_messages = self.convert_messages(messages)?;
        let converted_tools: Option<Vec<ChatCompletionTool>> = tools.map(|t| self.convert_tools(t));

        let mut request_builder = CreateChatCompletionRequestArgs::default();
        request_builder.model(&self.model);
        request_builder.messages(converted_messages);
        request_builder.stream(true);

        if let Some(tools) = converted_tools {
            request_builder.tools(tools);
        }

        if let Some(opts) = options {
            if let Some(max_tokens) = opts.max_tokens {
                request_builder.max_tokens(max_tokens);
            }
            if let Some(temperature) = opts.temperature {
                request_builder.temperature(temperature);
            }
            if let Some(top_p) = opts.top_p {
                request_builder.top_p(top_p);
            }
        }

        let request = request_builder
            .build()
            .map_err(|e| LlmError::InvalidRequest {
                message: format!("Failed to build request: {}", e),
            })?;

        let stream = self
            .client
            .chat()
            .create_stream(request)
            .await
            .map_err(|e| LlmError::ApiError {
                status: 500,
                message: e.to_string(),
            })?;

        let converted_stream = stream.map(|result| match result {
            Ok(chunk) => self.convert_stream_chunk(chunk),
            Err(e) => Err((LlmError::ApiError {
                status: 500,
                message: e.to_string(),
            })
            .into()),
        });

        Ok(Box::new(Box::pin(converted_stream)))
    }
}

impl OpenAiClient {
    /// Convert async-openai response to our internal format
    fn convert_response(
        &self,
        response: async_openai::types::CreateChatCompletionResponse,
    ) -> Result<LlmResponse> {
        let choice =
            response
                .choices
                .into_iter()
                .next()
                .ok_or_else(|| LlmError::InvalidRequest {
                    message: "No choices in response".to_string(),
                })?;

        let message_content = if let Some(content) = choice.message.content {
            if let Some(tool_calls) = choice.message.tool_calls {
                // Multi-modal content with text and tool calls
                let mut blocks = vec![ContentBlock::Text { text: content }];

                for tool_call in tool_calls {
                    let function = &tool_call.function;
                    let args: Value = serde_json::from_str(&function.arguments)
                        .unwrap_or_else(|_| Value::String(function.arguments.clone()));

                    blocks.push(ContentBlock::ToolUse {
                        id: tool_call.id,
                        name: function.name.clone(),
                        input: args,
                    });
                }

                MessageContent::MultiModal(blocks)
            } else {
                MessageContent::Text(content)
            }
        } else if let Some(tool_calls) = choice.message.tool_calls {
            // Only tool calls, no text content
            let mut blocks = Vec::new();

            for tool_call in tool_calls {
                let function = &tool_call.function;
                let args: Value = serde_json::from_str(&function.arguments)
                    .unwrap_or_else(|_| Value::String(function.arguments.clone()));

                blocks.push(ContentBlock::ToolUse {
                    id: tool_call.id,
                    name: function.name.clone(),
                    input: args,
                });
            }

            MessageContent::MultiModal(blocks)
        } else {
            MessageContent::Text(String::new())
        };

        let message = LlmMessage {
            role: MessageRole::Assistant,
            content: message_content,
            metadata: None,
        };

        let usage = response.usage.map(|u| Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        let finish_reason = choice.finish_reason.map(|reason| match reason {
            async_openai::types::FinishReason::Stop => FinishReason::Stop,
            async_openai::types::FinishReason::Length => FinishReason::Length,
            async_openai::types::FinishReason::ToolCalls => FinishReason::ToolCalls,
            async_openai::types::FinishReason::ContentFilter => FinishReason::ContentFilter,
            async_openai::types::FinishReason::FunctionCall => FinishReason::ToolCalls,
        });

        Ok(LlmResponse {
            message,
            usage,
            model: response.model,
            finish_reason,
            metadata: None,
        })
    }

    /// Convert async-openai stream chunk to our internal format
    fn convert_stream_chunk(
        &self,
        chunk: async_openai::types::CreateChatCompletionStreamResponse,
    ) -> Result<LlmStreamChunk> {
        let choice = chunk.choices.into_iter().next();

        let delta = choice.as_ref().and_then(|c| c.delta.content.clone());

        let tool_calls = choice
            .as_ref()
            .and_then(|c| {
                c.delta.tool_calls.as_ref().map(|tool_calls| {
                    tool_calls
                        .iter()
                        .map(|tool_call| {
                            // For streaming, we need to handle partial tool calls
                            // Don't try to parse JSON here, just pass the raw data
                            let id = tool_call.id.as_deref().unwrap_or("").to_string();
                            let name = tool_call
                                .function
                                .as_ref()
                                .and_then(|f| f.name.as_deref())
                                .unwrap_or("")
                                .to_string();
                            let arguments = tool_call
                                .function
                                .as_ref()
                                .and_then(|f| f.arguments.as_deref())
                                .unwrap_or("")
                                .to_string();

                            // Return raw tool call data for accumulation
                            ToolCall {
                                id,
                                name,
                                parameters: Value::String(arguments),
                                metadata: None,
                            }
                        })
                        .collect::<Vec<_>>()
                })
            })
            .filter(|calls| !calls.is_empty());

        let finish_reason = choice.and_then(|c| {
            c.finish_reason.map(|reason| match reason {
                async_openai::types::FinishReason::Stop => FinishReason::Stop,
                async_openai::types::FinishReason::Length => FinishReason::Length,
                async_openai::types::FinishReason::ToolCalls => FinishReason::ToolCalls,
                async_openai::types::FinishReason::ContentFilter => FinishReason::ContentFilter,
                async_openai::types::FinishReason::FunctionCall => FinishReason::ToolCalls,
            })
        });

        let usage = chunk.usage.map(|u| Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        Ok(LlmStreamChunk {
            delta,
            tool_calls,
            finish_reason,
            usage,
        })
    }
}
