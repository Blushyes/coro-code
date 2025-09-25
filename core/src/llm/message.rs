//! LLM message structures

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a message in an LLM conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmMessage {
    /// Role of the message sender
    pub role: MessageRole,

    /// Content of the message
    pub content: MessageContent,

    /// Optional metadata
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Role of the message sender
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// System message (instructions)
    System,

    /// User message (human input)
    User,

    /// Assistant message (AI response)
    Assistant,

    /// Tool message (tool execution result)
    Tool,
}

/// Content of a message
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text content
    Text(String),

    /// Multi-modal content with text and other media
    MultiModal(Vec<ContentBlock>),
}

/// A block of content within a message
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Text content
    Text { text: String },

    /// Image content
    Image {
        /// Image data (base64 encoded)
        data: String,
        /// MIME type of the image
        mime_type: String,
    },

    /// Tool use request
    ToolUse {
        /// Unique identifier for this tool use
        id: String,
        /// Name of the tool to use
        name: String,
        /// Input parameters for the tool
        input: serde_json::Value,
    },

    /// Tool result
    ToolResult {
        /// ID of the tool use this is a result for
        tool_use_id: String,
        /// Whether the tool execution was successful
        is_error: Option<bool>,
        /// Result content
        content: String,
    },
}

impl LlmMessage {
    /// Create a new system message
    pub fn system<S: Into<String>>(content: S) -> Self {
        Self {
            role: MessageRole::System,
            content: MessageContent::Text(content.into()),
            metadata: None,
        }
    }

    /// Create a new user message
    pub fn user<S: Into<String>>(content: S) -> Self {
        Self {
            role: MessageRole::User,
            content: MessageContent::Text(content.into()),
            metadata: None,
        }
    }

    /// Create a new assistant message
    pub fn assistant<S: Into<String>>(content: S) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: MessageContent::Text(content.into()),
            metadata: None,
        }
    }

    /// Create a new tool message
    pub fn tool<S: Into<String>>(content: S) -> Self {
        Self {
            role: MessageRole::Tool,
            content: MessageContent::Text(content.into()),
            metadata: None,
        }
    }

    /// Get the text content of the message
    pub fn get_text(&self) -> Option<String> {
        match &self.content {
            MessageContent::Text(text) => Some(text.clone()),
            MessageContent::MultiModal(blocks) => {
                let mut text_parts = Vec::new();
                for block in blocks {
                    if let ContentBlock::Text { text } = block {
                        text_parts.push(text.clone());
                    }
                }
                if text_parts.is_empty() {
                    None
                } else {
                    Some(text_parts.join("\n"))
                }
            }
        }
    }

    /// Check if the message contains tool use
    pub fn has_tool_use(&self) -> bool {
        match &self.content {
            MessageContent::Text(_) => false,
            MessageContent::MultiModal(blocks) => blocks
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolUse { .. })),
        }
    }

    /// Extract tool use blocks from the message
    pub fn get_tool_uses(&self) -> Vec<&ContentBlock> {
        match &self.content {
            MessageContent::Text(_) => Vec::new(),
            MessageContent::MultiModal(blocks) => blocks
                .iter()
                .filter(|block| matches!(block, ContentBlock::ToolUse { .. }))
                .collect(),
        }
    }
}

impl From<String> for MessageContent {
    fn from(text: String) -> Self {
        MessageContent::Text(text)
    }
}

impl From<&str> for MessageContent {
    fn from(text: &str) -> Self {
        MessageContent::Text(text.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_message_role_serialization() {
        assert_eq!(
            serde_json::to_string(&MessageRole::System).unwrap(),
            "\"system\""
        );
        assert_eq!(
            serde_json::to_string(&MessageRole::User).unwrap(),
            "\"user\""
        );
        assert_eq!(
            serde_json::to_string(&MessageRole::Assistant).unwrap(),
            "\"assistant\""
        );
        assert_eq!(
            serde_json::to_string(&MessageRole::Tool).unwrap(),
            "\"tool\""
        );
    }

    #[test]
    fn test_llm_message_constructors() {
        let system_msg = LlmMessage::system("You are a helpful assistant");
        assert_eq!(system_msg.role, MessageRole::System);
        assert_eq!(system_msg.get_text(), Some("You are a helpful assistant".to_string()));

        let user_msg = LlmMessage::user("Hello, world!");
        assert_eq!(user_msg.role, MessageRole::User);
        assert_eq!(user_msg.get_text(), Some("Hello, world!".to_string()));

        let assistant_msg = LlmMessage::assistant("Hello! How can I help you?");
        assert_eq!(assistant_msg.role, MessageRole::Assistant);
        assert_eq!(assistant_msg.get_text(), Some("Hello! How can I help you?".to_string()));
    }

    #[test]
    fn test_get_text_multimodal() {
        let message = LlmMessage {
            role: MessageRole::Assistant,
            content: MessageContent::MultiModal(vec![
                ContentBlock::Text { text: "First part".to_string() },
                ContentBlock::Text { text: "Second part".to_string() },
            ]),
            metadata: None,
        };

        assert_eq!(message.get_text(), Some("First part\nSecond part".to_string()));
    }

    #[test]
    fn test_has_tool_use() {
        let simple_message = LlmMessage::user("Hello");
        assert!(!simple_message.has_tool_use());

        let tool_message = LlmMessage {
            role: MessageRole::Assistant,
            content: MessageContent::MultiModal(vec![
                ContentBlock::ToolUse {
                    id: "tool_1".to_string(),
                    name: "test".to_string(),
                    input: json!({}),
                },
            ]),
            metadata: None,
        };
        assert!(tool_message.has_tool_use());
    }

    #[test]
    fn test_content_block_serialization() {
        let text_block = ContentBlock::Text {
            text: "Hello".to_string(),
        };
        let serialized = serde_json::to_value(&text_block).unwrap();
        assert_eq!(serialized["type"], "text");
        assert_eq!(serialized["text"], "Hello");

        let tool_block = ContentBlock::ToolUse {
            id: "tool_123".to_string(),
            name: "test_tool".to_string(),
            input: json!({"param": "value"}),
        };
        let serialized = serde_json::to_value(&tool_block).unwrap();
        assert_eq!(serialized["type"], "tool_use");
        assert_eq!(serialized["id"], "tool_123");
        assert_eq!(serialized["name"], "test_tool");
    }
}
