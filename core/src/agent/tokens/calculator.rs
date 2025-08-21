//! Token calculation utilities for conversation history management
//!
//! Provides estimation algorithms for calculating token usage in LLM conversations,
//! supporting multi-language content and different message types.

use crate::llm::{ContentBlock, LlmMessage, MessageContent, MessageRole};
use serde_json::Value;
use std::collections::HashMap;

/// Token calculator for estimating conversation token usage
pub struct TokenCalculator;

impl TokenCalculator {
    /// Estimate token count for a single message
    pub fn estimate_message_tokens(message: &LlmMessage) -> u32 {
        let content_tokens = match &message.content {
            MessageContent::Text(text) => Self::estimate_text_tokens(text),
            MessageContent::MultiModal(blocks) => {
                blocks.iter().map(Self::estimate_content_block_tokens).sum()
            }
        };

        // Add overhead for role and structure (approximately 3-5 tokens per message)
        let role_overhead = match message.role {
            MessageRole::System => 4,    // "system" role
            MessageRole::User => 3,      // "user" role
            MessageRole::Assistant => 5, // "assistant" role
            MessageRole::Tool => 4,      // "tool" role
        };

        content_tokens + role_overhead
    }

    /// Estimate token count for a content block
    fn estimate_content_block_tokens(block: &ContentBlock) -> u32 {
        match block {
            ContentBlock::Text { text } => Self::estimate_text_tokens(text),
            ContentBlock::Image { .. } => {
                // Images typically consume around 765-1024 tokens depending on size
                // Using conservative estimate
                1000
            }
            ContentBlock::ToolUse { name, input, .. } => {
                // Tool use overhead (name, id, structure) + input parameters
                let tool_overhead = 10;
                let input_tokens = Self::estimate_json_tokens(input);
                let name_tokens = Self::estimate_text_tokens(name);
                tool_overhead + input_tokens + name_tokens
            }
            ContentBlock::ToolResult { content, .. } => {
                // Tool result overhead + content
                let tool_result_overhead = 8;
                tool_result_overhead + Self::estimate_text_tokens(content)
            }
        }
    }

    /// Estimate token count for text content with language-aware calculation
    pub fn estimate_text_tokens(text: &str) -> u32 {
        if text.is_empty() {
            return 0;
        }

        let _char_count = text.chars().count() as u32;

        // Count different character types for better estimation
        let (cjk_chars, ascii_chars, other_chars) = Self::count_character_types(text);

        // Token estimation based on character types:
        // - CJK characters (Chinese, Japanese, Korean): ~2 characters per token
        // - ASCII letters/numbers: ~4 characters per token
        // - Other Unicode: ~3 characters per token
        let estimated_tokens =
            (cjk_chars as f64 / 2.0) + (ascii_chars as f64 / 4.0) + (other_chars as f64 / 3.0);

        // Minimum 1 token for non-empty text
        estimated_tokens.ceil().max(1.0) as u32
    }

    /// Count different types of characters for more accurate token estimation
    fn count_character_types(text: &str) -> (u32, u32, u32) {
        let mut cjk_chars = 0;
        let mut ascii_chars = 0;
        let mut other_chars = 0;

        for ch in text.chars() {
            if Self::is_cjk_character(ch) {
                cjk_chars += 1;
            } else if ch.is_ascii_alphanumeric()
                || ch.is_ascii_punctuation()
                || ch.is_ascii_whitespace()
            {
                ascii_chars += 1;
            } else {
                other_chars += 1;
            }
        }

        (cjk_chars, ascii_chars, other_chars)
    }

    /// Check if a character is CJK (Chinese, Japanese, Korean)
    fn is_cjk_character(ch: char) -> bool {
        let code = ch as u32;

        // Major CJK Unicode ranges
        (0x4E00..=0x9FFF).contains(&code) ||    // CJK Unified Ideographs
        (0x3400..=0x4DBF).contains(&code) ||    // CJK Extension A
        (0x20000..=0x2A6DF).contains(&code) ||  // CJK Extension B
        (0x2A700..=0x2B73F).contains(&code) ||  // CJK Extension C
        (0x2B740..=0x2B81F).contains(&code) ||  // CJK Extension D
        (0x2B820..=0x2CEAF).contains(&code) ||  // CJK Extension E
        (0x3040..=0x309F).contains(&code) ||    // Hiragana
        (0x30A0..=0x30FF).contains(&code) ||    // Katakana
        (0xAC00..=0xD7AF).contains(&code) // Hangul Syllables
    }

    /// Estimate tokens for JSON value
    pub fn estimate_json_tokens(json: &Value) -> u32 {
        match json {
            Value::Null => 1,
            Value::Bool(_) => 1,
            Value::Number(n) => {
                // Numbers vary in token count based on length
                let num_str = n.to_string();
                Self::estimate_text_tokens(&num_str)
            }
            Value::String(s) => {
                // Add overhead for JSON string quotes and escaping
                let string_overhead = 2;
                string_overhead + Self::estimate_text_tokens(s)
            }
            Value::Array(arr) => {
                // Array overhead + sum of elements
                let array_overhead = 2; // for brackets
                array_overhead + arr.iter().map(Self::estimate_json_tokens).sum::<u32>()
            }
            Value::Object(obj) => {
                // Object overhead + keys + values
                let object_overhead = 2; // for braces
                let content_tokens = obj
                    .iter()
                    .map(|(key, value)| {
                        let key_tokens = 1 + Self::estimate_text_tokens(key); // +1 for quotes
                        let value_tokens = Self::estimate_json_tokens(value);
                        key_tokens + value_tokens + 1 // +1 for colon
                    })
                    .sum::<u32>();
                object_overhead + content_tokens
            }
        }
    }

    /// Estimate total tokens for entire conversation history
    pub fn estimate_conversation_tokens(messages: &[LlmMessage]) -> u32 {
        messages.iter().map(Self::estimate_message_tokens).sum()
    }

    /// Calculate token usage statistics for a conversation
    pub fn calculate_conversation_stats(messages: &[LlmMessage]) -> ConversationTokenStats {
        let mut stats = ConversationTokenStats::default();

        for message in messages {
            let message_tokens = Self::estimate_message_tokens(message);
            stats.total_tokens += message_tokens;

            match message.role {
                MessageRole::System => {
                    stats.system_tokens += message_tokens;
                    stats.system_messages += 1;
                }
                MessageRole::User => {
                    stats.user_tokens += message_tokens;
                    stats.user_messages += 1;
                }
                MessageRole::Assistant => {
                    stats.assistant_tokens += message_tokens;
                    stats.assistant_messages += 1;
                }
                MessageRole::Tool => {
                    stats.tool_tokens += message_tokens;
                    stats.tool_messages += 1;
                }
            }
        }

        stats.total_messages = messages.len() as u32;
        stats
    }

    /// Estimate tokens for a batch of tool results (for compression planning)
    pub fn estimate_tool_results_tokens(results: &[&ContentBlock]) -> u32 {
        results
            .iter()
            .map(|block| Self::estimate_content_block_tokens(block))
            .sum()
    }

    /// Get rough token estimation for common model limits
    pub fn get_model_token_limits() -> HashMap<&'static str, u32> {
        let mut limits = HashMap::new();

        // Common model context limits
        limits.insert("gpt-4o", 128_000);
        limits.insert("gpt-4o-mini", 128_000);
        limits.insert("gpt-4-turbo", 128_000);
        limits.insert("gpt-4", 8_192);
        limits.insert("gpt-3.5-turbo", 16_385);
        limits.insert("claude-3-5-sonnet", 200_000);
        limits.insert("claude-3-5-haiku", 200_000);
        limits.insert("claude-3-opus", 200_000);
        limits.insert("gemini-1.5-pro", 2_097_152);
        limits.insert("gemini-1.5-flash", 1_048_576);

        limits
    }

    /// Estimate if adding content would exceed a token budget
    pub fn would_exceed_budget(current_tokens: u32, additional_content: &str, budget: u32) -> bool {
        let additional_tokens = Self::estimate_text_tokens(additional_content);
        current_tokens + additional_tokens > budget
    }
}

/// Statistics about token usage in a conversation
#[derive(Debug, Default, Clone)]
pub struct ConversationTokenStats {
    pub total_tokens: u32,
    pub total_messages: u32,
    pub system_tokens: u32,
    pub system_messages: u32,
    pub user_tokens: u32,
    pub user_messages: u32,
    pub assistant_tokens: u32,
    pub assistant_messages: u32,
    pub tool_tokens: u32,
    pub tool_messages: u32,
}

impl ConversationTokenStats {
    /// Calculate the percentage of tokens used by each role
    pub fn get_role_percentages(&self) -> HashMap<&'static str, f64> {
        let mut percentages = HashMap::new();

        if self.total_tokens == 0 {
            return percentages;
        }

        let total = self.total_tokens as f64;
        percentages.insert("system", (self.system_tokens as f64 / total) * 100.0);
        percentages.insert("user", (self.user_tokens as f64 / total) * 100.0);
        percentages.insert("assistant", (self.assistant_tokens as f64 / total) * 100.0);
        percentages.insert("tool", (self.tool_tokens as f64 / total) * 100.0);

        percentages
    }

    /// Get average tokens per message for each role
    pub fn get_average_tokens_per_message(&self) -> HashMap<&'static str, f64> {
        let mut averages = HashMap::new();

        if self.system_messages > 0 {
            averages.insert(
                "system",
                self.system_tokens as f64 / self.system_messages as f64,
            );
        }
        if self.user_messages > 0 {
            averages.insert("user", self.user_tokens as f64 / self.user_messages as f64);
        }
        if self.assistant_messages > 0 {
            averages.insert(
                "assistant",
                self.assistant_tokens as f64 / self.assistant_messages as f64,
            );
        }
        if self.tool_messages > 0 {
            averages.insert("tool", self.tool_tokens as f64 / self.tool_messages as f64);
        }

        averages
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_english_text_estimation() {
        // English text should be roughly 4 characters per token
        let text = "Hello world, this is a test message.";
        let tokens = TokenCalculator::estimate_text_tokens(text);

        // 36 characters / 4 ≈ 9 tokens
        assert!(
            (8..=12).contains(&tokens),
            "Expected ~9 tokens, got {}",
            tokens
        );
    }

    #[test]
    fn test_chinese_text_estimation() {
        // Chinese text should be roughly 2 characters per token
        let text = "你好世界，这是一个测试消息。";
        let tokens = TokenCalculator::estimate_text_tokens(text);

        // 14 characters / 2 ≈ 7 tokens
        assert!(
            (6..=9).contains(&tokens),
            "Expected ~7 tokens, got {}",
            tokens
        );
    }

    #[test]
    fn test_mixed_language_estimation() {
        let text = "Hello 世界! This is 测试 content.";
        let tokens = TokenCalculator::estimate_text_tokens(text);

        // Should handle mixed content appropriately
        assert!(tokens > 0, "Should estimate tokens for mixed content");
    }

    #[test]
    fn test_empty_text() {
        let tokens = TokenCalculator::estimate_text_tokens("");
        assert_eq!(tokens, 0);
    }

    #[test]
    fn test_json_estimation() {
        let json = json!({
            "name": "test",
            "count": 42,
            "items": ["a", "b", "c"],
            "enabled": true
        });

        let tokens = TokenCalculator::estimate_json_tokens(&json);
        assert!(tokens > 0, "Should estimate tokens for JSON");
    }

    #[test]
    fn test_message_estimation() {
        let message = LlmMessage::user("Hello, how are you?");
        let tokens = TokenCalculator::estimate_message_tokens(&message);

        // Should include content tokens + role overhead
        assert!(tokens > 4, "Should include role overhead");
    }

    #[test]
    fn test_conversation_stats() {
        let messages = vec![
            LlmMessage::system("You are a helpful assistant."),
            LlmMessage::user("Hello!"),
            LlmMessage::assistant("Hi there! How can I help?"),
        ];

        let stats = TokenCalculator::calculate_conversation_stats(&messages);

        assert_eq!(stats.total_messages, 3);
        assert_eq!(stats.system_messages, 1);
        assert_eq!(stats.user_messages, 1);
        assert_eq!(stats.assistant_messages, 1);
        assert!(stats.total_tokens > 0);
    }

    #[test]
    fn test_cjk_character_detection() {
        assert!(TokenCalculator::is_cjk_character('中'));
        assert!(TokenCalculator::is_cjk_character('あ'));
        assert!(TokenCalculator::is_cjk_character('한'));
        assert!(!TokenCalculator::is_cjk_character('A'));
        assert!(!TokenCalculator::is_cjk_character('1'));
    }

    #[test]
    fn test_character_type_counting() {
        let text = "Hello 世界 123!";
        let (cjk, ascii, _other) = TokenCalculator::count_character_types(text);

        assert_eq!(cjk, 2); // 世界
        assert!(ascii > 0); // Hello, 123, space, !
    }

    #[test]
    fn test_model_token_limits() {
        let limits = TokenCalculator::get_model_token_limits();

        assert!(limits.contains_key("gpt-4o"));
        assert!(limits.contains_key("claude-3-5-sonnet"));
        assert!(*limits.get("gpt-4o").unwrap() > 100_000);
    }

    #[test]
    fn test_budget_check() {
        let current = 1000;
        let content = "This is some additional content";
        let budget = 1050;

        // This should not exceed budget
        assert!(!TokenCalculator::would_exceed_budget(
            current, content, budget
        ));

        // This should exceed budget
        let small_budget = 1005;
        assert!(TokenCalculator::would_exceed_budget(
            current,
            content,
            small_budget
        ));
    }
}
