//! Token management and conversation compression system
//!
//! This module provides unified conversation management with automatic compression:
//! - **ConversationManager**: Single interface for all conversation token management
//! - **TokenCalculator**: Multi-language token calculation utilities
//!
//! ## Simple Usage
//! ```rust,no_run
//! use coro_core::agent::tokens::ConversationManager;
//! use std::sync::Arc;
//!
//! # async fn example() -> anyhow::Result<()> {
//! # let max_tokens = 8192u32;
//! # let llm_client: Arc<dyn coro_core::llm::LlmClient> = todo!();
//! # let messages = vec![];
//! # let context = None;
//! let mut manager = ConversationManager::new(max_tokens, llm_client);
//! let result = manager.maybe_compress(messages, context).await?;
//! // Use result.messages - compression is applied automatically if needed
//! # Ok(())
//! # }
//! ```

pub mod calculator;
pub mod conversation_manager;

// Public API
pub use calculator::{ConversationTokenStats, TokenCalculator};
pub use conversation_manager::{
    CompressionLevel, CompressionSummary, ConversationManager, MaybeCompressedResult,
};
