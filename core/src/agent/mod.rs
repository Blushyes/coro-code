//! Agent core logic and execution engine

pub mod base;
pub mod config;
pub mod core;
pub mod execution;
pub mod prompt;
pub mod tokens;

pub use base::{Agent, AgentResult};
pub use config::{AgentBuilder, AgentConfig, OutputMode};
pub use core::AgentCore;
pub use execution::AgentExecution;
pub use prompt::{build_system_prompt_with_context, build_user_message, TRAE_AGENT_SYSTEM_PROMPT};
pub use tokens::{
    CompressionLevel, CompressionSummary, ConversationManager, ConversationTokenStats,
    MaybeCompressedResult, TokenCalculator,
};
