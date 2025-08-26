//! Error types and handling for Coro Code Core

use thiserror::Error;

/// Result type alias for Coro Code operations
pub type Result<T> = std::result::Result<T, Error>;

/// Main error type for Coro Code Core
#[derive(Error, Debug)]
pub enum Error {
    /// Configuration-related errors
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    /// LLM client errors
    #[error("LLM error: {0}")]
    Llm(#[from] LlmError),

    /// Tool execution errors
    #[error("Tool error: {0}")]
    Tool(#[from] ToolError),

    /// Agent execution errors
    #[error("Agent error: {0}")]
    Agent(#[from] AgentError),

    /// Trajectory recording errors
    #[error("Trajectory error: {0}")]
    Trajectory(#[from] TrajectoryError),

    /// IO errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// HTTP request errors
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// SQLite database errors
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    /// Tree-sitter language errors
    #[error("Language error: {0}")]
    Language(#[from] tree_sitter::LanguageError),

    /// Timeout errors
    #[error("Timeout error: {0}")]
    Timeout(#[from] tokio::time::error::Elapsed),

    /// Generic error with message
    #[error("{0}")]
    Generic(String),
}

/// Configuration-specific errors
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Missing required field: {field}")]
    MissingField { field: String },

    #[error("Invalid value for field '{field}': {value}")]
    InvalidValue { field: String, value: String },

    #[error("File not found: {path}")]
    FileNotFound { path: String },

    #[error("Invalid configuration format")]
    InvalidFormat,

    #[error("No configuration found")]
    NoConfigFound,
}

/// LLM client errors
#[derive(Error, Debug)]
pub enum LlmError {
    #[error("Authentication failed: {message}")]
    Authentication { message: String },

    #[error("Rate limit exceeded")]
    RateLimit,

    #[error("Model not found: {model}")]
    ModelNotFound { model: String },

    #[error("Invalid request: {message}")]
    InvalidRequest { message: String },

    #[error("API error: {status} - {message}")]
    ApiError { status: u16, message: String },

    #[error("Network error: {message}")]
    Network { message: String },
}

/// Tool execution errors
#[derive(Error, Debug)]
pub enum ToolError {
    #[error("Tool not found: {name}")]
    NotFound { name: String },

    #[error("Tool execution failed: {name} - {message}")]
    ExecutionFailed { name: String, message: String },

    #[error("Invalid tool parameters: {message}")]
    InvalidParameters { message: String },

    #[error("Tool timeout: {name}")]
    Timeout { name: String },
}

/// Agent execution errors
#[derive(Error, Debug)]
pub enum AgentError {
    #[error("Maximum steps exceeded: {max_steps}")]
    MaxStepsExceeded { max_steps: usize },

    #[error("Task execution failed: {message}")]
    TaskFailed { message: String },

    #[error("Invalid task: {message}")]
    InvalidTask { message: String },

    #[error("Agent not initialized")]
    NotInitialized,
}

/// Trajectory recording errors
#[derive(Error, Debug)]
pub enum TrajectoryError {
    #[error("Failed to record trajectory: {message}")]
    RecordingFailed { message: String },

    #[error("Failed to load trajectory: {path}")]
    LoadFailed { path: String },

    #[error("Invalid trajectory format")]
    InvalidFormat,
}

impl From<String> for Error {
    fn from(msg: String) -> Self {
        Error::Generic(msg)
    }
}

impl From<&str> for Error {
    fn from(msg: &str) -> Self {
        Error::Generic(msg.to_string())
    }
}
