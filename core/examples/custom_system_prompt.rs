//! Example demonstrating how to use custom system prompts with coro-codeAgent
//!
//! This example shows three ways to set a custom system prompt:
//! 1. Through configuration file
//! 2. Through AgentConfig directly
//! 3. Dynamically at runtime

use coro_core::{
    agent::AgentCore, output::events::NullOutput, AgentConfig, OutputMode, Protocol,
    ResolvedLlmConfig,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== coro-codeAgent Custom System Prompt Example ===\n");

    // Method 1: Set system prompt through AgentConfig
    println!("1. Setting system prompt through AgentConfig:");
    let agent_config = AgentConfig {
        system_prompt: Some(
            "You are a specialized Rust programming assistant. \
             Focus on writing safe, efficient, and idiomatic Rust code. \
             Always explain memory safety considerations."
                .to_string(),
        ),
        ..Default::default()
    };

    // Create LLM configuration
    let llm_config = ResolvedLlmConfig::new(
        Protocol::OpenAICompat,
        "https://api.openai.com/v1".to_string(),
        "test-key".to_string(),
        "gpt-4o".to_string(),
    );

    // Create agent with custom system prompt
    let mut agent =
        AgentCore::new_with_llm_config(agent_config, llm_config, Box::new(NullOutput), None)
            .await?;

    // Verify the system prompt is set
    if let Some(prompt) = agent.get_configured_system_prompt() {
        println!("✓ Custom system prompt set: {}", prompt);
    } else {
        println!("✗ No custom system prompt found");
    }

    // Method 2: Dynamically change system prompt at runtime
    println!("\n2. Dynamically changing system prompt:");
    agent.set_system_prompt(Some(
        "You are now a Python expert assistant. \
         Focus on writing clean, Pythonic code following PEP 8 guidelines."
            .to_string(),
    ));

    if let Some(prompt) = agent.get_configured_system_prompt() {
        println!("✓ System prompt updated: {}", prompt);
    }

    // Method 3: Clear system prompt (use default)
    println!("\n3. Clearing system prompt (reverting to default):");
    agent.set_system_prompt(None);

    if agent.get_configured_system_prompt().is_none() {
        println!("✓ System prompt cleared, will use default CORO_AGENT_SYSTEM_PROMPT");
    }

    println!("\n=== Configuration Example ===");
    println!("You can also set system prompts in your configuration file:");
    println!(
        r#"
[agents.my_agent]
model = "claude-3-5-sonnet-20241022"
max_steps = 100
enable_lakeview = true
tools = ["bash", "str_replace_based_edit_tool"]
system_prompt = "You are a helpful coding assistant specialized in web development."
"#
    );

    println!("\n=== JSON Configuration Example ===");
    let example_config = AgentConfig {
        max_steps: 100,
        enable_lakeview: true,
        tools: vec![
            "bash".to_string(),
            "str_replace_based_edit_tool".to_string(),
        ],
        output_mode: OutputMode::Normal,
        system_prompt: Some("You are a specialized DevOps assistant.".to_string()),
    };

    let json = serde_json::to_string_pretty(&example_config)?;
    println!("{}", json);

    Ok(())
}
