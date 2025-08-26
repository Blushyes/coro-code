//! Tools listing command

use anyhow::Result;
use tracing::info;

/// Show available tools
pub async fn tools_command() -> Result<()> {
    info!("Listing available tools");

    use crate::tools::create_cli_tool_registry;

    println!("🛠️  Available Tools\n");

    let registry = create_cli_tool_registry();
    let tool_names = registry.list_tools();

    for name in tool_names {
        if let Some((tool_name, description)) = registry.get_tool_info(name) {
            println!("📦 {}", tool_name);
            // Show first line of description only for brevity
            let first_line = description.lines().next().unwrap_or(description);
            println!("   {}\n", first_line);
        }
    }

    println!("💡 Use these tools in your tasks to accomplish complex workflows!");
    println!(
        "📋 All tools follow the exact same specifications as the Python version of Coro Code."
    );

    Ok(())
}
