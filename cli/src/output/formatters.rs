//! Formatting utilities for CLI output

use coro_core::output::{ToolExecutionInfo, ToolExecutionStatus};
use std::path::Path;
use tracing::debug;

// ANSI color codes
const RED_BG: &str = "\x1b[41m"; // Red background
const GREEN_BG: &str = "\x1b[42m"; // Green background
const GRAY: &str = "\x1b[90m"; // Gray text for line numbers
const WHITE: &str = "\x1b[97m"; // White text for executing status
const GREEN: &str = "\x1b[92m"; // Green text for success status
const RED: &str = "\x1b[91m"; // Red text for error status
const BLACK: &str = "\x1b[30m"; // Black text for better contrast on colored backgrounds
const RESET: &str = "\x1b[0m";

/// Tool execution formatter
pub struct ToolFormatter;

impl ToolFormatter {
    pub fn new() -> Self {
        Self
    }

    /// Format tool execution status for CLI display
    pub fn format_tool_status(&self, tool_info: &ToolExecutionInfo) -> String {
        if tool_info.tool_name == "sequentialthinking" {
            let result = tool_info.result.as_ref().map_or("", |r| &r.content);
            debug!("result: {}", result);
            return String::from(result);
        }

        // Don't show status for task_done tool
        if tool_info.tool_name == "task_done" {
            return String::new();
        }

        let (dot_color, dot_char) = match tool_info.status {
            ToolExecutionStatus::Executing => (WHITE, "⏺"),
            ToolExecutionStatus::Success => (GREEN, "⏺"),
            ToolExecutionStatus::Error => (RED, "⏺"),
        };

        // Get friendly display name and command
        let display_name = self.get_tool_display_name(tool_info);
        let command = self.extract_tool_command(tool_info);

        if command.is_empty() {
            format!("{}{}{} {}", dot_color, dot_char, RESET, display_name)
        } else {
            format!(
                "{}{}{} {}({})",
                dot_color, dot_char, RESET, display_name, command
            )
        }
    }

    /// Get friendly display name for tools
    fn get_tool_display_name(&self, tool_info: &ToolExecutionInfo) -> String {
        match tool_info.tool_name.as_str() {
            "bash" => "Bash".to_string(),
            "str_replace_based_edit_tool" => {
                // Determine operation type based on parameters
                if tool_info.parameters.contains_key("file_text") {
                    "Create".to_string()
                } else if tool_info.parameters.contains_key("old_str") {
                    "Update".to_string()
                } else if tool_info.parameters.contains_key("view_range")
                    || tool_info.parameters.get("command").and_then(|v| v.as_str()) == Some("view")
                {
                    "Read".to_string()
                } else {
                    "Edit".to_string()
                }
            }
            "task_done" => "Complete".to_string(),
            "sequentialthinking" => "Think".to_string(),
            _ => {
                // For unknown tools, capitalize the first letter
                let name = tool_info.tool_name.as_str();
                if name.is_empty() {
                    "Tool".to_string()
                } else {
                    let mut chars = name.chars();
                    match chars.next() {
                        None => "Tool".to_string(),
                        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                    }
                }
            }
        }
    }

    /// Extract the main command/parameter from tool info for display
    fn extract_tool_command(&self, tool_info: &ToolExecutionInfo) -> String {
        let command = match tool_info.tool_name.as_str() {
            "bash" => tool_info
                .parameters
                .get("command")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            "str_replace_based_edit_tool" => tool_info
                .parameters
                .get("path")
                .and_then(|v| v.as_str())
                .map(|path| {
                    Path::new(path)
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or(path)
                        .to_string()
                })
                .unwrap_or_else(|| "file".to_string()),
            _ => {
                // For other tools, try to find a reasonable display parameter
                tool_info
                    .parameters
                    .get("path")
                    .or_else(|| tool_info.parameters.get("file"))
                    .or_else(|| tool_info.parameters.get("name"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string()
            }
        };

        // Truncate command if it's too long to prevent line wrapping
        // Assume terminal width of ~80 chars, leave space for tool name and formatting
        const MAX_COMMAND_LENGTH: usize = 60;
        if command.len() > MAX_COMMAND_LENGTH {
            format!("{}...", &command[..MAX_COMMAND_LENGTH - 3])
        } else {
            command
        }
    }

    /// Format tool result content for display
    pub fn format_tool_result(&self, tool_info: &ToolExecutionInfo) -> Option<String> {
        let result = tool_info.result.as_ref()?;

        match tool_info.tool_name.as_str() {
            "bash" => {
                // For bash commands, show the output directly without prefix
                if !result.content.trim().is_empty() {
                    let display_content = if result.content.len() > 200 {
                        format!("{}...", &result.content[..197])
                    } else {
                        result.content.clone()
                    };
                    Some(display_content)
                } else {
                    None
                }
            }
            "str_replace_based_edit_tool" => {
                // Determine operation type and show appropriate message
                if !result.success {
                    Some(format!("  ⎿  Error: {}", result.content))
                } else {
                    // Check operation type based on parameters
                    if tool_info.parameters.contains_key("file_text") {
                        // Create operation
                        None // Diff view will be shown separately
                    } else if tool_info.parameters.contains_key("old_str") {
                        // Update operation - no message needed, diff view will be shown
                        None
                    } else if tool_info.parameters.contains_key("view_range")
                        || tool_info.parameters.get("command").and_then(|v| v.as_str())
                            == Some("view")
                    {
                        // Read operation - show line count
                        let line_count = result.content.lines().count();
                        Some(format!("  ⎿  Read {} lines", line_count))
                    } else {
                        // Other edit operations
                        Some("  ⎿  File updated successfully".to_string())
                    }
                }
            }
            "task_done" => {
                // Extract details from the task_done result content
                // Format: "Summary: {summary}\n\nDetails:\n{details}"
                let content = &result.content;
                if let Some(details_start) = content.find("Details:\n") {
                    let details = &content[details_start + 9..]; // Skip "Details:\n"
                    if !details.trim().is_empty() {
                        Some(details.trim().to_string())
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            "sequentialthinking" => {
                // Thinking tool - no result message needed, thinking is shown separately
                None
            }
            _ => {
                if !result.content.trim().is_empty() {
                    Some(format!("  ⎿  {}", result.content))
                } else {
                    None
                }
            }
        }
    }
}

/// Diff formatter for file editing operations
pub struct DiffFormatter;

impl DiffFormatter {
    pub fn new() -> Self {
        Self
    }

    /// Format edit tool result with diff view
    pub fn format_edit_result(&self, tool_info: &ToolExecutionInfo) -> Option<String> {
        let result = tool_info.result.as_ref()?;

        if !result.success {
            return None;
        }

        let path = tool_info.parameters.get("path").and_then(|v| v.as_str())?;

        let file_name = Path::new(path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(path);

        // Determine operation type and create appropriate diff
        if let Some(old_str) = tool_info.parameters.get("old_str").and_then(|v| v.as_str()) {
            tool_info
                .parameters
                .get("new_str")
                .and_then(|v| v.as_str())
                .map(|new_str| {
                    self.create_unified_diff_view(file_name, Some(old_str), Some(new_str))
                })
        } else if let Some(new_str) = tool_info.parameters.get("new_str").and_then(|v| v.as_str()) {
            // insert operation
            Some(self.create_unified_diff_view(file_name, None, Some(new_str)))
        } else {
            tool_info
                .parameters
                .get("file_text")
                .and_then(|v| v.as_str())
                .map(|file_text| self.create_unified_diff_view(file_name, None, Some(file_text)))
        }
    }

    /// Create a unified diff view for all operations
    fn create_unified_diff_view(
        &self,
        file_name: &str,
        old_content: Option<&str>,
        new_content: Option<&str>,
    ) -> String {
        let mut result = String::new();

        result.push_str(&format!("{:<120}\n", file_name));
        result.push_str(&format!("{:<120}\n", ""));

        let old_lines: Vec<&str> = old_content.map(|s| s.lines().collect()).unwrap_or_default();
        let new_lines: Vec<&str> = new_content.map(|s| s.lines().collect()).unwrap_or_default();

        let max_lines = old_lines.len().max(new_lines.len());

        for i in 0..max_lines {
            let line_num = format!("{:>3}", i + 1);

            if i < old_lines.len() && i < new_lines.len() {
                if old_lines[i] != new_lines[i] {
                    // Changed line - show both old and new
                    let old_line =
                        self.format_line_with_background_and_prefix(old_lines[i], RED_BG, "-");
                    let new_line =
                        self.format_line_with_background_and_prefix(new_lines[i], GREEN_BG, "+");
                    result.push_str(&format!("   {}{}{} {}\n", GRAY, line_num, RESET, old_line));
                    result.push_str(&format!("   {}{}{} {}\n", GRAY, line_num, RESET, new_line));
                } else {
                    // Unchanged line
                    let line = self.truncate_line(old_lines[i]);
                    result.push_str(&format!(
                        "   {}{}{}    {:<100}\n",
                        GRAY, line_num, RESET, line
                    ));
                }
            } else if i < old_lines.len() {
                // Deleted line
                let line = self.format_line_with_background_and_prefix(old_lines[i], RED_BG, "-");
                result.push_str(&format!("   {}{}{} {}\n", GRAY, line_num, RESET, line));
            } else if i < new_lines.len() {
                // Added line
                let line = self.format_line_with_background_and_prefix(new_lines[i], GREEN_BG, "+");
                result.push_str(&format!("   {}{}{} {}\n", GRAY, line_num, RESET, line));
            }
        }

        result
    }

    /// Format a line with background color including prefix symbol
    fn format_line_with_background_and_prefix(
        &self,
        line: &str,
        bg_color: &str,
        prefix: &str,
    ) -> String {
        let truncated = self.truncate_line(line);
        let content_with_prefix = format!("{} {}", prefix, truncated);
        // Use black text on colored background for better contrast
        format!("{}{}{:<100}{}", bg_color, BLACK, content_with_prefix, RESET)
    }

    /// Truncate line if too long
    fn truncate_line(&self, line: &str) -> String {
        if line.len() > 100 {
            format!("{}...", &line[..97])
        } else {
            line.to_string()
        }
    }
}
