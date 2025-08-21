//! Output formatter for tool results
//!
//! This module provides formatted output display for different tool operations,
//! with unified formatting for all tools.

use crate::tools::{ToolCall, ToolResult};
use std::path::Path;

// ANSI color codes
pub const RED_BG: &str = "\x1b[41m"; // Red background
pub const GREEN_BG: &str = "\x1b[42m"; // Green background
pub const GRAY: &str = "\x1b[90m"; // Gray text for line numbers
pub const WHITE: &str = "\x1b[97m"; // White text for executing status
pub const GREEN: &str = "\x1b[92m"; // Green text for success status
pub const RED: &str = "\x1b[91m"; // Red text for error status
pub const YELLOW: &str = "\x1b[93m"; // Yellow text for warnings
pub const RESET: &str = "\x1b[0m";

/// Status of tool execution
#[derive(Debug, Clone, Copy)]
pub enum ToolStatus {
    Executing,
    Success,
    Error,
}

/// Unified formatter for all tool output
pub struct ToolOutputFormatter;

impl ToolOutputFormatter {
    /// Create a new formatter instance
    pub fn new() -> Self {
        Self
    }

    /// Format tool execution status with colored dot
    pub fn format_tool_status(&self, tool_name: &str, command: &str, status: ToolStatus) -> String {
        let (dot_color, dot_char) = match status {
            ToolStatus::Executing => (WHITE, "⏺"),
            ToolStatus::Success => (GREEN, "⏺"),
            ToolStatus::Error => (RED, "⏺"),
        };

        format!(
            "{}{}{} {}({})",
            dot_color, dot_char, RESET, tool_name, command
        )
    }

    /// Format tool result with unified output format
    pub fn format_tool_result_unified(
        &self,
        tool_name: &str,
        command: &str,
        content: &str,
        success: bool,
    ) -> String {
        let status = if success {
            ToolStatus::Success
        } else {
            ToolStatus::Error
        };
        let status_line = self.format_tool_status(tool_name, command, status);

        if content.trim().is_empty() {
            return status_line;
        }

        // Truncate content if too long
        let display_content = if content.len() > 200 {
            format!("{}...", &content[..197])
        } else {
            content.to_string()
        };

        format!("{}\n  ⎿  {}", status_line, display_content)
    }

    /// Format tool result with status update (overwrites previous line)
    pub fn format_tool_result_with_update(
        &self,
        tool_name: &str,
        command: &str,
        content: &str,
        success: bool,
    ) -> String {
        let status = if success {
            ToolStatus::Success
        } else {
            ToolStatus::Error
        };
        let mut result = String::new();

        // Clear current line and move cursor up to overwrite the executing line
        result.push_str("\x1b[1A\x1b[2K\r");
        result.push_str(&self.format_tool_status(tool_name, command, status));

        if !content.trim().is_empty() {
            // Truncate content if too long
            let display_content = if content.len() > 200 {
                format!("{}...", &content[..197])
            } else {
                content.to_string()
            };
            result.push_str(&format!("\n  ⎿  {}\n", display_content));
        }

        result
    }

    /// Format tool result based on the operation type
    pub fn format_tool_result(&self, tool_call: &ToolCall, tool_result: &ToolResult) -> String {
        // Extract command from tool call parameters
        let command = tool_call
            .parameters
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        match command {
            "view" => self.format_view_result(tool_call, tool_result),
            "str_replace" => self.format_str_replace_result(tool_call, tool_result),
            "insert" => self.format_insert_result(tool_call, tool_result),
            "create" => self.format_create_result(tool_call, tool_result),
            _ => {
                // Fallback to basic display
                if !tool_result.success {
                    format!("Error: {}", tool_result.content)
                } else {
                    String::new()
                }
            }
        }
    }

    /// Format view operation result
    fn format_view_result(&self, tool_call: &ToolCall, tool_result: &ToolResult) -> String {
        if !tool_result.success {
            return format!("Error: {}", tool_result.content);
        }

        let path = tool_call
            .parameters
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let file_name = Path::new(path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(path);

        // Count lines in the content
        let line_count = tool_result.content.lines().count();

        format!(
            "⏺ Read({})\n  ⎿  Read {} lines (ctrl+r to expand)",
            file_name, line_count
        )
    }

    /// Format str_replace operation result with diff view
    fn format_str_replace_result(&self, tool_call: &ToolCall, tool_result: &ToolResult) -> String {
        if !tool_result.success {
            return format!("Error: {}", tool_result.content);
        }

        let path = tool_call
            .parameters
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let old_str = tool_call
            .parameters
            .get("old_str")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let new_str = tool_call
            .parameters
            .get("new_str")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let file_name = Path::new(path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(path);

        // Show Update header and create diff view
        let mut result = format!("● Update({})\n", file_name);
        result.push_str(&self.create_unified_diff_view(file_name, Some(old_str), Some(new_str)));
        result
    }

    /// Format insert operation result
    fn format_insert_result(&self, tool_call: &ToolCall, tool_result: &ToolResult) -> String {
        if !tool_result.success {
            return format!("Error: {}", tool_result.content);
        }

        let path = tool_call
            .parameters
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let new_str = tool_call
            .parameters
            .get("new_str")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let file_name = Path::new(path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(path);

        // Show Update header and create diff view for insert (only new content)
        let mut result = format!("● Update({})\n", file_name);
        result.push_str(&self.create_unified_diff_view(file_name, None, Some(new_str)));
        result
    }

    /// Format create operation result
    fn format_create_result(&self, tool_call: &ToolCall, tool_result: &ToolResult) -> String {
        if !tool_result.success {
            return format!("Error: {}", tool_result.content);
        }

        let path = tool_call
            .parameters
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let file_text = tool_call
            .parameters
            .get("file_text")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let file_name = Path::new(path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(path);

        // Show Update header and create diff view for create (only new content)
        let mut result = format!("● Update({})\n", file_name);
        result.push_str(&self.create_unified_diff_view(file_name, None, Some(file_text)));
        result
    }

    /// Create a unified diff view for all operations
    /// old_content: None for create/insert operations, Some for replace operations
    /// new_content: None for delete operations, Some for replace/create/insert operations
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
        // Use fixed width of 100 characters for the content area
        format!("{}{:<100}{}", bg_color, content_with_prefix, RESET)
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

impl Default for ToolOutputFormatter {
    fn default() -> Self {
        Self::new()
    }
}
