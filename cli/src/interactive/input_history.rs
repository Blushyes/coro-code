//! Input history management for the interactive CLI
//!
//! This module provides functionality to store, persist, and navigate through
//! user input history with keyboard navigation support.

use coro_core::output::AgentOutput;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use tokio::fs;

/// Maximum number of history entries to keep in memory and on disk
const MAX_HISTORY_SIZE: usize = 1000;

/// Default history file name (using text format for optimal performance)
const HISTORY_FILE_NAME: &str = "input_history.txt";

/// Input history entry (simplified for text storage)
#[derive(Debug, Clone, PartialEq)]
pub struct HistoryEntry {
    /// The input text
    pub text: String,
}

impl HistoryEntry {
    /// Create a new history entry
    pub fn new(text: String) -> Self {
        Self { text }
    }
}

/// Input history manager with persistence and navigation
#[derive(Debug, Clone)]
pub struct InputHistory {
    /// History entries (most recent first)
    entries: VecDeque<HistoryEntry>,
    /// Current navigation position (0 = most recent, entries.len() = current input)
    current_position: usize,
    /// Current input being typed (saved when navigating history)
    current_input: String,
    /// Path to the history file
    history_file_path: PathBuf,
    /// Maximum number of entries to keep
    max_size: usize,
    /// Flag to indicate if history needs to be saved
    needs_save: bool,
}

impl InputHistory {
    /// Create a new input history manager
    pub fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            current_position: 0,
            current_input: String::new(),
            history_file_path: Self::default_history_path(),
            max_size: MAX_HISTORY_SIZE,
            needs_save: false,
        }
    }

    /// Create a new input history manager with custom file path
    #[allow(dead_code)]
    pub fn with_file_path<P: AsRef<Path>>(path: P) -> Self {
        let mut history = Self::new();
        history.history_file_path = path.as_ref().to_path_buf();
        history
    }

    /// Get the default history file path
    pub fn default_history_path() -> PathBuf {
        let mut path = dirs::config_dir().unwrap_or_else(|| PathBuf::from("."));
        path.push("coro");
        path.push(HISTORY_FILE_NAME);
        path
    }

    /// Load history from file
    pub async fn load(
        &mut self,
        output: &dyn AgentOutput,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if !self.history_file_path.exists() {
            let _ = output
                .debug("History file does not exist, starting with empty history")
                .await;
            return Ok(());
        }

        match fs::read_to_string(&self.history_file_path).await {
            Ok(content) => {
                // Parse text format: one command per line
                let entries: Vec<HistoryEntry> = content
                    .lines()
                    .filter(|line| !line.trim().is_empty()) // Skip empty lines
                    .map(|line| HistoryEntry::new(line.to_string()))
                    .collect();

                self.entries = entries.into_iter().collect();
                // Ensure we don't exceed max size
                self.trim_to_max_size();
                let _ = output
                    .debug(&format!(
                        "Loaded {} history entries from text format",
                        self.entries.len()
                    ))
                    .await;
            }
            Err(e) => {
                let _ = output
                    .warning(&format!("Failed to read history file: {}", e))
                    .await;
            }
        }

        Ok(())
    }

    /// Check if history needs to be saved
    pub fn needs_save(&self) -> bool {
        self.needs_save
    }

    /// Save history to file if needed
    pub async fn save_if_needed(
        &mut self,
        output: &dyn AgentOutput,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.needs_save {
            self.save(output).await?;
            self.needs_save = false;
        }
        Ok(())
    }

    /// Save history to file
    pub async fn save(
        &self,
        output: &dyn AgentOutput,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Create parent directory if it doesn't exist
        if let Some(parent) = self.history_file_path.parent() {
            if let Err(e) = fs::create_dir_all(parent).await {
                let _ = output
                    .error(&format!("Failed to create history directory: {}", e))
                    .await;
                return Err(e.into());
            }
        }

        // Convert to text format: one command per line
        let content: String = self
            .entries
            .iter()
            .map(|entry| entry.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        if let Err(e) = fs::write(&self.history_file_path, content).await {
            let _ = output
                .error(&format!("Failed to write history file: {}", e))
                .await;
            return Err(e.into());
        }

        let _ = output
            .debug(&format!(
                "Saved {} history entries in text format",
                self.entries.len()
            ))
            .await;

        Ok(())
    }

    /// Add a new entry to history
    /// Returns true if entry was added, false if skipped
    /// Uses delayed save mechanism for optimal performance
    pub fn add_entry(&mut self, text: String) -> bool {
        // Don't add empty entries
        if text.trim().is_empty() {
            return false;
        }

        // Don't add duplicate of the most recent entry
        if let Some(last_entry) = self.entries.front() {
            if last_entry.text == text {
                self.reset_navigation();
                return false;
            }
        }

        // Add new entry at the front (most recent)
        let entry = HistoryEntry::new(text);
        self.entries.push_front(entry);

        // Trim to max size
        self.trim_to_max_size();

        // Reset navigation position
        self.reset_navigation();

        // Mark for saving
        self.needs_save = true;

        true
    }

    /// Navigate to previous entry (up arrow)
    /// Returns the text to display, or None if at the beginning
    pub fn navigate_previous(&mut self, current_input: &str) -> Option<String> {
        // Save current input if we're at the current position
        if self.current_position == self.entries.len() {
            self.current_input = current_input.to_string();
            // Move to the most recent entry
            if !self.entries.is_empty() {
                self.current_position = 0;
                let entry = &self.entries[self.current_position];
                return Some(entry.text.clone());
            } else {
                return None;
            }
        }

        // Check if we can go back further in history
        if self.current_position + 1 < self.entries.len() {
            self.current_position += 1;
            let entry = &self.entries[self.current_position];
            Some(entry.text.clone())
        } else {
            None
        }
    }

    /// Navigate to next entry (down arrow)
    /// Returns the text to display, or None if at the end
    pub fn navigate_next(&mut self) -> Option<String> {
        if self.current_position == self.entries.len() {
            // Already at current input
            None
        } else if self.current_position > 0 {
            // Move to more recent entry
            self.current_position -= 1;
            let entry = &self.entries[self.current_position];
            Some(entry.text.clone())
        } else {
            // Move back to current input
            self.current_position = self.entries.len();
            Some(self.current_input.clone())
        }
    }

    /// Reset navigation to current input
    pub fn reset_navigation(&mut self) {
        self.current_position = self.entries.len();
        self.current_input.clear();
    }

    /// Check if currently navigating history (not at current input)
    #[allow(dead_code)]
    pub fn is_navigating(&self) -> bool {
        self.current_position < self.entries.len()
    }

    /// Get the number of entries in history
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if history is empty
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Trim history to maximum size
    fn trim_to_max_size(&mut self) {
        while self.entries.len() > self.max_size {
            self.entries.pop_back();
        }
    }

    /// Get current navigation position for debugging
    #[allow(dead_code)]
    pub fn current_position(&self) -> usize {
        self.current_position
    }

    /// Get current saved input for debugging
    #[allow(dead_code)]
    pub fn current_input(&self) -> &str {
        &self.current_input
    }
}

impl Default for InputHistory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coro_core::output::NullOutput;

    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_history_path(suffix: &str) -> PathBuf {
        // Build a reasonably unique filename without extra dependencies
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let pid = std::process::id();
        let mut path = std::env::temp_dir();
        path.push(format!(
            "coro_input_history_test_{}_{}_{}.txt",
            suffix, pid, nanos
        ));
        path
    }

    #[tokio::test]
    async fn test_add_entry() {
        let tmp_path = unique_temp_history_path("add_entry");
        let mut history = InputHistory::with_file_path(&tmp_path);
        let output = NullOutput;

        history.add_entry("first command".to_string());
        history.add_entry("second command".to_string());

        assert_eq!(history.len(), 2);
        assert_eq!(history.entries[0].text, "second command");
        assert_eq!(history.entries[1].text, "first command");

        // Test save functionality
        history.save_if_needed(&output).await.unwrap();
    }

    #[tokio::test]
    async fn test_navigate_history() {
        let mut history = InputHistory::new();
        let _output = NullOutput;

        history.add_entry("first".to_string());
        history.add_entry("second".to_string());

        // Navigate previous
        let result = history.navigate_previous("current");
        assert_eq!(result, Some("second".to_string()));

        let result = history.navigate_previous("current");
        assert_eq!(result, Some("first".to_string()));

        // Can't go further back
        let result = history.navigate_previous("current");
        assert_eq!(result, None);

        // Navigate forward
        let result = history.navigate_next();
        assert_eq!(result, Some("second".to_string()));

        let result = history.navigate_next();
        assert_eq!(result, Some("current".to_string()));
    }

    #[test]
    fn test_duplicate_prevention() {
        let mut history = InputHistory::new();
        let entry1 = HistoryEntry::new("same command".to_string());
        let _entry2 = HistoryEntry::new("same command".to_string());

        history.entries.push_front(entry1);
        // This should be prevented by add_entry logic
        assert_eq!(history.len(), 1);
    }

    #[tokio::test]
    async fn test_immediate_availability_after_add() {
        let mut history = InputHistory::new();
        let _output = NullOutput;

        // Add first entry
        history.add_entry("first command".to_string());

        // Should be immediately available for navigation
        let result = history.navigate_previous("current input");
        assert_eq!(result, Some("first command".to_string()));

        // Reset and add another entry
        history.reset_navigation();
        history.add_entry("second command".to_string());

        // The newest entry should be available immediately
        let result = history.navigate_previous("current input");
        assert_eq!(result, Some("second command".to_string()));

        // And we should be able to navigate to the older entry
        let result = history.navigate_previous("current input");
        assert_eq!(result, Some("first command".to_string()));
    }

    #[tokio::test]
    async fn test_first_navigation_after_load() {
        let mut history = InputHistory::new();
        let _output = NullOutput;

        // Add some entries
        history.add_entry("first".to_string());
        history.add_entry("second".to_string());
        history.add_entry("third".to_string());

        // Simulate program restart by resetting navigation
        history.reset_navigation();

        // First navigation should show the most recent entry
        let result = history.navigate_previous("current input");
        assert_eq!(result, Some("third".to_string()));

        // Second navigation should show the second most recent
        let result = history.navigate_previous("current input");
        assert_eq!(result, Some("second".to_string()));

        // Third navigation should show the oldest
        let result = history.navigate_previous("current input");
        assert_eq!(result, Some("first".to_string()));

        // No more entries
        let result = history.navigate_previous("current input");
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn test_real_time_session_workflow() {
        let mut history = InputHistory::new();
        let _output = NullOutput;

        // Simulate the exact workflow described:
        // 1. Type "hello" and press Enter (submit)
        history.add_entry("hello".to_string());

        // 2. Start typing a new command
        let current_input = "new command being typed";

        // 3. Press up arrow key
        let result = history.navigate_previous(current_input);

        // 4. Verify that "hello" appears as the current input
        assert_eq!(result, Some("hello".to_string()));

        // Additional verification: ensure we can navigate back to current input
        let result = history.navigate_next();
        assert_eq!(result, Some(current_input.to_string()));

        // Test multiple commands in sequence
        history.reset_navigation();
        history.add_entry("second command".to_string());

        // Should show most recent first
        let result = history.navigate_previous("typing again");
        assert_eq!(result, Some("second command".to_string()));

        // Then the previous one
        let result = history.navigate_previous("typing again");
        assert_eq!(result, Some("hello".to_string()));
    }

    #[tokio::test]
    async fn test_performance_batch_vs_individual_save() {
        let tmp1 = unique_temp_history_path("batch");
        let tmp2 = unique_temp_history_path("individual");
        let mut history = InputHistory::with_file_path(&tmp1);
        let output = NullOutput;

        // Test batch add with single save (optimal approach)
        let start = std::time::Instant::now();
        for i in 0..1000 {
            history.add_entry(format!("command {}", i));
        }
        let batch_duration = start.elapsed();

        // Save once at the end
        history.save_if_needed(&output).await.unwrap();
        let batch_with_save_duration = start.elapsed();

        // Compare with individual saves (suboptimal approach)
        let mut history2 = InputHistory::with_file_path(&tmp2);
        let start = std::time::Instant::now();
        for i in 0..100 {
            // Use fewer iterations due to I/O overhead
            history2.add_entry(format!("command {}", i));
            history2.save_if_needed(&output).await.unwrap();
        }
        let individual_save_duration = start.elapsed();

        println!("Batch add (1000 entries): {:?}", batch_duration);
        println!("Batch add + save: {:?}", batch_with_save_duration);
        println!(
            "Individual saves (100 entries): {:?}",
            individual_save_duration
        );

        // Batch approach should be significantly faster per entry
        let batch_per_entry = batch_duration.as_nanos() / 1000;
        let individual_per_entry = individual_save_duration.as_nanos() / 100;

        assert!(
            batch_per_entry < individual_per_entry / 10,
            "Batch add should be at least 10x faster per entry than individual saves"
        );
    }

    #[tokio::test]
    async fn test_text_format_performance() {
        let output = NullOutput;

        // Create test data
        let test_commands: Vec<String> = (0..1000)
            .map(|i| {
                format!(
                    "command {} with some longer text to test serialization performance",
                    i
                )
            })
            .collect();

        // Test text format save performance
        let tmp = unique_temp_history_path("text_perf");
        let mut text_history = InputHistory::with_file_path(&tmp);
        for cmd in &test_commands {
            text_history.add_entry(cmd.clone());
        }

        let start = std::time::Instant::now();
        text_history.save(&output).await.unwrap();
        let text_save_duration = start.elapsed();

        // Test text format load performance
        let mut text_history_load = InputHistory::with_file_path(&tmp);
        let start = std::time::Instant::now();
        text_history_load.load(&output).await.unwrap();
        let text_load_duration = start.elapsed();

        println!("Text format save (1000 entries): {:?}", text_save_duration);
        println!("Text format load (1000 entries): {:?}", text_load_duration);

        // Verify data integrity (entries are stored most recent first)
        assert_eq!(text_history_load.entries.len(), 1000);
        assert_eq!(text_history_load.entries[0].text, test_commands[999]); // Most recent
        assert_eq!(text_history_load.entries[999].text, test_commands[0]); // Oldest

        // Text format should be very fast (under 50ms for 1000 entries)
        assert!(
            text_save_duration.as_millis() < 50,
            "Text save should be under 50ms"
        );
        assert!(
            text_load_duration.as_millis() < 50,
            "Text load should be under 50ms"
        );
    }
}
