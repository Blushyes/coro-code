//! Input section component
//!
//! This module provides the input section component that handles
//! user input and displays the status bar.

use crate::interactive::file_search::{
    extract_existing_file_references, extract_search_query, should_show_file_search,
};
use crate::interactive::file_search::{
    DefaultFileSearchProvider, FileSearchProvider, FileSearchResult,
};
use crate::interactive::input_history::InputHistory;
use crate::interactive::message_handler::AppMessage;
use coro_core::ResolvedLlmConfig;
use coro_router::use_router;
use iocraft::prelude::*;
use std::cmp::min;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, Mutex};
use unicode_width::UnicodeWidthStr;

/// Find the nearest character boundary at or before the given byte position
/// This ensures we don't slice in the middle of a UTF-8 character
fn find_char_boundary(text: &str, byte_pos: usize) -> usize {
    if byte_pos >= text.len() {
        return text.len();
    }

    // Find the nearest character boundary at or before byte_pos
    text.char_indices()
        .map(|(i, _)| i)
        .filter(|&i| i <= byte_pos)
        .next_back()
        .unwrap_or(0)
}

/// Calculate cursor position (line, column) from text and byte position
/// Returns (line_number, column_number) where both are 1-based
fn calculate_cursor_position(text: &str, byte_pos: usize) -> (usize, usize) {
    if text.is_empty() {
        return (1, 1);
    }

    // Ensure we're at a valid character boundary
    let safe_pos = find_char_boundary(text, byte_pos);

    let text_before_cursor = &text[..safe_pos];

    // Count lines (number of newlines + 1)
    let line_number = text_before_cursor.matches('\n').count() + 1;

    // Find the start of the current line
    let current_line_start = text_before_cursor.rfind('\n').map(|i| i + 1).unwrap_or(0);
    let current_line_text = &text_before_cursor[current_line_start..];

    // Count characters in current line (not bytes)
    let column_number = current_line_text.chars().count() + 1;

    (line_number, column_number)
}

/// Calculate cursor display position for rendering with soft-wrapping awareness
/// Returns (display_line, display_column) where both are 0-based for UI positioning
fn calculate_cursor_display_position(
    text: &str,
    byte_pos: usize,
    max_width: usize,
) -> (usize, usize) {
    if text.is_empty() {
        return (0, 0);
    }

    let safe_pos = byte_pos.min(text.len());

    // Traverse characters up to the cursor and simulate wrapping using display width
    let mut display_line = 0usize;
    let mut current_line_width = 0usize;

    let iter = text.char_indices().peekable();
    for (idx, ch) in iter {
        if idx >= safe_pos {
            break;
        }
        if ch == '\n' {
            // Hard line break
            display_line += 1;
            current_line_width = 0;
            continue;
        }
        let char_width = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        // If adding this char would exceed the width and we already have something on this line, wrap before it
        if max_width > 0 && current_line_width > 0 && current_line_width + char_width > max_width {
            display_line += 1;
            current_line_width = 0;
        }
        current_line_width += char_width;
    }

    (display_line, current_line_width)
}

#[derive(Clone, Props)]
pub struct InputSectionProps {
    pub context: InputSectionContext,
}

impl Default for InputSectionProps {
    fn default() -> Self {
        Self {
            context: InputSectionContext {
                llm_config: ResolvedLlmConfig::new(
                    coro_core::Protocol::OpenAICompat,
                    "https://api.openai.com".to_string(),
                    "test-key".to_string(),
                    "gpt-4o".to_string(),
                ),
                project_path: PathBuf::new(),
                ui_sender: tokio::sync::broadcast::channel(1).0,
                agent: Arc::new(Mutex::new(None)),
            },
        }
    }
}

/// Context for the input section component
#[derive(Clone)]
pub struct InputSectionContext {
    pub llm_config: ResolvedLlmConfig,
    pub project_path: PathBuf,
    pub ui_sender: broadcast::Sender<AppMessage>,
    pub agent: Arc<Mutex<Option<coro_core::agent::AgentCore>>>,
}

/// Enhanced text input component that wraps iocraft's TextInput with submit handling
#[derive(Props)]
pub struct EnhancedTextInputProps {
    pub value: String,
    pub has_focus: bool,
    pub on_change: Handler<'static, String>,
    pub on_submit: Handler<'static, String>,
    pub on_cursor_position_change: Handler<'static, (usize, usize)>, // (line, column)
    pub on_file_list_state_change: Handler<'static, bool>,           // Track file list visibility
    pub width: u16,
    pub placeholder: String,
    pub color: Option<Color>,
    pub cursor_color: Option<Color>,
    pub project_path: PathBuf,
}

impl Default for EnhancedTextInputProps {
    fn default() -> Self {
        Self {
            value: String::new(),
            has_focus: false,
            on_change: Handler::default(),
            on_submit: Handler::default(),
            on_cursor_position_change: Handler::default(),
            on_file_list_state_change: Handler::default(),
            width: 80,
            placeholder: String::new(),
            color: None,
            cursor_color: None,
            project_path: PathBuf::new(),
        }
    }
}

/// Simple multiline text input component without internal scrolling
#[component]
pub fn EnhancedTextInput(
    mut hooks: Hooks,
    props: &mut EnhancedTextInputProps,
) -> impl Into<AnyElement<'static>> {
    let has_focus = props.has_focus;
    let width = props.width;
    let project_path = props.project_path.clone();

    // Local state for cursor position
    let cursor_pos = hooks.use_state(|| props.value.len());

    // State for file search popup
    let show_file_list = hooks.use_state(|| false);
    let search_results = hooks.use_state(Vec::<FileSearchResult>::new);
    let selected_file_index = hooks.use_state(|| 0usize);
    let current_query = hooks.use_state(String::new);
    // Track last text input time to disambiguate paste vs. manual Enter
    let last_text_time = hooks.use_state(|| Instant::now() - Duration::from_secs(10));

    // Cache for existing file references to avoid repeated parsing
    let cached_existing_refs = hooks.use_state(Vec::<String>::new);
    let last_input_for_refs = hooks.use_state(String::new);

    // Initialize search provider
    let search_provider =
        hooks.use_state(|| DefaultFileSearchProvider::new(project_path.clone()).ok());

    // Handle keyboard input
    hooks.use_terminal_events({
        let mut on_change = props.on_change.take();
        let mut on_submit = props.on_submit.take();
        let mut on_cursor_position_change = props.on_cursor_position_change.take();
        let mut on_file_list_state_change = props.on_file_list_state_change.take();
        let mut value = props.value.clone();
        let mut cursor_pos = cursor_pos;
        let mut show_file_list = show_file_list;
        let mut search_results = search_results;
        let mut selected_file_index = selected_file_index;
        let mut current_query = current_query;

        let _project_path = project_path.clone();
        let mut cached_existing_refs = cached_existing_refs;
        let mut last_input_for_refs = last_input_for_refs;
        let mut last_text_time = last_text_time;

        move |event| {
            if !has_focus {
                return;
            }

            match event {
                TerminalEvent::Key(KeyEvent {
                    code,
                    modifiers,
                    kind,
                    ..
                }) if kind != KeyEventKind::Release => {
                    let mut pos = cursor_pos.get();
                    let mut changed = false;

                    // Handle file list navigation when it's shown
                    if *show_file_list.read() {
                        match code {
                            KeyCode::Up => {
                                let current = selected_file_index.get();
                                if current > 0 {
                                    selected_file_index.set(current - 1);
                                }
                                return;
                            }
                            KeyCode::Down => {
                                let current = selected_file_index.get();
                                let max_index = search_results.read().len().saturating_sub(1);
                                if current < max_index {
                                    selected_file_index.set(current + 1);
                                }
                                return;
                            }
                            KeyCode::Char('p') if modifiers.contains(KeyModifiers::CONTROL) => {
                                // Ctrl+P: Move up (previous)
                                let current = selected_file_index.get();
                                if current > 0 {
                                    selected_file_index.set(current - 1);
                                }
                                return;
                            }
                            KeyCode::Char('n') if modifiers.contains(KeyModifiers::CONTROL) => {
                                // Ctrl+N: Move down (next)
                                let current = selected_file_index.get();
                                let max_index = search_results.read().len().saturating_sub(1);
                                if current < max_index {
                                    selected_file_index.set(current + 1);
                                }
                                return;
                            }
                            KeyCode::Enter | KeyCode::Tab => {
                                // If this Enter comes immediately after burst typing/paste, treat as newline (not file pick)
                                let recent_text = Instant::now()
                                    .duration_since(*last_text_time.read())
                                    <= Duration::from_millis(10);
                                if recent_text && matches!(code, KeyCode::Enter) {
                                    // Close popup and let the general handler below process Enter as newline
                                    show_file_list.set(false);
                                    on_file_list_state_change(false);
                                    // fallthrough to general handling by not returning
                                } else {
                                    // Insert selected file
                                    if let Some(selected_result) =
                                        search_results.read().get(selected_file_index.get())
                                    {
                                        // Find the @ position and replace the entire @search_term with @absolute_path + space
                                        if let Some(query) = extract_search_query(&value, pos) {
                                            if let Some(at_pos) = value.rfind('@') {
                                                // Find the end of the search term
                                                let search_end = at_pos + 1 + query.len();

                                                let before_at = &value[..at_pos];
                                                let after_search = &value[search_end..];

                                                // Create replacement: @absolute_path + space
                                                let replacement =
                                                    format!("@{} ", selected_result.insertion_path);

                                                value = format!(
                                                    "{}{}{}",
                                                    before_at, replacement, after_search
                                                );
                                                pos = at_pos + replacement.len();
                                                cursor_pos.set(pos);
                                                on_change(value.clone());

                                                // Update cursor position
                                                let (line, col) =
                                                    calculate_cursor_position(&value, pos);
                                                on_cursor_position_change((line, col));
                                            }
                                        }
                                    }
                                    show_file_list.set(false);
                                    on_file_list_state_change(false);
                                    return;
                                }
                            }
                            KeyCode::Esc => {
                                show_file_list.set(false);
                                on_file_list_state_change(false);
                                return;
                            }
                            _ => {}
                        }
                    }

                    match code {
                        KeyCode::Char(c) => {
                            // Record time of recent text input
                            last_text_time.set(Instant::now());
                            // Ensure we're at a character boundary before inserting
                            let safe_pos = find_char_boundary(&value, pos);
                            let char_pos = value[..safe_pos].chars().count();

                            let mut chars: Vec<char> = value.chars().collect();
                            chars.insert(char_pos, c);
                            value = chars.into_iter().collect();

                            // Update position to after the inserted character
                            pos = value
                                .char_indices()
                                .nth(char_pos + 1)
                                .map(|(i, _)| i)
                                .unwrap_or(value.len());
                            changed = true;

                            // Update cursor position
                            let (line, col) = calculate_cursor_position(&value, pos);
                            on_cursor_position_change((line, col));

                            // Check if we should show/update/hide file list after character input
                            let should_show = should_show_file_search(&value, pos);

                            if should_show {
                                if let Some(query) = extract_search_query(&value, pos) {
                                    // Show list and update search if needed
                                    if !*show_file_list.read() || query != *current_query.read() {
                                        if let Some(search_provider) =
                                            search_provider.read().as_ref()
                                        {
                                            // Get cached existing file references to exclude them
                                            let existing_refs = if value
                                                != *last_input_for_refs.read()
                                            {
                                                let refs =
                                                    extract_existing_file_references(&value, pos);
                                                cached_existing_refs.set(refs.clone());
                                                last_input_for_refs.set(value.clone());
                                                refs
                                            } else {
                                                cached_existing_refs.read().clone()
                                            };
                                            let exclude_paths: Vec<&str> =
                                                existing_refs.iter().map(|s| s.as_str()).collect();

                                            let results = if query.is_empty() {
                                                search_provider
                                                    .get_all_files_with_exclusions(&exclude_paths)
                                            } else {
                                                search_provider
                                                    .search_with_exclusions(&query, &exclude_paths)
                                            };
                                            search_results.set(results);
                                            selected_file_index.set(0);
                                            current_query.set(query);
                                            show_file_list.set(true);
                                            on_file_list_state_change(true);
                                        }
                                    }
                                }
                            } else {
                                // Should not show list, hide it
                                show_file_list.set(false);
                                on_file_list_state_change(false);
                            }
                        }
                        KeyCode::Backspace => {
                            if pos > 0 {
                                // Find the start of the previous character
                                let safe_pos = find_char_boundary(&value, pos);
                                let char_start = value[..safe_pos]
                                    .char_indices()
                                    .last()
                                    .map(|(i, _)| i)
                                    .unwrap_or(0);

                                // Convert to chars, remove the previous character, and rebuild string
                                let mut chars: Vec<char> = value.chars().collect();
                                let char_pos = value[..safe_pos].chars().count();
                                if char_pos > 0 {
                                    chars.remove(char_pos - 1);
                                    value = chars.into_iter().collect();
                                    pos = char_start;
                                    changed = true;

                                    // Update cursor position
                                    let (line, col) = calculate_cursor_position(&value, pos);
                                    on_cursor_position_change((line, col));

                                    // Check if we should show/update/hide file list after backspace
                                    let should_show = should_show_file_search(&value, pos);

                                    if should_show {
                                        if let Some(query) = extract_search_query(&value, pos) {
                                            // Show list and update search if needed
                                            if !*show_file_list.read()
                                                || query != *current_query.read()
                                            {
                                                if let Some(search_provider) =
                                                    search_provider.read().as_ref()
                                                {
                                                    // Get cached existing file references to exclude them
                                                    let existing_refs = if value
                                                        != *last_input_for_refs.read()
                                                    {
                                                        let refs = extract_existing_file_references(
                                                            &value, pos,
                                                        );
                                                        cached_existing_refs.set(refs.clone());
                                                        last_input_for_refs.set(value.clone());
                                                        refs
                                                    } else {
                                                        cached_existing_refs.read().clone()
                                                    };
                                                    let exclude_paths: Vec<&str> = existing_refs
                                                        .iter()
                                                        .map(|s| s.as_str())
                                                        .collect();

                                                    let results = if query.is_empty() {
                                                        search_provider
                                                            .get_all_files_with_exclusions(
                                                                &exclude_paths,
                                                            )
                                                    } else {
                                                        search_provider.search_with_exclusions(
                                                            &query,
                                                            &exclude_paths,
                                                        )
                                                    };
                                                    search_results.set(results);
                                                    selected_file_index.set(0);
                                                    current_query.set(query);
                                                    show_file_list.set(true);
                                                    on_file_list_state_change(true);
                                                }
                                            }
                                        }
                                    } else {
                                        // Should not show list, hide it
                                        show_file_list.set(false);
                                        on_file_list_state_change(false);
                                    }
                                }
                            }
                        }
                        KeyCode::Delete => {
                            if pos < value.len() {
                                // Find the next character boundary to delete safely
                                if value[pos..].chars().next().is_some() {
                                    // Convert to chars, remove one, and rebuild string
                                    let mut chars: Vec<char> = value.chars().collect();
                                    let safe_pos = find_char_boundary(&value, pos);
                                    let char_pos = value[..safe_pos].chars().count();
                                    if char_pos < chars.len() {
                                        chars.remove(char_pos);
                                        value = chars.into_iter().collect();
                                        changed = true;

                                        // Update cursor position
                                        let (line, col) = calculate_cursor_position(&value, pos);
                                        on_cursor_position_change((line, col));

                                        // Update search or hide file list
                                        if *show_file_list.read() {
                                            if let Some(query) = extract_search_query(&value, pos) {
                                                if query != *current_query.read() {
                                                    if let Some(search_provider) =
                                                        search_provider.read().as_ref()
                                                    {
                                                        // Get cached existing file references to exclude them
                                                        let existing_refs = if value
                                                            != *last_input_for_refs.read()
                                                        {
                                                            let refs =
                                                                extract_existing_file_references(
                                                                    &value, pos,
                                                                );
                                                            cached_existing_refs.set(refs.clone());
                                                            last_input_for_refs.set(value.clone());
                                                            refs
                                                        } else {
                                                            cached_existing_refs.read().clone()
                                                        };
                                                        let exclude_paths: Vec<&str> =
                                                            existing_refs
                                                                .iter()
                                                                .map(|s| s.as_str())
                                                                .collect();

                                                        let results = if query.is_empty() {
                                                            search_provider
                                                                .get_all_files_with_exclusions(
                                                                    &exclude_paths,
                                                                )
                                                        } else {
                                                            search_provider.search_with_exclusions(
                                                                &query,
                                                                &exclude_paths,
                                                            )
                                                        };
                                                        search_results.set(results);
                                                        selected_file_index.set(0);
                                                        current_query.set(query);
                                                    }
                                                }
                                            } else {
                                                // No valid query found, hide the list
                                                show_file_list.set(false);
                                                on_file_list_state_change(false);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        KeyCode::Enter => {
                            // Check if current line ends with backslash
                            let current_line_end_with_backslash = {
                                // Find the current line by looking backwards from cursor position
                                let safe_pos = find_char_boundary(&value, pos);
                                let before_cursor = &value[..safe_pos];
                                let current_line_start =
                                    before_cursor.rfind('\n').map(|i| i + 1).unwrap_or(0);

                                // Find the end of current line by looking forward from cursor
                                let after_cursor = &value[pos..];
                                let current_line_end = after_cursor
                                    .find('\n')
                                    .map(|i| pos + i)
                                    .unwrap_or(value.len());

                                // Get the current line content
                                let current_line = &value[current_line_start..current_line_end];
                                current_line.trim_end().ends_with('\\')
                            };

                            if current_line_end_with_backslash {
                                // Remove the trailing backslash and add newline
                                let safe_pos = find_char_boundary(&value, pos);
                                let before_cursor = &value[..safe_pos];
                                let current_line_start =
                                    before_cursor.rfind('\n').map(|i| i + 1).unwrap_or(0);

                                let after_cursor = &value[pos..];
                                let current_line_end = after_cursor
                                    .find('\n')
                                    .map(|i| pos + i)
                                    .unwrap_or(value.len());

                                let current_line = &value[current_line_start..current_line_end];
                                let trimmed_line = current_line.trim_end();

                                if let Some(backslash_pos) = trimmed_line.rfind('\\') {
                                    // Create new string parts to avoid borrowing conflicts
                                    let new_line = trimmed_line[..backslash_pos].to_string();
                                    let before_line = value[..current_line_start].to_string();
                                    let after_line = value[current_line_end..].to_string();

                                    value = format!("{}{}\n{}", before_line, new_line, after_line);
                                    pos = current_line_start + new_line.len() + 1; // Position after newline
                                    changed = true;

                                    // Update cursor position
                                    let (line, col) = calculate_cursor_position(&value, pos);
                                    on_cursor_position_change((line, col));
                                }
                            } else {
                                // Heuristic: if Enter comes right after a burst of text input (e.g., from a paste),
                                // treat it as a newline instead of submit.
                                let recent_text = Instant::now()
                                    .duration_since(*last_text_time.read())
                                    <= Duration::from_millis(10);
                                if modifiers.contains(KeyModifiers::SHIFT) || recent_text {
                                    // Insert newline - use safe character insertion
                                    let safe_pos = find_char_boundary(&value, pos);
                                    let char_pos = value[..safe_pos].chars().count();
                                    let mut chars: Vec<char> = value.chars().collect();
                                    chars.insert(char_pos, '\n');
                                    value = chars.into_iter().collect();

                                    // Update position to after the inserted newline
                                    pos = value
                                        .char_indices()
                                        .nth(char_pos + 1)
                                        .map(|(i, _)| i)
                                        .unwrap_or(value.len());
                                    changed = true;

                                    // Update cursor position
                                    let (line, col) = calculate_cursor_position(&value, pos);
                                    on_cursor_position_change((line, col));

                                    // Update recent text time since we effectively inserted text
                                    last_text_time.set(Instant::now());
                                } else {
                                    // Regular Enter submits
                                    on_submit(value.clone());
                                    return;
                                }
                            }
                        }
                        KeyCode::Left => {
                            if pos > 0 {
                                let safe_pos = find_char_boundary(&value, pos);
                                let char_start = value[..safe_pos]
                                    .char_indices()
                                    .last()
                                    .map(|(i, _)| i)
                                    .unwrap_or(0);
                                pos = char_start;
                                cursor_pos.set(pos);

                                // Update cursor position
                                let (line, col) = calculate_cursor_position(&value, pos);
                                on_cursor_position_change((line, col));
                            }
                        }
                        KeyCode::Right => {
                            if pos < value.len() {
                                let char_end = value[pos..]
                                    .char_indices()
                                    .nth(1)
                                    .map(|(i, _)| pos + i)
                                    .unwrap_or(value.len());
                                pos = char_end;
                                cursor_pos.set(pos);

                                // Update cursor position
                                let (line, col) = calculate_cursor_position(&value, pos);
                                on_cursor_position_change((line, col));
                            }
                        }
                        KeyCode::Up => {
                            // History navigation will be handled in InputSection
                            // For now, just handle normal text navigation
                        }
                        KeyCode::Down => {
                            // History navigation will be handled in InputSection
                            // For now, just handle normal text navigation
                        }
                        _ => {}
                    }

                    if changed {
                        cursor_pos.set(pos);
                        on_change(value.clone());
                    }
                }
                _ => {}
            }
        }
    });

    // Split text into display lines with wrapping
    let effective_width = (width as usize).saturating_sub(4); // Account for borders and padding
    let display_lines = if props.value.is_empty() {
        vec![String::new()]
    } else {
        let mut lines = Vec::new();
        for line in props.value.lines() {
            if line.is_empty() {
                lines.push(String::new());
            } else {
                // Simple wrapping: split long lines
                let line_width = UnicodeWidthStr::width(line);
                if line_width <= effective_width {
                    lines.push(line.to_string());
                } else {
                    // Need to wrap this line
                    let mut current_line = String::new();
                    let mut current_width = 0;

                    for ch in line.chars() {
                        let char_width = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
                        if current_width + char_width > effective_width && !current_line.is_empty()
                        {
                            lines.push(current_line);
                            current_line = String::new();
                            current_width = 0;
                        }
                        current_line.push(ch);
                        current_width += char_width;
                    }
                    if !current_line.is_empty() {
                        lines.push(current_line);
                    }
                }
            }
        }

        if props.value.ends_with('\n') {
            lines.push(String::new());
        }

        lines
    };

    let total_height = (display_lines.len() + 2) as u16; // +2 for borders

    element! {
        View(
            flex_direction: FlexDirection::Column,
            position: Position::Relative,
        ) {
            // Input box
            View(
                width: width,
                height: total_height,
                border_style: BorderStyle::Round,
                padding_left: 1,
                padding_right: 1,
                position: Position::Relative,
            ) {
                // Content area with cursor
                View(
                    flex_direction: FlexDirection::Column,
                    width: 100pct,
                    height: 100pct,
                    position: Position::Relative,
                ) {
                    #(display_lines.iter().enumerate().map(|(line_idx, line)| {
                        element! {
                            View(
                                key: format!("line-{}", line_idx),
                                height: 1,
                                width: 100pct,
                                position: Position::Relative,
                            ) {
                                #(if line.is_empty() && line_idx == 0 && props.value.is_empty() && !props.placeholder.is_empty() {
                                    Some(element! {
                                        Text(
                                            content: &props.placeholder,
                                            color: Color::DarkGrey,
                                        )
                                    })
                                } else {
                                    Some(element! {
                                        Text(
                                            content: line,
                                            color: props.color.unwrap_or(Color::White),
                                        )
                                    })
                                })
                            }
                        }
                    }))

                    // Render cursor if component has focus
                    #(if props.has_focus {
                        // Calculate cursor position in display coordinates (wrap-aware)
                        let cursor_pos = cursor_pos.get();
                        let (cursor_line, cursor_col) = calculate_cursor_display_position(&props.value, cursor_pos, effective_width);
                        // Clamp left within the drawable area to avoid off-by-one overflow
                        let clamped_left = min(cursor_col, effective_width.saturating_sub(1)) as u16;

                        // Get the character at cursor position for semi-transparent effect
                        let cursor_char = if props.value.is_empty() && !props.placeholder.is_empty() {
                            // Show placeholder character at cursor position with different styling
                            props.placeholder.chars().nth(cursor_col).unwrap_or(' ')
                        } else {
                            // Show space or character at cursor position
                            let chars: Vec<char> = props.value.chars().collect();
                            // Convert byte index to char index safely
                            // First, ensure cursor_pos is at a valid character boundary
                            let safe_cursor_pos = find_char_boundary(&props.value, cursor_pos);

                            let char_idx = props.value[..safe_cursor_pos].chars().count();
                            if char_idx < chars.len() {
                                chars[char_idx]
                            } else {
                                ' '
                            }
                        };

                        Some(element! {
                            View(
                                key: "cursor",
                                position: Position::Absolute,
                                top: cursor_line as u16,
                                left: clamped_left,
                                width: 1,
                                height: 1,
                                background_color: props.cursor_color.unwrap_or(Color::Rgb { r: 200, g: 200, b: 200 }), // Light grey background
                            ) {
                                Text(
                                    content: cursor_char.to_string(),
                                    color: Color::Rgb { r: 80, g: 80, b: 80 }, // Dark text for contrast
                                )
                            }
                        })
                    } else {
                        None
                    })
                }
            }

            // File list popup
            #(if *show_file_list.read() {
                let results = search_results.read();
                let selected_index = selected_file_index.get();
                let max_display_files = 10;
                let display_results: Vec<_> = results.iter().take(max_display_files).enumerate().collect();

                Some(element! {
                    View(
                        key: "file-list",
                        width: width,
                        height: min(results.len(), max_display_files) as u16,
                        position: Position::Relative,
                    ) {
                        View(
                            flex_direction: FlexDirection::Column,
                            width: 100pct,
                            height: 100pct,
                            padding_left: 2,
                            padding_right: 2,
                        ) {
                            #(display_results.iter().map(|(idx, result)| {
                                let is_selected = *idx == selected_index;
                                element! {
                                    View(
                                        key: format!("file-{}", idx),
                                        height: 1,
                                        width: 100pct,
                                    ) {
                                        Text(
                                            content: result.display_name.clone(),
                                            color: if is_selected {
                                                Color::Rgb { r: 100, g: 149, b: 237 }
                                            } else {
                                                Color::DarkGrey
                                            },
                                        )
                                    }
                                }
                            }))
                        }
                    }
                })
            } else {
                None
            })
        }
    }
}

/// Spawn agent task execution and broadcast UI events
pub fn spawn_ui_agent_task(
    input: String,
    llm_config: ResolvedLlmConfig,
    project_path: PathBuf,
    ui_sender: broadcast::Sender<AppMessage>,
) {
    use crate::interactive::message_handler::get_random_status_word;
    use crate::interactive::task_executor::execute_agent_task;

    // Start with a random status word
    let _ = ui_sender.send(AppMessage::AgentTaskStarted {
        operation: get_random_status_word(),
    });

    // Create a cancellation token for the timer
    let (cancel_sender, mut cancel_receiver) = tokio::sync::oneshot::channel::<()>();

    // Change status word once after 1 second (unless cancelled)
    let ui_sender_timer = ui_sender.clone();
    tokio::spawn(async move {
        tokio::select! {
            _ = tokio::time::sleep(tokio::time::Duration::from_secs(1)) => {
                let _ = ui_sender_timer.send(AppMessage::AgentTaskStarted {
                    operation: get_random_status_word(),
                });
            }
            _ = &mut cancel_receiver => {
                // Timer cancelled, do nothing
            }
        }
    });

    // Execute agent task
    tokio::spawn(async move {
        match execute_agent_task(input, llm_config, project_path, ui_sender.clone()).await {
            Ok(_) => {
                let _ = cancel_sender.send(()); // Cancel the timer
                let _ = ui_sender.send(AppMessage::AgentExecutionCompleted);
            }
            Err(e) => {
                let _ = cancel_sender.send(()); // Cancel the timer
                                                // Check if it's an interruption error
                if e.to_string().contains("Task interrupted by user") {
                    // Don't show error message for user interruptions
                } else {
                    let _ = ui_sender.send(AppMessage::SystemMessage(format!("Error: {}", e)));
                }
                let _ = ui_sender.send(AppMessage::AgentExecutionCompleted);
            }
        }
    });
}

/// Spawn agent task execution with persistent agent for conversation continuity
pub fn spawn_ui_agent_task_with_context(
    input: String,
    llm_config: ResolvedLlmConfig,
    project_path: PathBuf,
    ui_sender: broadcast::Sender<AppMessage>,
    agent: Arc<Mutex<Option<coro_core::agent::AgentCore>>>,
) {
    use crate::interactive::message_handler::get_random_status_word;
    use crate::interactive::task_executor::execute_agent_task_with_context;

    // Start with a random status word
    let _ = ui_sender.send(AppMessage::AgentTaskStarted {
        operation: get_random_status_word(),
    });

    // Create a cancellation token for the timer
    let (cancel_sender, mut cancel_receiver) = tokio::sync::oneshot::channel::<()>();

    // Change status word once after 1 second (unless cancelled)
    let ui_sender_timer = ui_sender.clone();
    tokio::spawn(async move {
        tokio::select! {
            _ = tokio::time::sleep(tokio::time::Duration::from_secs(1)) => {
                let _ = ui_sender_timer.send(AppMessage::AgentTaskStarted {
                    operation: get_random_status_word(),
                });
            }
            _ = &mut cancel_receiver => {
                // Timer cancelled, do nothing
            }
        }
    });

    // Execute agent task with persistent context
    tokio::spawn(async move {
        match execute_agent_task_with_context(
            input,
            llm_config,
            project_path,
            ui_sender.clone(),
            agent,
        )
        .await
        {
            Ok(_) => {
                let _ = cancel_sender.send(()); // Cancel the timer
                let _ = ui_sender.send(AppMessage::AgentExecutionCompleted);
            }
            Err(e) => {
                let _ = cancel_sender.send(()); // Cancel the timer
                                                // Check if it's an interruption error
                if e.to_string().contains("Task interrupted by user") {
                    // Don't show error message for user interruptions
                } else {
                    let _ = ui_sender.send(AppMessage::SystemMessage(format!("Error: {}", e)));
                }
                let _ = ui_sender.send(AppMessage::AgentExecutionCompleted);
            }
        }
    });
}

/// Input Section Component - Fixed bottom area for input and status
#[component]
pub fn InputSection(mut hooks: Hooks, props: &InputSectionProps) -> impl Into<AnyElement<'static>> {
    // Subscribe to keyboard and dispatch events
    let context = &props.context;

    // Get terminal width for fixed input width
    let (terminal_width, _height) = hooks.use_terminal_size();
    let input_width = if terminal_width > 6 {
        terminal_width - 4 // Reserve space for padding and borders
    } else {
        80 // Fallback width
    };

    // Local input state
    let input_value = hooks.use_state(String::new);
    let is_task_running = hooks.use_state(|| false);
    let current_user_input = hooks.use_state(String::new);
    let cursor_position = hooks.use_state(|| (1usize, 1usize)); // (line, column)

    // Input history state
    let input_history = hooks.use_state(InputHistory::new);
    let history_initialized = hooks.use_state(|| false);

    // Router for navigation - coro-router API
    let router = use_router(&mut hooks);

    // Initialize cursor position when input value changes
    let mut cursor_position_init = cursor_position;
    let input_value_for_init = input_value;
    hooks.use_future(async move {
        let current_input = input_value_for_init.read().clone();
        let (line, col) = calculate_cursor_position(&current_input, current_input.len());
        cursor_position_init.set((line, col));
    });

    // Load history from file on first render
    hooks.use_future({
        let mut input_history = input_history;
        let mut history_initialized = history_initialized;
        async move {
            if !*history_initialized.read() {
                // Clone the history to avoid borrowing issues
                let mut history_clone = input_history.read().clone();
                let null_output = coro_core::output::NullOutput;
                if let Ok(()) = history_clone.load(&null_output).await {
                    // Ensure navigation is properly reset after loading
                    history_clone.reset_navigation();
                    // Update the state with the loaded history
                    input_history.set(history_clone);
                }
                history_initialized.set(true);
            }
        }
    });

    // Periodic save mechanism (every 5 seconds if needed)
    hooks.use_future({
        let mut input_history = input_history;
        async move {
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;

                // Check if history needs saving
                let mut history_clone = input_history.read().clone();
                if history_clone.needs_save() {
                    let null_output = coro_core::output::NullOutput;
                    if let Ok(()) = history_clone.save_if_needed(&null_output).await {
                        input_history.set(history_clone);
                    }
                }
            }
        }
    });

    // Navigation is now handled directly in keyboard events - no async needed

    // Subscribe to UI events to track task status
    let ui_sender_status = context.ui_sender.clone();
    let mut is_task_running_clone = is_task_running;
    let mut current_user_input_clone = current_user_input;
    hooks.use_future(async move {
        let mut rx = ui_sender_status.subscribe();
        while let Ok(event) = rx.recv().await {
            match event {
                AppMessage::AgentTaskStarted { .. } => {
                    is_task_running_clone.set(true);
                }
                AppMessage::AgentExecutionCompleted
                | AppMessage::AgentExecutionInterrupted { .. } => {
                    is_task_running_clone.set(false);
                    current_user_input_clone.set(String::new());
                }
                AppMessage::UserMessage(input) => {
                    current_user_input_clone.set(input);
                }
                _ => {}
            }
        }
    });

    let llm_config = context.llm_config.clone();
    let project_path = context.project_path.clone();
    let ui_sender = context.ui_sender.clone();

    // State to track file list visibility for history navigation
    let file_list_visible = hooks.use_state(|| false);

    // Handle keyboard events for task interruption and history navigation
    hooks.use_terminal_events({
        let ui_sender = ui_sender.clone();

        let mut input_value = input_value;
        let mut cursor_position = cursor_position;
        let mut input_history = input_history;
        move |event| {
            match event {
                TerminalEvent::Key(KeyEvent { code, kind, .. })
                    if kind != KeyEventKind::Release =>
                {
                    match code {
                        KeyCode::Esc => {
                            // Handle ESC key - interrupt current task if running
                            if *is_task_running.read() {
                                let user_input = current_user_input.read().clone();
                                let _ = ui_sender
                                    .send(AppMessage::AgentExecutionInterrupted { user_input });
                            }
                        }
                        KeyCode::Up => {
                            // Navigate to previous history entry only if file list is not showing
                            if !*is_task_running.read() && !*file_list_visible.read() {
                                let current_input = input_value.read().clone();
                                if let Some(history_text) =
                                    input_history.write().navigate_previous(&current_input)
                                {
                                    input_value.set(history_text.clone());
                                    cursor_position.set((1, history_text.len() + 1));
                                }
                            }
                        }
                        KeyCode::Down => {
                            // Navigate to next history entry only if file list is not showing
                            if !*is_task_running.read() && !*file_list_visible.read() {
                                if let Some(history_text) = input_history.write().navigate_next() {
                                    input_value.set(history_text.clone());
                                    cursor_position.set((1, history_text.len() + 1));
                                }
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    });

    element! {
        View(
            key: "input-section",
            flex_shrink: 0.0,
            flex_grow: 0.0,
            flex_direction: FlexDirection::Column,
            position: Position::Relative,
        ) {
            // Enhanced multiline input area with fixed width and auto height
            EnhancedTextInput(
                key: "enhanced-text-input",
                value: input_value.to_string(),
                has_focus: !*is_task_running.read(),
                width: input_width,
                placeholder: "Type your message or @path/to/file (Enter to send, Shift+Enter for new line)".to_string(),
                color: Some(Color::White),
                cursor_color: Some(Color::Rgb { r: 100, g: 149, b: 237 }),
                project_path: project_path.clone(),
                on_change: {
                    let mut input_value = input_value;
                    move |new_value| {
                        input_value.set(new_value);
                    }
                },
                on_cursor_position_change: {
                    let mut cursor_position = cursor_position;
                    move |(line, col)| {
                        cursor_position.set((line, col));
                    }
                },
                on_file_list_state_change: {
                    let mut file_list_visible = file_list_visible;
                    move |is_visible| {
                        file_list_visible.set(is_visible);
                    }
                },
                on_submit: {
                    let mut input_value = input_value;
                    let mut cursor_position = cursor_position;
                    let mut input_history = input_history;
                    let mut router = router.clone();
                    let ui_sender = ui_sender.clone();
                    let llm_config = llm_config.clone();
                    let project_path = project_path.clone();
                    let agent = context.agent.clone();
                    move |input: String| {
                        if input.trim().is_empty() {
                            return;
                        }

                        // Check if input is "/test" and navigate to test page
                        if input.trim().to_lowercase() == "/test" {
                            // Navigate directly - no async needed!
                            let _ = router.navigate("router_test");
                            input_value.set(String::new());
                            cursor_position.set((1, 1));
                            return;
                        }

                        // Add to history before clearing input (fast, no I/O)
                        let input_for_history = input.clone();
                        let mut history_clone = input_history.read().clone();

                        // Add entry using fast method (no file I/O)
                        history_clone.add_entry(input_for_history);
                        input_history.set(history_clone);

                        // Clear input immediately and reset cursor position
                        input_value.set(String::new());
                        cursor_position.set((1, 1));

                        // Use enhanced task submission with file reference processing
                        crate::interactive::app::submit_task_with_file_processing(
                            input,
                            llm_config.clone(),
                            project_path.clone(),
                            ui_sender.clone(),
                            agent.clone(),
                        );
                    }
                },
            )

            // Status bar - 简约风格
            View(padding: 1) {
                Text(
                    content: {
                        let (line, col) = cursor_position.get();
                        format!(
                            "~/projects/trae-agent-rs (main*)  Ln {}, Col {}",
                            line, col
                        )
                    },
                    color: Color::DarkGrey,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_section_props_default() {
        let props = InputSectionProps::default();
        // Just ensure it compiles and creates successfully
        let _ = props;
    }

    #[test]
    fn test_calculate_cursor_position() {
        // Test empty string
        assert_eq!(calculate_cursor_position("", 0), (1, 1));

        // Test single line
        assert_eq!(calculate_cursor_position("hello", 0), (1, 1));
        assert_eq!(calculate_cursor_position("hello", 2), (1, 3));
        assert_eq!(calculate_cursor_position("hello", 5), (1, 6));

        // Test multiple lines
        let multiline = "line1\nline2\nline3";
        assert_eq!(calculate_cursor_position(multiline, 0), (1, 1)); // Start of first line
        assert_eq!(calculate_cursor_position(multiline, 5), (1, 6)); // End of first line
        assert_eq!(calculate_cursor_position(multiline, 6), (2, 1)); // Start of second line
        assert_eq!(calculate_cursor_position(multiline, 11), (2, 6)); // End of second line
        assert_eq!(calculate_cursor_position(multiline, 12), (3, 1)); // Start of third line
        assert_eq!(calculate_cursor_position(multiline, 17), (3, 6)); // End of third line

        // Test Unicode characters
        let unicode_text = "你好\n世界";
        assert_eq!(calculate_cursor_position(unicode_text, 0), (1, 1)); // Start
        assert_eq!(calculate_cursor_position(unicode_text, 3), (1, 2)); // After first char
        assert_eq!(calculate_cursor_position(unicode_text, 6), (1, 3)); // After second char
        assert_eq!(calculate_cursor_position(unicode_text, 7), (2, 1)); // Start of second line
        assert_eq!(calculate_cursor_position(unicode_text, 13), (2, 3)); // End
    }

    #[test]
    fn test_calculate_cursor_display_position() {
        // Test empty string
        assert_eq!(calculate_cursor_display_position("", 0, 80), (0, 0));

        // Test single line
        assert_eq!(calculate_cursor_display_position("hello", 0, 80), (0, 0));
        assert_eq!(calculate_cursor_display_position("hello", 2, 80), (0, 2));
        assert_eq!(calculate_cursor_display_position("hello", 5, 80), (0, 5));

        // Test multiple lines
        let multiline = "line1\nline2\nline3";
        assert_eq!(calculate_cursor_display_position(multiline, 0, 80), (0, 0)); // Start of first line
        assert_eq!(calculate_cursor_display_position(multiline, 5, 80), (0, 5)); // End of first line
        assert_eq!(calculate_cursor_display_position(multiline, 6, 80), (1, 0)); // Start of second line
        assert_eq!(calculate_cursor_display_position(multiline, 11, 80), (1, 5)); // End of second line
        assert_eq!(calculate_cursor_display_position(multiline, 12, 80), (2, 0)); // Start of third line
        assert_eq!(calculate_cursor_display_position(multiline, 17, 80), (2, 5)); // End of third line

        // Test Unicode characters (Chinese characters have width 2)
        let unicode_text = "你好\n世界";
        assert_eq!(
            calculate_cursor_display_position(unicode_text, 0, 80),
            (0, 0)
        ); // Start
        assert_eq!(
            calculate_cursor_display_position(unicode_text, 3, 80),
            (0, 2)
        ); // After first char (width 2)
        assert_eq!(
            calculate_cursor_display_position(unicode_text, 6, 80),
            (0, 4)
        ); // After second char (width 2)
        assert_eq!(
            calculate_cursor_display_position(unicode_text, 7, 80),
            (1, 0)
        ); // Start of second line
        assert_eq!(
            calculate_cursor_display_position(unicode_text, 13, 80),
            (1, 4)
        ); // End (width 2 per char)
    }

    #[test]
    fn test_calculate_cursor_display_position_soft_wrap_ascii() {
        // width = 4, ASCII widths = 1 per char
        let text = "abcdefgh";
        // Start
        assert_eq!(calculate_cursor_display_position(text, 0, 4), (0, 0));
        // Within first line
        assert_eq!(calculate_cursor_display_position(text, 1, 4), (0, 1));
        assert_eq!(calculate_cursor_display_position(text, 4, 4), (0, 4)); // exactly fits
                                                                           // Wrap to second display line
        assert_eq!(calculate_cursor_display_position(text, 5, 4), (1, 1));
        assert_eq!(calculate_cursor_display_position(text, 8, 4), (1, 4));
    }

    #[test]
    fn test_calculate_cursor_display_position_soft_wrap_unicode() {
        // width = 4, Chinese characters have width 2
        let text = "你好世界"; // 4 chars, each width 2
                               // Byte positions for each char boundary: 0, 3, 6, 9, 12
        assert_eq!(calculate_cursor_display_position(text, 0, 4), (0, 0));
        assert_eq!(calculate_cursor_display_position(text, 3, 4), (0, 2));
        assert_eq!(calculate_cursor_display_position(text, 6, 4), (0, 4)); // exactly fits
        assert_eq!(calculate_cursor_display_position(text, 9, 4), (1, 2)); // wrapped
        assert_eq!(calculate_cursor_display_position(text, 12, 4), (1, 4));
    }
}
