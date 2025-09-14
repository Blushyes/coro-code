//! Interactive application using iocraft

use crate::interactive::animation::UiAnimationConfig;
use crate::interactive::components::input_section::InputSectionContext;
use crate::interactive::components::logo::output_logo_to_terminal;
use crate::interactive::components::status_line::StatusLineContext;
use crate::interactive::message_handler::{app_message_to_ui_message, AppMessage};
use crate::interactive::pages::main_page::MainPage;
use crate::interactive::pages::router_test::RouterTestPage;
use crate::interactive::terminal_output::{output_content_block, overwrite_previous_lines};
use anyhow::Result;
use coro_core::ResolvedLlmConfig;
use coro_router as router;
use iocraft::prelude::*;
use regex::Regex;
use router::integration::{ReactiveRouterHandle, RouterContext};
use router::UIRouter;
use router::UIRouterBuilder;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex};

/// Represents a file reference found in user input
#[derive(Debug, Clone)]
struct FileReference {
    /// The original reference text (e.g., "@src/main.rs")
    pub original: String,
    /// The resolved absolute path
    pub path: PathBuf,
    /// Start position in the input string
    pub start: usize,
    /// End position in the input string
    pub end: usize,
}

/// Parse file references from user input
fn parse_file_references(input: &str, project_path: &Path) -> Vec<FileReference> {
    let mut references = Vec::new();

    // Regex to match @path patterns
    // Matches: @path/to/file, @path/to/file/, @path/to/file followed by space/end
    // Updated to stop at Chinese characters or other non-ASCII word characters
    let re = Regex::new(r"@([a-zA-Z0-9._/-]+/?)").expect("Invalid regex pattern");

    for cap in re.captures_iter(input) {
        if let Some(path_match) = cap.get(1) {
            let path_str = path_match.as_str();
            let full_match = cap.get(0).unwrap();

            // Resolve path relative to project root
            let resolved_path = if path_str.starts_with('/') {
                // Absolute path
                PathBuf::from(path_str)
            } else {
                // Relative path
                project_path.join(path_str)
            };

            references.push(FileReference {
                original: full_match.as_str().to_string(),
                path: resolved_path,
                start: full_match.start(),
                end: full_match.end(),
            });
        }
    }

    references
}

/// Read file content or directory metadata and return formatted content for AI context
async fn read_file_content(file_path: &PathBuf) -> Result<String> {
    use tokio::fs;

    // Check if path is a directory
    if file_path.is_dir() {
        return read_directory_metadata(file_path).await;
    }

    // Handle regular files
    let content = fs::read_to_string(file_path).await?;
    let line_count = content.lines().count();

    // Format content for AI context
    let formatted_content = format!(
        "File: {}\nLines: {}\nContent:\n```\n{}\n```",
        file_path.display(),
        line_count,
        content
    );

    Ok(formatted_content)
}

/// Read directory metadata and return formatted information for AI context
async fn read_directory_metadata(dir_path: &PathBuf) -> Result<String> {
    use tokio::fs;

    let mut entries = fs::read_dir(dir_path).await?;
    let mut files = Vec::new();
    let mut directories = Vec::new();
    let mut total_size = 0u64;

    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        let metadata = entry.metadata().await?;
        let file_name = entry.file_name().to_string_lossy().to_string();

        // Skip hidden files/directories
        if file_name.starts_with('.') {
            continue;
        }

        if metadata.is_dir() {
            // For directories, count subdirectories and files
            let mut subdir_count = 0;
            let mut subfile_count = 0;
            if let Ok(mut subentries) = fs::read_dir(&path).await {
                while let Ok(Some(subentry)) = subentries.next_entry().await {
                    let subname = subentry.file_name().to_string_lossy().to_string();
                    if !subname.starts_with('.') {
                        if let Ok(submeta) = subentry.metadata().await {
                            if submeta.is_dir() {
                                subdir_count += 1;
                            } else {
                                subfile_count += 1;
                            }
                        }
                    }
                }
            }

            directories.push((file_name, subdir_count, subfile_count));
        } else {
            let size = metadata.len();
            total_size += size;

            // Format file size in human readable format
            let size_str = if size < 1024 {
                format!("{} B", size)
            } else if size < 1024 * 1024 {
                format!("{:.1} KB", size as f64 / 1024.0)
            } else if size < 1024 * 1024 * 1024 {
                format!("{:.1} MB", size as f64 / (1024.0 * 1024.0))
            } else {
                format!("{:.1} GB", size as f64 / (1024.0 * 1024.0 * 1024.0))
            };

            files.push((file_name, size_str));
        }
    }

    // Format directory information for AI context
    let mut formatted_content = format!("Directory: {}\n", dir_path.display());
    formatted_content.push_str(&format!(
        "Total items: {} files, {} directories\n",
        files.len(),
        directories.len()
    ));

    if !files.is_empty() {
        let total_size_str = if total_size < 1024 {
            format!("{} B", total_size)
        } else if total_size < 1024 * 1024 {
            format!("{:.1} KB", total_size as f64 / 1024.0)
        } else if total_size < 1024 * 1024 * 1024 {
            format!("{:.1} MB", total_size as f64 / (1024.0 * 1024.0))
        } else {
            format!("{:.1} GB", total_size as f64 / (1024.0 * 1024.0 * 1024.0))
        };
        formatted_content.push_str(&format!("Total size: {}\n\n", total_size_str));
    } else {
        formatted_content.push('\n');
    }

    // List directories first
    if !directories.is_empty() {
        formatted_content.push_str("Directories:\n");
        for (name, subdir_count, subfile_count) in directories {
            formatted_content.push_str(&format!(
                "  📁 {} ({} dirs, {} files)\n",
                name, subdir_count, subfile_count
            ));
        }
        formatted_content.push('\n');
    }

    // List files
    if !files.is_empty() {
        formatted_content.push_str("Files:\n");
        for (name, size) in files {
            formatted_content.push_str(&format!("  📄 {} ({})\n", name, size));
        }
    }

    Ok(formatted_content)
}

/// Process user input with file references and return enhanced input for AI
async fn process_input_with_file_references(
    input: String,
    project_path: &Path,
    _ui_sender: &broadcast::Sender<AppMessage>,
) -> Result<(String, Vec<String>)> {
    let file_refs = parse_file_references(&input, project_path);

    if file_refs.is_empty() {
        // No file references, return original input with empty messages
        return Ok((input, Vec::new()));
    }

    let mut enhanced_input = input.clone();
    let mut file_contents = Vec::new();
    let mut file_read_messages = Vec::new();

    // Read all referenced files and collect read messages
    for file_ref in &file_refs {
        let file_name = file_ref
            .path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        match read_file_content(&file_ref.path).await {
            Ok(content) => {
                // Determine if this is a directory or file based on content format
                let is_directory = content.starts_with("Directory:");

                if is_directory {
                    // For directories, extract item count from the content
                    let item_info = if let Some(line) = content.lines().nth(1) {
                        // Extract the "Total items: X files, Y directories" line
                        if line.starts_with("Total items:") {
                            line.replace("Total items:", "").trim().to_string()
                        } else {
                            "directory".to_string()
                        }
                    } else {
                        "directory".to_string()
                    };

                    file_read_messages.push(format!("⎿ Listed {} ({})", file_name, item_info));
                } else {
                    // For files, count lines as before
                    let line_count = content.lines().count();
                    file_read_messages.push(format!("⎿ Read {} ({} lines)", file_name, line_count));
                }

                file_contents.push(content);
            }
            Err(e) => {
                // Collect error message instead of sending immediately
                file_read_messages.push(format!("⎿ Failed to read {}: {}", file_name, e));
            }
        }
    }

    // If we successfully read any files, append their content to the input
    if !file_contents.is_empty() {
        enhanced_input.push_str("\n\n--- Referenced Files ---\n");
        for content in file_contents {
            enhanced_input.push_str(&content);
            enhanced_input.push_str("\n\n");
        }
    }

    Ok((enhanced_input, file_read_messages))
}

/// Enhanced task submission with file reference processing
pub fn submit_task_with_file_processing(
    input: String,
    llm_config: ResolvedLlmConfig,
    project_path: PathBuf,
    ui_sender: broadcast::Sender<AppMessage>,
    agent: Arc<Mutex<Option<coro_core::agent::AgentCore>>>,
) {
    use crate::interactive::message_handler::get_random_status_word;

    // Process file references asynchronously and send combined message
    let ui_sender_clone = ui_sender.clone();
    let llm_config_clone = llm_config.clone();
    let project_path_clone = project_path.clone();
    let agent_clone = agent.clone();

    tokio::spawn(async move {
        let input_clone = input.clone();
        match process_input_with_file_references(input, &project_path_clone, &ui_sender_clone).await
        {
            Ok((enhanced_input, file_read_messages)) => {
                // Send combined user message with file read info
                let combined_message = if file_read_messages.is_empty() {
                    input_clone
                } else {
                    format!("{}\n{}", input_clone, file_read_messages.join("\n"))
                };

                let _ = ui_sender_clone.send(AppMessage::UserMessage(combined_message));

                // Start with a random status word
                let _ = ui_sender_clone.send(AppMessage::AgentTaskStarted {
                    operation: get_random_status_word(),
                });

                // Use the enhanced spawn_ui_agent_task_with_context with enhanced input
                crate::interactive::components::input_section::spawn_ui_agent_task_with_context(
                    enhanced_input,
                    llm_config_clone,
                    project_path_clone,
                    ui_sender_clone,
                    agent_clone.clone(),
                );
            }
            Err(e) => {
                // Send combined user message with error info
                let error_msg = format!("⎿ Error processing file references: {}", e);
                let combined_message = format!("{}\n{}", input_clone, error_msg);
                let _ = ui_sender_clone.send(AppMessage::UserMessage(combined_message));

                // Start with a random status word
                let _ = ui_sender_clone.send(AppMessage::AgentTaskStarted {
                    operation: get_random_status_word(),
                });

                // Fall back to original input
                crate::interactive::components::input_section::spawn_ui_agent_task_with_context(
                    input_clone,
                    llm_config_clone,
                    project_path_clone,
                    ui_sender_clone,
                    agent_clone,
                );
            }
        }
    });
}

/// Context for interactive mode - immutable application configuration
#[derive(Clone)]
struct AppContext {
    llm_config: ResolvedLlmConfig,
    project_path: PathBuf,
    ui_sender: broadcast::Sender<AppMessage>,
    ui_anim: UiAnimationConfig,
    debug_model: bool,
    // Persistent agent instance for conversation continuity
    agent: Arc<Mutex<Option<coro_core::agent::AgentCore>>>,
}

impl AppContext {
    fn new(
        llm_config: ResolvedLlmConfig,
        project_path: PathBuf,
        ui_sender: broadcast::Sender<AppMessage>,
        debug_model: bool,
    ) -> Self {
        let ui_anim = UiAnimationConfig::from_env();

        Self {
            llm_config,
            project_path,
            ui_sender,
            ui_anim,
            debug_model,
            agent: Arc::new(Mutex::new(None)),
        }
    }
}

/// Interactive mode using iocraft
pub async fn run_rich_interactive(
    llm_config: ResolvedLlmConfig,
    project_path: PathBuf,
    debug_model: bool,
) -> Result<()> {
    // Create UI broadcast channel and app context
    let (ui_sender, _) = broadcast::channel::<AppMessage>(256);
    let app_context = AppContext::new(llm_config, project_path, ui_sender, debug_model);

    // Run the iocraft-based UI with context provider in render loop mode
    tokio::task::spawn_blocking(move || {
        smol::block_on(async {
            (element! {
                ContextProvider(value: Context::owned(app_context)) {
                    CoroApp
                }
            })
            .render_loop()
            .await
        })
    })
    .await??;

    Ok(())
}

/// Main TRAE Interactive Application Component
#[component]
fn CoroApp(mut hooks: Hooks) -> impl Into<AnyElement<'static>> {
    // Get stdout handle for header and messages output
    let (stdout, _stderr) = hooks.use_output();

    // Local state for header and messages
    let show_tips = hooks.use_state(|| true);
    let header_rendered = hooks.use_state(|| false);
    let messages = hooks.use_state(Vec::<(String, String, Option<String>)>::new);
    // Track line counts for each message to enable proper overwriting
    let message_line_counts = hooks.use_state(std::collections::HashMap::<String, usize>::new);

    let (width, _height) = hooks.use_terminal_size();
    // Get current terminal width and reserve space for padding/borders
    let raw_width = if width as usize > 0 {
        width as usize
    } else {
        crossterm::terminal::size()
            .map(|(w, _)| w as usize)
            .unwrap_or(80)
    };
    let terminal_width = raw_width.saturating_sub(6);
    let terminal_width = std::cmp::max(terminal_width, 60);

    // Get app context
    let app_context = hooks.use_context::<AppContext>();
    let ui_sender = app_context.ui_sender.clone();

    // Subscribe to UI events for header tips management
    let ui_sender_tips = ui_sender.clone();
    let mut show_tips_clone = show_tips;
    hooks.use_future(async move {
        let mut rx = ui_sender_tips.subscribe();
        while let Ok(msg) = rx.recv().await {
            if app_message_to_ui_message(msg).is_some() {
                // If has any message, hide tips
                if *show_tips_clone.read() {
                    show_tips_clone.set(false);
                }
            }
        }
    });

    // Output header to stdout when component mounts
    let stdout_clone = stdout.clone();
    let show_tips_for_output = show_tips;
    let mut header_rendered_clone = header_rendered;
    hooks.use_future(async move {
        if !*header_rendered_clone.read() {
            // Use the logo output function from the logo module
            output_logo_to_terminal(&stdout_clone);

            // Output tips if they should be shown
            if *show_tips_for_output.read() {
                stdout_clone.println("Tips for getting started:");
                stdout_clone.println("1. Ask questions, edit files, or run commands.");
                stdout_clone.println("2. Be specific for the best results.");
                stdout_clone.println("3. /help for more information.");
                stdout_clone.println(""); // Empty line for spacing
            }

            header_rendered_clone.set(true);
        }
    });

    // Subscribe to UI events for messages output
    let ui_sender_messages = ui_sender.clone();
    let mut messages_clone = messages;
    let mut message_line_counts_clone: State<std::collections::HashMap<String, usize>> =
        message_line_counts;
    let stdout_messages = stdout.clone();
    hooks.use_future(async move {
        let mut rx = ui_sender_messages.subscribe();
        while let Ok(app_message) = rx.recv().await {
            if let Some((role, content, message_id, is_bash_output)) =
                app_message_to_ui_message(app_message)
            {
                use crate::interactive::message_handler::identify_content_block;

                let mut current: Vec<(String, String, Option<String>)> =
                    messages_clone.read().clone();
                let mut line_counts = message_line_counts_clone.read().clone();
                let is_new_message = if let Some(msg_id) = &message_id {
                    if let Some(pos) = current
                        .iter()
                        .position(|(_, _, id)| id.as_ref() == Some(msg_id))
                    {
                        current[pos] = (role.clone(), content.clone(), Some(msg_id.clone()));
                        false // Updated existing message
                    } else {
                        current.push((role.clone(), content.clone(), Some(msg_id.clone())));
                        true // New message
                    }
                } else {
                    current.push((role.clone(), content.clone(), None));
                    true // New message
                };

                // Output messages to stdout using block-based formatting
                // For new messages, output normally
                // For updated messages (like tool status changes), we need to handle the replacement
                if is_new_message {
                    let block_type = identify_content_block(&content, &role);
                    let total_lines = output_content_block(
                        &stdout_messages,
                        &content,
                        block_type,
                        terminal_width,
                        is_bash_output,
                    );

                    // Store line count for this message
                    if let Some(msg_id) = &message_id {
                        line_counts.insert(msg_id.clone(), total_lines);
                    }
                } else {
                    // This is an updated message (e.g., tool status change from executing to completed)
                    // We need to replace the previous line with the new content

                    // Get the previous line count for this message
                    let previous_lines = if let Some(msg_id) = &message_id {
                        line_counts.get(msg_id).copied().unwrap_or(2) // Default to 2 lines if not found
                    } else {
                        2 // Default fallback
                    };

                    // Use the helper function to overwrite previous lines
                    let new_total_lines = overwrite_previous_lines(
                        &stdout_messages,
                        &content,
                        &role,
                        terminal_width,
                        previous_lines,
                    );

                    // Update line count for this message
                    if let Some(msg_id) = &message_id {
                        line_counts.insert(msg_id.clone(), new_total_lines);
                    }
                }

                messages_clone.set(current);
                message_line_counts_clone.set(line_counts);
            }
        }
    });

    // Create contexts for child components
    let status_context = StatusLineContext {
        ui_sender: app_context.ui_sender.clone(),
        ui_anim: app_context.ui_anim.clone(),
    };

    let input_context = InputSectionContext {
        llm_config: app_context.llm_config.clone(),
        project_path: app_context.project_path.clone(),
        ui_sender: app_context.ui_sender.clone(),
        agent: app_context.agent.clone(),
    };

    // 使用 coro-router 构建路由并提供上下文
    let build_result = UIRouterBuilder::new()
        .route("main", "Main", move |_hooks| {
            element! {
                MainPage(
                    status_context: status_context.clone(),
                    input_context: input_context.clone()
                )
            }
            .into()
        })
        .route("router_test", "Router Test", move |_hooks| {
            element! { RouterTestPage() }.into()
        })
        .default("main")
        .build()
        .expect("Failed to build router");

    // 创建 RouterHandle 与 Context
    let handle = ReactiveRouterHandle::new_with_hooks(&mut hooks, build_result.config.clone());
    let router_context = RouterContext { handle };
    let shared_build_result = Arc::new(build_result);

    element! {
        ContextProvider(value: Context::owned(router_context)) {
            UIRouter(build_result: shared_build_result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::PathBuf;

    #[test]
    fn test_parse_file_references() {
        let project_path = PathBuf::from("/project");

        // Test single file reference
        let input = "请分析 @src/main.rs 这个文件";
        let refs = parse_file_references(input, &project_path);
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].original, "@src/main.rs");
        assert_eq!(refs[0].path, PathBuf::from("/project/src/main.rs"));

        // Test multiple file references
        let input = "@src/main.rs 和 @lib/utils.rs 这两个文件";
        let refs = parse_file_references(input, &project_path);
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].original, "@src/main.rs");
        assert_eq!(refs[1].original, "@lib/utils.rs");

        // Test absolute path
        let input = "查看 @/absolute/path/file.txt";
        let refs = parse_file_references(input, &project_path);
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].path, PathBuf::from("/absolute/path/file.txt"));

        // Test path ending with /
        let input = "@relative/path/file.py/请分析这个文件";
        let refs = parse_file_references(input, &project_path);
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].original, "@relative/path/file.py/");
    }
}
