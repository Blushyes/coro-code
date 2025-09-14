//! Router test page component
//!
//! This page provides a test interface for the router system with navigation options.

use coro_router::use_router;
use iocraft::prelude::*;

/// Navigation options available on the router test page
#[derive(Debug, Clone, Copy, PartialEq)]
enum NavigationOption {
    GoBack,
    GoHome,
}

impl NavigationOption {
    /// Get the display text for this option
    fn display_text(&self) -> &'static str {
        match self {
            NavigationOption::GoBack => "Go Back",
            NavigationOption::GoHome => "Go Home",
        }
    }

    /// Get the route ID for this option
    fn route_id(&self) -> &'static str {
        match self {
            NavigationOption::GoBack => "main", // For now, go back means go to main
            NavigationOption::GoHome => "main",
        }
    }

    /// Get all available options
    fn all_options() -> Vec<NavigationOption> {
        vec![NavigationOption::GoBack, NavigationOption::GoHome]
    }
}

/// Properties for the router test page component
#[derive(Props, Default)]
pub struct RouterTestPageProps {}

/// Router test page component that provides navigation testing functionality
#[component]
pub fn RouterTestPage(
    mut hooks: Hooks,
    _props: &RouterTestPageProps,
) -> impl Into<AnyElement<'static>> {
    // Get router from coro-router
    let router = use_router(&mut hooks);

    // State for currently selected option
    let selected_option = hooks.use_state(|| NavigationOption::GoBack);

    // Handle keyboard input
    hooks.use_terminal_events({
        let mut selected_option = selected_option;
        let mut router = router.clone();
        move |event| {
            if let TerminalEvent::Key(key_event) = event {
                match key_event.code {
                    KeyCode::Up => {
                        // Move selection up
                        let current = *selected_option.read();
                        let options = NavigationOption::all_options();
                        if let Some(current_index) = options.iter().position(|&opt| opt == current)
                        {
                            let new_index = if current_index == 0 {
                                options.len() - 1
                            } else {
                                current_index - 1
                            };
                            selected_option.set(options[new_index]);
                        }
                    }
                    KeyCode::Down => {
                        // Move selection down
                        let current = *selected_option.read();
                        let options = NavigationOption::all_options();
                        if let Some(current_index) = options.iter().position(|&opt| opt == current)
                        {
                            let new_index = (current_index + 1) % options.len();
                            selected_option.set(options[new_index]);
                        }
                    }
                    KeyCode::Enter => {
                        // Execute selected navigation
                        let current = *selected_option.read();
                        let route_id = current.route_id();

                        // Navigate using the reactive router system
                        if let Err(e) = router.navigate(route_id) {
                            // Handle navigation error (could log or show error message)
                            eprintln!("Navigation error: {:?}", e);
                        }
                    }
                    _ => {}
                }
            }
        }
    });

    let current_selection = *selected_option.read();

    element! {
        View(
            key: "router-test-page",
            flex_direction: FlexDirection::Column,
            height: 100pct,
            width: 100pct,
            padding: 2,
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
        ) {
            // Page title
            Text(
                content: "Router Test Page",
                weight: Weight::Bold,
                color: Color::Cyan
            )

            // Spacer
            View(height: 2)

            // Description
            Text(
                content: "Use ↑/↓ arrows to navigate, Enter to select",
                color: Color::Grey
            )

            // Spacer
            View(height: 1)

            // Navigation options
            View(
                flex_direction: FlexDirection::Column,
                gap: 1,
                align_items: AlignItems::Center,
            ) {
                #(NavigationOption::all_options().into_iter().map(|option| {
                    let is_selected = option == current_selection;
                    let text_color = if is_selected { Color::Yellow } else { Color::White };
                    let prefix = if is_selected { "► " } else { "  " };

                    element! {
                        Text(
                            key: format!("nav-option-{:?}", option),
                            content: format!("{}{}", prefix, option.display_text()),
                            color: text_color,
                            weight: if is_selected { Weight::Bold } else { Weight::Normal }
                        )
                    }
                }).collect::<Vec<_>>())
            }

            // Spacer
            View(height: 3)

            // Instructions
            View(
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
            ) {
                Text(
                    content: "This page demonstrates the reactive router system",
                    color: Color::Grey
                )
                Text(
                    content: "Navigation is handled synchronously without async complexity",
                    color: Color::Grey
                )
            }
        }
    }
}
