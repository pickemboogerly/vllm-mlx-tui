"""
Root Textual application.

Handles the outer navigation flow:
  Launcher → Chat (loop on model-switch) → Quit
"""
from __future__ import annotations

import asyncio
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header

from .launcher import LauncherScreen
from .help import HelpModal

APP_VERSION = "0.1.0"


class VLLMMlxTUIApp(App):
    """Root application shell."""

    CSS = """
    /* ---- Global design tokens ---- */
    /* We use Textual's standard variables here so they can be overridden by themes. */
    
    Screen {
        background: $background;
        color: $text;
    }

    /* Scrollbars */
    ScrollBar {
        background: $surface;
        color: $border;
    }
    ScrollBar:hover {
        color: $accent;
    }

    .hidden {
        display: none;
    }
    .error-text {
        color: $error;
    }
    .success-text {
        color: $success;
    }
    .muted {
        color: $text-muted;
    }
    """

    TITLE = f"vllm-mlx-tui  v{APP_VERSION}"

    BINDINGS = [
        Binding("f1,ctrl+h", "show_help", "Help Reference"),
        Binding("ctrl+q", "quit_app", "Exits"),
    ]

    def on_mount(self) -> None:
        self.push_screen(LauncherScreen())

    def action_show_help(self) -> None:
        self.push_screen(HelpModal())

    def action_quit_app(self) -> None:
        # Standardize app shutdown
        if hasattr(self.screen, "manager") and self.screen.manager:
             self.screen.manager.stop()
        self.exit()
