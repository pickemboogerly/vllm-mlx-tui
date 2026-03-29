"""
Embellished shortcut reference for a premium TUI experience.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, ScrollableContainer, Vertical, Horizontal, Grid
import textual.screen
from textual.widgets import Footer, Label, Static


class HelpModal(textual.screen.ModalScreen):
    """Universal shortcut reference."""

    BINDINGS = [
        Binding("escape,f1,q", "dismiss", "Close Help"),
    ]

    DEFAULT_CSS = """
    HelpModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.7);
    }
    #help-outer {
        width: 80;
        height: auto;
        min-height: 20;
        max-height: 40;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
        layout: vertical;
    }
    #help-header {
        height: auto;
        border-bottom: solid $border;
        margin-bottom: 1;
        padding-bottom: 1;
        layout: vertical;
        content-align: center middle;
    }
    #help-title {
        text-style: bold;
        color: $primary;
        width: 100%;
        content-align: center middle;
        margin-bottom: 0;
    }
    #help-scroll {
        height: auto;
        max-height: 25;
        padding: 1;
    }
    .help-section {
        height: auto;
        margin-bottom: 1;
        border-left: round $primary-darken-2;
        padding-left: 2;
    }
    .help-section-title {
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }
    .shortcut-row {
        height: 1;
        layout: horizontal;
    }
    .shortcut-key {
        width: 14;
        color: $success;
        text-style: bold;
    }
    .shortcut-desc {
        width: 1fr;
        color: $text;
    }
    .help-hint {
        color: $text-muted;
        text-style: italic;
        content-align: center middle;
        margin-top: 1;
    }
    .palette-info {
        border: dashed $border;
        padding: 1;
        margin: 1 0;
        color: $accent;
        text-align: center;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="help-outer"):
            with Vertical(id="help-header"):
                yield Label("⌨️  VLLM-MLX COMMAND REFERENCE", id="help-title")
                yield Label("Keyboard-powered interface for local AI server lifecycle.", classes="help-hint")

            with ScrollableContainer(id="help-scroll"):
                # System Category
                with Vertical(classes="help-section"):
                    yield Label("Global / System", classes="help-section-title")
                    yield from self._shortcut("ctrl+q", "Quit / Exit application")
                    yield from self._shortcut("f1", "Close / Show help")
                    yield from self._shortcut("ctrl+h", "Help Reference")
                    yield from self._shortcut("ctrl+\\", "Native Command Palette")

                # Navigation
                with Vertical(classes="help-section"):
                    yield Label("Navigation", classes="help-section-title")
                    yield from self._shortcut("tab", "Move focus forward")
                    yield from self._shortcut("shift+tab", "Move focus backward")
                    yield from self._shortcut("enter", "Execute Action / Send")

                # Launcher Specific
                with Vertical(classes="help-section"):
                    yield Label("Launcher / Server", classes="help-section-title")
                    yield from self._shortcut("enter", "Launch selected model server")
                    yield from self._shortcut("ctrl+s", "Save configuration to profile")
                    yield from self._shortcut("ctrl+b", "Focus & scroll boot log")
                    yield from self._shortcut("ctrl+c", "Copy boot log to clipboard")
                    yield from self._shortcut("ctrl+m", "Monitor server metrics")

                # Chat Specific
                with Vertical(classes="help-section"):
                    yield Label("Chat Interaction", classes="help-section-title")
                    yield from self._shortcut("enter", "Send message to assistant")
                    yield from self._shortcut("ctrl+n", "Start new chat session")
                    yield from self._shortcut("ctrl+l", "Clear chat history")
                    yield from self._shortcut("f2", "Rename chat session (auto-save)")
                    yield from self._shortcut("ctrl+e", "Export chat to Markdown")
                    yield from self._shortcut("ctrl+r", "Switch back to model launcher")

                # Branding/Extra
                yield Label("💡 Pro-Tip: Use GFM table syntax (|---|) for beautiful native tables in chat.", classes="palette-info")

            yield Label("Press ESC or Q to resume", classes="help-hint")

    def _shortcut(self, key: str, desc: str) -> ComposeResult:
        with Horizontal(classes="shortcut-row"):
            yield Label(key.upper(), classes="shortcut-key")
            yield Label(desc, classes="shortcut-desc")
