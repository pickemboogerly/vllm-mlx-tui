"""
Modal help display showing all current keyboard shortcuts and their functions.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, ScrollableContainer, Vertical, Horizontal
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Label, Static


class HelpModal(ModalScreen):
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
        width: 85;
        height: 35;
        background: #161b22;
        border: thick #58a6ff;
        padding: 1 2;
        layout: vertical;
    }
    #help-header {
        height: auto;
        border-bottom: solid #30363d;
        margin-bottom: 1;
        padding-bottom: 1;
    }
    #help-title {
        text-style: bold;
        color: #58a6ff;
        width: 100%;
        content-align: center middle;
    }
    #help-table-container {
        height: 1fr;
    }
    HelpModal DataTable {
        height: 1fr;
        border: none;
        background: transparent;
    }
    .help-hint {
        color: #8b949e;
        text-style: italic;
        content-align: center middle;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="help-outer"):
            with Vertical(id="help-header"):
                yield Label(f"⌨️  VLLM-MLX TUI Shortcut Reference", id="help-title")
                yield Label("Navigate the interface entirely with your keyboard.", classes="help-hint")

            with Container(id="help-table-container"):
                yield DataTable(id="help-table")

            yield Label("Press ESC, F1, or Q to resume", classes="help-hint")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Shortcut", "Action", "Context")
        table.cursor_type = "row"

        # Get all bindings from the active screen and the app
        app_bindings = self.app.BINDINGS
        screen_bindings = getattr(self.app.screen, "BINDINGS", [])

        seen = set()

        def add_bindings(bindings, context: str):
            for b in bindings:
                # Handle comma-separated keys
                key_display = b.key.upper().replace(",", " / ")
                if key_display in seen: continue
                table.add_row(key_display, b.description, context)
                seen.add(key_display)

        # Show screen-specific first, then global
        if screen_bindings:
            add_bindings(screen_bindings, "Active View")
        add_bindings(app_bindings, "System Wide")
