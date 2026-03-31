"""
Launcher wizard screen.

Lets the user pick a cached model, configure options, and boot vllm-mlx.
After successful boot it pushes ChatScreen and registers a model-switch
callback so the user can return here without restarting the CLI.
"""
from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Optional

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
import textual.screen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Input,
    Label,
    LoadingIndicator,
    Log,
    Static,
    TabbedContent,
    TabPane,
)

from ..cache import discover_cached_models, CachedModel
from ..profiles import Profile, list_profiles, save_profile, load_default_profile
from ..server import ServerManager
from ..downloader import ensure_model_is_downloaded
from .statusbar import StatusBar

_BOOT_LOG_PATH = Path.home() / ".cache" / "vllm-mlx-tui" / "boot_log.txt"


class LauncherScreen(textual.screen.Screen):
    """Model selection + configuration wizard."""

    BINDINGS = [
        Binding("enter", "launch_server", "🚀 Launch"),
        Binding("ctrl+c", "copy_log", "Copy Log"),
        Binding("ctrl+b", "view_boot_server", "Boot Server"),
        Binding("ctrl+s", "save_profile", "Save Profile"),
        Binding("ctrl+m", "view_metrics", "Metrics"),
        Binding("f1", "app.show_help", "Help Reference"),
        Binding("ctrl+q", "app.quit_app", "Exits"),
    ]

    CSS = """
    LauncherScreen {
        background: $background;
        layout: vertical;
    }

    /* ---- Column layout ---- */
    #launcher-body {
        layout: horizontal;
        height: 1fr;
    }
    #left-panel {
        width: 1fr;
        min-width: 40;
        border-right: solid $border;
        layout: vertical;
        padding: 0;
    }
    /* Scrollable options area fills remaining height */
    #options-scroll {
        height: 1fr;
        padding: 0 1;
        overflow-y: auto;
    }
    /* Pinned action bar always visible at bottom of left panel */
    #action-bar {
        height: auto;
        min-height: 7;
        padding: 0 1 1 1;
        border-top: solid $border;
        background: transparent;
        layout: vertical;
    }
    #options-grid {
        height: auto;
        layout: horizontal;
    }
    #options-grid .column {
        width: 1fr;
        height: auto;
        padding-right: 2;
    }
    #right-panel {
        width: 36;
        padding: 0 1;
        layout: vertical;
    }

    /* ---- Section headers ---- */
    .section-title {
        text-style: bold;
        color: $primary;
        padding: 1 0 0 0;
    }

    /* ---- Model table ---- */
    #model-table {
        height: 10;
        border: solid $border;
        margin-bottom: 1;
    }
    DataTable > .datatable--header {
        background: transparent;
        color: $text-muted;
        text-style: bold;
    }
    DataTable > .datatable--cursor {
        background: transparent;
        border: solid $primary;
        color: $text;
    }

    /* ---- Inputs ---- */
    .field-label {
        color: $text-muted;
        padding-top: 1;
    }
    Input {
        border: solid $border;
        background: transparent;
        color: $text;
        margin-bottom: 1;
    }
    Input:focus {
        border: solid $primary;
    }

    /* ---- Checkboxes ---- */
    Checkbox {
        color: $text;
        padding: 0;
        margin-bottom: 0;
        background: transparent;
    }
    Checkbox:focus {
        border: solid $primary;
    }

    /* ---- Action bar buttons ---- */
    #btn-row {
        layout: horizontal;
        height: auto;
        margin-top: 1;
        margin-bottom: 1;
    }
    #launch-btn {
        width: 1fr;
        margin-right: 1;
    }
    #save-profile-btn {
        width: 1fr;
    }
    #load-profile-btn {
        width: 100%;
        margin-bottom: 1;
    }

    /* ---- Boot log ---- */
    #boot-log {
        height: 1fr;
        border: solid $border;
        background: transparent;
        color: $text-muted;
        margin-top: 1;
    }

    /* ---- Status ---- */
    #status-label {
        color: $error;
        text-style: bold;
        height: 1;
    }
    LoadingIndicator {
        height: 1;
        color: $primary;
    }

    /* ---- No-models notice ---- */
    #no-models-msg {
        color: $warning;
        padding: 1;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.manager: Optional[ServerManager] = None
        self._models: list[CachedModel] = []

    # ---- Layout -------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield StatusBar()

        with Container(id="launcher-body"):
            with Vertical(id="left-panel"):

                # ── Scrollable options area ──────────────────────────────
                with ScrollableContainer(id="options-scroll"):
                    yield Label("Cached Models", classes="section-title")
                    yield DataTable(id="model-table", cursor_type="row", zebra_stripes=True)
                    yield Static("", id="no-models-msg", classes="hidden")

                    yield Label("Manual Model Override (path or Hub id)", classes="field-label")
                    yield Input(placeholder="e.g. mlx-community/Mistral-7B-v0.1-4bit", id="manual-model")

                    yield Label("Options", classes="section-title")
                    with Horizontal(id="options-grid"):
                        with Vertical(classes="column"):
                            yield Checkbox("Multimodal  (--mllm)", id="opt-mllm")
                            yield Checkbox("Ngrok tunnel  (public URL)", id="opt-ngrok")
                            yield Checkbox("Disable tool-call flags", id="opt-no-tool")

                            yield Label("Port", classes="field-label")
                            yield Input(value="8001", id="opt-port")

                            yield Label("Host", classes="field-label")
                            yield Input(value="127.0.0.1", id="opt-host")

                        with Vertical(classes="column"):
                            yield Label("Max Model Len  (0 = srv default)", classes="field-label")
                            yield Input(value="0", id="opt-max-len")

                            yield Label("Quantization  (opt.)", classes="field-label")
                            yield Input(placeholder="e.g. awq", id="opt-quant")

                            yield Label("Dtype  (opt.)", classes="field-label")
                            yield Input(placeholder="e.g. float16", id="opt-dtype")

                    yield Label("Parser override (leave blank for default)", classes="field-label")
                    yield Input(placeholder="e.g. hermes", id="opt-parser")

                # ── Pinned action bar — always visible ───────────────────
                with Vertical(id="action-bar"):
                    with Horizontal(id="btn-row"):
                        yield Button("🚀  Launch", id="launch-btn", variant="primary")
                        yield Button("💾  Save Profile", id="save-profile-btn", variant="default")
                    yield Label("", id="status-label")
                    yield LoadingIndicator(id="loading", classes="hidden")

            with Vertical(id="right-panel"):
                yield Label("Profiles", classes="section-title")
                yield Button("📂  Load Profile", id="load-profile-btn", variant="default")
                yield DataTable(id="profile-table", cursor_type="row")

                yield Label("Boot Log", classes="section-title")
                yield Log(id="boot-log", highlight=True)

        yield Footer()

    # ---- Mount --------------------------------------------------------

    def on_mount(self) -> None:
        self._populate_model_table()
        self._populate_profile_table()
        self._prefill_default_profile()

    def _populate_model_table(self) -> None:
        table = self.query_one("#model-table", DataTable)
        table.add_columns("Model", "Size")

        self._models = discover_cached_models()
        no_msg = self.query_one("#no-models-msg", Static)

        if not self._models:
            no_msg.remove_class("hidden")
            no_msg.update(
                "⚠  No models found in ~/.cache/huggingface/hub.\n"
                "Run: huggingface-cli download <model-id>"
            )
            self.query_one("#launch-btn", Button).disabled = True
            return

        for m in self._models:
            size_str = m.param_hint or "—"
            table.add_row(m.display_name, size_str, key=m.repo_id)

        # Select first row
        table.move_cursor(row=0)

    def _populate_profile_table(self) -> None:
        pt = self.query_one("#profile-table", DataTable)
        pt.add_columns("Profile", "Saved")
        profiles = list_profiles()
        for p in profiles:
            default_marker = " ★" if p.is_default else ""
            pt.add_row(p.name + default_marker, p.saved_date, key=p.name)

    def _prefill_default_profile(self) -> None:
        p = load_default_profile()
        if p:
            self._apply_profile(p)

    # ---- Helpers -------------------------------------------------------

    def _selected_model_id(self) -> Optional[str]:
        """Return Hub id from table selection or manual field."""
        manual = self.query_one("#manual-model", Input).value.strip()
        if manual:
            return manual

        table = self.query_one("#model-table", DataTable)
        if table.cursor_row is not None and self._models:
            idx = table.cursor_row
            if 0 <= idx < len(self._models):
                return self._models[idx].repo_id
        return None

    def _selected_snapshot_path(self) -> str:
        """Return absolute snapshot path for the selected model."""
        manual = self.query_one("#manual-model", Input).value.strip()
        if manual:
            return manual

        table = self.query_one("#model-table", DataTable)
        if table.cursor_row is not None and self._models:
            idx = table.cursor_row
            if 0 <= idx < len(self._models):
                return self._models[idx].snapshot_path
        return ""

    def _selected_display_name(self) -> str:
        manual = self.query_one("#manual-model", Input).value.strip()
        if manual:
            return manual

        table = self.query_one("#model-table", DataTable)
        if table.cursor_row is not None and self._models:
            idx = table.cursor_row
            if 0 <= idx < len(self._models):
                return self._models[idx].display_name
        return ""

    def _build_profile_from_ui(self, name: str = "Unnamed") -> Profile:
        def _int(widget_id: str, default: int, min_val: int = 0, max_val: int = 65535) -> int:
            try:
                val = int(self.query_one(widget_id, Input).value.strip() or default)
                return max(min_val, min(max_val, val))
            except ValueError:
                return default

        return Profile(
            name=name,
            model=self._selected_model_id() or "",
            port=_int("#opt-port", 8001, 1, 65535),
            host=self.query_one("#opt-host", Input).value.strip() or "127.0.0.1",
            mllm=self.query_one("#opt-mllm", Checkbox).value,
            ngrok=self.query_one("#opt-ngrok", Checkbox).value,
            no_tool_parser=self.query_one("#opt-no-tool", Checkbox).value,
            tool_parser=self.query_one("#opt-parser", Input).value.strip(),
            max_model_len=_int("#opt-max-len", 0, 0, 1000000),
            quantization=self.query_one("#opt-quant", Input).value.strip(),
            dtype=self.query_one("#opt-dtype", Input).value.strip(),
        )

    def _apply_profile(self, p: Profile) -> None:
        try:
            self.query_one("#manual-model", Input).value = p.model
            self.query_one("#opt-port", Input).value = str(p.port)
            self.query_one("#opt-host", Input).value = p.host
            self.query_one("#opt-mllm", Checkbox).value = p.mllm
            self.query_one("#opt-ngrok", Checkbox).value = p.ngrok
            self.query_one("#opt-no-tool", Checkbox).value = p.no_tool_parser
            self.query_one("#opt-parser", Input).value = p.tool_parser
            self.query_one("#opt-max-len", Input).value = str(p.max_model_len)
            self.query_one("#opt-quant", Input).value = p.quantization
            self.query_one("#opt-dtype", Input).value = p.dtype
        except Exception:
            pass

    def _set_status(self, msg: str, is_error: bool = False) -> None:
        label = self.query_one("#status-label", Label)
        label.update(msg)
        if is_error:
            label.add_class("error-text")
        else:
            label.remove_class("error-text")

    # ---- Actions -------------------------------------------------------

    def action_launch_server(self) -> None:
        """Trigger the launch button logic."""
        self.query_one("#launch-btn", Button).press()

    def action_save_profile(self) -> None:
        """Trigger the save profile button logic."""
        self.query_one("#save-profile-btn", Button).press()

    def action_view_metrics(self) -> None:
        """Open the metrics dashboard."""
        from .metrics import MetricsScreen
        self.app.push_screen(MetricsScreen())

    def action_view_boot_server(self) -> None:
        """Focus the boot log widget."""
        self.query_one("#boot-log").focus()

    def action_quit_app(self) -> None:
        if self.manager:
            self.manager.stop()
        self.app.exit()

    def action_copy_log(self) -> None:
        try:
            log_widget = self.query_one("#boot-log", Log)
            text = "\n".join(log_widget._lines) if hasattr(log_widget, "_lines") else ""
            if not text:
                text = _BOOT_LOG_PATH.read_text() if _BOOT_LOG_PATH.exists() else ""
            if text:
                self.app.copy_to_clipboard(text)
                self.notify("Boot log copied to clipboard.")
            else:
                self.notify("Nothing to copy yet.", severity="warning")
        except Exception as e:
            self.notify(f"Copy failed: {e}", severity="error")

    # ---- Button handlers -----------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id

        if btn_id == "launch-btn":
            model_id = self._selected_model_id()
            if not model_id:
                self._set_status("Select a model first.", is_error=True)
                return
            snapshot_path = self._selected_snapshot_path()
            display_name = self._selected_display_name()
            try:
                port = int(self.query_one("#opt-port", Input).value.strip() or "8001")
                host = self.query_one("#opt-host", Input).value.strip() or "127.0.0.1"
            except ValueError:
                port = 8001
                host = "127.0.0.1"
            self._boot_server(model_id, snapshot_path, display_name, port, host)

        elif btn_id == "save-profile-btn":
            model_id = self._selected_model_id()
            if not model_id:
                self._set_status("Select a model first.", is_error=True)
                return
            # Use model short name as default profile name
            name = self._selected_display_name().split("/")[-1]
            p = self._build_profile_from_ui(name)
            save_profile(p)
            self.notify(f"Profile '{p.name}' saved.")
            # Refresh profile table
            pt = self.query_one("#profile-table", DataTable)
            pt.clear()
            for prof in list_profiles():
                marker = " ★" if prof.is_default else ""
                pt.add_row(prof.name + marker, prof.saved_date, key=prof.name)

        elif btn_id == "load-profile-btn":
            pt = self.query_one("#profile-table", DataTable)
            if pt.cursor_row is None:
                return
            profiles = list_profiles()
            idx = pt.cursor_row
            if 0 <= idx < len(profiles):
                self._apply_profile(profiles[idx])
                self.notify(f"Profile '{profiles[idx].name}' loaded.")

    # ---- Boot worker ---------------------------------------------------

    @work(exclusive=True)
    async def _boot_server(
        self,
        model_id: str,
        snapshot_path: str,
        display_name: str,
        port: int,
        host: str,
    ) -> None:
        btn = self.query_one("#launch-btn", Button)
        load = self.query_one("#loading", LoadingIndicator)
        log = self.query_one("#boot-log", Log)

        btn.disabled = True
        load.remove_class("hidden")
        log.clear()
        self._set_status("Starting vllm-mlx…")

        # Ensure log dir
        _BOOT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log_file = _BOOT_LOG_PATH.open("w")

        # Build extra args from UI
        extra_args: list[str] = []
        # (Values passed as args take priority)
        mllm = self.query_one("#opt-mllm", Checkbox).value
        ngrok = self.query_one("#opt-ngrok", Checkbox).value
        no_tool = self.query_one("#opt-no-tool", Checkbox).value
        parser = self.query_one("#opt-parser", Input).value.strip()
        max_len_str = self.query_one("#opt-max-len", Input).value.strip()
        quant = self.query_one("#opt-quant", Input).value.strip()
        dtype = self.query_one("#opt-dtype", Input).value.strip()

        if mllm:
            extra_args.append("--mllm")
        if no_tool:
            extra_args += ["--disable-auto-tool-choice"]
        elif parser:
            extra_args += ["--enable-auto-tool-choice", "--tool-call-parser", parser]
        try:
            max_len = int(max_len_str)
            if max_len > 0:
                extra_args += ["--max-model-len", str(max_len)]
        except ValueError:
            pass
        if quant:
            extra_args += ["--quantization", quant]
        if dtype:
            extra_args += ["--dtype", dtype]

        # Define log callback early
        async def on_log_line(line: str) -> None:
            log.write_line(line)
            log_file.write(line + "\n")

        try:
            # 1. ENSURE MODEL IS DOWNLOADED (Pre-flight)
            self._set_status(f"Pre-flight: Checking results for model '{model_id}'...")
            # This ensures we have a fully populated local HF snapshot before hitting serve
            local_snapshot = await ensure_model_is_downloaded(model_id, on_log_line)
            
            # 2. START SERVER
            self._set_status("Starting vllm-mlx kernel…")
            self.manager = ServerManager(
                model_id=model_id,
                snapshot_path=local_snapshot,
                port=port,
                host=host,
                extra_args=extra_args,
                use_ngrok=ngrok,
            )

            ready, error_msg = await self.manager.start(log_callback=on_log_line)

        except Exception as exc:
            ready = False
            error_msg = str(exc)
        finally:
            log_file.close()
            load.add_class("hidden")
            btn.disabled = False

        if not ready:
            self._set_status(f"Failed: {error_msg}", is_error=True)
            return

        self._set_status("Server ready! Launching chat…")

        # Update status bar
        try:
            self.app.query_one(StatusBar).model_name = display_name
        except Exception:
            pass

        # Push ChatScreen
        from .chat import ChatScreen
        await self.app.push_screen(
            ChatScreen(
                manager=self.manager,
                model_id=model_id,
                display_name=display_name,
            )
        )
