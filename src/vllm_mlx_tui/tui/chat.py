"""
Chat screen — the main interaction surface.

Layout:
  ┌──── StatusBar (top) ────────────────────────────────┐
  │  [Chat viewport]  │  [Side panel]                   │
  │  ─────────────    │   Options / Session management  │
  │  [Input row]      │                                 │
  ├──── Nav tabs (bottom) ──────────────────────────────┤
  └──── Footer ─────────────────────────────────────────┘

Architecture: all I/O runs as async tasks; UI updated via messages.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Optional
from rich.markdown import Markdown as RichMarkdown

import httpx
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.message import Message
from textual.reactive import reactive
import textual.screen
from textual.widget import Widget
from textual.widgets import (
    Button,
    Collapsible,
    Footer,
    Input,
    Label,
    ListItem,
    ListView,
    Markdown,
    Static,
    TabbedContent,
    TabPane,
)

from ..server import ServerManager
from ..sessions import (
    ChatSession,
    delete_session,
    list_sessions,
    save_session,
    session_to_markdown,
)
from .statusbar import StatusBar

# Streaming throttle: accumulate tokens for this many ms before re-render
_RENDER_THROTTLE_MS = 80


# ── Messages ──────────────────────────────────────────────────────────────────

class TokenReceived(Message):
    def __init__(self, token: str) -> None:
        super().__init__()
        self.token = token

class StreamDone(Message):
    pass

class StreamError(Message):
    def __init__(self, error: str) -> None:
        super().__init__()
        self.error = error

class MetricsUpdate(Message):
    def __init__(self, mem_mb: float, tps: float) -> None:
        super().__init__()
        self.mem_mb = mem_mb
        self.tps = tps


# ── Chat bubble widgets ────────────────────────────────────────────────────────

class UserBubble(Widget):
    """Right-aligned user message bubble.

    Right-alignment is achieved by making width=100% (fills the scroll
    column) and using content-align: right. The visible bubble colour/border
    sits inside via padding so it looks like a right-aligned chat bubble.
    """
    DEFAULT_CSS = """
    UserBubble {
        width: 100%;
        height: auto;
        content-align: right top;
        padding: 0;
        margin: 0;
        background: transparent;
    }
    UserBubble .bubble-inner {
        background: transparent;
        border: round $primary;
        padding: 0 1;
        margin: 0;
        color: $text;
        width: auto;
        max-width: 75%;
    }
    """

    def __init__(self, text: str, **kwargs):
        super().__init__(**kwargs)
        self._text = text

    def compose(self) -> ComposeResult:
        yield Static(self._text, classes="bubble-inner")


class SystemMessage(Static):
    """Centered, muted system messages."""
    DEFAULT_CSS = """
    SystemMessage {
        width: 100%;
        height: auto;
        content-align: center top;
        color: $text-muted;
        text-style: italic;
        margin: 0;
        padding: 0 1;
    }
    """
    def __init__(self, text: str, **kwargs):
        super().__init__(text, **kwargs)


class AssistantBubble(Widget):
    """Left-aligned assistant message with optional <think> collapsible."""

    DEFAULT_CSS = """
    AssistantBubble {
        width: 100%;
        height: auto;
        max-width: 90%;
        margin: 0;
        layout: vertical;
    }
    AssistantBubble Collapsible {
        background: transparent;
        border: round $border;
        color: $text-muted;
        margin: 0;
    }
    AssistantBubble Markdown {
        border-left: round $success;
        padding: 0 2;
        margin: 0 0 1 1;
        background: transparent;
        height: auto;
    }
    """

    def __init__(self, reasoning: str = "", body: str = "", **kwargs):
        super().__init__(**kwargs)
        self._reasoning = reasoning
        self._body = body

    def compose(self) -> ComposeResult:
        if self._reasoning:
            with Collapsible(title="🧠 Reasoning (collapsed)", collapsed=True):
                yield Static(self._reasoning, markup=False)
        yield Markdown(self._body or "…")

    def _fix_markdown(self, text: str) -> str:
        """Sanitize 'lazy' Markdown from LLMs. 
        Ensures tables have separator rows and handles prefixes like 'Example: |'.
        """
        lines = text.split("\n")
        new_lines = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Handle cases where LLM prefixes a table row with "Example: |"
            if " | " in stripped and stripped.count("|") >= 2:
                 if not stripped.startswith("|"):
                      idx = stripped.find("|")
                      if idx > 0:
                           line = stripped[idx:]
                           stripped = line.strip()

            new_lines.append(line)
            
            # Inject separator if missing
            if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
                # If we're at the end of text OR the next line isn't a separator row...
                is_last = (i + 1 == len(lines))
                has_next = not is_last
                is_sep_next = has_next and ("---" in lines[i+1])
                
                if is_last or not is_sep_next:
                    # Construct a separator row
                    cols = stripped.count("|") - 1
                    sep = "|" + (" --- |" * cols)
                    new_lines.append(sep)
        return "\n".join(new_lines)

    def update_body(self, body: str) -> None:
        self._body = body
        try:
            fixed = self._fix_markdown(body or "…")
            self.query_one(Markdown).update(fixed)
        except Exception:
            pass

    def finalize(self, reasoning: str, body: str) -> None:
        self._reasoning = reasoning
        self._body = body
        try:
            fixed = self._fix_markdown(body or "")
            self.query_one(Markdown).update(fixed)
        except Exception:
            pass


# ── Parsing helpers ────────────────────────────────────────────────────────────

def _split_think(content: str) -> tuple[str, str]:
    """Split <think>…</think> from the final reply.
    Supports <think>, <thought>, and [thinking] variants.
    """
    patterns = [
        r"<think>(.*?)</think>",
        r"<thought>(.*?)</thought>",
        r"\[thinking\](.*?)\[/thinking\]",
    ]
    for p in patterns:
        m = re.search(p, content, re.DOTALL | re.IGNORECASE)
        if m:
            reasoning = m.group(1).strip()
            body = (content[: m.start()] + content[m.end() :]).strip()
            return reasoning, body
    return "", content


# ── Side panel ────────────────────────────────────────────────────────────────

class SidePanel(Widget):
    """Chat options + session management side panel."""

    DEFAULT_CSS = """
    SidePanel {
        width: 30;
        border-left: solid $border;
        background: $surface;
        padding: 0;
        layout: vertical;
    }
    SidePanel ScrollableContainer {
        height: 1fr;
        padding: 0 1;
    }
    SidePanel .panel-title {
        text-style: bold;
        color: $primary;
        padding: 1 0 0 0;
    }
    SidePanel Label {
        color: $text-muted;
        padding: 0;
        margin: 0;
    }
    SidePanel Input {
        border: solid $border;
        background: $background;
        color: $text;
        margin-bottom: 0;
        height: 3;
    }
    SidePanel Input:focus {
        border: solid $primary;
    }
    SidePanel Button {
        width: 100%;
        height: 3;
        border: none;
        margin: 0;
        background: transparent;
        content-align: left middle;
        padding-left: 2;
    }
    SidePanel Button:hover {
        background: $primary-darken-1;
        color: $primary;
    }
    SidePanel ListView {
        height: 1fr;
        min-height: 10;
        border: solid $border;
        background: $background;
        margin-top: 1;
    }
    SidePanel ListItem {
        color: $text;
        padding: 0 1;
    }
    SidePanel ListItem.--highlight {
        background: $primary-darken-1;
    }
    """

    class NewChatRequested(Message):
        pass

    class SwitchModelRequested(Message):
        pass

    class MetricsRequested(Message):
        pass

    class SessionSelected(Message):
        def __init__(self, session_id: str) -> None:
            super().__init__()
            self.session_id = session_id

    class DeleteSessionRequested(Message):
        def __init__(self, session_id: str) -> None:
            super().__init__()
            self.session_id = session_id

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._sessions: list[ChatSession] = []

    def compose(self) -> ComposeResult:
        with ScrollableContainer():
            yield Label("Chat Options", classes="panel-title")

            yield Label("Temperature (0–2)")
            yield Input(value="0.7", id="opt-temperature")

            yield Label("Top P (0–1)")
            yield Input(value="0.9", id="opt-top-p")

            yield Label("Max Output Tokens")
            yield Input(value="2048", id="opt-max-tokens")

            yield Label("System Prompt")
            yield Input(
                value="You are a helpful assistant. Provide detailed, structured responses. "
                      "Use GFM Markdown for tables (always include |---| separator rows) and code blocks.",
                id="opt-system-prompt"
            )

            yield Label("Chat History")
            yield ListView(id="session-list")

    def on_mount(self) -> None:
        self.refresh_sessions()

    def refresh_sessions(self) -> None:
        self._sessions = list_sessions()
        lv = self.query_one("#session-list", ListView)
        lv.clear()
        for s in self._sessions:
            lv.append(ListItem(Label(f"{s.title}  [{s.updated_date}]")))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        pass

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx = self.query_one("#session-list", ListView).index
        if idx is not None and 0 <= idx < len(self._sessions):
            sid = self._sessions[idx].id
            self.post_message(SidePanel.SessionSelected(sid))

    def _rename_selected(self) -> None:
        lv = self.query_one("#session-list", ListView)
        idx = lv.index
        if idx is not None and 0 <= idx < len(self._sessions):
            s = self._sessions[idx]
            self.app.push_screen(_RenameModal(session=s, panel=self))

    def _delete_selected(self) -> None:
        lv = self.query_one("#session-list", ListView)
        idx = lv.index
        if idx is not None and 0 <= idx < len(self._sessions):
            sid = self._sessions[idx].id
            self.post_message(SidePanel.DeleteSessionRequested(sid))

    def _export_selected(self) -> None:
        lv = self.query_one("#session-list", ListView)
        idx = lv.index
        if idx is not None and 0 <= idx < len(self._sessions):
            s = self._sessions[idx]
            md = session_to_markdown(s)
            out_path = f"/tmp/{s.id}.md"
            with open(out_path, "w") as f:
                f.write(md)
            self.app.notify(f"Exported to {out_path}")

    def get_options(self) -> dict:
        def _f(wid: str, default: float) -> float:
            try:
                return float(self.query_one(wid, Input).value.strip())
            except ValueError:
                return default

        def _i(wid: str, default: int) -> int:
            try:
                return int(self.query_one(wid, Input).value.strip())
            except ValueError:
                return default

        return {
            "temperature": _f("#opt-temperature", 0.7),
            "top_p": _f("#opt-top-p", 0.9),
            "max_tokens": _i("#opt-max-tokens", 2048),
            "system_prompt": self.query_one("#opt-system-prompt", Input).value.strip(),
        }


class _RenameModal(textual.screen.ModalScreen):
    """Simple inline rename dialog."""
    DEFAULT_CSS = """
    _RenameModal {
        align: center middle;
        background: rgba(0,0,0,0.6);
    }
    #rename-box {
        width: 50;
        height: auto;
        background: $background;
        border: solid $primary;
        padding: 1 2;
        layout: vertical;
    }
    #rename-box Label {
        text-style: bold;
        margin-bottom: 1;
    }
    #rename-box Input {
        margin-bottom: 1;
    }
    #rename-box Horizontal {
        height: 3;
        content-align: center middle;
    }
    #rename-box Button {
        margin: 0 1;
    }
    """

    def __init__(self, session: ChatSession, panel: SidePanel, **kwargs):
        super().__init__(**kwargs)
        self._session = session
        self._panel = panel

    def compose(self) -> ComposeResult:
        with Container(id="rename-box"):
            yield Label("Rename Chat Session")
            yield Input(value=self._session.title, id="rename-input")
            with Horizontal():
                yield Button("Save", id="btn-ok", variant="primary")
                yield Button("Cancel", id="btn-cancel")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._perf_rename()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-ok":
            self._perf_rename()
        else:
            self.dismiss()

    def _perf_rename(self) -> None:
        new_name = self.query_one("#rename-input", Input).value.strip()
        if new_name:
            self._session.title = new_name
            save_session(self._session)
            self._panel.refresh_sessions()
        self.dismiss()

    def on_key(self, event) -> None:
        if event.key == "escape":
            self._perf_rename()


# ── Chat screen ───────────────────────────────────────────────────────────────

class ChatScreen(textual.screen.Screen):
    """Primary chat interface."""

    BINDINGS = [
        Binding("ctrl+n", "new_chat", "New Chat"),
        Binding("ctrl+l", "clear_chat", "Clear"),
        Binding("ctrl+s", "save", "Save"),
        Binding("ctrl+x", "stop_stream", "Stop"),
        Binding("ctrl+m", "show_metrics", "Metrics"),
        Binding("ctrl+r", "switch_model", "Switch Model"),
        Binding("ctrl+e", "export_session", "Export MD"),
        Binding("ctrl+d", "delete_session", "Delete Chat"),
        Binding("f2", "rename_session", "Rename Chat"),
        Binding("ctrl+q", "quit_app", "Quit"),
        Binding("ctrl+c", "copy_selection", "Copy"),
    ]

    CSS = """
    ChatScreen {
        background: $background;
        layout: vertical;
    }

    /* ── Nav tabs (Removed) ── */
    #nav-tabs {
        display: none;
    }

    /* ── Main body ── */
    #chat-body {
        layout: horizontal;
        height: 1fr;
    }
    #chat-main {
        layout: vertical;
        width: 1fr;
    }

    /* ── Scroll area ── */
    #chat-scroll {
        height: 1fr;
        padding: 0 2;
        overflow-y: scroll;
        layout: vertical;
    }

    /* ── Input row ── */
    #input-row {
        height: 5;
        layout: horizontal;
        border-top: solid $border;
        padding: 1;
        background: transparent;
    }
    #message-input {
        width: 1fr;
        border: solid $border;
        background: transparent;
        color: $text;
    }
    #message-input:focus {
        border: solid $primary;
    }
    #btn-send {
        width: 10;
        margin-left: 1;
    }
    #btn-stop {
        width: 10;
        margin-left: 1;
        display: none;
    }
    #btn-stop.visible {
        display: block;
    }

    /* ── Stop active ── */
    .streaming #btn-send {
        display: none;
    }
    .streaming #btn-stop {
        display: block;
    }
    """

    def __init__(
        self,
        manager: ServerManager,
        model_id: str,
        display_name: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.manager = manager
        self.model_id = model_id
        self.display_name = display_name or model_id

        self._session: ChatSession = ChatSession(model_id=model_id)
        self._streaming = False
        self._stop_flag = False
        self._stream_task: Optional[asyncio.Task] = None
        self._metrics_task_handle: Optional[object] = None
        # Buffer for in-progress assistant response
        self._current_content = ""
        self._current_bubble: Optional[AssistantBubble] = None
        # Throttle render
        self._last_render = 0.0

    # ── Layout ──────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield StatusBar()

        with Horizontal(id="chat-body"):
            with Vertical(id="chat-main"):
                yield ScrollableContainer(id="chat-scroll")
                with Horizontal(id="input-row"):
                    yield Input(
                        placeholder="Type a message…  (Enter to send)",
                        id="message-input",
                    )
                    yield Button("Send", id="btn-send", variant="primary")
                    yield Button("⏹ Stop", id="btn-stop", variant="error")

            yield SidePanel(id="side-panel")
        yield Footer()

    # ── Mount ───────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        # Update status bar
        try:
            self.query_one(StatusBar).model_name = self.display_name
        except Exception:
            pass

        # Start metrics polling
        self._metrics_task_handle = self.set_interval(2.0, self._poll_metrics)

        # Show welcome message
        self._add_system_message(
            f"Connected to **{self.display_name}**. "
            f"Listening on `{self.manager.base_url}`."
        )
        # Load sessions in panel
        self.query_one(SidePanel).refresh_sessions()

    def on_unmount(self) -> None:
        self._cancel_stream()
        if self._metrics_task_handle:
            self._metrics_task_handle.stop()

    # ── Message handlers (Textual) ───────────────────────────────────────

    def on_side_panel_new_chat_requested(self, event: SidePanel.NewChatRequested) -> None:
        self._save_current_session()
        self._start_new_session()

    def on_side_panel_switch_model_requested(self, event: SidePanel.SwitchModelRequested) -> None:
        self.action_switch_model()

    def on_side_panel_session_selected(self, event: SidePanel.SessionSelected) -> None:
        self._load_session(event.session_id)

    def on_side_panel_delete_session_requested(self, event: SidePanel.DeleteSessionRequested) -> None:
        delete_session(event.session_id)
        self.query_one(SidePanel).refresh_sessions()
        self.notify("Chat deleted.")

    def on_side_panel_metrics_requested(self, event: SidePanel.MetricsRequested) -> None:
        self._push_metrics()

    # ── Token streaming messages ─────────────────────────────────────────

    def on_token_received(self, msg: TokenReceived) -> None:
        self._current_content += msg.token
        now = time.monotonic()
        if now - self._last_render >= _RENDER_THROTTLE_MS / 1000.0:
            self._last_render = now
            if self._current_bubble:
                _, body = _split_think(self._current_content)
                self._current_bubble.update_body(body or "…")
                self.query_one("#chat-scroll").scroll_end(animate=False)

    def on_stream_done(self, msg: StreamDone) -> None:
        self._finalize_stream()

    def on_stream_error(self, msg: StreamError) -> None:
        if self._current_bubble:
            self._current_bubble.update_body(f"⚠  Error: {msg.error}")
        self._finalize_stream()

    def on_metrics_update(self, msg: MetricsUpdate) -> None:
        try:
            sb = self.query_one(StatusBar)
            sb.mem_mb = msg.mem_mb
            sb.tokens_per_sec = msg.tps
        except Exception:
            pass

    # ── Actions ────────────────────────────────────────────────────────

    def action_new_chat(self) -> None:
        self._save_current_session()
        self._start_new_session()

    def action_clear_chat(self) -> None:
        self._start_new_session()

    def action_switch_model(self) -> None:
        self._cancel_stream()
        self._save_current_session()
        if self.manager:
            self.manager.stop()
        self.app.pop_screen()

    def action_quit_app(self) -> None:
        self._cancel_stream()
        self._save_current_session()
        if self.manager:
            self.manager.stop()
        self.app.exit()

    def action_save(self) -> None:
        self._save_current_session()
        self.notify("Session saved.")

    def action_show_metrics(self) -> None:
        self._push_metrics()

    def action_stop_stream(self) -> None:
        self._cancel_stream()
        self.notify("Generation stopped.")

    def action_export_session(self) -> None:
        self.query_one(SidePanel)._export_selected()

    def action_delete_session(self) -> None:
        # Prompt for confirmation? No, just delete for now then refresh.
        lv = self.query_one("#session-list", ListView)
        idx = lv.index
        sess_list = self.query_one(SidePanel)._sessions
        if idx is not None and 0 <= idx < len(sess_list):
            sid = sess_list[idx].id
            delete_session(sid)
            self.query_one(SidePanel).refresh_sessions()
            self.notify("Session deleted.")

    def action_rename_session(self) -> None:
        self.query_one(SidePanel)._rename_selected()

    def action_copy_selection(self) -> None:
        # Copy the most recent assistant message
        for msg in reversed(self._session.messages):
            if msg["role"] == "assistant" and msg.get("content"):
                try:
                    self.app.copy_to_clipboard(msg["content"])
                    self.notify("Copied last response to clipboard.")
                except Exception:
                    self.notify("Clipboard unavailable; try Option+Drag.", severity="warning")
                return

    # ── Button clicks ──────────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if btn_id == "btn-send":
            self._handle_send()
        elif btn_id == "btn-stop":
            self._cancel_stream()
        elif btn_id == "nav-chat":
            pass  # already here
        elif btn_id == "nav-metrics":
            self._push_metrics()
        elif btn_id == "nav-launcher":
            self.action_switch_model()

    def _push_metrics(self) -> None:
        from .metrics import MetricsScreen
        self.app.push_screen(MetricsScreen(manager=self.manager))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "message-input":
            self._handle_send()

    # ── Internal helpers ───────────────────────────────────────────────

    def _handle_send(self) -> None:
        inp = self.query_one("#message-input", Input)
        prompt = inp.value.strip()
        if not prompt or self._streaming:
            return
        inp.value = ""
        # Kick off as a work task (async, non-blocking)
        self._do_send(prompt)

    @work(exclusive=False)
    async def _do_send(self, prompt: str) -> None:
        """Mount user + assistant bubbles then launch the stream."""
        self._cancel_stream()
        self._streaming = True
        self._stop_flag = False
        self._set_streaming_ui(True)

        # Record user message
        self._session.messages.append({"role": "user", "content": prompt})
        if len(self._session.messages) == 1:
            self._session.title = (prompt[:40] + "…") if len(prompt) > 40 else prompt

        scroll = self.query_one("#chat-scroll")

        # Mount user bubble (self-right-aligning via CSS)
        user_bubble = UserBubble(prompt)
        await scroll.mount(user_bubble)

        # Mount streaming assistant bubble
        assistant_bubble = AssistantBubble()
        self._current_bubble = assistant_bubble
        self._current_content = ""
        await scroll.mount(assistant_bubble)
        scroll.scroll_end(animate=False)

        # Get generation options
        opts = self.query_one(SidePanel).get_options()
        sys_prompt = opts.get("system_prompt", "")

        # Kick off the HTTP stream task
        self._stream_task = asyncio.create_task(
            self._stream_chat(prompt, opts, sys_prompt)
        )

    async def _stream_chat(self, prompt: str, opts: dict, sys_prompt: str) -> None:
        """Async task: streams chat completions and posts messages to UI."""
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.extend(self._session.messages)

        payload = {
            "model": self.model_id,
            "messages": messages,
            "stream": True,
            "temperature": opts.get("temperature", 0.7),
            "top_p": opts.get("top_p", 0.9),
            "max_tokens": opts.get("max_tokens", 2048),
        }

        start_time = time.monotonic()
        token_count = 0

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{self.manager.base_url}/v1/chat/completions",
                    json=payload,
                    headers={"Accept": "text/event-stream"},
                ) as resp:
                    resp.raise_for_status()
                    buffer: list[str] = []
                    async for line in resp.aiter_lines():
                        if self._stop_flag or self._stream_task.cancelled():
                            break
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0]["delta"].get("content", "")
                            if delta:
                                buffer.append(delta)
                                if len(buffer) >= 5:
                                    for tok in buffer:
                                        self.post_message(TokenReceived(tok))
                                        token_count += 1
                                    buffer.clear()
                        except (KeyError, ValueError, json.JSONDecodeError):
                            # Skip malformed chunks but continue the stream
                            continue

                    # Flush remaining buffer tokens
                    if buffer:
                        for tok in buffer:
                            self.post_message(TokenReceived(tok))
                            token_count += 1

            # End of successful stream
            self.post_message(StreamDone())

            # Update metrics
            elapsed = time.monotonic() - start_time
            if elapsed > 0:
                tps = token_count / elapsed
                mem = self.manager.get_memory_usage()
                self.post_message(MetricsUpdate(mem_mb=mem, tps=tps))

        except asyncio.CancelledError:
            # Task was cancelled (likely via _cancel_stream)
            self.post_message(StreamDone())
        except httpx.HTTPError as exc:
            self.post_message(StreamError(f"HTTP error: {exc}"))
        except Exception as exc:
            self.post_message(StreamError(f"Stream failed: {exc}"))
        finally:
            # Ensure we don't leave the UI in a streaming state
            self._streaming = False

    def _finalize_stream(self) -> None:
        """Commit the streamed content into the session message list."""
        content = self._current_content
        reasoning, body = _split_think(content)

        if self._current_bubble:
            self._current_bubble.finalize(reasoning=reasoning, body=body)

        # Record in session
        self._session.messages.append({
            "role": "assistant",
            "content": content,
        })

        # Update context-window estimate in status bar
        ctx_chars = sum(len(m.get("content", "")) for m in self._session.messages)
        try:
            self.app.query_one(StatusBar).ctx_chars = ctx_chars
        except Exception:
            pass

        self._streaming = False
        self._set_streaming_ui(False)
        self._current_bubble = None
        self._current_content = ""
        self.query_one("#chat-scroll").scroll_end(animate=False)

    def _cancel_stream(self) -> None:
        if self._stream_task and not self._stream_task.done():
            self._stop_flag = True
            self._stream_task.cancel()
        self._streaming = False
        self._set_streaming_ui(False)

    def _set_streaming_ui(self, active: bool) -> None:
        try:
            inp = self.query_one("#message-input", Input)
            inp.disabled = active
            send = self.query_one("#btn-send", Button)
            stop = self.query_one("#btn-stop", Button)
            if active:
                send.add_class("hidden")
                stop.remove_class("hidden")
            else:
                send.remove_class("hidden")
                stop.add_class("hidden")
        except Exception:
            pass

    def _add_system_message(self, msg: str) -> None:
        scroll = self.query_one("#chat-scroll")
        scroll.mount(SystemMessage(msg))

    def _start_new_session(self) -> None:
        self._cancel_stream()
        self._session = ChatSession(model_id=self.model_id)
        scroll = self.query_one("#chat-scroll")
        scroll.remove_children()
        self._add_system_message(
            f"New chat started with **{self.display_name}**."
        )
        self.query_one(SidePanel).refresh_sessions()

    def _save_current_session(self) -> None:
        if self._session.messages:
            save_session(self._session)
            self.query_one(SidePanel).refresh_sessions()

    def _load_session(self, session_id: str) -> None:
        sessions = list_sessions()
        for s in sessions:
            if s.id == session_id:
                self._cancel_stream()
                self._session = s
                self._rebuild_transcript()
                return

    def _rebuild_transcript(self) -> None:
        scroll = self.query_one("#chat-scroll")
        scroll.remove_children()
        self._add_system_message(f"**{self._session.title}**")
        for msg in self._session.messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                scroll.mount(UserBubble(content))
            elif role == "assistant":
                reasoning, body = _split_think(content)
                scroll.mount(AssistantBubble(reasoning=reasoning, body=body))
        scroll.scroll_end(animate=False)

    # ── Background metrics poll ─────────────────────────────────────────

    def _poll_metrics(self) -> None:
        if self.manager:
            mem = self.manager.get_memory_usage()
            self.post_message(MetricsUpdate(mem_mb=mem, tps=0.0))
