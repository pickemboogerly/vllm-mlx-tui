import httpx
from httpx_sse import aconnect_sse
import json
import time
import asyncio
from textual.screen import Screen
from textual.widgets import Header, Footer, Input, Static
from textual.containers import VerticalScroll, Horizontal, Vertical
from textual import work
from rich.markdown import Markdown
from rich.text import Text as RichText
from .metrics import MetricsPane
from ..server import ServerManager

class ChatScreen(Screen):
    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+c", "copy_text", "Copy Latest Response"),
    ]

    CSS = """
    #chat-scroll {
        height: 1fr;
        border: solid green;
        overflow-y: scroll;
        padding: 1;
    }
    #input-box {
        dock: bottom;
    }
    Horizontal {
        height: 100%;
    }
    MetricsPane {
        width: 30;
        border-left: solid green;
        padding: 1;
    }
    .metric-label {
        padding-top: 1;
        padding-bottom: 1;
        text-style: bold;
        color: magenta;
    }
    .msg-user {
        border: round $accent;
        background: $accent 15%;
        width: auto;
        max-width: 80%;
        padding: 0 1;
        margin-bottom: 1;
    }
    .msg-assistant {
        border: round $primary;
        background: $primary 15%;
        width: auto;
        max-width: 80%;
        padding: 0 1;
        margin-bottom: 1;
    }
    #footer-label {
        dock: bottom;
        width: 100%;
        text-align: center;
        background: $surface;
        color: $text-muted;
    }
    """

    def __init__(self, manager: ServerManager, model_id: str, **kwargs):
        super().__init__(**kwargs)
        self.manager = manager
        self.model_id = model_id
        self.messages = []
        self.memory_poller = None

    def compose(self):
        yield Header()
        yield Horizontal(
            Vertical(
                VerticalScroll(id="chat-scroll"),
                Input(placeholder="Type your message...", id="input-box")
            ),
            MetricsPane(id="metrics")
        )
        yield Static(f"Model: {self.model_id} | Text Select: Option-Drag (Mac) / Shift-Drag (PC)", id="footer-label")
        yield Footer()

    def action_quit(self):
        if self.manager:
            self.manager.stop()
        self.app.exit()

    def action_copy_text(self):
        for msg in reversed(self.messages):
            if msg["role"] == "assistant" and msg["content"]:
                try:
                    self.app.copy_to_clipboard(msg["content"])
                    self.notify("Copied last response to clipboard!")
                except Exception:
                    self.notify("Failed to copy. Try native Option+Drag to select.", severity="error")
                break

    def on_mount(self):
        self.memory_poller = self.set_interval(2.0, self.update_memory)
        # H5: system greeting uses RichText (not Markdown) to avoid markup injection
        greeting = RichText()
        greeting.append("System: ", style="bold")
        greeting.append(f"Successfully connected to loaded model ")
        greeting.append(self.model_id, style="italic")
        self.query_one("#chat-scroll").mount(
            Static(greeting, classes="msg-assistant")
        )

    def update_memory(self):
        if self.manager:
            mem = self.manager.get_memory_usage()
            self.query_one("#metrics", MetricsPane).memory_mb = f"{mem:.1f} MB"

    @work(exclusive=True)
    async def request_completion(self, prompt: str):
        input_widget = self.query_one("#input-box", Input)
        chat_scroll = self.query_one("#chat-scroll")

        # H3: Disable input widget for the duration of generation
        input_widget.disabled = True

        try:
            # H5: User message rendered as RichText (plain), not Markdown,
            # so user-supplied content cannot inject markup or Markdown syntax.
            user_text = RichText()
            user_text.append("User: ", style="bold")
            user_text.append(prompt)
            self.messages.append({"role": "user", "content": prompt})
            await chat_scroll.mount(Static(user_text, classes="msg-user"))

            # Assistant placeholder
            assistant_md = Static(Markdown("**Assistant:** "), classes="msg-assistant")
            await chat_scroll.mount(assistant_md)
            chat_scroll.scroll_end(animate=False)

            self.messages.append({"role": "assistant", "content": ""})

            start_time = time.time()
            tokens = 0
            try:
                async with httpx.AsyncClient() as client:
                    payload = {
                        "model": self.model_id,
                        "messages": self.messages,
                        "stream": True,
                    }
                    port = self.manager.port if self.manager else 8000
                    async with aconnect_sse(
                        client, "POST", f"http://127.0.0.1:{port}/v1/chat/completions", json=payload
                    ) as event_source:
                        async for sse in event_source.aiter_sse():
                            if sse.data == "[DONE]":
                                break
                            try:
                                data = json.loads(sse.data)
                                delta = data["choices"][0]["delta"].get("content", "")
                                self.messages[-1]["content"] += delta
                                tokens += 1

                                # Re-render markdown for assistant (AI content is trusted)
                                assistant_md.update(Markdown(f"**Assistant:** {self.messages[-1]['content']}"))
                                chat_scroll.scroll_end(animate=False)

                                elapsed = time.time() - start_time
                                if elapsed > 0:
                                    tps = tokens / elapsed
                                    self.query_one("#metrics", MetricsPane).tokens_s = f"{tps:.1f} t/s"

                            except Exception as loop_e:
                                # L4: log SSE parse errors instead of silently swallowing them
                                self.log.warning(f"SSE parse error: {loop_e}")
            except Exception as e:
                # H5: Error text rendered as RichText to prevent server-controlled
                # content from injecting Markdown/markup syntax.
                err_text = RichText()
                err_text.append("System: Connection error: ", style="bold red")
                err_text.append(str(e), style="red")
                assistant_md.update(err_text)

        finally:
            # H3: Always re-enable input, even if an exception occurred
            input_widget.value = ""
            input_widget.disabled = False

    def on_input_submitted(self, event: Input.Submitted):
        prompt = event.value.strip()
        # H3: Don't submit if input is disabled (generation in progress)
        if prompt and not self.query_one("#input-box", Input).disabled:
            self.request_completion(prompt)
