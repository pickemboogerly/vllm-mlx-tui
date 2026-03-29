"""
Global status header bar shown on every screen.

Layout:
  [Left: App name + version]  [Center: Model name]  [Right: Mem | t/s | Ctx% | CPU%]
"""
from __future__ import annotations

import os
import time
from typing import Optional

import psutil
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

APP_VERSION = "0.1.0"

_t0 = time.monotonic()


def _cpu_pct() -> float:
    """Non-blocking: returns last measured CPU%."""
    return psutil.cpu_percent(interval=None)


class StatusBar(Widget):
    """Top status bar with live metrics."""

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        background: #1c2128;
        color: #8b949e;
        dock: top;
        layout: horizontal;
    }
    StatusBar .status-left {
        width: 1fr;
        content-align: left middle;
        color: #58a6ff;
        text-style: bold;
        padding-left: 1;
    }
    StatusBar .status-center {
        width: 1fr;
        content-align: center middle;
        color: #e6edf3;
    }
    StatusBar .status-right {
        width: 1fr;
        content-align: right middle;
        padding-right: 1;
        color: #8b949e;
    }
    """

    model_name: reactive[str] = reactive("")
    mem_mb: reactive[float] = reactive(0.0)
    tokens_per_sec: reactive[float] = reactive(0.0)
    ctx_chars: reactive[int] = reactive(0)
    ctx_window: reactive[int] = reactive(8192)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._poll_timer: Optional[object] = None

    def compose(self) -> ComposeResult:
        yield Static(f"vllm-mlx-tui  v{APP_VERSION}", classes="status-left")
        yield Static("", id="status-center", classes="status-center")
        yield Static("", id="status-right", classes="status-right")

    def on_mount(self) -> None:
        # Seed CPU measurement
        psutil.cpu_percent(interval=None)
        self._poll_timer = self.set_interval(1.5, self._refresh_metrics)

    def on_unmount(self) -> None:
        if self._poll_timer:
            self._poll_timer.stop()

    def watch_model_name(self, val: str) -> None:
        self._update_center()

    def _update_center(self) -> None:
        label = self.model_name or "—  no model loaded"
        try:
            self.query_one("#status-center", Static).update(label)
        except Exception:
            pass

    def _refresh_metrics(self) -> None:
        """Update right-side metrics. Called on interval."""
        mem_str = f"Mem {self.mem_mb:.0f}MB" if self.mem_mb > 0 else "Mem —"
        tps_str = f"{self.tokens_per_sec:.1f}t/s" if self.tokens_per_sec > 0 else "—t/s"

        ctx_pct = 0
        if self.ctx_window > 0 and self.ctx_chars > 0:
            # Rough token estimate: chars / 4
            ctx_pct = min(100, int(self.ctx_chars / 4 / self.ctx_window * 100))
        ctx_str = f"Ctx {ctx_pct}%"

        cpu = _cpu_pct()
        cpu_str = f"CPU {cpu:.0f}%"

        right = f"{mem_str}  {tps_str}  {ctx_str}  {cpu_str}"
        try:
            self.query_one("#status-right", Static).update(right)
        except Exception:
            pass
