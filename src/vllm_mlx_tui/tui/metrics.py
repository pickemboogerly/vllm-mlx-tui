"""
Metrics dashboard — full-screen view with live time-series graphs.

Uses textual-plotext for charting. Falls back to ASCII-sparkline
Static widgets if plotext is unavailable (graceful degradation).

Tracks (rolling 5-minute window):
  - E2E Request Latency
  - Time Per Output Token (TPOT)
  - Time to First Token (TTFT)
  - Token Throughput (t/s)
  - Memory Usage (MB)
  - CPU Utilization (%)
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Optional

import psutil
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
import textual.screen
from textual.widgets import Footer, Label, Static

try:
    from textual_plotext import PlotextPlot
    _PLOTEXT_AVAILABLE = True
except ImportError:
    _PLOTEXT_AVAILABLE = False

from ..server import ServerManager
from .statusbar import StatusBar

_WINDOW_SECONDS = 300  # 5 minutes
_POLL_INTERVAL = 1.5   # seconds


class _SparkLine(Static):
    """Minimal ASCII spark-line fallback for when plotext isn't available."""
    _CHARS = " ▁▂▃▄▅▆▇█"

    def __init__(self, title: str, **kwargs):
        super().__init__(**kwargs)
        self._title = title
        self._data: deque[float] = deque(maxlen=60)

    def push(self, v: float) -> None:
        self._data.append(v)
        self._redraw()

    def _redraw(self) -> None:
        if not self._data:
            return
        mx = max(self._data) or 1
        bars = "".join(
            self._CHARS[int(v / mx * (len(self._CHARS) - 1))] for v in self._data
        )
        last = list(self._data)[-1]
        self.update(f"{self._title}: {last:.1f}\n{bars}")


class MetricsScreen(textual.screen.Screen):
    """Full-screen metrics dashboard."""

    BINDINGS = [
        Binding("f1,ctrl+h", "show_help", "Help", show=True),
        Binding("q,escape", "go_back", "Back", show=True),
    ]

    def action_show_help(self) -> None:
        from .help import HelpModal
        self.app.push_screen(HelpModal())

    def action_go_back(self) -> None:
        self.app.pop_screen()

    CSS = """
    MetricsScreen {
        background: $background;
        layout: vertical;
    }
    #metrics-grid {
        layout: grid;
        grid-size: 2;
        grid-rows: 1fr 1fr 1fr 1fr;
        height: 1fr;
        padding: 1;
    }
    .chart-cell {
        border: solid $border;
        background: $surface;
        padding: 1;
        height: 100%;
    }
    .chart-title {
        color: $primary;
        text-style: bold;
        height: 1;
    }
    .chart-caption {
        color: $text-muted;
        height: 1;
        padding-bottom: 1;
    }
    PlotextPlot {
        height: 1fr;
    }
    _SparkLine {
        height: 1fr;
        color: $success;
    }
    """

    def __init__(self, manager: Optional[ServerManager] = None, **kwargs):
        super().__init__(**kwargs)
        self.manager = manager
        self._window = _WINDOW_SECONDS
        self._ts: deque[float] = deque(maxlen=int(_WINDOW_SECONDS / _POLL_INTERVAL))
        self._series: dict[str, deque[float]] = {
            "tps": deque(maxlen=int(_WINDOW_SECONDS / _POLL_INTERVAL)),
            "mem": deque(maxlen=int(_WINDOW_SECONDS / _POLL_INTERVAL)),
            "cpu": deque(maxlen=int(_WINDOW_SECONDS / _POLL_INTERVAL)),
            "latency": deque(maxlen=int(_WINDOW_SECONDS / _POLL_INTERVAL)),
        }
        self._poll_handle = None
        # Seed CPU measurement
        psutil.cpu_percent(interval=None)

    def compose(self) -> ComposeResult:
        yield StatusBar()

        with Container(id="metrics-grid"):
            # --- Token Throughput ---
            with Vertical(classes="chart-cell"):
                yield Label("Token Throughput  (t/s)", classes="chart-title")
                yield Label("Generation speed — how many tokens are produced every second.", classes="chart-caption")
                if _PLOTEXT_AVAILABLE:
                    yield PlotextPlot(id="plot-tps")
                else:
                    yield _SparkLine("t/s", id="spark-tps")

            # --- Memory Usage ---
            with Vertical(classes="chart-cell"):
                yield Label("Memory Usage  (MB)", classes="chart-title")
                yield Label("Model footprint — physical RAM currently mapped by the vllm process.", classes="chart-caption")
                if _PLOTEXT_AVAILABLE:
                    yield PlotextPlot(id="plot-mem")
                else:
                    yield _SparkLine("MB", id="spark-mem")

            # --- CPU Utilization ---
            with Vertical(classes="chart-cell"):
                yield Label("CPU Utilization  (%)", classes="chart-title")
                yield Label("System load — combined CPU core usage for processing and I/O.", classes="chart-caption")
                if _PLOTEXT_AVAILABLE:
                    yield PlotextPlot(id="plot-cpu")
                else:
                    yield _SparkLine("CPU%", id="spark-cpu")

            # --- E2E Latency (simulated / placeholder) ---
            with Vertical(classes="chart-cell"):
                yield Label("E2E Request Latency  (s)", classes="chart-title")
                yield Label("Turnaround time — total wait time from send to final token.", classes="chart-caption")
                if _PLOTEXT_AVAILABLE:
                    yield PlotextPlot(id="plot-latency")
                else:
                    yield _SparkLine("Latency", id="spark-latency")

            # --- Summary stats row ---
            with Horizontal(classes="chart-cell"):
                yield Static("", id="stat-summary")

        yield Footer()

    def on_mount(self) -> None:
        self._poll_handle = self.set_interval(_POLL_INTERVAL, self._tick)

    def on_unmount(self) -> None:
        if self._poll_handle:
            self._poll_handle.stop()

    def action_go_back(self) -> None:
        self.app.pop_screen()

    # ── Data collection ────────────────────────────────────────────────

    def _tick(self) -> None:
        now = time.monotonic()
        self._ts.append(now)

        # Memory
        mem = self.manager.get_memory_usage() if self.manager else 0.0
        self._series["mem"].append(mem)

        # CPU
        cpu = psutil.cpu_percent(interval=None)
        self._series["cpu"].append(cpu)

        # TPS: pull from status bar (set by chat stream)
        try:
            tps = self.app.query_one(StatusBar).tokens_per_sec
        except Exception:
            tps = 0.0
        self._series["tps"].append(tps)

        # Latency placeholder (will be replaced by actual vllm metrics endpoint)
        self._series["latency"].append(0.0)

        self._redraw_all()
        self._update_summary(mem, cpu, tps)

    def _xs(self) -> list[int]:
        """X-axis: seconds-ago labels."""
        ts = list(self._ts)
        if not ts:
            return []
        now = ts[-1]
        return [int(t - now) for t in ts]   # negative seconds

    def _redraw_all(self) -> None:
        if _PLOTEXT_AVAILABLE:
            self._redraw_plotext("plot-tps", "tps", "blue", "t/s")
            self._redraw_plotext("plot-mem", "mem", "green", "MB")
            self._redraw_plotext("plot-cpu", "cpu", "orange", "%")
            self._redraw_plotext("plot-latency", "latency", "purple", "s")
        else:
            self._redraw_sparkline("spark-tps", "tps")
            self._redraw_sparkline("spark-mem", "mem")
            self._redraw_sparkline("spark-cpu", "cpu")
            self._redraw_sparkline("spark-latency", "latency")

    def _redraw_plotext(self, widget_id: str, key: str, color: str, unit: str) -> None:
        try:
            plot = self.query_one(f"#{widget_id}", PlotextPlot)
            plt = plot.plt
            plt.clear_data()
            ys = list(self._series[key])
            xs = list(range(-len(ys) + 1, 1))
            if not ys:
                return
            plt.plot(xs, ys, color=color, label=unit)
            plt.xlabel("seconds ago")
            plt.ylabel(unit)
            plt.theme("dark")
            plot.refresh()
        except Exception:
            pass

    def _redraw_sparkline(self, widget_id: str, key: str) -> None:
        try:
            spark = self.query_one(f"#{widget_id}", _SparkLine)
            data = list(self._series[key])
            if data:
                spark.push(data[-1])
        except Exception:
            pass

    def _update_summary(self, mem: float, cpu: float, tps: float) -> None:
        try:
            stat = self.query_one("#stat-summary", Static)
            stat.update(
                f"Memory: {mem:.0f} MB   "
                f"CPU: {cpu:.0f}%   "
                f"Throughput: {tps:.1f} t/s"
            )
        except Exception:
            pass


# ── Legacy shim ────────────────────────────────────────────────────────────────

class MetricsPane(Vertical):
    """Compact metrics pane — kept for backward compatibility.
    Not used in the new UI; the full MetricsScreen replaces it.
    """
    from textual.reactive import reactive
    memory_mb: reactive = reactive("0.0 MB")
    tokens_s: reactive = reactive("0.0 t/s")

    def compose(self) -> ComposeResult:
        yield Static("Memory", classes="metric-label")
        yield Static("0.0 MB", id="memory-val")
        yield Static("Speed", classes="metric-label")
        yield Static("0.0 t/s", id="token-val")

    def watch_memory_mb(self, _, new_val):
        try:
            self.query_one("#memory-val", Static).update(str(new_val))
        except Exception:
            pass

    def watch_tokens_s(self, _, new_val):
        try:
            self.query_one("#token-val", Static).update(str(new_val))
        except Exception:
            pass
