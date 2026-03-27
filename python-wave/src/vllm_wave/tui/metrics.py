from textual.app import ComposeResult
from textual.widgets import Static, Digits
from textual.containers import Vertical
from textual.reactive import reactive

class MetricsPane(Vertical):
    memory_mb = reactive("0.0 MB")
    tokens_s = reactive("0.0 t/s")
    
    def compose(self) -> ComposeResult:
        yield Static("Memory Usage", classes="metric-label")
        yield Digits("0.0", id="memory-val")
        yield Static("Generation Speed", classes="metric-label")
        yield Digits("0.0", id="token-val")

    def watch_memory_mb(self, old_val, new_val):
        self.query_one("#memory-val").update(str(new_val).replace("MB","").strip())

    def watch_tokens_s(self, old_val, new_val):
        self.query_one("#token-val").update(str(new_val).replace("t/s","").strip())
