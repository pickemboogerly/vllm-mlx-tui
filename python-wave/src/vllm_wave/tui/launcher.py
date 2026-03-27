from textual.screen import Screen
from textual.widgets import Header, Footer, ListView, ListItem, Label, Button, LoadingIndicator
from textual.containers import Vertical, Horizontal
from textual import work
from ..cache import list_cached_models
from ..server import ServerManager
from .chat import ChatScreen
import asyncio

class LauncherScreen(Screen):
    manager = None

    def compose(self):
        yield Header()
        yield Vertical(
            Label("Select a model from your HuggingFace cache:", id="prompt"),
            ListView(id="model-list"),
            Horizontal(
                Button("Launch Model", id="launch-btn", variant="primary"),
                LoadingIndicator(id="loading", classes="hidden")
            ),
            Label("", id="error-text", classes="error")
        )
        yield Footer()

    def on_mount(self):
        models = list_cached_models()
        model_list = self.query_one("#model-list", ListView)
        for model in models:
            import re
            safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", model)
            li = ListItem(Label(model), id=f"model_{safe_id}")
            li.model_id = model
            model_list.append(li)
            
    def switch_to_chat(self, model_id: str):
        self.app.push_screen(ChatScreen(manager=self.manager, model_id=model_id))

    @work(exclusive=True)
    async def boot_server(self, model_id: str):
        # UI updates
        btn = self.query_one("#launch-btn", Button)
        load = self.query_one("#loading", LoadingIndicator)
        err = self.query_one("#error-text", Label)
        
        btn.disabled = True
        load.remove_class("hidden")
        err.update("")
        
        self.manager = ServerManager(model_id)
        ready, error_msg = await self.manager.start()
        
        load.add_class("hidden")
        btn.disabled = False
        
        if not ready:
            err.update(f"Failed to start vllm-mlx: {error_msg}")
        else:
            self.switch_to_chat(model_id)

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "launch-btn":
            list_view = self.query_one("#model-list", ListView)
            if list_view.index is not None:
                item = list_view.highlighted_child
                if item:
                    self.boot_server(str(item.model_id))
