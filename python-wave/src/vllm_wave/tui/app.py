from textual.app import App
from .launcher import LauncherScreen

class VLLMWaveApp(App):
    CSS = """
    .hidden {
        display: none;
    }
    .error {
        color: red;
    }
    """
    
    def on_mount(self):
        self.push_screen(LauncherScreen())
