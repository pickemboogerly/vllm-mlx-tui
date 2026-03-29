# vllm-mlx-tui 🚀

> A premium, high-contrast Terminal User Interface (TUI) for discovering, serving, and chatting with local models on Apple Silicon.

Built for the **MLX** ecosystem, **vllm-mlx-tui** is a keyboard-centric launcher and chat client that wraps `vllm-mlx`. It transforms your terminal into a powerful local AI workstation with a polished, interactive interface that respects your system themes.

---

## ✨ Features

- **🎨 High-Contrast "Outlined" Aesthetic**: A premium, border-driven design that eliminates distracting background fills for maximum text legibility across all themes.
- **🌗 Native Theme Integration**: Fully compatible with Textual's native theme engine (Dark/Light/etc.) via the standard Command Palette (`Ctrl + \`).
- **📝 Robust Markdown Support**: Enhanced GFM rendering with an auto-healing "fixer" for LLM-generated tables and code blocks.
- **🔍 Auto-Discovery**: Automatically scans your Hugging Face cache for compatible MLX / safetensors models.
- **⚡ Supercharged Launcher**: A two-column configuration wizard with a "Launch on Enter" shortcut and full profile management.
- **📊 Real-time Metrics**: A live header status bar showing Memory consumption (MB), Generation speed (TPS), CPU usage (%), and Context window (%) saturation.
- **⌨️ Premium Command Reference**: A beautifully organized help system (`F1`) to master the full keyboard-powered experience.
- **🧠 Session Context**: Full support for system prompts, temperature tuning, and session-based chat management.

---

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/vllm-mlx-tui.git
   cd vllm-mlx-tui
   ```

2. **Set up a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the package**:
   ```bash
   pip install -e .
   ```

### Running the TUI

Once installed, simply run the command:

```bash
vllm-mlx-tui
```

---

## 📖 Keyboard Shortcuts

| Shortcut | Action | Context |
|----------|--------|---------|
| **Enter** | Launch Model / Send Message | Global / Chat |
| **Ctrl + \** | Open Command Palette (Themes) | Global |
| **F1 / Ctrl + H** | Open Command Reference | Global |
| **Ctrl + S** | Save Current Profile / Session | Launcher / Chat |
| **Ctrl + N** | Start New Chat Session | Chat |
| **Ctrl + R** | Switch Back to Launcher | Chat |
| **Ctrl + L** | Clear Chat History | Chat |
| **F2** | Rename Active Session (Autosave) | Chat |
| **Ctrl + E** | Export Chat to Markdown | Chat |
| **Ctrl + M** | Toggle Metrics Dashboard | Global |
| **Ctrl + Q** | Graceful Shutdown | Global |

---

## 🛠 Tech Stack

- **Core UI**: [Textual](https://textual.textualize.io/) (Python TUI Framework)
- **Rendering**: [Rich](https://rich.readthedocs.io/) (Markdown, Tables, and Logging)
- **Networking**: [httpx](https://www.python-httpx.org/) + [httpx-sse](https://github.com/florimondmanca/httpx-sse)
- **Inference**: [vllm-mlx](https://github.com/ml-explore/vllm-mlx)
- **Discovery**: `huggingface_hub` scan cache API

---

## ⚖️ License

MIT License. See `LICENSE` for details.
