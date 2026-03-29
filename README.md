# vllm-mlx-tui 🚀

> A premium Terminal User Interface (TUI) for discovering, serving, and chatting with local models on Apple Silicon.

Built for the **MLX** ecosystem, **vllm-mlx-tui** is a lightweight Python-based launcher and chat client that wraps `vllm-mlx`. It transforms your terminal into a powerful local AI workstation with a polished, interactive interface.

---

## ✨ Features

- **🔍 Auto-Discovery**: Automatically scans your Hugging Face cache for compatible MLX / safetensors models.
- **⚡ Supercharged Serving**: Configure and launch `vllm-mlx serve` as a background process directly from the UI.
- **💬 Polished Chat Interface**: A rich [Textual](https://textual.textualize.io/)-based chat UI with streaming replies and markdown rendering.
- **🧠 Profile Management**: Easily manage model-specific system prompts and generation settings.
- **📈 Live Metrics**: Monitor your model's performance and memory consumption in real-time.
- **🛠 Subprocess Model**: Keeps the UI process light by running inference in a separate process, ensuring a smooth, non-blocking experience.

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

## 📖 Usage

### 1. The Launcher
Upon startup, `vllm-mlx-tui` will list all locally cached models it finds in your Hugging Face directory. You can select a model, set serving parameters (like KV cache fraction), and hit **Start**.

### 2. The Chat
Once the server is ready, you'll be handed off to the chat console.
- **Ctrl+N**: New chat session
- **Ctrl+R**: Switch model (restarts the launcher)
- **Ctrl+Q**: Quit

---

## 🛠 Tech Stack

- **Core UI**: [Textual](https://textual.textualize.io/) (Python TUI Framework)
- **Networking**: [httpx](https://www.python-httpx.org/) + [httpx-sse](https://github.com/florimondmanca/httpx-sse)
- **Inference Engine**: [vllm-mlx](https://github.com/ml-explore/vllm-mlx)
- **Model Discovery**: `huggingface_hub` scan cache API

---

## 📝 Configuration

You can override default behavior using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_PORT` | Port for the vLLM server | `8001` |
| `VLLM_HOST` | Host for the vLLM server | `127.0.0.1` |
| `VLLM_MLX_BIN` | Binary name for the server | `vllm-mlx` |

---

## ⚖️ License

MIT License. See `LICENSE` for details.
