# Common to all

- Top pane status line:
    - Left: Application Name v0.0.0
    - Center: Model Name
    - Right:
        - Memory Usage
        - Generation Speed: t/s
        - Context Window % (Estimated by character count / 4)
            - CPU %
- Bottom Status Line (Global Navigation Tabs)
    - Start Session
    - Chat
    - Metrics
- Second Bottom Status Line (Textual Footer)
    - Command Glossary
    - Theme

# Chat UI

- Chat Window
    - Boxed chat text
    - Markdown Formatting
    - Left Align Assistant
    - Right Align You
    - Enable Responses Copying
    - Enable entire dialogue Copying
    - Hide / Collapse reasoning to a single line in a different color than the final message (Using `<think>` tag parsing and Textual `Collapsible`). Allow the reasoning to be shown by a command or configuration option.
    - Stop Response button / Command
- Side Panel
    - Chat Options:
        - Temperature (Applies instantly to next message)
        - Top P (Applies instantly to next message)
        - Max Output Tokens (Applies instantly to next message)
    - Chat Management
    - New Chat Button
    - Chat list history List (Reads from JSON files in `~/.cache/vllm-wave/chats/`)
        - chat name
    - Rename Chat
    - Delete Chat
    - Save Chat (Persists to JSON)
    - Export as Markdown
    - Switch Profile (Terminates current subprocess and boots new one)
    - Metrics button
    - Quit

# vLLM-MLX Launcher

- Cached Model Selection Menu
    - List by model name only, no path
    - If possible show Parameter Size (Estimated via Safetensors disk size proxy)
- Manual Model Override: Manual model (path or Hub id; overrides table when non-empty)
- Options:
    - Multimodal (--mllm) \[chk\]
    - Ngrok tunnel (public URL) \[chk\]
    - Disable tool-call flags \[chk\]
    - Manual Parser Override \[text\]
    - Port (8000 default)
    - Max Model Len (default)
    - Quantization (opt)
    - Dtype (opt)
- Output boot log (savable)
- Launch Button (init model & web server); Switch to Chat upon launch
- Save Profile
- Load Profile

# Profile Management Modal

&nbsp;

- List of Profile Names & Save Date. (A Profile saves Model + Options + Port)
- Upon Select:
    - Name editable
    - Load / Save / Cancel / Delete / Set As Default (Pre-fills Launcher on next startup)

# Metric Dashboard (Full Screen)

*(Note: Must be exotically beautiful using terminal graphing plugins like `textual-plotext`)*
- Metrics Graph over five minutes
    - E2E Request Latency
    - Time Per Output Token Latency
    - Time to First Token Latency
    - Token Throughput
    - Scheduler State
    - Cache Utilization
    - CPU Utlilization
    - Memory Utilization.