# vLLM-MLX TUI – Reference Architecture

## Overview

This project provides a macOS-native TUI for managing and interacting with local LLM inference powered by `vllm-mlx`.

The system is designed as a **service-oriented control plane**, where the TUI manages lifecycle, configuration, and interaction with a locally running inference server.

---

## Architectural Principles

1. **Service-first design**
   - All inference runs through a local HTTP server
   - The TUI never directly executes model inference

2. **Loose coupling**
   - Backend interactions occur via API contracts, not internal imports

3. **Profile-driven configuration**
   - All model + runtime parameters are declarative

4. **Replaceable backend layer**
   - Future support for alternative backends without UI changes

5. **Mac-first constraints**
   - Optimized for MLX and Apple Silicon memory behavior

---

## High-Level Architecture

```

+----------------------+
|        TUI           |
| (Textual Interface)  |
+----------+-----------+
|
v
+----------------------+
|   Controller Layer   |
|  (State + Actions)   |
+----------+-----------+
|
v
+----------------------+

| Backend Adapter          |
| ------------------------ |
| VLLM-MLX Service         |
| +----------+-----------+ |

```
       |
       v
```

+----------------------+
| Local HTTP Server    |
| (vllm-mlx runtime)   |
+----------------------+

````

---

## Core Components

### 1. TUI Layer

**Responsibilities:**
- Render views (chat, metrics, profiles)
- Handle user input
- Display streaming outputs

**Recommended Stack:**
- `textual`
- `rich`

---

### 2. Controller Layer

**Responsibilities:**
- Application state management
- Command routing
- Async orchestration

**Key Concepts:**
- Active profile
- Server state (running/stopped)
- Chat session state

---

### 3. Backend Adapter Interface

Defines a stable contract between the UI and inference backend.

```python
class InferenceBackend:
    async def start(self, profile): ...
    async def stop(self): ...
    async def health_check(self): ...
    async def generate(self, prompt): ...
    async def stream(self, prompt): ...
    async def metrics(self): ...
````

---

### 4. VLLM-MLX Backend Implementation

**Responsibilities:**

* Launch `vllm-mlx` via subprocess
* Manage ports and environment
* Handle HTTP communication

---

## Process Management

### Lifecycle

```
START:
  spawn subprocess
  wait for health endpoint
  mark as ready

RUNNING:
  stream logs
  collect metrics

STOP:
  terminate gracefully
  fallback to force kill
```

### Implementation Notes

* Use `asyncio.subprocess`
* Capture stdout/stderr for logs
* Implement retry + timeout logic

---

## Profile System

Profiles define runtime configuration.

### Example

```yaml
name: mistral-mlx
model: mistral-7b
backend: mlx
port: 8000
max_model_len: 8192
gpu_memory_utilization: 0.9
dtype: float16
```

### Storage

```
~/.vllm-tui/profiles/
```

---

## Metrics System

### Sources

* HTTP endpoints (if available)
* Log parsing fallback

### Metrics to Track

* tokens/sec
* request latency
* queue depth
* memory usage

### Refresh Strategy

* Poll every 0.5–2 seconds
* Non-blocking async updates

---

## Chat Subsystem

### Features

* Streaming completions
* Multi-turn sessions
* Per-profile session isolation

### API

* OpenAI-compatible `/v1/chat/completions`

### Design

```
User Input → Controller → Backend.stream() → UI Renderer
```

---

## Logging

### Goals

* Debug server issues
* Surface errors in UI

### Approach

* Structured log parsing
* Optional raw log viewer panel

---

## Error Handling

### Categories

1. Startup failures
2. Runtime crashes
3. API errors
4. Resource exhaustion (MLX memory)

### Strategy

* Surface actionable errors in UI
* Auto-restart (optional)
* Graceful degradation

---

## macOS / MLX Considerations

* Memory pressure is critical
* Model swaps should:

  * fully stop previous process
  * release memory before restart
* Avoid concurrent large models

---

## Future Extensions

* Multiple concurrent servers
* Remote host support
* Plugin system for metrics
* Lightweight benchmarking mode

---

## Non-Goals

* Modifying vLLM internals
* Replacing inference engine
* Supporting non-MLX backends (initially)

---

You’re solving a classic concurrency problem:

> **Continuously rendering a TUI while simultaneously handling:**
>
> * user input
> * streaming HTTP responses
> * background metrics polling
> * subprocess lifecycle events

In a macOS + MLX context, correctness (no UI blocking, no orphan processes) matters more than raw throughput.

---

# Core Design Constraint

**The TUI must never block on I/O.**

That leads to a strict rule:

> All external interactions (HTTP, subprocess, metrics) must run in **async tasks**, coordinated through a central event loop.

---

# Event Loop Model (Textual + asyncio)

Using **Textual**, you already operate inside an asyncio loop. The correct approach is:

* Use Textual as the **UI scheduler**
* Use `asyncio.create_task()` for all concurrent work
* Communicate via **message passing**, not shared mutable state

---

# High-Level Concurrency Topology

```
                ┌────────────────────┐
                │   Textual App      │
                │  (Main Event Loop) │
                └─────────┬──────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        v                 v                 v
 Chat Stream Task   Metrics Poller   Process Monitor
        │                 │                 │
        └──────────┬──────┴──────┬──────────┘
                   v             v
              Message Bus / Event Dispatch
                          │
                          v
                     UI Components
```

---

# Core Patterns

## 1. Task-Oriented Concurrency

Each responsibility becomes a long-lived async task:

| Task            | Responsibility           |
| --------------- | ------------------------ |
| Chat Stream     | Handle streaming tokens  |
| Metrics Poller  | Periodic backend polling |
| Process Monitor | Watch subprocess health  |
| Log Streamer    | Parse stdout/stderr      |

---

## 2. Message Passing (Critical)

Avoid directly mutating UI from background tasks.

Instead:

* Emit events/messages
* Let UI handle rendering

Textual provides:

* `post_message()`
* custom `Message` classes

---

# Concrete Structure

## App Skeleton

```python
from textual.app import App
import asyncio

class VLLMTUI(App):

    async def on_mount(self):
        self.backend = VLLMBackend()

        # Start background tasks
        self.metrics_task = asyncio.create_task(self.metrics_loop())
        self.process_task = asyncio.create_task(self.process_monitor())

    async def on_unmount(self):
        self.metrics_task.cancel()
        self.process_task.cancel()
        await self.backend.stop()
```

---

# Metrics Loop

## Design Goals

* Non-blocking
* Resilient to failures
* Adjustable frequency

```python
async def metrics_loop(self):
    while True:
        try:
            metrics = await self.backend.metrics()
            self.post_message(MetricsUpdate(metrics))
        except Exception as e:
            self.post_message(BackendError(str(e)))

        await asyncio.sleep(1.0)
```

---

# Chat Streaming Loop

This is the most sensitive part.

## Key Requirement

* Stream tokens incrementally
* Avoid blocking UI
* Allow cancellation

---

## Pattern: Producer → UI Messages

```python
async def stream_chat(self, prompt: str):
    try:
        async for token in self.backend.stream(prompt):
            self.post_message(TokenStream(token))

        self.post_message(StreamComplete())

    except Exception as e:
        self.post_message(StreamError(str(e)))
```

---

## Triggering It

```python
def handle_user_prompt(self, prompt):
    asyncio.create_task(self.stream_chat(prompt))
```

---

# Process Monitor

Tracks whether the backend is alive.

```python
async def process_monitor(self):
    while True:
        alive = await self.backend.health_check()

        self.post_message(ProcessStatus(alive))

        await asyncio.sleep(2.0)
```

---

# Subprocess Log Streaming

If you capture stdout:

```python
async def log_reader(self, stream):
    while True:
        line = await stream.readline()
        if not line:
            break

        self.post_message(LogLine(line.decode()))
```

Spawned during backend start:

```python
self.log_task = asyncio.create_task(
    self.log_reader(process.stdout)
)
```

---

# Cancellation Strategy (Important)

Streaming + model switching introduces race conditions.

## Rules

* Keep task references
* Cancel before starting new ones

```python
if self.current_stream_task:
    self.current_stream_task.cancel()

self.current_stream_task = asyncio.create_task(
    self.stream_chat(prompt)
)
```

---

# Backpressure Handling

Streaming tokens can overwhelm UI rendering.

## Mitigation

* Buffer tokens in small batches:

```python
buffer = []
async for token in stream:
    buffer.append(token)

    if len(buffer) >= 5:
        self.post_message(TokenBatch(buffer))
        buffer.clear()
```

---

# Error Isolation

Never let a task crash silently.

Wrap all loops:

```python
async def safe_loop(self):
    try:
        while True:
            ...
    except asyncio.CancelledError:
        pass
    except Exception as e:
        self.post_message(SystemError(str(e)))
```

---

# State Model

Avoid scattering state across tasks.

Use a central state object:

```python
class AppState:
    server_running: bool
    active_profile: str
    current_model: str
    tokens_per_sec: float
```

UI updates react to messages → update state → re-render.

---

# Timing Model Summary

| Loop           | Frequency    |
| -------------- | ------------ |
| Metrics        | 0.5–2s       |
| Process Health | ~2s          |
| Chat Streaming | event-driven |
| Logs           | continuous   |

---

# Subtle macOS/MLX Considerations

* **Startup latency is high**
  → show “warming up” state
* **Memory pressure spikes**
  → block new requests until stable
* **Process teardown must complete**
  → await termination before restart

---

# Minimal Execution Flow

### User sends prompt:

1. UI event fires
2. Cancel existing stream
3. Spawn `stream_chat` task
4. Tokens arrive → messages → UI updates

### Metrics:

1. Background loop polls
2. Sends updates
3. UI refreshes panels

### Model switch:

1. Cancel all tasks
2. Stop backend
3. Start new process
4. Resume loops

---

# Key Takeaways

* Treat everything as **independent async workers**
* Communicate via **messages, not shared state**
* Always **own your task lifecycle**
* Design for **cancellation first**, not after
