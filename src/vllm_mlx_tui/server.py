"""
Server lifecycle manager.

Responsible for:
  - Starting vllm-mlx as an async subprocess
  - Streaming stderr to a caller-supplied callback (boot log)
  - Polling the /v1/models readiness endpoint
  - Tracking memory usage of the server process tree
  - Graceful (SIGTERM → SIGKILL) shutdown
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import shutil
import signal
import socket
import sys
import time
from collections import deque
from typing import Awaitable, Callable, List, Optional, Tuple

import httpx
import psutil

LogCallback = Callable[[str], Awaitable[None]]

VLLM_MLX_BIN = os.environ.get("VLLM_MLX_BIN", "vllm-mlx")
API_READY_TIMEOUT = int(os.environ.get("API_READY_TIMEOUT", "1200"))
_DEFAULT_VLLM_MLX_BIN = "vllm-mlx"


def _validate_model_id(model_id: str) -> None:
    """Strict validation for HF model IDs or local paths."""
    import re
    # repo_id: namespace/model-name or model-name (with dots, dashes, underscores)
    # paths: absolute or relative with common safe chars
    if not re.match(r"^[a-zA-Z0-9/._\-\\]+$", model_id) or model_id.startswith("-"):
        raise ValueError(f"Invalid model ID or path: {model_id!r}")


def _vllm_mlx_command_prefix() -> List[str]:
    """Return the command prefix used to launch vllm-mlx."""
    if VLLM_MLX_BIN != _DEFAULT_VLLM_MLX_BIN:
        return [VLLM_MLX_BIN]

    # Use a tiny Python launcher when possible so we can apply compatibility
    # patches for known vllm-mlx releases before its CLI imports the engine.
    if importlib.util.find_spec("vllm_mlx") and importlib.util.find_spec("vllm_mlx_tui"):
        return [sys.executable, "-m", "vllm_mlx_tui._vllm_mlx_launcher"]

    return [VLLM_MLX_BIN]


class ServerManager:
    """Manages the lifecycle of one vllm-mlx server subprocess."""

    def __init__(
        self,
        model_id: str,
        snapshot_path: str = "",
        port: int = 8001,
        host: str = "127.0.0.1",
        extra_args: Optional[List[str]] = None,
        use_ngrok: bool = False,
    ):
        _validate_model_id(model_id)
        if snapshot_path:
            _validate_model_id(snapshot_path)

        self.model_id = model_id
        # Prefer snapshot path for the actual launch arg (avoids HF re-download)
        self.model_arg = snapshot_path if snapshot_path else model_id
        self.port = port
        self.host = host
        self.extra_args = extra_args or []
        self.use_ngrok = use_ngrok

        self.process: Optional[asyncio.subprocess.Process] = None
        self._ngrok_proc: Optional[asyncio.subprocess.Process] = None
        self.stderr_log: deque[str] = deque(maxlen=500)
        self._stderr_task: Optional[asyncio.Task] = None
        self._log_cb: Optional[LogCallback] = None

    @property
    def base_url(self) -> str:
        h = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{h}:{self.port}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(
        self,
        log_callback: Optional[LogCallback] = None,
    ) -> Tuple[bool, str]:
        """Start the server and wait until /v1/models responds.

        Returns (ready: bool, error_message: str).
        """
        self._log_cb = log_callback

        command_prefix = _vllm_mlx_command_prefix()
        if command_prefix == [VLLM_MLX_BIN] and not shutil.which(VLLM_MLX_BIN):
            return False, f"'{VLLM_MLX_BIN}' not found on PATH."

        cmd = [
            *command_prefix, "serve", self.model_arg,
            "--host", self.host,
            "--port", str(self.port),
        ] + self.extra_args

        await self._emit_log(f"$ {' '.join(cmd)}")

        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError:
            return False, f"'{VLLM_MLX_BIN}' not found."
        except Exception as e:
            return False, str(e)

        # Pre-flight check: is the port already bound?
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((self.host, self.port)) == 0:
                return False, f"Port {self.port} is already in use. Please stop the existing server first."

        # Stream stderr in background
        self._stderr_task = asyncio.create_task(self._stream_stderr())

        # Race: process exit vs HTTP readiness
        poll_task = asyncio.create_task(self._poll_readiness())
        exit_task = asyncio.create_task(self.process.wait())

        done, pending = await asyncio.wait(
            [poll_task, exit_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for t in pending:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

        if exit_task in done:
            # Process exited before becoming ready
            code = exit_task.result()
            lines = list(self.stderr_log)[-20:]
            return False, f"Process exited (code {code}):\n" + "\n".join(lines)

        # HTTP poll finished
        try:
            ready = poll_task.result()
        except Exception as e:
            return False, f"Readiness poll error: {e}"

        if not ready:
            return False, f"Server not ready after {API_READY_TIMEOUT}s."

        if self.use_ngrok:
            asyncio.create_task(self._start_ngrok())

        return True, ""

    def stop(self) -> None:
        """Graceful SIGTERM → SIGKILL shutdown with process group awareness."""
        if self._ngrok_proc:
            try:
                # Ngrok doesn't usually need a group kill in this simple setup
                self._ngrok_proc.terminate()
            except Exception:
                pass

        if not self.process or self.process.returncode is not None:
            return

        try:
            pgid = os.getpgid(self.process.pid)
            # Send SIGTERM to the entire process group
            os.killpg(pgid, signal.SIGTERM)

            # Wait up to 3 seconds for exit
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if self.process.returncode is not None:
                    return
                # Check if process is still alive without waiting (which would block)
                try:
                    os.kill(self.process.pid, 0)
                except ProcessLookupError:
                    return
                time.sleep(0.1)

            # Escalate to SIGKILL if still running
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        except Exception:
            # Last ditch: try to kill just the main process
            try:
                self.process.kill()
            except Exception:
                pass

    def get_memory_usage(self) -> float:
        """Return RSS of the entire process group in MB. Optimized via caching."""
        if not self.process or self.process.returncode is not None:
            return 0.0

        try:
            # We use psutil to find descendants of the main process
            main_proc = psutil.Process(self.process.pid)
            total_rss = main_proc.memory_info().rss
            for child in main_proc.children(recursive=True):
                try:
                    total_rss += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return total_rss / (1024 * 1024)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _emit_log(self, line: str) -> None:
        self.stderr_log.append(line)
        if self._log_cb:
            try:
                await self._log_cb(line)
            except Exception:
                pass

    async def _stream_stderr(self) -> None:
        """Read stderr and forward each line via _emit_log."""
        if not self.process or not self.process.stderr:
            return
        try:
            while True:
                raw = await self.process.stderr.readline()
                if not raw:
                    break
                line = raw.decode(errors="replace").rstrip()
                await self._emit_log(line)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            await self._emit_log(f"[stderr reader error: {e}]")

    async def _poll_readiness(self) -> bool:
        """Poll /v1/models until it lists the requested model, or timeout."""
        # vllm-mlx usually lists the full hub ID, but sometimes just the basename
        short_id = self.model_id.split("/")[-1]

        async with httpx.AsyncClient() as client:
            for _ in range(API_READY_TIMEOUT):
                try:
                    res = await client.get(
                        f"{self.base_url}/v1/models", timeout=2.0
                    )
                    if res.status_code == 200:
                        data = res.json().get("data", [])
                        current_ids = [m.get("id", "") for m in data]
                        
                        # Match full path, short name, or the exact absolute local path we booted with
                        if (self.model_id in current_ids or 
                            short_id in current_ids or 
                            self.model_arg in current_ids):
                            await self._emit_log(f"[TUI] ✅ Model '{self.model_id}' found in registry! Initializing chat...")
                            # Final 1s grace to ensure engine is truly stable
                            await asyncio.sleep(1.0)
                            return True
                        else:
                            # Log every ~5s to not spam but keep user informed
                            if _ % 5 == 0:
                                await self._emit_log(f"[TUI] ⏳ Waiting for engine metadata (ID '{self.model_id}' not listed yet)...")
                except (httpx.RequestError, ValueError, KeyError):
                    if _ % 5 == 0:
                        await self._emit_log(f"[TUI] ⏳ Still waiting for port {self.port} to become responsive...")
                    pass
                await asyncio.sleep(1.0)
        return False

    async def _start_ngrok(self) -> None:
        import shutil as _shutil
        if not _shutil.which("ngrok"):
            await self._emit_log("[ngrok not on PATH — tunnel skipped]")
            return
        try:
            self._ngrok_proc = await asyncio.create_subprocess_exec(
                "ngrok", "http", str(self.port),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.sleep(2.0)
            # Try to read the public URL from the ngrok API
            async with httpx.AsyncClient() as c:
                resp = await c.get("http://127.0.0.1:4040/api/tunnels", timeout=3.0)
                tunnels = resp.json().get("tunnels", [])
                for t in tunnels:
                    url = t.get("public_url", "")
                    if url.startswith("https"):
                        await self._emit_log(f"[ngrok] Public URL: {url}")
                        return
        except Exception as e:
            await self._emit_log(f"[ngrok] Failed: {e}")

    @staticmethod
    def _same_group(proc: psutil.Process, pgid: int) -> bool:
        try:
            return os.getpgid(proc.info["pid"]) == pgid
        except Exception:
            return False
