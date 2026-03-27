import asyncio
import os
import signal
import time
import sys
import psutil
import httpx
from typing import Optional, Tuple
import re
import socket

class ServerManager:
    def __init__(self, model_id: str):
        if not re.match(r"^[a-zA-Z0-9/._-]+$", model_id) or model_id.startswith("-"):
            raise ValueError("Invalid model ID format.")
        self.model_id = model_id
        self.process: Optional[asyncio.subprocess.Process] = None
        self.stderr_tail: list[str] = []

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', 0))
            self.port = s.getsockname()[1]

    async def start(self) -> Tuple[bool, str]:
        """Starts the server and waits for it to become ready via a timeout race."""
        cmd = ["vllm-mlx", "serve", self.model_id, "--port", str(self.port)]

        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=os.setpgrp  # Create a new process group for clean kill
            )
        except Exception as e:
            if isinstance(e, FileNotFoundError):
                return False, "vllm-mlx executable not found on PATH."
            raise

        # Race process completion vs polling HTTP endpoint.
        # M5: _read_stderr no longer calls process.wait() itself, preventing
        # the double-wait race between start() and _read_stderr().
        read_stderr_task = asyncio.create_task(self._read_stderr())
        poll_task = asyncio.create_task(self._poll_readiness())

        done, pending = await asyncio.wait(
            [read_stderr_task, poll_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel the pending task
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Check which task finished first
        if read_stderr_task in done:
            # The process exited prematurely before HTTP became ready.
            # M5: process.wait() is only called here, not inside _read_stderr.
            try:
                await self.process.wait()
            except Exception:
                pass
            return False, "\n".join(self.stderr_tail[-20:])

        elif poll_task in done:
            try:
                poll_result = poll_task.result()
                if poll_result:
                    # C2: validate at least one model is served (TOCTOU mitigation)
                    if not await self._validate_model_loaded():
                        return False, "Server responded but returned no models (possible port collision?)"
                    return True, ""
            except Exception as e:
                return False, f"Timeout or connection issue: {str(e)}"

        return False, "Unknown initialization error"

    async def _validate_model_loaded(self) -> bool:
        """C2: Best-effort check that /v1/models returns at least one model entry."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"http://127.0.0.1:{self.port}/v1/models", timeout=5.0
                )
                data = resp.json()
                return len(data.get("data", [])) > 0
        except Exception:
            return False

    async def _read_stderr(self):
        """Consume stderr to avoid blocking the subprocess pipe.
        M5: Does NOT call self.process.wait() — that is the caller's responsibility.
        """
        if not self.process or not self.process.stderr:
            return

        try:
            while True:
                line = await self.process.stderr.readline()
                if not line:
                    break  # EOF — process exited; return without calling wait()
                decoded = line.decode().strip()
                self.stderr_tail.append(decoded)
                if len(self.stderr_tail) > 100:
                    self.stderr_tail.pop(0)
        except (asyncio.CancelledError, Exception):
            pass
        # M5: intentionally NOT calling await self.process.wait() here.

    async def _poll_readiness(self) -> bool:
        """Poll the /v1/models endpoint."""
        async with httpx.AsyncClient() as client:
            for _ in range(120):  # 120 seconds timeout for large models
                try:
                    res = await client.get(
                        f"http://127.0.0.1:{self.port}/v1/models", timeout=1.0
                    )
                    if res.status_code == 200:
                        return True
                except httpx.RequestError:
                    pass
                await asyncio.sleep(1)
        return False

    def stop(self):
        """Gracefully terminates the server process group.
        C3: Sends SIGTERM and polls for exit up to 3 seconds before SIGKILL.
        """
        if self.process:
            try:
                pgid = os.getpgid(self.process.pid)
                os.killpg(pgid, signal.SIGTERM)

                # C3: Poll for graceful exit (up to 3 seconds) instead of
                # sending SIGKILL immediately after SIGTERM.
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline:
                    try:
                        os.kill(self.process.pid, 0)  # 0 = process-existence check
                    except ProcessLookupError:
                        return  # Exited cleanly
                    time.sleep(0.1)

                # Timeout reached — force kill
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def get_memory_usage(self) -> float:
        """Returns the memory usage in MB for the process tree."""
        if not self.process:
            return 0.0

        try:
            pgid = os.getpgid(self.process.pid)
            total_rss = 0

            for proc in psutil.process_iter(['pid', 'memory_info']):
                try:
                    if os.getpgid(proc.info['pid']) == pgid:
                        total_rss += proc.info['memory_info'].rss
                except (ProcessLookupError, psutil.NoSuchProcess, PermissionError):
                    continue

            return total_rss / (1024 * 1024)
        except (ProcessLookupError, psutil.NoSuchProcess):
            return 0.0
