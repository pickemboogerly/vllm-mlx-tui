"""
HuggingFace model downloader for vllm-mlx-tui.

Leverages the huggingface_hub API to perform full snapshot downloads
with real-time progress reporting.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Awaitable, Callable, Optional

from huggingface_hub import snapshot_download

ProgressCallback = Callable[[float, str], Awaitable[None]]


class ModelDownloader:
    """Manages the download of a model from HuggingFace Hub."""

    def __init__(self, model_id: str):
        self.model_id = model_id

    async def download(self, log_cb: Optional[Callable[[str], Awaitable[None]]] = None) -> str:
        """Download the model snapshot via CLI subprocess and return the local path."""
        import shutil
        cli_path = shutil.which("hf") or shutil.which("huggingface-cli")
        if not cli_path:
            raise RuntimeError("hf/huggingface-cli not found in PATH. Are the python requirements installed?")

        if log_cb:
            await log_cb(f"[TUI] Starting fetching subprocess for '{self.model_id}'...")

        cmd = [cli_path, "download", self.model_id]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def _stream_output(stream: asyncio.StreamReader):
            import re
            buffer = bytes()
            while True:
                chunk = await stream.read(1)
                if not chunk:
                    if buffer:
                        text = buffer.decode(errors="replace").strip()
                        clean_text = re.sub(r'\x1b\[[0-9;]*m', '', text)
                        if clean_text and log_cb:
                            await log_cb(f"[HF-CLI] {clean_text}")
                    break
                    
                buffer += chunk
                # tqdm updates frequently end in \r (or \n for new files)
                if chunk == b'\r' or chunk == b'\n':
                    text = buffer.decode(errors="replace").strip()
                    buffer = bytes()
                    
                    if text:
                        clean_text = re.sub(r'\x1b\[[0-9;]*m', '', text)
                        if clean_text and log_cb:
                            await log_cb(f"[HF-CLI] {clean_text}")

        # Gather both stderr and stdout concurrently
        if process.stdout and process.stderr:
            await asyncio.gather(
                _stream_output(process.stdout),
                _stream_output(process.stderr)
            )

        await process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"HuggingFace CLI download failed with code {process.returncode}")

        # Once downloaded successfully, getting the path is instant
        from huggingface_hub import snapshot_download
        loop = asyncio.get_running_loop()
        path = await loop.run_in_executor(None, lambda: snapshot_download(self.model_id, local_files_only=True))
        return str(path)

async def ensure_model_is_downloaded(model_id: str, log_cb: Optional[Callable[[str], Awaitable[None]]] = None) -> str:
    """Helper to ensure a model is locally available, downloading it if needed."""
    if log_cb:
        await log_cb(f"[TUI] Validating model cache for '{model_id}'...")
    
    downloader = ModelDownloader(model_id)
    path = await downloader.download(log_cb)
    
    if log_cb:
        # Avoid showing the full path (privacy/noise)
        await log_cb(f"[TUI] Pre-flight check complete. Local weights are locked and loaded.")
    
    return path
