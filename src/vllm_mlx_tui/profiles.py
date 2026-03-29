"""
User Profile persistence for persistent launch configurations.

This module handles the storage, retrieval, and deletion of named 'Profiles'.
A profile encapsulates all CLI flags and environment settings (model name, port,
quantization, etc.) required to launch a `vllm-mlx` server instance.

Storage location: ~/.cache/vllm-mlx-tui/profiles/*.json
Each profile is stored as an individual JSON file for simplified local management.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional
from datetime import datetime


# Root location for all user-defined configuration profiles
PROFILES_DIR = Path.home() / ".cache" / "vllm-mlx-tui" / "profiles"


@dataclass
class Profile:
    """Represents a saved server configuration profile.
    
    Attributes:
        name: Unique identifier for this configuration file.
        model: HuggingFace Hub ID or absolute path to the local model weights.
        port: Local networking port for the server.
        host: Local networking host (usually 127.0.0.1).
        cache_pct: MLX KV cache fraction (0.0 to 1.0).
        mllm: Enable Multi-modal LLM flags if supported.
        ngrok: Enable public tunnel via ngrok.
        no_tool_parser: Flag to explicitly disable tool-call parsing.
        tool_parser: Optional name of the tool-call parser to use.
        max_model_len: Maximum sequence length (0 = server default).
        quantization: Specific quantization override (e.g. awq).
        dtype: Data type override (e.g. float16).
        saved_at: POSIX timestamp of when this profile was last modified.
        is_default: If true, this profile loads automatically on TUI startup.
    """
    name: str
    model: str
    port: int = 8001
    host: str = "127.0.0.1"
    cache_pct: float = 0.75
    mllm: bool = False
    ngrok: bool = False
    no_tool_parser: bool = False
    tool_parser: str = ""
    max_model_len: int = 0
    quantization: str = ""
    dtype: str = ""
    saved_at: float = field(default_factory=time.time)
    is_default: bool = False

    @property
    def saved_date(self) -> str:
        """Helper to return a human-readable localized date string."""
        return datetime.fromtimestamp(self.saved_at).strftime("%Y-%m-%d %H:%M")


def _ensure_dir() -> Path:
    """Proactively creates the profiles storage directory if it doesn't exist."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    return PROFILES_DIR


def list_profiles() -> list[Profile]:
    """Retrieve all saved profiles from the filesystem, sorted by filename."""
    d = _ensure_dir()
    profiles: list[Profile] = []
    for f in sorted(d.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            profiles.append(Profile(**data))
        except Exception:
            # Skip corrupted or malformed JSON profiles
            continue
    return profiles


def save_profile(profile: Profile) -> None:
    """Persist a profile to a JSON file. Use atomic write-and-replace strategy.
    
    Includes rudimentary path-traversal sanitization for the filename.
    """
    d = _ensure_dir()
    # Basic filename sanitization
    safe_name = profile.name.replace("/", "_").replace(" ", "_").replace("..", "_")
    safe_id = Path(safe_name).name
    path = d / f"{safe_id}.json"
    
    # Atomic save: write to temp file then replace target to prevent corruption on crash
    temp_path = path.with_suffix(".tmp")
    try:
        temp_path.write_text(json.dumps(asdict(profile), indent=2))
        os.replace(temp_path, path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def delete_profile(name: str) -> None:
    """Remove a saved profile from the filesystem."""
    d = _ensure_dir()
    safe_name = name.replace("/", "_").replace(" ", "_").replace("..", "_")
    safe_id = Path(safe_name).name
    (d / f"{safe_id}.json").unlink(missing_ok=True)


def load_default_profile() -> Optional[Profile]:
    """Return the first profile marked as default, or None if no defaults exist."""
    for p in list_profiles():
        if p.is_default:
            return p
    return None


def set_default_profile(name: str) -> None:
    """Set the specified profile as default, and unset it for all others."""
    profiles = list_profiles()
    for p in profiles:
        p.is_default = (p.name == name)
        save_profile(p)
