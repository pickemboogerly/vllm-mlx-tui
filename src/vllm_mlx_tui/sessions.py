"""
Chat session persistence.

Chats are stored as JSON under ~/.cache/vllm-mlx-tui/chats/<id>.json.
Each file contains the full message history plus metadata.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional


CHATS_DIR = Path.home() / ".cache" / "vllm-mlx-tui" / "chats"


@dataclass
class ChatSession:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = "New Chat"
    model_id: str = ""
    system_prompt: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def updated_date(self) -> str:
        from datetime import datetime
        return datetime.fromtimestamp(self.updated_at).strftime("%m/%d %H:%M")


def _ensure_dir() -> Path:
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    return CHATS_DIR


def list_sessions() -> list[ChatSession]:
    d = _ensure_dir()
    sessions: list[ChatSession] = []
    for f in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text())
            sessions.append(ChatSession(**data))
        except Exception:
            continue
    return sessions


def save_session(session: ChatSession) -> None:
    d = _ensure_dir()
    session.updated_at = time.monotonic()
    
    # Path traversal prevention: ensure session.id is a simple filename
    safe_id = Path(session.id).name
    path = d / f"{safe_id}.json"
    
    # Atomic save: write to temp file then replace
    temp_path = path.with_suffix(".tmp")
    try:
        temp_path.write_text(json.dumps(asdict(session), indent=2))
        os.replace(temp_path, path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def delete_session(session_id: str) -> None:
    d = _ensure_dir()
    safe_id = Path(session_id).name
    (d / f"{safe_id}.json").unlink(missing_ok=True)


def session_to_markdown(session: ChatSession) -> str:
    """Export session as Markdown text."""
    lines = [f"# {session.title}", f"Model: {session.model_id}", ""]
    if session.system_prompt:
        lines += [f"**System:** {session.system_prompt}", ""]
    for msg in session.messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            lines += [f"**You:** {content}", ""]
        elif role == "assistant":
            lines += [f"**Assistant:** {content}", ""]
    return "\n".join(lines)
