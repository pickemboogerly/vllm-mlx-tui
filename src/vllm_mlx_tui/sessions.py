"""
Chat session persistence layer.

Handles the long-term storage, retrieval, and modification of chat histories.
Each session is stored as a single JSON file containing the message history,
system prompt, and session-level metadata.

Storage location: ~/.cache/vllm-mlx-tui/chats/*.json
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


# Root location for all chat session history files
CHATS_DIR = Path.home() / ".cache" / "vllm-mlx-tui" / "chats"


@dataclass
class ChatSession:
    """Represents a single conversation history and its metadata.
    
    Attributes:
        id: Unique 8-character prefix generated via UUID4.
        title: User-facing title for the session (default 'New Chat').
        model_id: The specific MLX model identifier used for this session.
        system_prompt: The system Persona/Instructions used during the chat.
        messages: A list of OpenAI-formatted message dictionaries {'role', 'content'}.
        created_at: POSIX timestamp of session creation.
        updated_at: Monotonic timestamp of the last message update.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = "New Chat"
    model_id: str = ""
    system_prompt: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def updated_date(self) -> str:
        """Helper to return a localized date string (e.g. '01/01 12:00')."""
        return datetime.fromtimestamp(self.updated_at).strftime("%m/%d %H:%M")


def _ensure_dir() -> Path:
    """Ensure the chat session directory exists on disk."""
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    return CHATS_DIR


def list_sessions() -> list[ChatSession]:
    """Retrieve all available chat sessions, sorted by last modified (newest first)."""
    d = _ensure_dir()
    sessions: list[ChatSession] = []
    # Sort files by actual modification time to ensure 'Recent' chats are at top
    for f in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text())
            sessions.append(ChatSession(**data))
        except Exception:
            # Silently skip malformed or corrupted session files
            continue
    return sessions


def save_session(session: ChatSession) -> None:
    """Persist the chat session state using an atomic write-and-replace strategy.
    
    Includes simple path-traversal prevention by ensuring the session.id is 
    treated as a direct filename in the target directory.
    """
    d = _ensure_dir()
    # Update modification time to ensure sorting stays fresh
    session.updated_at = time.time()
    
    # Path traversal prevention: treat session ID as a local filename
    safe_id = Path(session.id).name
    path = d / f"{safe_id}.json"
    
    # Atomic save: write to temp file then replace target
    temp_path = path.with_suffix(".tmp")
    try:
        temp_path.write_text(json.dumps(asdict(session), indent=2))
        os.replace(temp_path, path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def delete_session(session_id: str) -> None:
    """Permanently remove a chat session from the filesystem."""
    d = _ensure_dir()
    safe_id = Path(session_id).name
    (d / f"{safe_id}.json").unlink(missing_ok=True)


def session_to_markdown(session: ChatSession) -> str:
    """Export a chat session as formatted Markdown text for easy consumption.
    
    Includes Header metadata and identifies participants clearly.
    """
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
