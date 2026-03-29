"""
Profile management: load/save/delete named launch configurations.

A profile captures: model path, port, all CLI options.
Stored as JSON under ~/.cache/vllm-mlx-tui/profiles/.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


PROFILES_DIR = Path.home() / ".cache" / "vllm-mlx-tui" / "profiles"


@dataclass
class Profile:
    name: str
    model: str                   # Hub id or absolute path
    port: int = 8001
    host: str = "127.0.0.1"
    cache_pct: float = 0.75
    mllm: bool = False
    ngrok: bool = False
    no_tool_parser: bool = False
    tool_parser: str = ""
    max_model_len: int = 0       # 0 = use server default
    quantization: str = ""
    dtype: str = ""
    saved_at: float = field(default_factory=time.time)
    is_default: bool = False

    @property
    def saved_date(self) -> str:
        from datetime import datetime
        return datetime.fromtimestamp(self.saved_at).strftime("%Y-%m-%d %H:%M")


def _ensure_dir() -> Path:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    return PROFILES_DIR


def list_profiles() -> list[Profile]:
    d = _ensure_dir()
    profiles: list[Profile] = []
    for f in sorted(d.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            profiles.append(Profile(**data))
        except Exception:
            continue
    return profiles


def save_profile(profile: Profile) -> None:
    d = _ensure_dir()
    # Path traversal prevention: sanitize filename and use .name
    safe_name = profile.name.replace("/", "_").replace(" ", "_").replace("..", "_")
    safe_id = Path(safe_name).name
    path = d / f"{safe_id}.json"
    
    # Atomic save: write to temp file then replace
    temp_path = path.with_suffix(".tmp")
    try:
        temp_path.write_text(json.dumps(asdict(profile), indent=2))
        os.replace(temp_path, path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def delete_profile(name: str) -> None:
    d = _ensure_dir()
    safe_name = name.replace("/", "_").replace(" ", "_").replace("..", "_")
    safe_id = Path(safe_name).name
    (d / f"{safe_id}.json").unlink(missing_ok=True)


def load_default_profile() -> Optional[Profile]:
    for p in list_profiles():
        if p.is_default:
            return p
    return None


def set_default_profile(name: str) -> None:
    profiles = list_profiles()
    for p in profiles:
        p.is_default = (p.name == name)
        save_profile(p)
