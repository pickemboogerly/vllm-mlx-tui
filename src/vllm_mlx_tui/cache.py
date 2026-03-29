"""
Model cache discovery utilities.

Scans the local HuggingFace Hub cache and returns structured model info
including estimated parameter size via safetensors disk-size heuristics.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from huggingface_hub import scan_cache_dir


@dataclass
class CachedModel:
    repo_id: str          # e.g. "mlx-community/Mistral-7B-v0.1-4bit"
    snapshot_path: str    # Absolute path to the snapshot dir used by vllm-mlx
    display_name: str     # Human-readable label shown in UI
    size_gb: Optional[float] = None   # Estimated disk size
    param_hint: Optional[str] = None  # e.g. "~7B" derived from disk size


def _estimate_params(size_bytes: int) -> Optional[str]:
    """Heuristic: safetensors for 4-bit models ≈ 0.5 bytes/param.
    Return a human label like '~7B', '~13B', etc."""
    gb = size_bytes / (1024**3)
    # Rough mapping for 4-bit quantized MLX models
    thresholds = [
        (0.4,  "~1B"),
        (1.2,  "~3B"),
        (3.0,  "~7B"),
        (6.5,  "~13B"),
        (20.0, "~30B"),
        (35.0, "~65B"),
    ]
    for threshold, label in thresholds:
        if gb <= threshold:
            return label
    return f"~{gb:.0f}GB"


def _make_display_name(repo_id: str) -> str:
    """Convert HF repo_id to a clean display label.

    'mlx-community/Mistral-7B-v0.1-4bit' → 'mlx-community/Mistral-7B-v0.1-4bit'
    (already pretty). For cache paths like 'models--mlx-community--Mistral-7B'
    we reconstruct the org/name form.
    """
    if repo_id.startswith("models--"):
        parts = repo_id.removeprefix("models--").split("--", 1)
        return "/".join(parts)
    return repo_id


def discover_cached_models() -> list[CachedModel]:
    """Scan local HF cache and return structured model info, sorted by name.

    Returns an empty list if cache is inaccessible. Caller must handle that.
    """
    results: list[CachedModel] = []
    try:
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_type != "model":
                continue

            # Pick the newest revision snapshot path
            snapshot_path = ""
            best_ts = -1
            for rev in repo.revisions:
                ts = rev.last_modified
                if ts > best_ts:
                    best_ts = ts
                    snapshot_path = str(rev.snapshot_path)

            if not snapshot_path:
                continue

            size_bytes = repo.size_on_disk
            size_gb = size_bytes / (1024**3)
            param_hint = _estimate_params(size_bytes)

            results.append(CachedModel(
                repo_id=repo.repo_id,
                snapshot_path=snapshot_path,
                display_name=_make_display_name(repo.repo_id),
                size_gb=size_gb,
                param_hint=param_hint,
            ))
    except Exception:
        pass

    results.sort(key=lambda m: m.display_name.lower())
    return results


# Legacy compat: list_cached_models() used by old launcher
def list_cached_models() -> list[str]:
    return [m.repo_id for m in discover_cached_models()]
