"""
Model cache discovery utilities for the vllm-mlx-tui.

This module leverages the huggingface_hub API to scan local model caches
and extract relevant metadata (paths, sizes, parameter estimates). It provides
a structured view of what models are available to the TUI without requiring
additional configuration.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from huggingface_hub import scan_cache_dir


@dataclass
class CachedModel:
    """Represents a locally available MLX model snapshot.
    
    Attributes:
        repo_id: The original HuggingFace shorthand (e.g. 'mlx-community/Mistral-7B').
        snapshot_path: The absolute path to the actual model revision weights.
        display_name: The human-readable label shown in the TUI selection table.
        size_gb: The total estimated disk size of the model files.
        param_hint: A heuristic estimation of parameters (e.g. '~7B') derived from size.
    """
    repo_id: str
    snapshot_path: str
    display_name: str
    size_gb: Optional[float] = None
    param_hint: Optional[str] = None


def _estimate_params(size_bytes: int) -> Optional[str]:
    """Heuristic logic to estimate model parameters from disk size.
    
    Note: Safetensors for 4-bit MLX models typically occupy ~0.5 bytes per parameter.
    """
    gb = size_bytes / (1024**3)
    # Mapping for common 4-bit quantized MLX models
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
    """Normalize repo_id into a clean display label for UI tables."""
    if repo_id.startswith("models--"):
        parts = repo_id.removeprefix("models--").split("--", 1)
        return "/".join(parts)
    return repo_id


def discover_cached_models() -> list[CachedModel]:
    """Scan local HF hub cache and return a list of structured CachedModel objects.

    Iterates through all known model repositories in the default HF hub path,
    picking the most recently modified revision/snapshot for each.
    
    Returns:
        A list of CachedModel objects, sorted alphabetically by display_name.
    """
    results: list[CachedModel] = []
    try:
        # scan_cache_dir is provided by huggingface_hub
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_type != "model":
                continue

            # Pick the newest revision snapshot path for this repo
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
        # If cache scan fails (e.g. no models downloaded), yield an empty list
        pass

    results.sort(key=lambda m: m.display_name.lower())
    return results


def list_cached_models() -> list[str]:
    """Legacy compatibility helper to return raw repo IDs."""
    return [m.repo_id for m in discover_cached_models()]
