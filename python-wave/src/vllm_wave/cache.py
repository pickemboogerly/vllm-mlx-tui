import os
from huggingface_hub import scan_cache_dir

def list_cached_models() -> list[str]:
    """Scans the local HuggingFace cache and returns a list of model IDs.
    
    Returns an empty list if the cache is empty or unavailable. The caller
    is responsible for handling the empty case (L6: no fake fallback models
    that don't exist on disk).
    """
    models = []
    try:
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            models.append(repo.repo_id)
    except Exception:
        # Offline or cache directory doesn't exist — return empty list
        pass
    return models
