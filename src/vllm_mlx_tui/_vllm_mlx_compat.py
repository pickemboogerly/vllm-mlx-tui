"""Compatibility fixes for known vllm-mlx releases.

This module is imported only by the server subprocess launcher. It keeps
workarounds out of the main TUI process and lets newer vllm-mlx versions run
untouched once they include the upstream fix.
"""
from __future__ import annotations

import inspect
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _has_missing_normal_return(func: Any) -> bool:
    """Detect the vllm-mlx tokenizer helper version that drops mlx_lm.load()."""
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return False

    return (
        "model, tokenizer = load(model_name, tokenizer_config=tokenizer_config)" in source
        and "return model, tokenizer" not in source
        and "def load_model_with_fallback" in source
    )


def install() -> bool:
    """Install compatibility patches.

    Returns True when a patch was applied.
    """
    try:
        from vllm_mlx.utils import tokenizer as tokenizer_mod
    except Exception:
        return False

    original = getattr(tokenizer_mod, "load_model_with_fallback", None)
    if original is None or not _has_missing_normal_return(original):
        return False

    def load_model_with_fallback(model_name: str, tokenizer_config: dict | None = None):
        """Patched copy of vllm_mlx.utils.tokenizer.load_model_with_fallback."""
        from mlx_lm import load

        tokenizer_config = tokenizer_config or {}

        if tokenizer_mod._needs_tokenizer_fallback(model_name):
            logger.info(
                "Model %s requires tokenizer fallback, loading directly...",
                model_name,
            )
            return tokenizer_mod._load_with_tokenizer_fallback(model_name)

        try:
            return load(model_name, tokenizer_config=tokenizer_config)
        except ValueError as exc:
            message = str(exc)
            if "TokenizersBackend" in message or "Tokenizer class" in message:
                logger.warning("Standard tokenizer loading failed, using fallback: %s", exc)
                return tokenizer_mod._load_with_tokenizer_fallback(model_name)
            if "parameters not in model" in message:
                logger.warning(
                    "Extra parameters found (e.g., vision tower / MTP weights), "
                    "retrying with strict=False: %s",
                    exc,
                )
                return tokenizer_mod._load_strict_false(model_name, tokenizer_config)
            raise

    tokenizer_mod.load_model_with_fallback = load_model_with_fallback
    logger.info("Applied vllm-mlx tokenizer load_model_with_fallback return-value patch.")
    return True
