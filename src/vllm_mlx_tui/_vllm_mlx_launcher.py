"""Launch vllm-mlx with local compatibility patches applied."""
from __future__ import annotations

import sys

from ._vllm_mlx_compat import install


def main() -> int:
    install()

    from vllm_mlx.cli import main as vllm_mlx_main

    return int(vllm_mlx_main() or 0)


if __name__ == "__main__":
    sys.exit(main())
