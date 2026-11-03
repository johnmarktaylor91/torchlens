"""Import-light probe of the MLX wrapper state for the protocol shell.

The shell may not import ``torchlens.backends.mlx`` at construction (the
registry validates shells while the backend packages are still importing),
so the wrapper-installed check is resolved lazily here, at call time.
"""

from __future__ import annotations


def mlx_wrappers_installed() -> bool:
    """Whether the process-global MLX capture wrappers are installed right now."""

    from ..mlx.wrappers import is_mlx_wrapped

    return is_mlx_wrapped()


__all__ = ["mlx_wrappers_installed"]
