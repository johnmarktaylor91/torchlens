"""depyf bridge helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def dump(model: Any, x: Any, path: str | Path | None = None, **kwargs: Any) -> Any:
    """Dump depyf source/graph context for a model and example input.

    This is a companion bridge, not TorchLens native ``torch.compile`` support.
    TorchLens still reports ``torch.compile`` capture as SCOPE in
    ``tl.compat.report``.

    Parameters
    ----------
    model:
        Model or compiled model to inspect with depyf.
    x:
        Example input passed through to depyf when the installed API accepts it.
    path:
        Optional output directory.
    **kwargs:
        Additional keyword arguments forwarded to depyf's dump function.

    Returns
    -------
    Any
        depyf return value.

    Raises
    ------
    ImportError
        If depyf is unavailable.
    RuntimeError
        If the installed depyf package does not expose a supported dump entrypoint.
    """

    try:
        import depyf as depyf_module
    except ImportError as exc:
        raise ImportError(
            "depyf bridge requires the `depyf` extra: install torchlens[depyf]."
        ) from exc

    output_dir = Path(path) if path is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for attr_name in ("dump", "decompile", "prepare_debug"):
        candidate = getattr(depyf_module, attr_name, None)
        if callable(candidate):
            try:
                if output_dir is None:
                    return candidate(model, x, **kwargs)
                return candidate(model, x, output_dir, **kwargs)
            except TypeError:
                if output_dir is None:
                    return candidate(model, **kwargs)
                return candidate(model, output_dir, **kwargs)

    raise RuntimeError("Installed depyf does not expose dump, decompile, or prepare_debug.")


__all__ = ["dump"]
