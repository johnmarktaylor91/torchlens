"""depyf bridge helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .._errors import CaptureContextError


def dump(model: Any, x: Any, path: str | Path, **kwargs: Any) -> list[Path]:
    """Compile ``model`` under depyf and return the source files depyf dumped.

    Runs ``with depyf.prepare_debug(path): torch.compile(model)(*args)`` so
    depyf writes the decompiled Dynamo bytecode, the captured FX graphs and
    the generated kernels for this example input into ``path``. This is a
    companion bridge, not TorchLens native ``torch.compile`` support:
    TorchLens still reports ``torch.compile`` capture as SCOPE in
    ``tl.compat.report``.

    Parameters
    ----------
    model:
        Eager ``nn.Module`` (or callable) to compile and run once.
    x:
        Example input. A tuple is unpacked as positional arguments; any other
        value is passed as the single argument.
    path:
        Output directory for depyf's dump (created when missing). Required:
        depyf writes files, and this bridge never picks a location for you.
    **kwargs:
        Keyword arguments forwarded to ``depyf.prepare_debug`` (for example
        ``clean_wild_fx_code`` or ``log_bytecode``).

    Returns
    -------
    list[Path]
        Sorted paths of the files written or rewritten under ``path`` by this
        call.

    Raises
    ------
    ImportError
        If depyf is unavailable.
    RuntimeError
        If the installed depyf has no ``prepare_debug``, or the compiled run
        dumped nothing (Dynamo reused an existing compile cache entry).
    """

    try:
        import depyf as depyf_module
    except ImportError as exc:
        raise ImportError(
            "depyf bridge requires the `depyf` extra: install torchlens[depyf]."
        ) from exc
    import torch

    prepare_debug = getattr(depyf_module, "prepare_debug", None)
    if not callable(prepare_debug):
        raise CaptureContextError(
            "Installed depyf does not expose prepare_debug(dump_src_dir)",
            code="bridge_depyf_prepare_debug_missing",
            remedy='install depyf>=0.18 (pip install "torchlens[depyf]")',
        )
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    before = _file_stamps(output_dir)
    args = x if isinstance(x, tuple) else (x,)
    with prepare_debug(str(output_dir), **kwargs):
        torch.compile(model)(*args)
    after = _file_stamps(output_dir)
    written = sorted(file for file, stamp in after.items() if before.get(file) != stamp)
    if not written:
        raise CaptureContextError(
            f"depyf dumped no files into {output_dir}: torch.compile reused a cached "
            "compile of this model, so nothing was recompiled",
            code="bridge_depyf_nothing_dumped",
            remedy="call torch.compiler.reset() before dump() to force a fresh compile",
        )
    return written


def _file_stamps(directory: Path) -> dict[Path, tuple[int, int]]:
    """Return ``{file: (mtime_ns, size)}`` for every file under ``directory``."""

    return {
        file: (file.stat().st_mtime_ns, file.stat().st_size)
        for file in directory.rglob("*")
        if file.is_file()
    }


__all__ = ["dump"]
