"""Human-readable formatting, print overrides, and environment detection.

Formatting helpers used by the display/print paths of ModelLog and LayerLog,
plus environment checks (Jupyter detection, parallel-processing guard).
"""

import multiprocessing as mp
import sys
import time
from collections.abc import Iterable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator, List, TypeVar, cast

import torch

if TYPE_CHECKING:
    from ..data_classes.model_log import ModelLog

_T = TypeVar("_T")


def identity(x: Any) -> Any:
    """Return the input unchanged.

    Used as a no-op placeholder where a callable is expected (e.g.
    ``activation_postfunc`` when no user postprocessing is desired).
    """
    return x


def int_list_to_compact_str(int_list: List[int]) -> str:
    """Collapse a list of integers into a compact range string.

    Contiguous runs are collapsed into ``"start-end"`` ranges, separated
    by commas.  Example: ``[1, 2, 3, 7, 8, 10]`` becomes ``"1-3,7-8,10"``.

    Args:
        int_list: List of integers (need not be sorted).

    Returns:
        Compact string representation.
    """
    int_list = sorted(int_list)
    if len(int_list) == 0:
        return ""
    if len(int_list) == 1:
        return str(int_list[0])
    ranges = []
    start = int_list[0]
    end = int_list[0]
    for i in range(1, len(int_list)):
        if int_list[i] == end + 1:
            # Extend the current contiguous run.
            end = int_list[i]
        else:
            # Current run ended — flush it and start a new one.
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = int_list[i]
            end = int_list[i]
    # Flush the final run.
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")
    return ",".join(ranges)


def human_readable_size(size: float, decimal_places: int = 1) -> str:
    """Convert a byte count into a human-readable string (e.g. ``"1.5 MB"``).

    Args:
        size: Number of bytes.
        decimal_places: Number of decimal places for non-byte units.

    Returns:
        String with human-readable size and unit suffix.
    """
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if size < 1024.0 or unit == "PB":
            break
        size /= 1024.0
    if unit == "B":
        size = int(size)  # No fractional bytes.
    else:
        size = round(size, decimal_places)
    return f"{size} {unit}"


def format_size(size: float, decimal_places: int = 1) -> str:
    """Format a byte count using binary units.

    Parameters
    ----------
    size:
        Number of bytes.
    decimal_places:
        Number of decimal places for KB and larger units.

    Returns
    -------
    str
        Human-readable byte string such as ``"1.2 MB"``.
    """

    if size < 0:
        raise ValueError("size must be non-negative.")
    return human_readable_size(size, decimal_places=decimal_places)


def format_flops(
    flops: float,
    decimal_places: int = 1,
    *,
    count_fma_as_two: bool = False,
) -> str:
    """Format a FLOP count using SI units.

    Parameters
    ----------
    flops:
        Number of floating-point operations.
    decimal_places:
        Number of decimal places for kilo-FLOPs and larger units.
    count_fma_as_two:
        Convention marker for callers that expose FLOP/MAC counting choices.
        The value does not rescale ``flops``; it records which convention the
        provided count already follows.

    Returns
    -------
    str
        Human-readable FLOP string such as ``"3.4 GFLOPs"``.
    """

    del count_fma_as_two
    if flops < 0:
        raise ValueError("flops must be non-negative.")
    value = float(flops)
    units = ["FLOPs", "KFLOPs", "MFLOPs", "GFLOPs", "TFLOPs", "PFLOPs"]
    unit = units[-1]
    for unit in units:
        if value < 1000.0 or unit == units[-1]:
            break
        value /= 1000.0
    if unit == "FLOPs":
        return f"{int(value)} {unit}"
    return f"{round(value, decimal_places)} {unit}"


def _format_number(value: float | None) -> str:
    """Format a compact tensor statistic value.

    Parameters
    ----------
    value:
        Numeric value or ``None`` for unavailable statistics.

    Returns
    -------
    str
        Compact display string.
    """

    if value is None:
        return "n/a"
    return f"{value:.4g}"


def _format_percent(value: float) -> str:
    """Format a percentage compactly.

    Parameters
    ----------
    value:
        Percent value in the range 0-100.

    Returns
    -------
    str
        Percentage string.
    """

    if value == 0:
        return "0%"
    if value < 0.1:
        return f"{value:.3g}%"
    if value < 10:
        return f"{value:.2g}%"
    return f"{value:.3g}%"


def tensor_stats_summary(tensor: torch.Tensor) -> str:
    """Return a lovely-style one-line tensor statistics summary.

    Parameters
    ----------
    tensor:
        Tensor to summarize.

    Returns
    -------
    str
        One-line summary with shape, dtype, device, scalar stats, and
        NaN/Inf warning flags when present.
    """

    shape = ", ".join(str(dim) for dim in tuple(tensor.shape))
    shape_text = f"Tensor[{shape}]" if shape else "Tensor[]"
    dtype_text = str(tensor.dtype).replace("torch.", "")
    prefix = f"{shape_text} {dtype_text} {tensor.device}"
    if tensor.numel() == 0:
        return f"{prefix} empty"

    try:
        work = tensor.detach()
        if work.is_complex():
            stat_tensor = work.abs().to(torch.float64)
            negative_percent = 0.0
        else:
            stat_tensor = work.to(torch.float64)
            negative_percent = float((stat_tensor < 0).sum().item()) / work.numel() * 100
        nan_percent = float(torch.isnan(stat_tensor).sum().item()) / work.numel() * 100
        inf_percent = float(torch.isinf(stat_tensor).sum().item()) / work.numel() * 100
        zero_percent = float((stat_tensor == 0).sum().item()) / work.numel() * 100
        finite = stat_tensor[torch.isfinite(stat_tensor)]
        if finite.numel() == 0:
            mean_value = std_value = min_value = max_value = None
        else:
            mean_value = float(finite.mean().item())
            std_value = float(finite.std(unbiased=False).item())
            min_value = float(finite.min().item())
            max_value = float(finite.max().item())
    except (RuntimeError, TypeError, ValueError):
        return prefix

    summary = (
        f"{prefix} mean={_format_number(mean_value)} std={_format_number(std_value)} "
        f"min={_format_number(min_value)} max={_format_number(max_value)} "
        f"nan={_format_percent(nan_percent)} inf={_format_percent(inf_percent)} "
        f"neg={_format_percent(negative_percent)} zero={_format_percent(zero_percent)}"
    )
    if nan_percent > 0:
        summary += f" [⚠ {_format_percent(nan_percent)} NaN]"
    if inf_percent > 0:
        summary += f" [⚠ {_format_percent(inf_percent)} Inf]"
    return summary


def progress_bar(
    iterable: Iterable[_T],
    *,
    total: int | None,
    desc: str,
    enabled: bool = True,
    threshold: int = 10,
) -> Iterable[_T]:
    """Wrap an iterable with an environment-appropriate progress bar.

    Parameters
    ----------
    iterable:
        Iterable to wrap.
    total:
        Total iteration count when known.
    desc:
        Progress-bar label.
    enabled:
        Whether the caller requested progress reporting.
    threshold:
        Minimum total count required before a progress bar is shown.

    Returns
    -------
    Iterable[_T]
        Original iterable or a tqdm-wrapped iterable.
    """

    if not enabled or total is None or total <= threshold:
        return iterable
    try:
        if in_notebook():
            from tqdm.notebook import tqdm

            return cast(Iterable[_T], tqdm(iterable, total=total, desc=desc))
        from tqdm import tqdm

        return cast(
            Iterable[_T],
            tqdm(iterable, total=total, desc=desc, disable=not sys.stderr.isatty()),
        )
    except ImportError:
        return iterable


def in_notebook() -> bool:
    """Return True if running inside a Jupyter notebook kernel.

    Checks for the IPython kernel app in the running IPython instance's
    config.  Returns False in plain Python, IPython terminal, or when
    IPython is not installed.
    """
    try:
        from IPython import get_ipython  # type: ignore[attr-defined]

        ipython = get_ipython()  # type: ignore[no-untyped-call]
        if ipython is None or "IPKernelApp" not in ipython.config:
            return False
    except (ImportError, AttributeError):
        return False
    return True


def _vprint(model_log: "ModelLog", message: str) -> None:
    """Print a progress message if verbose mode is enabled on the ModelLog."""
    if getattr(model_log, "verbose", False):
        print(f"[torchlens] {message}")


@contextmanager
def _vtimed(model_log: "ModelLog", description: str) -> Iterator[None]:
    """Context manager that prints a timed progress message if verbose mode is enabled.

    Prints ``[torchlens] description...`` on entry, then appends `` done (X.XXs)``
    on exit.
    """
    if not getattr(model_log, "verbose", False):
        yield
        return
    print(f"[torchlens] {description}...", end="", flush=True)
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f" done ({elapsed:.2f}s)")


def warn_parallel() -> None:
    """Raise ``RuntimeError`` if called from a child process.

    TorchLens is single-threaded by design — its global toggle state and
    ordered tensor counter are not safe for concurrent access.  This guard
    is called early in ``log_forward_pass`` to fail fast rather than
    produce silently corrupted logs.
    """
    if mp.current_process().name != "MainProcess":
        raise RuntimeError(
            "WARNING: It looks like you are using parallel execution; only run "
            "torchlens in the main process, since certain operations "
            "depend on execution order."
        )
