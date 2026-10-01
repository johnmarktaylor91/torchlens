"""Human-readable formatting, print overrides, and environment detection.

Formatting helpers used by the display/print paths of Trace and Layer,
plus environment checks (Jupyter detection, parallel-processing guard).
"""

import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
import weakref
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import torch

from ..quantities import Bytes, Flops

if TYPE_CHECKING:
    from types import FrameType

    from ..data_classes.trace import Trace

_T = TypeVar("_T")

# Package directory, resolved once at import: `user_stacklevel` compares frame
# filenames against it to find the first non-torchlens frame. Matches the
# already-established approach in utils/introspection.py's stack filter.
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> None:
    """Atomically replace a text file with fully written content.

    Parameters
    ----------
    path:
        Destination path.
    text:
        Complete text payload.
    encoding:
        Text encoding used for the temporary file.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.tmp.",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def ensure_trace_visualizer_dir(trace: Any) -> Path:
    """Return a trace-owned visualizer scratch directory.

    Parameters
    ----------
    trace:
        Trace-like owner receiving the scratch path.

    Returns
    -------
    pathlib.Path
        Existing or newly created scratch directory.
    """

    current = getattr(trace, "_visualizer_dir", None)
    if current is not None and Path(current).is_dir():
        return Path(current)
    output_dir = Path(tempfile.mkdtemp(prefix="torchlens_visualizers_"))
    trace._visualizer_dir = str(output_dir)
    weakref.finalize(trace, shutil.rmtree, output_dir, ignore_errors=True)
    return output_dir


def cleanup_trace_visualizer_dir(trace: Any) -> None:
    """Remove a trace-owned visualizer scratch directory if one exists.

    Parameters
    ----------
    trace:
        Trace-like owner whose scratch directory should be removed.
    """

    output_dir = getattr(trace, "_visualizer_dir", None)
    if output_dir is not None:
        shutil.rmtree(output_dir, ignore_errors=True)
        trace._visualizer_dir = None


def _record_phase_timing(trace: "Trace", bucket: str, elapsed_s: float) -> None:
    """Accumulate one phase-timing sample on a trace.

    Parameters
    ----------
    trace:
        Trace receiving timing data.
    bucket:
        Stable timing bucket name.
    elapsed_s:
        Elapsed wall-clock seconds.
    """

    timings = trace.__dict__.setdefault("_phase_timings", {})
    bucket_stats = timings.setdefault(bucket, {"total_s": 0.0, "count": 0})
    bucket_stats["total_s"] += elapsed_s
    bucket_stats["count"] += 1


@contextmanager
def _timed_phase(trace: "Trace", bucket: str) -> Iterator[None]:
    """Record elapsed time for a capture or postprocess phase.

    Parameters
    ----------
    trace:
        Trace receiving timing data.
    bucket:
        Stable timing bucket name.

    Yields
    ------
    None
        Context body.
    """

    start = time.perf_counter()
    try:
        yield
    finally:
        _record_phase_timing(trace, bucket, time.perf_counter() - start)


def identity(x: Any) -> Any:
    """Return the input unchanged.

    Used as a no-op placeholder where a callable is expected (e.g.
    ``activation_transform`` when no user postprocessing is desired).
    """
    return x


def int_list_to_compact_str(int_list: list[int]) -> str:
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
    if size < 0:
        raise ValueError("size must be non-negative.")
    if decimal_places == 1:
        return str(Bytes(size))
    return format(Bytes(size), f".{decimal_places}f")


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

    return human_readable_size(size, decimal_places=decimal_places)


def format_flops(
    flops: float,
    decimal_places: int = 1,
) -> str:
    """Format a FLOP count using SI units.

    A pure formatter: it never converts between FMA conventions (the
    ``count_fma_as_two`` kwarg it once accepted-and-discarded is REMOVED --
    convention selection lives on the counting surfaces such as
    ``Trace.summary``/``flop_count``, which honor it or refuse typed).

    Parameters
    ----------
    flops:
        Number of floating-point operations.
    decimal_places:
        Number of decimal places for kilo-FLOPs and larger units.

    Returns
    -------
    str
        Human-readable FLOP string such as ``"3.4 GFLOPs"``.
    """

    if flops < 0:
        raise ValueError("flops must be non-negative.")
    if decimal_places == 1:
        return str(Flops(flops))
    return format(Flops(flops), f".{decimal_places}f")


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


def _non_torch_array_summary(array: Any) -> str:
    """Return a backend-generic one-line summary for a non-``torch.Tensor`` array.

    Used for saved activations captured under a preview (non-torch) backend
    -- e.g. MLX, tinygrad, TensorFlow, JAX, Paddle -- whose array types do not
    share ``torch.Tensor``'s ``.device``/``grad_fn``/dtype-cast surface. Only
    duck-typed, near-universal attributes (``.shape``, ``.dtype``) are read;
    no torch-only numeric stats (mean/std/min/max/nan/inf/zero counts) are
    attempted, since those rely on torch-specific reductions that do not
    exist -- or behave differently -- across array libraries.

    Parameters
    ----------
    array:
        Non-torch array-like object to summarize.

    Returns
    -------
    str
        One-line summary with whatever shape/dtype information the object
        exposes. Never raises.
    """

    # Reject callables (e.g. a ``.shape``/``.dtype`` *method* rather than a
    # value attribute): reading them unguarded stringifies a bound-method repr
    # as if it were a shape/dtype. Treat a callable -- or an absent attribute --
    # as "unknown" so this helper never surfaces garbage and never raises.
    shape = getattr(array, "shape", None)
    if shape is not None and not callable(shape):
        try:
            shape_text = "Tensor[" + ", ".join(str(dim) for dim in tuple(shape)) + "]"
        except TypeError:
            shape_text = f"Tensor[{shape}]"
    else:
        shape_text = "Tensor[?]"
    dtype = getattr(array, "dtype", None)
    dtype_text = str(dtype) if dtype is not None and not callable(dtype) else "unknown dtype"
    return f"{shape_text} {dtype_text}"


def tensor_stats_summary(tensor: Any) -> str:
    """Return a lovely-style one-line tensor statistics summary.

    Parameters
    ----------
    tensor:
        Tensor to summarize. Non-``torch.Tensor`` arrays (saved activations
        from a preview backend such as MLX/tinygrad/TF/JAX/Paddle) are
        handled defensively via :func:`_non_torch_array_summary`: shape and
        dtype are reported when available and torch-only numeric stats are
        omitted, rather than raising ``AttributeError``. The ``torch.Tensor``
        path below is unchanged.

    Returns
    -------
    str
        One-line summary with shape, dtype, device, scalar stats, and
        NaN/Inf warning flags when present (torch tensors), or a reduced
        shape/dtype-only summary (non-torch arrays).
    """

    if not isinstance(tensor, torch.Tensor):
        return _non_torch_array_summary(tensor)

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
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is None or "IPKernelApp" not in ipython.config:
            return False
    except (ImportError, AttributeError):
        return False
    return True


def _vprint(trace: "Trace", message: str) -> None:
    """Print a progress message if verbose mode is enabled on the Trace."""
    if getattr(trace, "verbose", False):
        print(f"[torchlens] {message}")


@contextmanager
def _vtimed(trace: "Trace", description: str) -> Iterator[None]:
    """Context manager that prints a timed progress message if verbose mode is enabled.

    Prints ``[torchlens] description...`` on entry, then appends `` done (X.XXs)``
    on exit.
    """
    bucket = f"postprocess:{description.strip()}"
    if not getattr(trace, "verbose", False):
        with _timed_phase(trace, bucket):
            yield
        return
    print(f"[torchlens] {description}...", end="", flush=True)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        _record_phase_timing(trace, bucket, elapsed)
        print(f" done ({elapsed:.2f}s)")


def user_stacklevel(extra: int = 0) -> int:
    """Return the ``warnings.warn`` stacklevel that blames the caller's caller.

    A warning about the user's model or arguments should point at the user's own
    line, not at whichever torchlens internal happened to notice. A fixed
    ``stacklevel=`` cannot do that from deep in the pipeline: the capture path is
    ~10 frames below ``tl.trace`` and the depth is not even constant (the rescue
    re-run adds a frame), so a hardcoded number is wrong about as often as it is
    right.

    This walks outward from the caller and returns the depth of the first frame
    outside the torchlens package, which is what ``stacklevel`` wants. Falls back
    to ``2`` (the caller's caller) when every frame is internal -- e.g. under a
    test that drives postprocess directly.

    Parameters
    ----------
    extra:
        Frames between the caller of this function and the ``warnings.warn``
        call, for helpers that warn on someone else's behalf.

    Returns
    -------
    int
        Stacklevel to pass to ``warnings.warn``.
    """

    frame: FrameType | None = sys._getframe(1)
    depth = 1
    while frame is not None:
        if not frame.f_code.co_filename.startswith(_PACKAGE_ROOT):
            return depth + extra
        frame = frame.f_back
        depth += 1
    return 2 + extra


#: PID of the interpreter that imported this module (r-b6 R40-3b). A raw
#: ``os.fork()`` child inherits the stamp with a different ``getpid()``, which
#: is how ``warn_parallel`` sees through multiprocessing-invisible forks.
_WARN_PARALLEL_IMPORT_PID: int = os.getpid()

#: PID that first observed an initialized process group at capture entry.
#: Inheriting another process's stamp marks a fork child of a rank, never a
#: rank. Dict-in-a-slot so the fork-inherited copy stays readable.
_DIST_GROUP_OBSERVED_PID: dict[str, int] = {}


def warn_parallel() -> None:
    """Refuse capture from a non-rank CHILD PROCESS.

    TorchLens capture is single-owner by design — its global toggle state and
    ordered tensor counter are not safe for concurrent access. This guard is
    called early in ``trace`` to fail fast rather than produce silently
    corrupted logs.

    Scope, exactly (the name is historical and broader than the check):

    * Refused HERE: capture in a child process, i.e. any process whose name is
      not ``MainProcess`` and which is not a distributed rank.
    * NOT refused here, and covered elsewhere: THREAD concurrency. A second
      capture entering while one is active is refused atomically by
      ``_state.active_logging`` (``ReentrantTraceError``); a non-owner thread's
      ``pause_logging()`` is a no-op that cannot blind the owner's capture; a
      non-owner thread's torch ops are skipped by the wrapper's owner-thread
      fast path and disclosed by the thread-count witness. ``nn.DataParallel``
      (which parallelizes over threads, not processes) is unwrapped to its
      ``.module`` at capture entry and traced single-threaded, so it never
      reaches a concurrent-capture path at all.

    A DISTRIBUTED RANK process is the deliberate exception: each rank is its
    own interpreter with its own torchlens state, capturing its own rank-local
    forward (merge-ranks tier (b)). ``torchrun`` ranks are independent
    ``MainProcess`` es and never hit this guard; ``multiprocessing.spawn``
    ranks are recognized by having an initialized process group in a
    non-daemonic process. Daemonic children (DataLoader workers) stay refused
    even when a forked flag claims an initialized group.

    Raises
    ------
    CaptureContextError
        If called from a child process that is not a distributed rank
        (code ``child_process_capture_unsupported``).
    """
    # r-b6 R40-3b: child detection does NOT trust ``process.name`` — it is a
    # user-assignable constructor kwarg (``mp.Process(name="MainProcess")``),
    # and a raw ``os.fork()`` child is invisible to multiprocessing entirely
    # (it inherits the parent's process object wholesale). ``parent_process()``
    # catches every multiprocessing child regardless of name; the import-time
    # PID stamp catches raw-forked children, which inherit the module state
    # (and the wrapped-torch toggle state that makes their captures unsafe).
    process = mp.current_process()
    if mp.parent_process() is None and os.getpid() == _WARN_PARALLEL_IMPORT_PID:
        # r5 b6-fable R40 (4th round): stamp rank ownership on THIS return
        # too. The early-return skipped the stamp, so a MAIN-process rank
        # that captured first never claimed it -- a raw ``os.fork()`` child
        # then setdefault'ed its OWN pid below and was accepted as a rank.
        try:
            import torch.distributed as dist

            if not process.daemon and dist.is_available() and dist.is_initialized():
                _DIST_GROUP_OBSERVED_PID.setdefault("pid", os.getpid())
        except Exception:
            pass
        return
    try:
        import torch.distributed as dist

        is_rank_process = not process.daemon and dist.is_available() and dist.is_initialized()
        if is_rank_process:
            # A FORKED child inherits the parent's initialized-group flag, so
            # "initialized" alone does not prove this process is a rank. The
            # first process to reach this check with an initialized group
            # stamps its PID; an inheritor of somebody else's stamp is a fork
            # child of a rank, not a rank (r-b6 R40-3b). A rank whose parent
            # never captured is stamped here on ITS first capture — the stamp
            # is per-interpreter state, reset by spawn's fresh import.
            owner_pid = _DIST_GROUP_OBSERVED_PID.setdefault("pid", os.getpid())
            if owner_pid != os.getpid() and os.getpid() == _WARN_PARALLEL_IMPORT_PID:
                # R40 steal closure: first-observer stamping must not be
                # first-FORK-CHILD stamping. A rank that raw-forks BEFORE its
                # first capture would otherwise lose the stamp to the child
                # (child accepted as "rank", the REAL rank then refused). The
                # interpreter's original importer can never be a fork child —
                # a fork child inherits the parent's import-PID value, which
                # differs from its own pid — so the import-PID process
                # reclaims the stamp unconditionally. Non-importer processes
                # (raw-fork children) still cannot displace an existing stamp.
                _DIST_GROUP_OBSERVED_PID["pid"] = os.getpid()
                owner_pid = os.getpid()
            is_rank_process = owner_pid == os.getpid()
    except Exception:
        is_rank_process = False
    if not is_rank_process:
        from .._errors import CaptureContextError

        raise CaptureContextError(
            "TorchLens capture was started in child process "
            f"{process.name!r}, which is not a distributed rank; capture state "
            "is per-interpreter and ordered, so a child-process capture "
            "produces a silently corrupted Trace",
            code="child_process_capture_unsupported",
            remedy=(
                "run the capture in the main process (a distributed rank with an "
                "initialized, non-daemonic process group is the one exception)"
            ),
            process_name=process.name,
            process_daemon=bool(process.daemon),
        )
