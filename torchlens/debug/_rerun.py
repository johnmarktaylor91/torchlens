"""Isolated reruns: run the same forward more than once under controlled state.

``tl.debug.isolated_capture`` is the harness behind ``tl.debug.bisect_precision`` and
``tl.debug.check_determinism``. Every run executes on a fresh deep copy of the model,
released from any TorchLens preparation it inherited first (a no-op on a never-traced
copy; a capture leaves no forward wrapper behind, so the release only normalizes plain
attributes holding torch functions), with inputs cloned by ``tl.debug.clone_input_tree``.
``tl.debug.preserved_rng_state`` saves and restores the caller's RNG streams (Python,
NumPy, torch CPU and initialized CUDA devices) around every run, so a diagnostic never
perturbs the experiment that called it. All three spellings are DOCUMENTED-UNSTABLE.
"""

from __future__ import annotations

import contextlib
import copy
import random
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torchlens._data_classes.trace import Trace

__all__ = ["clone_input_tree", "isolated_capture", "preserved_rng_state"]


def _initialized_cuda_devices() -> list[int]:
    """Return CUDA device ordinals safe to touch for RNG bookkeeping.

    Mirrors the ``bisect_precision`` gate: reading RNG state for devices this
    process never initialized would allocate their CUDA contexts for nothing.

    Returns
    -------
    list[int]
        Device ordinals with a live CUDA context, or an empty list.
    """

    from .._utils.tensor_utils import _is_cuda_initialized

    if _is_cuda_initialized() and torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


@contextlib.contextmanager
def preserved_rng_state() -> Iterator[None]:
    """Preserve and restore every host RNG a diagnostic rerun may consume.

    Covers the Python ``random`` module, NumPy's legacy global generator when
    NumPy is importable, the torch CPU generator, and every initialized CUDA
    device generator. The states are restored when the block exits, normally or
    by an exception, so the caller's RNG streams observe no consumption from
    anything executed inside it.

    Examples
    --------
    >>> import torch
    >>> import torchlens as tl
    >>> before = torch.get_rng_state()
    >>> with tl.debug.preserved_rng_state():
    ...     _ = torch.randn(4)
    >>> torch.equal(torch.get_rng_state(), before)
    True

    Yields
    ------
    None
        Control returns with all captured states restored.
    """

    python_state = random.getstate()
    numpy_state: Any = None
    numpy_module: Any = None
    try:
        import numpy

        numpy_module = numpy
        numpy_state = numpy.random.get_state()
    except ImportError:
        pass
    torch_cpu_state = torch.get_rng_state()
    cuda_devices = _initialized_cuda_devices()
    cuda_states = [torch.cuda.get_rng_state(device) for device in cuda_devices]
    try:
        yield
    finally:
        random.setstate(python_state)
        if numpy_module is not None:
            numpy_module.random.set_state(numpy_state)
        torch.set_rng_state(torch_cpu_state)
        for device, state in zip(cuda_devices, cuda_states, strict=True):
            torch.cuda.set_rng_state(state, device)


def clone_input_tree(value: Any) -> Any:
    """Return a detached clone of an input tree for one isolated run.

    Tensors are detached and cloned (keeping ``requires_grad``) so in-place
    mutation by the model during one run cannot leak into another run or back to
    the caller. Dicts, lists and tuples are rebuilt recursively as new
    containers; any other leaf passes through as the same object.

    Examples
    --------
    >>> import torch
    >>> import torchlens as tl
    >>> inputs = {"x": torch.zeros(2), "scale": 2.0}
    >>> copied = tl.debug.clone_input_tree(inputs)
    >>> _ = copied["x"].add_(1.0)
    >>> inputs["x"].tolist()
    [0.0, 0.0]

    Parameters
    ----------
    value:
        Input value, container tree, or non-tensor leaf.

    Returns
    -------
    Any
        Cloned tree safe to hand to one isolated forward.
    """

    if isinstance(value, torch.Tensor):
        return value.detach().clone().requires_grad_(value.requires_grad)
    if isinstance(value, dict):
        return {key: clone_input_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(clone_input_tree(item) for item in value)
    if isinstance(value, list):
        return [clone_input_tree(item) for item in value]
    return value


def isolated_capture(
    model: Any,
    input_args: Any,
    input_kwargs: dict[Any, Any] | None = None,
    *,
    seed: int,
    prepare: Callable[[Any], Any] | None = None,
    **trace_kwargs: Any,
) -> Trace:
    """Run one fully isolated seeded capture of a fresh model copy.

    The source model is deep-copied, the COPY is released from any persistent
    TorchLens preparation it inherited (the deepcopy-after-trace ``KeyError``
    repair), inputs are cloned, and the capture runs under preserved RNG state
    with ``random`` / NumPy / torch seeded to ``seed``. The caller's model,
    inputs, and RNG streams are untouched.

    Parameters
    ----------
    model:
        Source model. Never mutated; each call copies it afresh.
    input_args:
        Forward input value or positional-argument list/tuple, exactly as
        accepted by ``tl.trace``.
    input_kwargs:
        Optional forward keyword arguments.
    seed:
        Seed applied to Python, NumPy, and torch generators inside the
        preserved-RNG scope before the forward runs. It is also routed to
        ``CaptureOptions(seed=...)`` (unless the caller's ``capture``
        options carry an explicit ``random_seed``), because unseeded captures
        pick a FRESH private seed per capture by design (R57) and would
        defeat same-seed reproduction.
    prepare:
        Optional callable applied to the fresh copy AFTER release and BEFORE
        tracing (e.g. ``lambda m: m.to(torch.float64)`` for a reference-dtype
        run). May return the model or ``None`` for in-place preparation.
    **trace_kwargs:
        Additional keyword arguments passed through to ``tl.trace``.

    Returns
    -------
    Trace
        The completed capture of the isolated copy. The caller owns cleanup.

    Raises
    ------
    Exception
        Whatever the underlying ``tl.trace`` raises; the RNG restore still
        runs.
    """

    from .._user_funcs import release_model, trace
    from ..options import CaptureOptions

    run_model = copy.deepcopy(model)
    # Releasing the copy starts its preparation from scratch (a no-op on
    # never-traced models; captures leave no forward wrapper to strip).
    release_model(run_model)
    if prepare is not None:
        prepared = prepare(run_model)
        if prepared is not None:
            run_model = prepared
    cloned_args = clone_input_tree(input_args)
    cloned_kwargs = clone_input_tree(input_kwargs) if input_kwargs is not None else None
    run_trace_kwargs = dict(trace_kwargs)
    capture_options = run_trace_kwargs.get("capture")
    if capture_options is None:
        run_trace_kwargs["capture"] = CaptureOptions(seed=seed)
    elif not capture_options.is_field_explicit("seed"):
        explicit_values = {
            field_name: value
            for field_name, value in capture_options.as_dict().items()
            if capture_options.is_field_explicit(field_name)
        }
        explicit_values["seed"] = seed
        run_trace_kwargs["capture"] = CaptureOptions(**explicit_values)
    with preserved_rng_state():
        random.seed(seed)
        try:
            import numpy

            numpy.random.seed(seed % (2**32))
        except ImportError:
            pass
        torch.manual_seed(seed)
        return trace(run_model, cloned_args, cloned_kwargs, **run_trace_kwargs)
