"""Torch-only keywords of the forward validation doors: check, collect and route them."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ._options_validation import _validate_output_device
from ._save_budget import SaveBudgetOption, resolve_save_budget
from .backends.registry import BackendUnsupportedError, resolve_backend_spec

if TYPE_CHECKING:
    from ._data_classes.trace import Trace
    from ._literals import OutputDeviceLiteral
    from .backends.registry import BackendName, BackendSpec


def resolve_forward_validation_entry(
    backend: BackendName | None,
    model: Any,
    inputs: Any,
    input_kwargs: dict[Any, Any] | None,
    output_device: OutputDeviceLiteral,
    save_budget: SaveBudgetOption,
    num_threads: int | None,
    trace_observer: Callable[[Trace | None], None] | None,
) -> tuple[BackendSpec, dict[str, Any]]:
    """Resolve the validation backend and the torch-only keywords to forward to it.

    Save options are checked with the capture entry's validators before the
    backend resolves; only non-default values are forwarded, so a default call
    reaches every backend's entry with exactly the keywords it always received.
    ``trace_observer`` is forwarded as the torch harness's ``_trace_observer``.

    Parameters
    ----------
    backend, model, inputs, input_kwargs:
        Backend selector and the model and inputs it resolves for.
    output_device, save_budget, num_threads, trace_observer:
        The door's torch-only keywords, as passed.

    Returns
    -------
    tuple[BackendSpec, dict[str, Any]]
        The resolved backend spec and the keywords to forward to its entry.

    Raises
    ------
    BackendUnsupportedError
        ``backend_unsupported`` if any of them is non-default on a non-torch backend.
    """

    _validate_output_device(output_device)
    resolve_save_budget(save_budget)
    spec = resolve_backend_spec(backend, model, inputs, input_kwargs)
    chosen: dict[str, Any] = {}
    if output_device != "same":
        chosen["output_device"] = output_device
    if save_budget != "auto":
        chosen["save_budget"] = save_budget
    if num_threads is not None:
        chosen["num_threads"] = num_threads
    if trace_observer is not None:
        chosen["trace_observer"] = trace_observer
    if chosen and spec.name != "torch":
        raise BackendUnsupportedError(
            f"Backend {spec.name!r} validation does not accept {sorted(chosen)!r}; "
            "only the torch backend's validator honors these keywords.",
            code="backend_unsupported",
            remedy=f"omit {', '.join(sorted(chosen))}, or validate on the torch backend",
            backend=spec.name,
            options=sorted(chosen),
        )
    if "trace_observer" in chosen:
        chosen["_trace_observer"] = chosen.pop("trace_observer")
    return spec, chosen


__all__ = ["resolve_forward_validation_entry"]
