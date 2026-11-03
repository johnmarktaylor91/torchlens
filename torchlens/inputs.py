"""Input preprocessors: teach ``tl.trace`` to accept your model's raw input.

Register a function that turns a raw input (a string, a file path, a record)
into what your model's ``forward`` takes. Every later ``tl.trace(model, raw)``
call whose model and input the function accepts runs it first, keeps the raw
value as ``trace.input`` and records the preprocessor in
``trace.provenance.input_preprocessor``. Registration affects future traces
only; a trace already made is never changed.

Examples
--------
>>> import torch
>>> import torchlens as tl
>>> class CharModel(torch.nn.Module):
...     def forward(self, codes):
...         return codes.float() * 2
>>> @tl.inputs.register(name="char_codes", class_name="CharModel")
... def char_codes(model, raw):
...     if not isinstance(raw, str):
...         return None
...     return torch.tensor([[ord(ch) for ch in raw]])
>>> trace = tl.trace(CharModel(), "hi")
>>> trace.input
'hi'
>>> tl.inputs.unregister("char_codes")
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torchlens._autoroute import input as _input_registry
from torchlens._autoroute._registry import class_names_tuple, matches_class_name
from torchlens._data_classes.trace import ResolvedPreprocessing
from torchlens._user_funcs import trace as _trace
from torchlens.options import merge_capture_options

#: Provenance ``source`` stamped on traces whose input a registered preprocessor built.
_PREPROCESSOR_SOURCE = "registered_input_preprocessor"


def register(
    *,
    name: str,
    priority: int = 100,
    class_name: str | tuple[str, ...] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return a decorator that registers an input preprocessor for future traces.

    The decorated function is called as ``fn(model, raw)`` and returns the
    model input built from ``raw``: a tensor (or a tuple or list of positional
    arguments), or a mapping of keyword arguments. Returning ``None`` declines,
    and the next preprocessor (or the plain ``tl.trace`` path) handles the call.
    Preprocessors run in ascending ``priority`` order; built-in ones (Hugging Face
    text, image and multimodal input) sit at priorities 10 to 30.

    Parameters
    ----------
    name:
        Unique preprocessor name, used by ``unregister`` and stamped in the
        trace's input provenance.
    priority:
        Sort key. Lower values run first.
    class_name:
        Optional model class name, a glob pattern, or a tuple of them; the
        preprocessor runs only for models whose class name matches.

    Returns
    -------
    Callable[[Callable[..., Any]], Callable[..., Any]]
        Decorator that registers and returns the original function.

    Raises
    ------
    ValueError
        If a preprocessor or built-in input detector named ``name`` is
        already registered.

    Examples
    --------
    >>> import torch
    >>> import torchlens as tl
    >>> @tl.inputs.register(name="doc_numbers", class_name="Linear")
    ... def doc_numbers(model, raw):
    ...     return torch.tensor([raw], dtype=torch.float32) if isinstance(raw, list) else None
    >>> trace = tl.trace(torch.nn.Linear(2, 1), [1.0, 2.0])
    >>> trace.provenance.input_preprocessor.identifier
    'doc_numbers'
    >>> tl.inputs.unregister("doc_numbers")
    """

    class_names = class_names_tuple(class_name)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        """Register ``func`` behind an input-route detector and return it unchanged."""

        def detector(model: Any, payload: Any, **trace_kwargs: Any) -> Any:
            """Trace ``model`` on ``func``'s preprocessed input, or decline with ``None``."""

            if not matches_class_name(class_names, type(model).__name__):
                return None
            prepared = func(model, payload)
            if prepared is None:
                return None
            transform = _prepared_transform(model, payload, prepared, func, name)
            return _trace_preprocessed(model, payload, transform, trace_kwargs)

        detector.__doc__ = func.__doc__
        detector.__qualname__ = getattr(func, "__qualname__", name)
        _input_registry.register(name=name, priority=priority)(detector)
        return func

    return decorator


def unregister(name: str) -> None:
    """Remove a registered input preprocessor; later traces no longer use it.

    Parameters
    ----------
    name:
        The name given to ``register``.

    Raises
    ------
    KeyError
        If no preprocessor named ``name`` is registered.

    Examples
    --------
    >>> import torchlens as tl
    >>> @tl.inputs.register(name="doc_noop")
    ... def doc_noop(model, raw):
    ...     return None
    >>> tl.inputs.unregister("doc_noop")
    """

    _input_registry.unregister(name)


def _prepared_transform(
    model: Any, payload: Any, prepared: Any, func: Callable[..., Any], name: str
) -> Callable[[Any], Any]:
    """Return the input transform that runs ``func``, stamped with its provenance record.

    The transform reuses the already prepared value for the raw payload the
    route saw, so ``func`` runs once per trace.
    """

    def transform(raw: Any) -> Any:
        """Return the model input for ``raw``."""

        return prepared if raw is payload else func(model, raw)

    transform._tl_preprocessing_record = ResolvedPreprocessing(  # type: ignore[attr-defined]
        source=_PREPROCESSOR_SOURCE,
        identifier=name,
        verified=False,
        config={"preprocessor": name},
        description=f"registered input preprocessor {name!r}",
    )
    return transform


def _trace_preprocessed(
    model: Any, payload: Any, transform: Callable[[Any], Any], trace_kwargs: dict[str, Any]
) -> Any:
    """Trace ``model`` with ``transform`` as the capture's input transform.

    The raw value is kept as ``trace.input`` exactly as an explicit
    ``input_transform=`` would keep it.
    """

    capture = merge_capture_options(
        capture=trace_kwargs.pop("capture", None),
        input_transform=transform,
    )
    return _trace(model, payload, capture=capture, **trace_kwargs)


__all__ = ["register", "unregister"]
