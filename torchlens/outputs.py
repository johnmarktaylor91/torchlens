"""Output decoders: give your model's outputs human-readable labels.

Register a function that names the classes your model's logits score. Every
later ``tl.trace`` of a matching model decodes its output once at the end of
capture into ``trace.decode_output()`` (top-k rows per batch item) and records
the decoder in ``trace.provenance.output_postprocessor``; ``trace.output_table()``
renders the same rows as a dataframe. Registration affects future traces only:
a trace made before ``register`` keeps no decode, and asking an older trace to
decode with ``decode_output(style=<decoder name>)`` is an explicit request that
reads the registry as it is at that moment.

Examples
--------
>>> import torch
>>> import torchlens as tl
>>> class Pets(torch.nn.Module):
...     def forward(self, x):
...         return torch.stack([x.sum(-1), -x.sum(-1)], dim=-1)
>>> @tl.outputs.register(name="pets", class_name="Pets")
... def pets(outputs):
...     return ["cat", "dog"]
>>> trace = tl.trace(Pets(), torch.ones(1, 3))
>>> trace.decode_output()["rows"][0]["label"]
'cat'
>>> tl.outputs.unregister("pets")
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from torchlens._autoroute import output as _output_registry
from torchlens._autoroute._decode_spellings import HEAD_KW, STYLE_KW
from torchlens._autoroute._registry import class_names_tuple, matches_class_name
from torchlens._data_classes.trace import ResolvedPostprocessing

#: Provenance ``source`` stamped on traces whose output a registered decoder labelled.
_DECODER_SOURCE = "registered_output_decoder"


def register(
    *,
    name: str,
    priority: int = 100,
    class_name: str | tuple[str, ...] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return a decorator that registers an output decoder for future traces.

    The decorated function is called as ``fn(outputs)`` with the model's output
    (the live return value at the end of capture; at read time the retained
    output tensor, or a dict of them keyed by output head) and returns the class
    labels in logit order: a sequence of strings, or a mapping from each class
    index (0 to n-1) to its label. Returning ``None`` declines, and the next decoder is asked.
    Decoders run in ascending ``priority`` order; the built-in ones (Hugging Face
    ``config.id2label``, verified ImageNet-1k widths, Hugging Face text) sit at
    priorities 10 to 30, so use a priority below 10 to run before them.

    The decoder also answers ``trace.decode_output(style=name)`` on demand.

    Parameters
    ----------
    name:
        Unique decoder name, used by ``unregister``, as the on-demand
        ``style=`` and in the trace's output provenance.
    priority:
        Sort key. Lower values run first.
    class_name:
        Optional model class name, a glob pattern, or a tuple of them; the
        decoder runs only for traces of models whose class name matches.

    Returns
    -------
    Callable[[Callable[..., Any]], Callable[..., Any]]
        Decorator that registers and returns the original function.

    Raises
    ------
    ValueError
        If a decoder or built-in output detector named ``name`` is already
        registered.

    Examples
    --------
    >>> import torch
    >>> import torchlens as tl
    >>> @tl.outputs.register(name="doc_bits", class_name="Linear")
    ... def doc_bits(outputs):
    ...     return {0: "off", 1: "on"}
    >>> model = torch.nn.Linear(3, 2)
    >>> trace = tl.trace(model, torch.ones(1, 3))
    >>> trace.provenance.output_postprocessor.identifier
    'doc_bits'
    >>> tl.outputs.unregister("doc_bits")
    """

    class_names = class_names_tuple(class_name)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        """Register ``func`` behind an output detector and return it unchanged."""

        detector = _decoder_detector(func, name, class_names)
        _output_registry.register(name=name, priority=priority)(detector)
        return func

    return decorator


def _decoder_detector(
    func: Callable[..., Any], name: str, class_names: tuple[str, ...]
) -> Callable[[Any, dict[str, Any]], ResolvedPostprocessing | None]:
    """Wrap a public decoder as an output-registry detector."""

    def detector(outputs: Any, meta: dict[str, Any]) -> ResolvedPostprocessing | None:
        """Resolve ``func``'s labels into a classification decode, or decline."""

        if meta.get(STYLE_KW) not in (None, name):
            return None
        if not matches_class_name(class_names, meta.get("model_class_name")):
            return None
        labels = func(outputs)
        if labels is None:
            return None
        if isinstance(labels, ResolvedPostprocessing):
            return labels
        return _labels_record(name, labels, meta.get(HEAD_KW))

    detector.__doc__ = func.__doc__
    detector.__qualname__ = getattr(func, "__qualname__", name)
    return detector


def unregister(name: str) -> None:
    """Remove a registered output decoder; later traces no longer use it.

    Parameters
    ----------
    name:
        The name given to ``register``.

    Raises
    ------
    KeyError
        If no decoder named ``name`` is registered.

    Examples
    --------
    >>> import torchlens as tl
    >>> @tl.outputs.register(name="doc_none")
    ... def doc_none(outputs):
    ...     return None
    >>> tl.outputs.unregister("doc_none")
    """

    _output_registry.unregister(name)


def _labels_record(name: str, labels: Any, head: str | None) -> ResolvedPostprocessing:
    """Build the classification decode record for a decoder's labels."""

    if isinstance(labels, Mapping):
        id2label = {int(index): str(label) for index, label in labels.items()}
    else:
        id2label = {index: str(label) for index, label in enumerate(labels)}
    return ResolvedPostprocessing(
        source=_DECODER_SOURCE,
        identifier=name,
        verified=True,
        config={"num_labels": len(id2label), "id2label": id2label},
        description=f"registered output decoder {name!r}",
        style="classification",
        selected_output_head=head,
        label_source=name,
        label_source_version=_DECODER_SOURCE,
        confidence=1.0,
    )


__all__ = ["register", "unregister"]
