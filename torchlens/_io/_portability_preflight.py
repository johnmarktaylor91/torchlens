"""Save-side portability preflight for ``metadata.pkl`` (AUD-CODE 2.20 / 3.11g).

Write/read symmetry law: TorchLens never writes metadata bytes that its own
DEFAULT loader cannot read back. ``tl.load`` unpickles ``metadata.pkl``
through the default-deny allowlist (``_safe_unpickle``), so an arbitrary
user object sitting in ``Trace.annotations`` (or a captured tensor still
carrying its ``tl_*`` session attributes under ``logged_values``) used to
SAVE silently and then make the WHOLE artifact unloadable with a misleading
"corrupt or malformed pickle stream" refusal. Two doors close that:

* :func:`sanitize_annotation_tensors` COERCES the one shape TorchLens itself
  produces -- a tensor logged during the forward (``tl.observers.log_value``)
  or dropped into an annotation while still carrying capture attributes /
  autograd history / a subclass -- into the plain detached tensor the loader
  already admits. The numbers are identical; only session-time attributes
  (never meaningful after save) and autograd history (never persisted) go.
* :func:`preflight_metadata_portability` runs the exact bytes about to be
  written through the SAME restricted unpickler the loader uses (default
  trust flags, governed window open) and REFUSES typed, naming the offending
  key path, when the dry run fails. Nothing is written on refusal.

Codes: ``annotation_value_unportable`` (the located value lives under an
``annotations`` mapping -- trace-level or per-record) and
``metadata_value_unportable`` (anywhere else in the persisted state).
"""

from __future__ import annotations

import io
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import torch

from . import TorchLensIOError
from ._canonical_pickle import dump_canonical_metadata
from .state_contract import governed_artifact_load

__all__ = [
    "canonical_metadata_bytes",
    "preflight_metadata_portability",
    "sanitize_annotation_tensors",
    "sanitize_state_annotations",
]

UnpicklerFactory = Callable[[io.BytesIO], Any]

#: Recursion ceiling for the culprit walk: deeper trees report the deepest
#: located container rather than walking an adversarial structure forever.
_LOCATE_MAX_DEPTH = 64

_ANNOTATION_REMEDY = (
    "store plain data under annotations (str / int / float / bool / None / "
    "list / tuple / dict, torch.dtype / torch.Size, or a plain detached tensor), "
    "or drop the key before saving; tl.load() rehydrates metadata through a "
    "default-deny allowlist and never constructs user classes"
)
_METADATA_REMEDY = (
    "remove the unportable value from the trace before saving, or report the "
    "field: a persisted TorchLens field that the default loader cannot read "
    "back is a writer defect"
)


def canonical_metadata_bytes(state: Any) -> bytes:
    """Return the canonical ``metadata.pkl`` bytes for ``state``."""

    buffer = io.BytesIO()
    dump_canonical_metadata(state, buffer)
    return buffer.getvalue()


def sanitize_annotation_tensors(annotations: Any) -> Any:
    """Return a copy of an annotations tree whose tensors are load-portable.

    A ``torch.Tensor`` pickles through the admitted ``_rebuild_tensor_v2``
    reconstructor only when it is a plain ``torch.Tensor`` with an empty
    instance ``__dict__``; a subclass (``nn.Parameter``), a tensor carrying
    Python attributes (every activation captured in a forward carries its
    ``tl_*`` bookkeeping), or one with grad history pickles through the
    denied ``_rebuild_from_type_v2``. Those are replaced by a detached plain
    tensor with identical values. Containers are rebuilt, never mutated, so
    the live trace keeps its session-time objects; unknown leaves pass
    through untouched for the preflight to judge.
    """

    return _sanitize(annotations, 0)


def sanitize_state_annotations(state: dict[str, Any]) -> None:
    """Make every annotation tensor in a SCRUBBED state load-portable (AUD-CODE 3.11g).

    A tensor logged during the forward (``tl.observers.log_value``) or dropped
    into an annotation still carries its ``tl_*`` capture attributes, so it
    pickles through the loader-denied ``_rebuild_from_type_v2`` and the saved
    artifact could never be read back. The scrubbed trace-level mapping and
    each scrubbed record's mapping (``layer_list`` ops, ``layer_logs`` layers,
    ...) are rebuilt with plain detached tensors through
    :func:`sanitize_annotation_tensors`; containers are copied, never mutated
    in place, so the live trace keeps its session-time objects.
    :func:`preflight_metadata_portability` then proves the final bytes load.
    """

    annotations = state.get("annotations")
    if isinstance(annotations, dict) and annotations:
        state["annotations"] = sanitize_annotation_tensors(annotations)
    for value in list(state.values()):
        if isinstance(value, dict):
            records: Any = value.values()
        elif isinstance(value, (list, tuple)):
            records = value
        else:
            continue
        for record in records:
            record_annotations = getattr(record, "annotations", None)
            if isinstance(record_annotations, dict) and record_annotations:
                record.annotations = sanitize_annotation_tensors(record_annotations)


def _sanitize(value: Any, depth: int) -> Any:
    """Recursive worker for :func:`sanitize_annotation_tensors` (depth-bounded)."""

    if depth > _LOCATE_MAX_DEPTH:
        return value
    if isinstance(value, torch.Tensor):
        return _plain_tensor(value)
    if isinstance(value, Mapping) and type(value) in (dict,):
        return {key: _sanitize(item, depth + 1) for key, item in value.items()}
    if type(value) is list:
        return [_sanitize(item, depth + 1) for item in value]
    if type(value) is tuple:
        return tuple(_sanitize(item, depth + 1) for item in value)
    return value


def _plain_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return ``tensor`` as a plain, attribute-free, detached ``torch.Tensor``."""

    state = getattr(tensor, "__dict__", None)
    if type(tensor) is torch.Tensor and not state and not tensor.requires_grad:
        return tensor
    try:
        with torch.no_grad():
            plain = tensor.detach()
            if type(plain) is not torch.Tensor:
                plain = plain.as_subclass(torch.Tensor)
            if getattr(plain, "__dict__", None):
                plain = plain.clone()
        return plain
    except Exception:  # noqa: BLE001 -- exotic subclass; the preflight then refuses typed
        return tensor


def preflight_metadata_portability(
    scrubbed_state: Mapping[str, Any],
    *,
    unpickler_factory: UnpicklerFactory,
    bundle_path: Any,
    data: bytes | None = None,
) -> bytes:
    """Return the ``metadata.pkl`` bytes for ``scrubbed_state`` iff the loader reads them.

    Parameters
    ----------
    scrubbed_state:
        The fully scrubbed trace state about to be written.
    unpickler_factory:
        The restricted unpickler class the LOADER uses (``_RenameAwareUnpickler``),
        constructed with its default trust flags so the dry run mirrors a plain
        ``tl.load(path)``.
    bundle_path:
        Destination named in the refusal.
    data:
        The canonical bytes of ``scrubbed_state`` when the caller already
        produced them (the writers dump once through their own
        ``dump_canonical_metadata`` seam and write exactly the bytes checked
        here); ``None`` dumps them here.

    Raises
    ------
    TorchLensIOError
        ``annotation_value_unportable`` / ``metadata_value_unportable`` when
        the default loader would refuse the bytes; the ``field`` names the
        located key path and ``reason`` carries the loader's own message.
    """

    if data is None:
        data = canonical_metadata_bytes(scrubbed_state)
    failure = _dry_run_failure(data, unpickler_factory)
    if failure is None:
        return data
    if isinstance(failure, TorchLensIOError):
        raise failure
    located, in_annotations = _locate_unportable(scrubbed_state, unpickler_factory)
    reason = _root_cause_text(failure)
    message = (
        f"Refusing to write {bundle_path}: {located} holds a value that tl.load() "
        f"cannot rehydrate under the default-deny metadata allowlist ({reason}). "
        "Nothing was written. Remedy: "
    )
    if in_annotations:
        raise TorchLensIOError(
            message + _ANNOTATION_REMEDY + ".",
            code="annotation_value_unportable",
            field=located,
            reason=reason,
            remedy=_ANNOTATION_REMEDY,
            path=str(bundle_path),
        ) from failure
    raise TorchLensIOError(
        message + _METADATA_REMEDY + ".",
        code="metadata_value_unportable",
        field=located,
        reason=reason,
        remedy=_METADATA_REMEDY,
        path=str(bundle_path),
    ) from failure


def _root_cause_text(failure: BaseException) -> str:
    """Name the innermost chained failure (the loader normalizes VM errors on top)."""

    root: BaseException = failure
    seen: set[int] = set()
    while id(root) not in seen:
        seen.add(id(root))
        nxt = root.__cause__ or root.__context__
        if nxt is None:
            break
        root = nxt
    text = f"{type(root).__name__}: {str(root)[:300]}"
    if root is not failure:
        text = f"{type(failure).__name__} <- {text}"
    return text


def _dry_run_failure(data: bytes, unpickler_factory: UnpicklerFactory) -> BaseException | None:
    """Unpickle ``data`` exactly as the loader would; return the failure, if any."""

    try:
        with governed_artifact_load():
            unpickler_factory(io.BytesIO(data)).load()
    except Exception as exc:  # noqa: BLE001 -- any failure means unloadable
        return exc
    return None


def _portable(value: Any, unpickler_factory: UnpicklerFactory) -> bool:
    """True iff ``value`` alone survives a canonical dump + loader dry run."""

    try:
        data = canonical_metadata_bytes(value)
    except Exception:  # noqa: BLE001 -- unpicklable == unportable
        return False
    return _dry_run_failure(data, unpickler_factory) is None


def _locate_unportable(
    state: Mapping[str, Any], unpickler_factory: UnpicklerFactory
) -> tuple[str, bool]:
    """Name the deepest container/leaf that fails the loader dry run."""

    located = _descend(state, [], False, 0, unpickler_factory)
    if located is None:  # whole state failed but every part passes alone
        return ("Trace (whole persisted state)", False)
    return located


def _descend(
    value: Any,
    path: list[str],
    in_annotations: bool,
    depth: int,
    unpickler_factory: UnpicklerFactory,
) -> tuple[str, bool] | None:
    """Depth-first culprit walk: (rendered path, under-annotations flag) or None."""

    if depth > 0 and _portable(value, unpickler_factory):
        return None
    if depth >= _LOCATE_MAX_DEPTH:
        return (_render(path), in_annotations)
    for segment, item, under_annotations in _walkable_children(value, depth):
        found = _descend(
            item,
            [*path, segment],
            in_annotations or under_annotations,
            depth + 1,
            unpickler_factory,
        )
        if found is not None:
            return found
    # The depth-0 state mapping itself is never the culprit (the caller names
    # the whole state); every deeper container/leaf that failed the dry run is.
    return None if depth == 0 else (_render(path), in_annotations)


def _walkable_children(value: Any, depth: int) -> Iterator[tuple[str, Any, bool]]:
    """Yield ``(path segment, child, enters-annotations)`` for one walkable value.

    Mappings walk their items (an ``annotations`` key flips the flag), lists /
    tuples their elements, and record objects their ``annotations`` mapping;
    plain leaves yield nothing.
    """

    if isinstance(value, Mapping):
        for key, item in value.items():
            segment = f".{key}" if depth == 0 else f"[{key!r}]"
            yield segment, item, key == "annotations"
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield f"[{index}]", item, False
        return
    record_annotations = _record_annotations(value)
    if record_annotations is not None:
        label = getattr(value, "label", None)
        yield f" ({type(value).__name__} {label!r}).annotations", record_annotations, True


def _record_annotations(value: Any) -> Mapping[str, Any] | None:
    """Return a record object's ``annotations`` mapping, or None for plain leaves."""

    if isinstance(value, (str, bytes, int, float, complex, bool, type(None), torch.Tensor)):
        return None
    annotations = getattr(value, "annotations", None)
    return annotations if isinstance(annotations, Mapping) else None


def _render(path: list[str]) -> str:
    """Render a located key path as ``Trace<segments>``."""

    return "Trace" + "".join(path)
