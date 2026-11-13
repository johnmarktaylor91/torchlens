"""Payload retention decided during a Paddle capture (``save=`` lowers peak memory).

A full Paddle capture holds every op output until the trace is finished, then
``save=`` hides the unselected payloads (``_postprocess/_selective_save.py``).
Replay validation reads every hidden payload, so the peak is the full capture.

Here the capture decides, as each wrapped call returns and after its edges are
recorded, whether the output payload stays in RAM. A payload the ``save=``
selector does not select is written to a private spool directory and the live
tensor is released; the trace's replay-visible payload table reads it back on
demand. Validation therefore sees exactly the witnesses a full capture keeps
(no check changes and no regenerated outputs), while peak memory holds only the
selected payloads.

The decision only affects where a payload lives, never what the trace returns:
the selector is evaluated on the capture-time context (function name, layer
type, module stack and the predicted label spellings), and after the trace is
finished every op the selector selects on its final labels gets its payload
back in RAM, so a wrong prediction costs a reload, not a value.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import weakref
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ... import _state

STATE_KEY = "_paddle_retention_state"
"""Trace ``__dict__`` key holding the capture's :class:`PaddleRetention`."""

STREAM_KEY = "_paddle_stream_state"
"""Trace ``__dict__`` key holding the capture's streaming state (``_stream_capture``)."""


@dataclass(frozen=True)
class SpilledPayload:
    """One op payload moved out of RAM during capture.

    Attributes
    ----------
    path
        ``.npy`` file in the capture's spool directory.
    place
        The Paddle place the payload lived on.
    """

    path: Path
    place: Any

    def load(self) -> Any:
        """Read the payload back as a Paddle tensor on its original place."""

        import numpy as np
        import paddle

        with _state.pause_logging():
            return paddle.to_tensor(np.load(self.path, allow_pickle=False), place=self.place)


class SpilledPayloads(Mapping[str, Any]):
    """Replay-visible payloads by label; spilled entries load on access.

    Parameters
    ----------
    payloads
        Hidden payloads by raw and final label; spilled ones are
        :class:`SpilledPayload` handles.
    spool
        The spool directory, removed when this table is collected.
    """

    def __init__(self, payloads: Mapping[str, Any], spool: Path) -> None:
        self._payloads = dict(payloads)
        self._spool = spool
        self._cleanup = weakref.finalize(self, shutil.rmtree, str(spool), True)

    def __getitem__(self, label: str) -> Any:
        value = self._payloads[label]
        return value.load() if isinstance(value, SpilledPayload) else value

    def __iter__(self) -> Iterator[str]:
        return iter(self._payloads)

    def __len__(self) -> int:
        return len(self._payloads)

    def __reduce__(self) -> tuple[Any, ...]:
        """Copies and pickles carry the loaded payloads, never spool paths."""

        return (dict, (dict(self.items()),))


@dataclass
class PaddleRetention:
    """Retention state of one capture.

    Attributes
    ----------
    predicate
        The static-label ``save=`` selector.
    spool
        Spool directory, created on the first spill.
    spilled
        Number of payloads moved out of RAM.
    handles
        Spilled payloads by raw label.
    """

    predicate: Callable[[Any], Any]
    spool: Path | None = None
    spilled: int = 0
    handles: dict[str, SpilledPayload] = field(default_factory=dict)

    def keeps(self, ctx: Any) -> bool:
        """Whether the selector may select the op ``ctx`` describes (True keeps it in RAM)."""

        return _may_select(self.predicate, ctx)

    def spill(self, label_raw: str, tensor: Any) -> None:
        """Write ``tensor`` to the spool under ``label_raw``."""

        import numpy as np

        if self.spool is None:
            self.spool = Path(tempfile.mkdtemp(prefix="torchlens-paddle-retention-"))
        self.spilled += 1
        path = self.spool / f"{self.spilled}.npy"
        with _state.pause_logging():
            np.save(path, tensor.numpy(), allow_pickle=False)
        self.handles[label_raw] = SpilledPayload(path=path, place=tensor.place)


_FINAL_LABEL = re.compile(r"^(?P<stem>.+?_\d+)(?:_\d+)?(?::(?:\d+|last|all))?$")
"""A final label: ``{type}_{type_index}[_{layer_index}][:{pass}]``."""


def _may_select(selector: Any, ctx: Any) -> bool:
    """Whether ``selector`` may select the op the capture context ``ctx`` describes.

    Unions and intersections are decided child by child. A leaf is evaluated on
    the context with the label spellings capture can predict. Capture refuses
    final-label literals (their layer index is assigned when the trace is
    finished), so such a ``tl.label`` leaf is decided by its type and type
    index; any other undecidable leaf keeps the op. Only RAM placement depends
    on the answer: the final labels decide what the trace keeps.
    """

    operator = getattr(selector, "operator", None)
    children = getattr(selector, "selectors", None)
    if operator in ("and", "or") and children is not None:
        answers = (_may_select(child, ctx) for child in children)
        return any(answers) if operator == "or" else all(answers)
    variants = (
        ctx,
        replace(ctx, label=f"{ctx.layer_type}_{ctx.type_index}_{ctx.raw_index}"),
        replace(ctx, label=f"{ctx.layer_type}_{ctx.type_index}"),
    )
    try:
        return any(bool(selector(variant)) for variant in variants)
    except Exception:  # noqa: BLE001 - undecidable at capture; final labels decide
        match = _FINAL_LABEL.match(str(getattr(selector, "selector_value", "")))
        if getattr(selector, "selector_kind", None) != "label" or match is None:
            return True
        return match["stem"] == f"{ctx.layer_type}_{ctx.type_index}"


def start_retention(trace: Any, predicate: Callable[[Any], Any]) -> None:
    """Arm capture-time retention on ``trace`` for a static-label ``save=`` selector.

    The caller arms it only when the final keep set is ``predicate(op)`` on
    each finished op (a static-label selector) and no option needs every
    payload in RAM.

    Parameters
    ----------
    trace
        The trace under construction.
    predicate
        The static-label ``save=`` selector.
    """

    trace.__dict__[STATE_KEY] = PaddleRetention(predicate)


def retention_policy(session: Any) -> Callable[[Any], bool]:
    """Return the capture's per-op keep decision (wave B hook contract).

    Parameters
    ----------
    session
        The trace under construction.

    Returns
    -------
    Callable[[Any], bool]
        Maps a capture-time ``RecordContext`` to True when its payload stays in RAM.
    """

    state = session.__dict__.get(STATE_KEY)
    return (lambda _ctx: True) if state is None else state.keeps


def retained_payload(
    backend: Any, session: Any, site: Any, call: Any, output: Any, policy: Any
) -> Any:
    """Return the payload one emitted output carries: the tensor, or ``None`` when unsaved.

    Parameters
    ----------
    backend
        The Paddle backend (builds the capture-time context).
    session
        The trace under construction.
    site
        The output's reserved label.
    call
        The wrapped call's ``FunctionEventInput``.
    output
        The output tensor.
    policy
        The capture's ``CapturePolicy``; no payload when it saves none.

    Returns
    -------
    Any
        ``output`` when it stays in RAM, else ``None`` (unsaved, or the spool holds it).
    """

    if not policy.save_payload:
        return None
    state = session.__dict__.get(STATE_KEY)
    if state is None or site.layer_type == "input" or not _spillable(backend, output):
        return output
    if state.keeps(backend.build_record_context(session, site, call, output)):
        return output
    state.spill(site.label_raw, output)
    return None


def saved_ref(backend: Any, session: Any, site: Any, call: Any, output: Any, policy: Any) -> Any:
    """Build the output's tensor reference with the payload :func:`retained_payload` keeps.

    When the capture streams (``storage=``), the kept payload is handed to the
    stream here, the one point every emitted output passes once its edges are
    recorded; the reference then names the bundle blob and drops the payload
    unless ``retain_in_memory``.
    """

    payload = retained_payload(backend, session, site, call, output, policy)
    ref = backend.tensor_ref(session, output, payload, policy)
    stream = session.__dict__.get(STREAM_KEY)
    if stream is None or payload is None or not backend.is_tensor(payload):
        return ref
    return stream.submit(session, ref, payload, site.label_raw)


def _spillable(backend: Any, output: Any) -> bool:
    """Tensors NumPy round-trips exactly: not bfloat16 (NumPy carries it as uint16)."""

    return bool(backend.is_tensor(output)) and "bfloat16" not in str(backend._dtype(output))


def apply_save(trace: Any, predicate: Any, apply_policy: Callable[..., None]) -> None:
    """Apply ``save=`` to the finished trace, using the payloads retention spilled.

    Ops the selector selects on their final labels get their payload back in
    RAM before filtering; every other spilled payload stays replay-visible
    through the trace's hidden payload table.

    Parameters
    ----------
    trace
        The finished trace.
    predicate
        The static ``save=`` predicate, or ``None``.
    apply_policy
        ``apply_static_label_save_policy``: hides the unselected payloads.
    """

    state = trace.__dict__.pop(STATE_KEY, None)
    handles = {} if state is None else state.handles
    for op in trace._layer_list if handles else ():
        handle = handles.get(str(op._label_raw))
        if handle is not None and bool(predicate(op)):
            _set_out(op, handle.load())
    apply_policy(trace, predicate, backend_name="paddle")
    if not handles or state is None or state.spool is None:
        return
    hidden = dict(trace._selective_save_hidden_payloads or {})  # set by apply_policy above
    for op in trace._layer_list:
        handle = handles.get(str(op._label_raw))
        if handle is not None and getattr(op, "out", None) is None:
            hidden.setdefault(str(op._label_raw), handle)
            hidden.setdefault(str(op.label), handle)
    trace._selective_save_hidden_payloads = SpilledPayloads(hidden, state.spool)


def _set_out(op: Any, payload: Any) -> None:
    """Set ``op.out`` without marking trace state dirty, as the save filter does."""

    setter = getattr(op, "_internal_set", None)
    if callable(setter):
        setter("out", payload)
    else:
        op.out = payload


def discard_retention(trace: Any) -> None:
    """Remove retention state and its spool after a failed capture."""

    state = trace.__dict__.pop(STATE_KEY, None)
    if state is not None and state.spool is not None:
        shutil.rmtree(state.spool, ignore_errors=True)
