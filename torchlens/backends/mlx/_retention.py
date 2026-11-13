"""Capture-time retention for MLX ``save=`` and structure-only captures (private).

A plain MLX capture keeps every op's output and realizes all of them in one
``mx.eval`` after the forward (SPEC section 3), so peak memory holds every
payload at once. Under an active :class:`RetentionRequest` the capture instead
decides per op, while the forward runs, whether the value stays public:

* selected values (a live static-selector match) keep their public payload;
* every other value is a replay witness: it is realized in topological,
  byte-bounded chunks after the forward, written to a private spill directory
  and dropped from memory, so validation still re-executes each op against
  real parent values;
* a structure-only request keeps neither payloads nor witnesses; the forward
  still executes concretely and only the outputs are realized.

This module imports nothing from the capture stack: the backend passes the
selector matcher and the keep-set filter in :class:`RetentionDeps`, which keeps
the module out of the package's import cycle.

The live match only decides what stays in memory. After the trace is finished
the public keep set is recomputed with the post-capture rule on final ops, so
the result equals a full capture filtered by the same ``save=`` selector: a
selected op the live match missed is restored from its spilled witness.
"""

from __future__ import annotations

import shutil
import tempfile
import weakref
from collections import ChainMap, deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

#: Default byte budget for one chunked evaluation of witnesses.
DEFAULT_CHUNK_BYTES = 64 * 2**20

#: Selector kinds a capture-time ``RecordContext`` can decide. Module kinds need
#: the finished module hierarchy, so their live answer is "keep".
LIVE_DECIDABLE_KINDS = frozenset({"label", "func", "contains", "and", "or", "not"})


@dataclass(frozen=True)
class RetentionRequest:
    """What one MLX capture should retain.

    Parameters
    ----------
    selector:
        Static-label ``save=`` selector, or ``None`` with ``structure_only``.
    structure_only:
        Keep no payloads or witnesses; validation then refuses.
    chunk_bytes:
        Byte budget of one witness evaluation chunk.
    """

    selector: Any | None = None
    structure_only: bool = False
    chunk_bytes: int = DEFAULT_CHUNK_BYTES


@dataclass(frozen=True)
class RetentionDeps:
    """Capture-stack callables a retention run uses.

    Parameters
    ----------
    undecidable:
        ``selector -> bool``: whether a capture-time context cannot decide it.
    live_match:
        ``(selector, RecordContext) -> bool``: the live static-selector match.
    apply_keep_set:
        ``(trace, keep) -> None``: the post-capture public keep-set filter.
    """

    undecidable: Callable[[Any], bool]
    live_match: Callable[[Any, Any], bool]
    apply_keep_set: Callable[[Any, set[str]], None]


_ACTIVE: ContextVar[RetentionRequest | None] = ContextVar(
    "torchlens_mlx_retention_request", default=None
)


@contextmanager
def retention_scope(request: RetentionRequest | None) -> Iterator[None]:
    """Make ``request`` the retention request of captures run inside the block.

    Parameters
    ----------
    request:
        Request to apply, or ``None`` for a plain capture.

    Yields
    ------
    None
        Control while the request is active.
    """

    token = _ACTIVE.set(request)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def active_request() -> RetentionRequest | None:
    """Return the retention request of the current context, if any."""

    return _ACTIVE.get()


def retention_policy(request: RetentionRequest, deps: RetentionDeps) -> Callable[[Any], bool]:
    """Return the live decision ``RecordContext -> keep public payload``.

    Parameters
    ----------
    request:
        Active retention request.
    deps:
        Selector matcher supplied by the backend.

    Returns
    -------
    Callable[[Any], bool]
        Decision for one capture-time record context. It may keep more than
        the final rule (that costs memory only); a miss is restored from the
        op's witness when the trace is settled.
    """

    if request.structure_only:
        return lambda _ctx: False
    selector = request.selector
    if selector is None or deps.undecidable(selector):
        return lambda _ctx: True
    return lambda ctx: deps.live_match(selector, ctx)


class SpillStore(Mapping[str, Any]):
    """Read-only mapping from op labels to spilled MLX witnesses.

    Values are written once with :meth:`put` and loaded from disk on each
    lookup, so holding the store costs no device memory. The spill directory is
    removed when the store is garbage collected or :meth:`close` is called.
    Copies of a trace share the store (``copy`` and ``deepcopy`` return it).

    Parameters
    ----------
    mx:
        Imported ``mlx.core`` module.
    """

    def __init__(self, mx: Any) -> None:
        self._mx = mx
        self._dir = Path(tempfile.mkdtemp(prefix="torchlens-mlx-spill-"))
        self._files: dict[str, str] = {}
        self._memory: dict[str, Any] = {}
        self._finalizer = weakref.finalize(self, shutil.rmtree, str(self._dir), True)

    def put(self, label: str, value: Any) -> None:
        """Spill one realized witness under ``label``."""

        path = str(self._dir / f"{len(self._files) + len(self._memory)}.safetensors")
        try:
            self._mx.save_safetensors(path, {"v": value})
        except (TypeError, ValueError, RuntimeError):
            # A dtype safetensors cannot hold stays in memory: correct, not lean.
            self._memory[label] = value
            return
        self._files[label] = path

    def alias(self, label: str, existing: str) -> None:
        """Make ``label`` resolve to the witness stored under ``existing``."""

        if existing in self._files:
            self._files[label] = self._files[existing]
        elif existing in self._memory:
            self._memory[label] = self._memory[existing]

    def alias_final_labels(self, ops: Any) -> None:
        """Make each op's final label resolve to the witness stored under its raw label."""

        for op in ops:
            if isinstance(op.label, str) and isinstance(op._label_raw, str):
                self.alias(op.label, op._label_raw)

    @property
    def in_memory_count(self) -> int:
        """Number of witnesses kept in memory because they could not spill."""

        return len(set(map(id, self._memory.values())))

    def __getitem__(self, label: str) -> Any:
        if label in self._memory:
            return self._memory[label]
        return self._mx.load(self._files[label])["v"]

    def __iter__(self) -> Iterator[str]:
        yield from self._files
        yield from self._memory

    def __len__(self) -> int:
        return len(self._files) + len(self._memory)

    def __contains__(self, label: object) -> bool:
        return label in self._files or label in self._memory

    def __copy__(self) -> SpillStore:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> SpillStore:
        return self

    def close(self) -> None:
        """Remove the spill directory now."""

        self._finalizer()


class RetentionRun:
    """Per-capture retention state: live decisions, chunked realize, settle.

    Parameters
    ----------
    request:
        Active retention request.
    mx:
        Imported ``mlx.core`` module.
    deps:
        Capture-stack callables supplied by the backend.
    """

    def __init__(self, request: RetentionRequest, mx: Any, deps: RetentionDeps) -> None:
        self.request = request
        self._mx = mx
        self._decide = retention_policy(request, deps)
        self._apply_keep_set = deps.apply_keep_set
        self._pending: deque[tuple[str, Any, bool]] = deque()
        self.store: SpillStore | None = None if request.structure_only else SpillStore(mx)

    def note(self, label_raw: str, value: Any, context: Callable[[], Any]) -> bool:
        """Record one op output and return whether its payload stays public.

        Parameters
        ----------
        label_raw:
            Raw label of the output.
        value:
            Lazy MLX output array.
        context:
            Builds the capture-time ``RecordContext`` (called only when needed).

        Returns
        -------
        bool
            ``True`` when the op keeps its public payload.
        """

        if self.request.structure_only:
            return False
        public = bool(self._decide(context()))
        self._pending.append((label_raw, value, public))
        return public

    def realize(self, extra_values: tuple[Any, ...]) -> None:
        """Realize recorded values in byte-bounded chunks, spilling witnesses.

        Each value is computed once. Witnesses leave memory after their chunk;
        public payloads and ``extra_values`` (sources, outputs) stay realized.
        """

        chunk: list[tuple[str, Any, bool]] = []
        chunk_bytes = 0
        while self._pending:
            entry = self._pending.popleft()
            chunk.append(entry)
            chunk_bytes += int(getattr(entry[1], "nbytes", 0))
            del entry
            if chunk_bytes >= self.request.chunk_bytes:
                self._flush(chunk)
                chunk_bytes = 0
        self._flush(chunk)
        if extra_values:
            self._mx.eval(*extra_values)

    def _flush(self, chunk: list[tuple[str, Any, bool]]) -> None:
        """Evaluate one chunk, spill its witnesses and drop every reference."""

        if not chunk:
            return
        self._mx.eval(*(value for _label, value, _public in chunk))
        if self.store is not None:
            for label, value, public in chunk:
                if not public:
                    self.store.put(label, value)
        chunk.clear()

    def settle(self, trace: Any) -> None:
        """Apply the final ``save=`` keep set and attach witnesses for replay.

        Parameters
        ----------
        trace:
            Finished MLX trace.
        """

        store = self.store
        if store is None:
            trace._options_structure_only = True
            return
        store.alias_final_labels(trace._layer_list)
        keep = _final_keep_set(trace._layer_list, self.request.selector)
        _restore_live_misses(trace._layer_list, keep, store)
        self._apply_keep_set(trace, keep)
        public: Mapping[str, Any] = trace._selective_save_hidden_payloads
        # ChainMap writes only to maps[0]; the read-only spill store is a lookup tail.
        trace._selective_save_hidden_payloads = ChainMap(dict(public), cast(Any, store))


def start_retention(mx: Any, deps: RetentionDeps) -> RetentionRun | None:
    """Return the retention run for a capture starting now, or ``None``."""

    request = active_request()
    return None if request is None else RetentionRun(request, mx, deps)


def _final_keep_set(ops: Any, selector: Any | None) -> set[str]:
    """Return the labels the post-capture ``save=`` rule keeps on final ops."""

    return {op.label for op in ops if selector is None or bool(selector(op))}


def _restore_live_misses(ops: Any, keep: set[str], store: SpillStore) -> None:
    """Restore the public payload of each kept op the live match turned into a witness."""

    for op in ops:
        if op.label in keep and op.out is None and op._label_raw in store:
            _restore_public_payload(op, store[op._label_raw])


def _restore_public_payload(op: Any, value: Any) -> None:
    """Make ``value`` the public payload of ``op`` (inverse of the post drop)."""

    for name, field_value in (("out", value), ("has_saved_activation", True)):
        setter = getattr(op, "_internal_set", None)
        if callable(setter):
            setter(name, field_value)
        else:
            setattr(op, name, field_value)
