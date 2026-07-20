"""Round-47 hon1_1 immunizer -- generators held on a container SUBCLASS's own attribute.

r45 hon1_1 descends model-held containers by PROTOCOL (every ``Mapping`` keys+values, every non-leaf
``Collection`` elements). But a container SUBCLASS can hold a generator as its OWN ``__dict__`` /
``__slots__`` attribute (``self.rng`` on a ``Sequence`` / ``Mapping`` / ``UserList`` / ``UserDict``
subclass), which is NOT among its keys/values/elements -- so the early ``continue`` in the Mapping /
Collection branch skipped the holder walk and the generator went uninventoried -> false ``VERIFIED``
(probe M4/M7 returned 0).

r47 gives a value that is BOTH a container AND a custom inert holder BOTH walks: its keys/values or
elements, PLUS its own ``__dict__`` / ``__slots__`` children when ``_is_recursable_custom_holder``
is true (which excludes stdlib/torch/numpy, so a plain ``dict`` / ``list`` is never double-walked).
"""

from __future__ import annotations

import collections
import collections.abc as abc
import shutil
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness
from torchlens.utils.rng import (
    _rng_exempt_instances,
    host_nondeterminism_monitor,
)

_CAP = dict(intervention_ready=True, capture_container_structure=True, cache=False)


# --- Container SUBCLASSES holding a generator as an ATTRIBUTE (not an element) -----------------
class MappingSub(abc.Mapping):  # type: ignore[type-arg]
    def __init__(self, gen: Any) -> None:
        self.rng = gen
        self._d: dict[Any, Any] = {}

    def __getitem__(self, k: Any) -> Any:
        return self._d[k]

    def __iter__(self) -> Any:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)


class MutableMappingSub(abc.MutableMapping):  # type: ignore[type-arg]
    def __init__(self, gen: Any) -> None:
        self.rng = gen
        self._d: dict[Any, Any] = {}

    def __getitem__(self, k: Any) -> Any:
        return self._d[k]

    def __setitem__(self, k: Any, v: Any) -> None:
        self._d[k] = v

    def __delitem__(self, k: Any) -> None:
        del self._d[k]

    def __iter__(self) -> Any:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)


class SequenceSub(abc.Sequence):  # type: ignore[type-arg]
    def __init__(self, gen: Any) -> None:
        self.rng = gen
        self._items: list[Any] = []

    def __getitem__(self, i: Any) -> Any:
        return self._items[i]

    def __len__(self) -> int:
        return len(self._items)


class MutableSequenceSub(abc.MutableSequence):  # type: ignore[type-arg]
    def __init__(self, gen: Any) -> None:
        self.rng = gen
        self._items: list[Any] = []

    def __getitem__(self, i: Any) -> Any:
        return self._items[i]

    def __setitem__(self, i: Any, v: Any) -> None:
        self._items[i] = v

    def __delitem__(self, i: Any) -> None:
        del self._items[i]

    def __len__(self) -> int:
        return len(self._items)

    def insert(self, i: int, v: Any) -> None:
        self._items.insert(i, v)


class SetSub(abc.Set):  # type: ignore[type-arg]
    def __init__(self, gen: Any) -> None:
        self.rng = gen
        self._s: set[Any] = set()

    def __contains__(self, x: object) -> bool:
        return x in self._s

    def __iter__(self) -> Any:
        return iter(self._s)

    def __len__(self) -> int:
        return len(self._s)


class UserListSub(collections.UserList):  # type: ignore[type-arg]
    def __init__(self, gen: Any) -> None:
        super().__init__()
        self.rng = gen


class UserDictSub(collections.UserDict):  # type: ignore[type-arg]
    def __init__(self, gen: Any) -> None:
        super().__init__()
        self.rng = gen


class SlotsSequenceSub(abc.Sequence):  # type: ignore[type-arg]
    __slots__ = ("rng", "_items")

    def __init__(self, gen: Any) -> None:
        self.rng = gen
        self._items = []

    def __getitem__(self, i: Any) -> Any:
        return self._items[i]

    def __len__(self) -> int:
        return len(self._items)


_HOLDER_TYPES = [
    MappingSub,
    MutableMappingSub,
    SequenceSub,
    MutableSequenceSub,
    SetSub,
    UserListSub,
    UserDictSub,
    SlotsSequenceSub,
]


def _sweep_count(model: nn.Module) -> int:
    """Return how many model-held generators ``_sweep_model_generators`` snapshots."""

    mon = host_nondeterminism_monitor(model)
    mon._exempt_ids = frozenset(id(inst) for inst in _rng_exempt_instances())
    return len(mon._sweep_model_generators())


@pytest.mark.parametrize("holder_type", _HOLDER_TYPES, ids=lambda t: t.__name__)
def test_container_subclass_attribute_generator_is_swept(holder_type: type) -> None:
    """Direct sweep: a generator held as ``self.rng`` on a container SUBCLASS is found (was 0)."""

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.holder: Any = holder_type(np.random.default_rng(0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x).relu()

    assert _sweep_count(_Model()) == 1


class _PreexistingWorker:
    """A worker thread started BEFORE any capture window (non-hooked, foreign)."""

    def __init__(self) -> None:
        import queue

        self._jobs: "queue.Queue[Any]" = queue.Queue()
        self._results: "queue.Queue[Any]" = queue.Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            try:
                self._results.put(("ok", job()))
            except BaseException as exc:  # noqa: BLE001 - relayed to test thread
                self._results.put(("err", exc))

    def run(self, job: Callable[[], Any]) -> Any:
        self._jobs.put(job)
        kind, value = self._results.get()
        if kind == "err":
            raise value
        return value

    def stop(self) -> None:
        self._jobs.put(None)
        self._thread.join(timeout=5)


@pytest.fixture()
def preexisting_worker() -> Any:
    worker = _PreexistingWorker()
    try:
        yield worker
    finally:
        worker.stop()


def _roundtrip(model: nn.Module, x: torch.Tensor, tmp: Path) -> tl.RunResult:
    trace = tl.trace(model, x, capture=CaptureOptions(random_seed=1, **_CAP))
    path = tmp / "r47cont.tlspec"
    shutil.rmtree(path, ignore_errors=True)
    trace.save(path, level="runnable", include_weights=True, include_activations=True)
    return tl.load(path).run(inputs=x)


@pytest.mark.parametrize(
    "holder_type",
    [MappingSub, SequenceSub, UserListSub, UserDictSub, SlotsSequenceSub],
    ids=lambda t: t.__name__,
)
def test_container_subclass_generator_drawn_on_worker_ceilings(
    holder_type: type, preexisting_worker: Any, tmp_path: Path
) -> None:
    """End-to-end: drawing the container-subclass-held generator on a PRE-EXISTING worker thread
    is digest-witnessed -> UNVERIFIABLE (never a false VERIFIED)."""

    class _DrawModel(nn.Module):
        def __init__(self, worker: Any) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.holder: Any = holder_type(np.random.default_rng(777))
            self._worker = worker

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            v = self._worker.run(lambda: float(self.holder.rng.random()))
            h = self.lin(x)
            return h * 2.0 if v < 0.5 else h * 3.0

    result = _roundtrip(_DrawModel(preexisting_worker), torch.randn(2, 4), tmp_path)
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE


@pytest.mark.parametrize(
    "holder_type", [SequenceSub, UserDictSub, SlotsSequenceSub], ids=lambda t: t.__name__
)
def test_undrawn_container_subclass_generator_stays_verified(
    holder_type: type, tmp_path: Path
) -> None:
    """Over-trigger pin: a container-subclass-held generator present but NEVER drawn stays
    VERIFIED (the digest sees no state change)."""

    class _UndrawnModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.holder: Any = holder_type(np.random.default_rng(3))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x).relu()

    result = _roundtrip(_UndrawnModel(), torch.randn(2, 4), tmp_path)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


def test_oneshot_iterator_attribute_on_subclass_not_consumed(tmp_path: Path) -> None:
    """A one-shot python generator stored on a container-subclass holder is NEVER iterated by the
    sweep (the ``Collection`` (Sized) gate, not bare ``Iterable``): the marker stays False and the
    model stays VERIFIED."""

    marker = {"consumed": False}

    def _oneshot() -> Any:
        marker["consumed"] = True
        yield 1

    class _Holder(SequenceSub):
        def __init__(self) -> None:
            super().__init__(np.random.default_rng(0))
            self.it = _oneshot()  # a bare one-shot iterator attribute

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.holder: Any = _Holder()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x).relu()

    result = _roundtrip(_Model(), torch.randn(2, 4), tmp_path)
    assert marker["consumed"] is False
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
