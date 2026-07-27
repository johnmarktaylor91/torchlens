"""Round-45 hon1_1 immunizer -- structural model-held-generator inventory by container PROTOCOL.

r42 corr2_1 promised to catch a generator the model itself HOLDS (even drawn on a pre-existing,
non-hooked thread). r44 hon1_1 found the container recursion incomplete: the sweep descended only
``list`` / ``tuple`` / ``set`` / ``frozenset`` / ``dict``, so a generator inside a ``deque`` /
``ChainMap`` / ``UserList`` / ``UserDict`` / ``queue.Queue`` fell through -> false ``VERIFIED`` +
false ``ATTESTED``.

r45 descends by container PROTOCOL: every ``collections.abc.Mapping`` (keys+values) and every
non-leaf ``collections.abc.Collection`` (elements), gated on ``Collection`` (Sized) NOT bare
``Iterable`` so a one-shot generator/iterator attribute is never consumed. Safe queues
(``Queue`` / ``LifoQueue`` / ``PriorityQueue``) contribute a non-mutating snapshot of their
internal ``.queue`` deque; an opaque queue (``SimpleQueue`` / ``mp.Queue``) fails closed to
``inventory_opaque_container`` (INCOMPLETE) rather than proving absence.

This immunizer parametrizes EVERY container kind holding a model-owned numpy ``Generator`` drawn
on a PRE-EXISTING worker (all must ceiling), pins the one-shot-iterator non-consumption safety,
defines the custom ``Sequence`` / ``Mapping`` classes IN the test (a by-name ``deque``/``Queue``
implementation stays RED), and pins the over-trigger controls (undrawn generators, tensor-only
containers, deterministic model) VERIFIED.
"""

from __future__ import annotations

import collections
import collections.abc as abc
import queue
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
from torchlens.runnable import NumericAttestationStatus, PathFaithfulness

_CAP = dict(intervention_ready=True, capture_container_structure=True, cache=False)


def _capture(model: nn.Module, x: torch.Tensor) -> tl.Trace:
    return tl.trace(model, x, capture=CaptureOptions(random_seed=1, **_CAP))


def _roundtrip(model: nn.Module, x: torch.Tensor, tmp: Path) -> tuple[tl.Trace, tl.RunResult]:
    trace = _capture(model, x)
    path = tmp / "r45rng.tlspec"
    shutil.rmtree(path, ignore_errors=True)
    trace.save(path, level="runnable", include_weights=True, include_activations=True)
    result = tl.load(path).run(inputs=x)
    return trace, result


class _PreexistingWorker:
    """A worker thread started BEFORE any capture window (a non-hooked, foreign thread)."""

    def __init__(self) -> None:
        self.jobs: "queue.Queue[Any]" = queue.Queue()
        self.results: "queue.Queue[Any]" = queue.Queue()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while True:
            job = self.jobs.get()
            if job is None:
                return
            try:
                self.results.put(("ok", job()))
            except BaseException as exc:  # noqa: BLE001 - relayed to the test thread
                self.results.put(("err", exc))

    def run(self, job: Callable[[], Any]) -> Any:
        self.jobs.put(job)
        kind, value = self.results.get()
        if kind == "err":
            raise value
        return value

    def stop(self) -> None:
        self.jobs.put(None)


@pytest.fixture
def preexisting_worker() -> Any:
    worker = _PreexistingWorker()
    try:
        yield worker
    finally:
        worker.stop()


# Custom ABC containers DEFINED IN THE TEST -- a by-name ``deque``/``Queue`` implementation that
# does not descend arbitrary ``Collection`` / ``Mapping`` leaves these RED.
class _CustomSeq(abc.Sequence):  # type: ignore[type-arg]
    def __init__(self, items: Any) -> None:
        self._items = list(items)

    def __getitem__(self, i: Any) -> Any:
        return self._items[i]

    def __len__(self) -> int:
        return len(self._items)


class _CustomMap(abc.Mapping):  # type: ignore[type-arg]
    def __init__(self, d: Any) -> None:
        self._d = dict(d)

    def __getitem__(self, k: Any) -> Any:
        return self._d[k]

    def __iter__(self) -> Any:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)


_NamedTup = collections.namedtuple("_NamedTup", ["g"])


def _make_container(kind: str, gen: Any) -> tuple[Any, Callable[[Any], Any]]:
    """Return ``(holder, draw)`` where ``holder`` holds ``gen`` and ``draw(holder)`` draws it."""

    if kind == "deque":
        return collections.deque([gen]), lambda h: h[0].random()
    if kind == "ordereddict":
        return collections.OrderedDict(g=gen), lambda h: h["g"].random()
    if kind == "counter":
        c: Any = collections.Counter()
        c["g"] = gen
        return c, lambda h: h["g"].random()
    if kind == "defaultdict":
        d: Any = collections.defaultdict(lambda: None)
        d["g"] = gen
        return d, lambda h: h["g"].random()
    if kind == "chainmap":
        return collections.ChainMap({"g": gen}), lambda h: h["g"].random()
    if kind == "userlist":
        return collections.UserList([gen]), lambda h: h[0].random()
    if kind == "userdict":
        return collections.UserDict({"g": gen}), lambda h: h["g"].random()
    if kind == "namedtuple":
        return _NamedTup(g=gen), lambda h: h[0].random()
    if kind == "custom_seq":
        return _CustomSeq([gen]), lambda h: h[0].random()
    if kind == "custom_map":
        return _CustomMap({"g": gen}), lambda h: h["g"].random()
    if kind == "queue":
        q: Any = queue.Queue()
        q.put(gen)
        return q, lambda h: h.queue[0].random()
    if kind == "lifoqueue":
        lq: Any = queue.LifoQueue()
        lq.put(gen)
        return lq, lambda h: h.queue[0].random()
    raise AssertionError(kind)


class _ContainerGenModel(nn.Module):
    """Holds a numpy Generator ONLY inside ``holder`` and draws it on a pre-existing worker."""

    def __init__(self, kind: str, worker: Any) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.holder, self._draw = _make_container(kind, np.random.default_rng(777))
        self._worker = worker

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self._worker.run(lambda: float(self._draw(self.holder)))
        h = self.lin(x)
        return h * 2.0 if v < 0.5 else h * 3.0


_INSPECTABLE_CONTAINERS = [
    "deque",
    "ordereddict",
    "counter",
    "defaultdict",
    "chainmap",
    "userlist",
    "userdict",
    "namedtuple",
    "custom_seq",
    "custom_map",
    "queue",
    "lifoqueue",
]


@pytest.mark.parametrize("kind", _INSPECTABLE_CONTAINERS)
def test_container_held_generator_drawn_on_worker_ceilings(
    kind: str, preexisting_worker: Any, tmp_path: Path
) -> None:
    """Every inspectable container holding a model-owned generator drawn on a pre-existing worker
    is inventoried -> ``model_attribute_generator`` -> UNVERIFIABLE + not ATTESTED."""

    trace, result = _roundtrip(
        _ContainerGenModel(kind, preexisting_worker), torch.randn(2, 4), tmp_path
    )
    assert "model_attribute_generator" in getattr(trace, "_runnable_host_rng_channels", ()), (
        f"{kind}: model-held generator draw was not inventoried"
    )
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is not NumericAttestationStatus.ATTESTED


def test_simplequeue_held_generator_fails_closed_opaque(
    preexisting_worker: Any, tmp_path: Path
) -> None:
    """A ``queue.SimpleQueue`` has no non-mutating buffer snapshot -> fail closed to
    ``inventory_opaque_container`` -> UNVERIFIABLE (never a silent no-consumption VERIFIED)."""

    class _SimpleQueueModel(nn.Module):
        def __init__(self, worker: Any) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.holder: Any = queue.SimpleQueue()
            self.holder.put(np.random.default_rng(777))
            self._worker = worker

        def _draw(self) -> float:
            g = self.holder.get()
            r = float(g.random())
            self.holder.put(g)
            return r

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            v = self._worker.run(self._draw)
            h = self.lin(x)
            return h * 2.0 if v < 0.5 else h * 3.0

    _trace, result = _roundtrip(_SimpleQueueModel(preexisting_worker), torch.randn(2, 4), tmp_path)
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is not NumericAttestationStatus.ATTESTED


@pytest.mark.parametrize("queue_factory", [queue.SimpleQueue, queue.Queue, queue.LifoQueue])
def test_empty_opaque_queue_stays_verified(queue_factory: Any, tmp_path: Path) -> None:
    """r47 hon1_2 precision pin: a deterministic model holding an EMPTY opaque queue is
    NON-MUTATINGLY provably empty (``empty()``/``qsize()``), so it CANNOT hold a generator and must
    stay VERIFIED + ATTESTED -- no ``inventory_opaque_container`` over-trigger. The non-empty
    opaque-queue fail-closed residual (above) is unchanged."""

    class _EmptyQueueModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.holder: Any = queue_factory()  # created empty, never populated

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x).relu()

    trace, result = _roundtrip(_EmptyQueueModel(), torch.randn(2, 4), tmp_path)
    assert "inventory_opaque_container" not in getattr(
        trace, "_runnable_rng_monitor_uncertain_detail", ()
    ), "an empty opaque queue must not fail closed to inventory_opaque_container"
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED


def test_oneshot_generator_attribute_is_not_consumed(tmp_path: Path) -> None:
    """A model attribute that is a one-shot python generator is NEVER iterated by the sweep (the
    ``Collection`` gate, not ``Iterable``): the side-effect marker must not fire and the model
    stays VERIFIED (a bare uninspectable iterator is the documented residual, not a false read)."""

    marker = {"consumed": False}

    def _oneshot() -> Any:
        marker["consumed"] = True
        yield 1

    class _OneShotModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.it = _oneshot()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x).relu()

    _trace, result = _roundtrip(_OneShotModel(), torch.randn(2, 4), tmp_path)
    assert marker["consumed"] is False
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


@pytest.mark.parametrize("kind", ["deque", "chainmap", "userlist", "queue", "namedtuple"])
def test_undrawn_container_generator_stays_verified(kind: str, tmp_path: Path) -> None:
    """Over-trigger pin: a container-held generator that is present but NEVER drawn stays
    VERIFIED with no ``model_attribute_generator`` channel."""

    class _UndrawnModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.holder, _draw = _make_container(kind, np.random.default_rng(3))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x).relu()

    trace, result = _roundtrip(_UndrawnModel(), torch.randn(2, 4), tmp_path)
    assert "model_attribute_generator" not in getattr(trace, "_runnable_host_rng_channels", ())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


@pytest.mark.parametrize("kind", ["deque", "queue"])
def test_tensor_only_container_stays_verified(kind: str, tmp_path: Path) -> None:
    """Over-trigger pin: a container holding only tensors is not a host-RNG channel and stays
    VERIFIED (descending it enqueues tensors, which are leaves)."""

    class _TensorContainerModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            if kind == "deque":
                self.holder: Any = collections.deque([torch.ones(4), torch.zeros(4)])
            else:
                self.holder = queue.Queue()
                self.holder.put(torch.ones(4))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x).relu()

    trace, result = _roundtrip(_TensorContainerModel(), torch.randn(2, 4), tmp_path)
    assert "model_attribute_generator" not in getattr(trace, "_runnable_host_rng_channels", ())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


def test_deterministic_model_stays_verified(tmp_path: Path) -> None:
    """A host-RNG-free model with no containers stays VERIFIED + ATTESTED."""

    class _Det(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seq = nn.Sequential(*[nn.Linear(8, 8) for _ in range(6)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.seq(x)

    _trace, result = _roundtrip(_Det(), torch.randn(2, 8), tmp_path)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED
