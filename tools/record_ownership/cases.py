"""Live traces the ownership audit runs on.

Each case returns the roots to walk: one trace, or a trace and the object
derived from it (a fork, a loaded copy). The cases cover the states in which
the trace is known to gain a holder: lookup doors read, facets read, a
backward capture, registered buffers written during the forward, a
conditional, user annotations, a fork, and the three copies (file, pickle,
deepcopy), plus the state pickle itself carries.
"""

from __future__ import annotations

import contextlib
import copy
import gc
import pickle
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

Roots = list[tuple[str, Any]]


class DoorModel(nn.Module):
    """One module per record kind: a shared module, buffers, params, a branch."""

    def __init__(self) -> None:
        """Build the layers."""

        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.shared = nn.Linear(4, 4)
        self.bn = nn.BatchNorm1d(4)
        self.fc2 = nn.Linear(4, 2)
        self.emb = nn.Embedding(3, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the shared linear twice, branch once, normalise."""

        x = x + self.emb(torch.tensor([0, 1, 2]))
        x = torch.relu(self.fc1(x))
        x = self.shared(x)
        x = torch.relu(self.shared(x))
        if x.sum() > -1e9:
            x = x * 2
        x = self.bn(x)
        return self.fc2(x)


def _trace(train: bool = False, **options: Any) -> Any:
    """Trace the door model on a fixed input."""

    import torchlens as tl

    torch.manual_seed(0)
    model = DoorModel()
    model.train(train)
    return tl.trace(model, torch.randn(3, 4), **options)


def _read_doors(trace: Any) -> None:
    """Take one record through every lookup door, then drop it."""

    for door in ("ops", "layers", "modules", "module_calls", "params", "buffers"):
        accessor = getattr(trace, door)
        for key in list(accessor.keys()) if hasattr(accessor, "keys") else []:
            accessor[key]
        list(accessor)
    for label in list(trace.layers.keys()):
        trace[label]


def _read_facets(trace: Any) -> None:
    """Read the facet view of every op and module."""

    for op in list(trace.ops):
        _ = op.facets
    for module in list(trace.modules):
        if len(module.calls) == 1:
            _ = module.facets
    for call in list(trace.module_calls):
        _ = call.facets


def case_plain() -> Roots:
    """A fresh trace nobody has read."""

    return [("Trace", _trace())]


def case_save_none() -> Roots:
    """A fresh trace that saved no activations."""

    return [("Trace", _trace(save="none"))]


def case_doors_read() -> Roots:
    """Every lookup door read once."""

    trace = _trace()
    _read_doors(trace)
    return [("Trace", trace)]


def case_facets_read() -> Roots:
    """Facets read on every op and module."""

    trace = _trace()
    _read_facets(trace)
    return [("Trace", trace)]


def case_readers() -> Roots:
    """The bulk readers run: ``repr``, ``to_pandas`` and the summary."""

    trace = _trace()
    repr(trace)
    trace.to_pandas()
    for op in list(trace.ops):
        repr(op)
    return [("Trace", trace)]


def case_alias_lookup() -> Roots:
    """Lookups by a bare layer label, which build the alias indexes of the ops door."""

    trace = _trace()
    for label in list(trace.layers.keys()):
        with contextlib.suppress(LookupError, ValueError):
            trace.ops[label]
    return [("Trace", trace)]


def case_annotated() -> Roots:
    """User annotations written on every op, and one on the trace itself."""

    trace = _trace()
    for label in list(trace.ops.keys()):
        trace.annotate(label, data={"note": label})
    trace.annotations["note"] = "kept by the user"
    _read_doors(trace)
    return [("Trace", trace)]


def case_buffers_written() -> Roots:
    """Train mode: batch norm writes its registered buffers during the forward."""

    trace = _trace(train=True)
    _read_doors(trace)
    return [("Trace", trace)]


def case_backward() -> Roots:
    """A backward capture, with the gradient doors read."""

    trace = _trace()
    trace.backward(trace.output.sum())
    _read_doors(trace)
    list(trace.autograd.grad_fns)
    for op in list(trace.ops):
        getattr(op, "grad", None)
    return [("Trace", trace)]


def case_fork() -> Roots:
    """A fork taken after the parent's doors were read, then read itself."""

    trace = _trace()
    _read_doors(trace)
    fork = trace.fork()
    _read_doors(fork)
    return [("Trace", trace), ("Trace", fork)]


def case_fork_cold() -> Roots:
    """A fork taken before any read; only the fork is held."""

    fork = _trace().fork()
    gc.collect()
    return [("Trace", fork)]


def case_loaded() -> Roots:
    """A trace saved to a file and loaded back."""

    import torchlens as tl

    trace = _trace()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "trace.tlspec"
        trace.save(path)
        loaded = tl.load(path)
    _read_doors(loaded)
    return [("Trace", loaded)]


def case_pickled() -> Roots:
    """A pickle round trip of a trace with a backward capture."""

    trace = _trace()
    trace.backward(trace.output.sum())
    restored = pickle.loads(pickle.dumps(trace))
    _read_doors(restored)
    return [("Trace", restored)]


def case_deepcopied() -> Roots:
    """A deep copy of a read trace."""

    trace = _trace()
    _read_doors(trace)
    _read_facets(trace)
    clone = copy.deepcopy(trace)
    return [("Trace", trace), ("Trace", clone)]


def _state_root(value: Any) -> Any:
    """Wrap a pickle state so the walk names its keys as attributes."""

    state = value[2] if isinstance(value, tuple) and len(value) > 2 else value
    if isinstance(state, dict):
        return SimpleNamespace(**{str(key): item for key, item in state.items()})
    return value


def case_pickle_state() -> Roots:
    """What pickle carries: the trace's state, and the state of one record per kind.

    A deep copy is a pickle round trip, so this is its state as well.
    """

    from . import record_types

    trace = _trace()
    trace.backward(trace.output.sum())
    _read_doors(trace)
    _read_facets(trace)
    roots: Roots = [("TraceState", _state_root(trace.__getstate__()))]
    doors = (trace.ops, trace.layers, trace.modules, trace.module_calls, trace.params)
    samples = [next(iter(door)) for door in doors]
    samples += [next(iter(trace.buffers)), next(iter(trace.autograd.grad_fns))]
    kinds = record_types()
    for record in samples:
        roots.append((f"{kinds[type(record)]}State", _state_root(record.__reduce_ex__(4))))
    return roots


CASES: dict[str, Callable[[], Roots]] = {
    "plain": case_plain,
    "save_none": case_save_none,
    "doors_read": case_doors_read,
    "facets_read": case_facets_read,
    "readers": case_readers,
    "alias_lookup": case_alias_lookup,
    "annotated": case_annotated,
    "buffers_written": case_buffers_written,
    "backward": case_backward,
    "fork": case_fork,
    "fork_cold": case_fork_cold,
    "loaded": case_loaded,
    "pickled": case_pickled,
    "deepcopied": case_deepcopied,
    "pickle_state": case_pickle_state,
}
