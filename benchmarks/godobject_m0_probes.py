"""M0 measurement probes for the trace-core columnar re-plumbing.

Measures the evidence items the converged design gates on
(docs/reference/trace_core_design.md sections 4 and 7, wave M0):

- A2: retained journal bytes vs semantic-graph bytes on a large capture
  (journal-columnarization reopening condition: journal > 2x semantic store).
- A3: retained ``RecordContext`` count vs event count (O(events) retention?).
- Scale tiers: deep-retained structural bytes/op and objects/op at 1K and
  10K ops, payloads excluded and reported separately.
- Post-finish mutation inventory: which Op fields are written after
  ``_set_tracing_finished`` on ordinary + backward + intervention flows
  (this list seeds the overlay-policy / sanctioned-writer spec).

Run: python benchmarks/godobject_m0_probes.py
"""

from __future__ import annotations

import gc
import sys
import tracemalloc
from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes.op import Op

_SEED = 20260812


def _deep_size(root: Any, *, skip_tensors: bool = True) -> tuple[int, int, int]:
    """Return (structural_bytes, object_count, tensor_payload_bytes).

    Traverses ``__dict__``/``__slots__``/containers with an id-visited set.
    Tensor payload storage is counted separately (alias-deduplicated by
    storage id), never in the structural figure.
    """

    visited: set[int] = set()
    tensor_storages: dict[int, int] = {}
    structural = 0
    objects = 0
    stack = [root]
    while stack:
        obj = stack.pop()
        oid = id(obj)
        if oid in visited:
            continue
        visited.add(oid)
        if isinstance(obj, torch.Tensor):
            if skip_tensors:
                try:
                    storage = obj.untyped_storage()
                    tensor_storages[id(storage)] = storage.nbytes()
                except RuntimeError:
                    pass
                structural += sys.getsizeof(obj)
                objects += 1
                continue
        if isinstance(obj, (type, type(sys), type(_deep_size))):
            continue
        try:
            structural += sys.getsizeof(obj)
        except TypeError:
            continue
        objects += 1
        if isinstance(obj, dict):
            stack.extend(obj.keys())
            stack.extend(obj.values())
            continue
        if isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
            continue
        if isinstance(obj, (str, bytes, bytearray, int, float, complex, bool)):
            continue
        inst_dict = getattr(obj, "__dict__", None)
        if inst_dict is not None:
            stack.append(inst_dict)
        for klass in type(obj).__mro__:
            for slot in getattr(klass, "__slots__", ()):
                if slot in ("__dict__", "__weakref__"):
                    continue
                try:
                    stack.append(object.__getattribute__(obj, slot))
                except AttributeError:
                    continue
    return structural, objects, sum(tensor_storages.values())


class _ScaleModel(nn.Module):
    """Sequential linear+relu blocks scaled to a target op count."""

    def __init__(self, blocks: int) -> None:
        """Create ``blocks`` linear layers (2 ops per block + overhead)."""

        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(8, 8) for _ in range(blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Chain the blocks with relu."""

        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


def probe_scale(target_ops: int) -> None:
    """Report deep structural bytes/op and objects/op at one scale tier."""

    blocks = max(1, target_ops // 2)
    torch.manual_seed(_SEED)
    model = _ScaleModel(blocks)
    x = torch.zeros(1, 8)
    gc.collect()
    tracemalloc.start()
    trace = tl.trace(model, x)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    n_ops = len(trace.ops.keys())

    events = getattr(trace, "_capture_events", None)
    journal_bytes = journal_objects = 0
    if events is not None:
        journal_bytes, journal_objects, _ = _deep_size(events)

    # One traversal over the whole op family with a SHARED visited set, so
    # shared containers/labels count once (per-op sums overstate sharing).
    op_bytes, op_objects, payload = _deep_size(list(trace.ops.values()))

    print(f"[scale {target_ops}] ops={n_ops} capture_peak={peak / 1e6:.1f}MB")
    print(
        f"[scale {target_ops}] semantic op graph: {op_bytes / n_ops:.0f} B/op "
        f"structural, {op_objects / n_ops:.1f} objects/op, "
        f"payload {payload / 1e6:.1f}MB (alias-deduped)"
    )
    print(
        f"[scale {target_ops}] A2 journal: {journal_bytes / 1e6:.2f}MB "
        f"({journal_objects} objects) vs semantic {op_bytes / 1e6:.2f}MB "
        f"-> ratio {journal_bytes / max(op_bytes, 1):.2f}x"
    )

    from torchlens._ir.predicate import RecordContext

    gc.collect()
    contexts = sum(1 for obj in gc.get_objects() if isinstance(obj, RecordContext))
    n_events = getattr(events, "event_seq", None)
    print(f"[scale {target_ops}] A3 RecordContext retained: {contexts} (events={n_events})")
    del trace
    gc.collect()


def probe_mutation_inventory() -> None:
    """Inventory Op fields written after tracing is finished."""

    writes: dict[str, int] = {}
    original = Op.__setattr__

    def recording(self: Op, name: str, value: Any) -> None:
        if self._slot("_construction_done", False):
            trace_ref = self._slot("_source_trace_ref")
            trace = trace_ref() if trace_ref is not None else None
            if trace is not None and getattr(trace, "_tracing_finished", False):
                writes[name] = writes.get(name, 0) + 1
        original(self, name, value)

    Op.__setattr__ = recording  # type: ignore[method-assign]
    try:
        torch.manual_seed(_SEED)
        model = _ScaleModel(4)
        x = torch.zeros(1, 8, requires_grad=True)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = tl.trace(
                model,
                x,
                capture=tl.options.CaptureOptions(backward_ready=True),
                save_mode="reference",
            )
            loss = trace[trace.ops.keys()[-1]].out.sum()
            trace.log_backward(loss)

            torch.manual_seed(_SEED)
            ablated = tl.trace(
                _ScaleModel(4),
                torch.zeros(1, 8),
                intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
            )
            del ablated
            fork = trace.fork()
            del fork
    finally:
        Op.__setattr__ = original  # type: ignore[method-assign]
    print("[mutation inventory] post-finish Op writes (field: count):")
    for name in sorted(writes):
        print(f"  {name}: {writes[name]}")
    if not writes:
        print("  (none recorded)")


def main() -> None:
    """Run every probe."""

    for tier in (1_000, 10_000):
        probe_scale(tier)
    probe_mutation_inventory()


if __name__ == "__main__":
    main()
