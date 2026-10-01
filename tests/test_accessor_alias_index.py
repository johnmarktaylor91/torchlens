"""Result-identity coverage for the Trace accessor alias reverse indexes.

``TraceOpAccessor``/``TraceModuleCallAccessor``/``TraceGradFnCallAccessor`` resolve
non-exact keys through reverse indexes built once per accessor instead of scanning
every item per lookup. These tests pin the indexed answers to the scanning answers
for EVERY alias, miss, and ambiguous key of several models -- including the
``AmbiguousOpLookupError`` message verbatim -- and pin the ``_trace_stats`` edge
counters to unmemoized reference computations.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.data_classes._trace_accessors import (
    TraceGradFnCallAccessor,
    TraceModuleCallAccessor,
    TraceOpAccessor,
)
from torchlens.data_classes.op import Op


class Nested(nn.Module):
    """Nested model with a BatchNorm buffer and a literal-arg op."""

    def __init__(self) -> None:
        """Build the submodules."""

        super().__init__()
        self.a = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.BatchNorm1d(8))
        self.b = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the nested stack.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return self.b(self.a(x) + 1.0)


class Recurrent(nn.Module):
    """Single module reused across passes, giving multi-pass (ambiguous) layers."""

    def __init__(self, num_passes: int = 4) -> None:
        """Build the reused cell.

        Parameters
        ----------
        num_passes:
            Number of times the cell is called.
        """

        super().__init__()
        self.cell = nn.Linear(6, 6)
        self.num_passes = num_passes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the cell repeatedly.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        for _ in range(self.num_passes):
            x = torch.relu(self.cell(x))
        return x


class Branchy(nn.Module):
    """Shared submodule, registered buffer, and a concatenating join."""

    def __init__(self) -> None:
        """Build the submodules and buffer."""

        super().__init__()
        self.shared = nn.Sequential(nn.Linear(5, 5), nn.ReLU())
        self.register_buffer("scale", torch.ones(5) * 2)
        self.out = nn.Linear(10, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run both branches through the shared stack.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        left = self.shared(x) * self.scale
        right = self.shared(x + 0.5)
        return self.out(torch.cat([left, right], dim=1))


def _traces() -> list[tuple[str, nn.Module, tl.Trace]]:
    """Capture the probe traces.

    Returns
    -------
    list[tuple[str, nn.Module, tl.Trace]]
        ``(name, model, trace)`` triples the caller must release and clean up.
    """

    torch.manual_seed(0)
    cases: list[tuple[str, nn.Module, torch.Tensor]] = [
        ("nested", Nested(), torch.randn(4, 8)),
        ("recurrent", Recurrent(), torch.randn(3, 6)),
        ("branchy", Branchy(), torch.randn(2, 5)),
    ]
    captured = []
    for name, model, x in cases:
        model.eval()
        captured.append((name, model, tl.trace(model, x)))
    return captured


def _outcome(call: Any) -> Any:
    """Return a comparable outcome for a lookup, exception text included.

    Parameters
    ----------
    call:
        Zero-argument callable performing one lookup.

    Returns
    -------
    Any
        The resolved object, or ``"<type>: <message>"`` when the lookup raised.
    """

    try:
        return call()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _op_probe_keys(accessor: TraceOpAccessor) -> list[str]:
    """Return every alias, stored relationship label, and miss key to probe.

    Parameters
    ----------
    accessor:
        Op accessor under test.

    Returns
    -------
    list[str]
        Deduplicated probe keys in discovery order.
    """

    keys: list[Any] = []
    for op in accessor:
        keys += [
            op.label,
            op.label_short,
            op._label_raw,
            op.raw_label,
            op.layer_label,
            op.layer_label_short,
        ]
        keys += list(op.children) + list(op.parents)
    keys += ["nope", "", ":", "1", "self", "linear", "relu_1_1:99"]
    return list(dict.fromkeys(key for key in keys if isinstance(key, str)))


def _reference_module_call_match(accessor: TraceModuleCallAccessor, key: str) -> Any:
    """Resolve a ModuleCall key by scanning, mirroring the pre-index behavior."""

    parent_matches = [call for call in accessor._list if key == getattr(call, "address", None)]
    if len(parent_matches) == 1:
        return parent_matches[0]
    if len(parent_matches) > 1:
        from torchlens._errors import AmbiguousOpLookupError

        raise AmbiguousOpLookupError(
            f"Module '{key}' has {len(parent_matches)} calls. Use a 0-based integer "
            f"position or a call-qualified label like '{key}:1'."
        )
    return None


def _reference_grad_fn_call_match(accessor: TraceGradFnCallAccessor, key: str) -> Any:
    """Resolve a GradFnCall key by scanning, mirroring the pre-index behavior."""

    parent_matches = [call for call in accessor._list if key == getattr(call, "label", None)]
    if len(parent_matches) == 1:
        return parent_matches[0]
    if len(parent_matches) > 1:
        from torchlens._errors import AmbiguousOpLookupError

        raise AmbiguousOpLookupError(
            f"GradFn '{key}' has {len(parent_matches)} calls. Use a 0-based integer "
            f"position or a call-qualified label like '{key}:1'."
        )
    return None


@pytest.mark.smoke
def test_op_alias_index_matches_scan_for_every_key() -> None:
    """Indexed Op resolution equals the scan for every alias, miss, and ambiguity."""

    probed = {"hit": 0, "miss": 0, "ambiguous": 0}
    for name, model, trace in _traces():
        try:
            accessor = trace.ops
            keys = _op_probe_keys(accessor)
            assert len(keys) > 10, name
            for key in keys:
                indexed = _outcome(lambda k=key, a=accessor: a._resolve_substring(k))
                scanned = _outcome(lambda k=key, a=accessor: a._resolve_substring_by_scan(k))
                if isinstance(scanned, str):
                    # Exception text, including the ambiguity message, verbatim.
                    assert indexed == scanned, (name, key)
                    probed["ambiguous"] += 1
                elif scanned is None:
                    assert indexed is None, (name, key)
                    probed["miss"] += 1
                else:
                    assert indexed is scanned, (name, key)
                    probed["hit"] += 1
                # ``__getitem__`` must agree with both.
                via_getitem = _outcome(lambda k=key, a=accessor: a[k])
                if isinstance(indexed, Op):
                    assert via_getitem is indexed, (name, key)
                elif isinstance(indexed, str):
                    assert via_getitem == indexed, (name, key)
                else:
                    assert isinstance(via_getitem, str), (name, key)
        finally:
            tl.release_model(model)
            trace.cleanup()
    assert probed["hit"] > 0
    assert probed["miss"] > 0
    assert probed["ambiguous"] > 0


@pytest.mark.smoke
def test_module_call_and_grad_fn_call_indexes_match_scan() -> None:
    """Indexed ModuleCall/GradFnCall resolution equals the scanning form."""

    torch.manual_seed(0)
    model = Recurrent(3)
    model.eval()
    x = torch.randn(3, 6, requires_grad=True)
    trace = tl.trace(
        model, x, capture=tl.options.CaptureOptions(layers_to_save="all", save_grads="all")
    )
    try:
        loss = trace[trace.output_layers[0]].out.sum()
        trace.log_backward(loss, retain_graph=True)
        trace.log_backward(loss, retain_graph=True)

        module_calls = trace.module_calls
        module_keys = [getattr(call, "address", None) for call in module_calls]
        module_keys += [getattr(call, "label", None) for call in module_calls]
        module_keys += ["nope", "self", "cell"]
        ambiguous_modules = 0
        for key in dict.fromkeys(k for k in module_keys if isinstance(k, str)):
            indexed = _outcome(lambda k=key: module_calls._resolve_substring(k))
            scanned = _outcome(lambda k=key: _reference_module_call_match(module_calls, k))
            if isinstance(scanned, str):
                assert indexed == scanned, key
                ambiguous_modules += 1
            else:
                assert indexed is scanned, key
        assert ambiguous_modules > 0

        grad_fn_calls = trace.grad_fn_calls
        assert len(grad_fn_calls) > 0
        grad_fn_keys = [getattr(call, "label", None) for call in grad_fn_calls]
        grad_fn_keys += [getattr(call, "call_label", None) for call in grad_fn_calls]
        grad_fn_keys += ["nope", "AddBackward0"]
        ambiguous_grad_fns = 0
        for key in dict.fromkeys(k for k in grad_fn_keys if isinstance(k, str)):
            indexed = _outcome(lambda k=key: grad_fn_calls._resolve_substring(k))
            scanned = _outcome(lambda k=key: _reference_grad_fn_call_match(grad_fn_calls, k))
            if isinstance(scanned, str):
                assert indexed == scanned, key
                ambiguous_grad_fns += 1
            else:
                assert indexed is scanned, key
        assert ambiguous_grad_fns > 0
    finally:
        tl.release_model(model)
        trace.cleanup()


@pytest.mark.smoke
def test_edge_counters_match_unmemoized_reference() -> None:
    """Memoized edge counting equals the direct per-edge accessor resolution."""

    for name, model, trace in _traces():
        try:
            ops = trace.ops
            direct_edges = {
                (op.label, ops[child_label].label) for op in ops for child_label in op.children
            }
            compute_ops = trace.compute_ops
            compute_labels = {op.label for op in compute_ops}
            direct_compute = {
                (op.label, ops[child_label].label)
                for op in compute_ops
                for child_label in op.children
                if ops[child_label].label in compute_labels
            }
            direct_buffer = {
                (op.label, ops[child_label].label)
                for op in ops
                for child_label in op.children
                if op.is_buffer or ops[child_label].is_buffer
            }
            assert trace.num_edges == len(direct_edges), name
            assert trace.num_compute_edges == len(direct_compute), name
            assert trace.num_buffer_edges == len(direct_buffer), name
            # Repeat reads must be stable once the memo is warm.
            assert trace.num_edges == len(direct_edges), name
            assert trace.num_buffer_edges == len(direct_buffer), name
        finally:
            tl.release_model(model)
            trace.cleanup()


@pytest.mark.smoke
def test_alias_index_build_failure_falls_back_to_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreadable alias on a later Op degrades to the scan, not to an error.

    The scan can return before ever touching a later Op's labels, so eager index
    construction must not turn a lookup that used to succeed into a raise.

    Parameters
    ----------
    monkeypatch:
        Pytest patch helper.
    """

    torch.manual_seed(0)
    model = Nested()
    model.eval()
    trace = tl.trace(model, torch.randn(4, 8))
    try:
        ops = list(trace.ops)
        assert len(ops) > 2
        first, last = ops[0], ops[-1]
        probe_key = first.label_short
        assert isinstance(probe_key, str)

        original = Op.raw_label

        def raising_raw_label(op: Op) -> str:
            """Fail only for the final Op, mimicking an unreadable label slot."""

            if op is last:
                raise RuntimeError("unreadable raw_label")
            return original.fget(op)

        monkeypatch.setattr(Op, "raw_label", property(raising_raw_label))
        accessor = TraceOpAccessor(ops, trace.layer_num_calls)
        assert accessor._resolve_substring(probe_key) is first
        assert accessor._alias_index_unavailable is True
        assert accessor._alias_index is None
        # The degraded accessor stays degraded and stays correct.
        assert accessor._resolve_substring(probe_key) is first
        # A full-miss scan does reach the unreadable Op -- and must raise there
        # exactly as the pre-index scan did, rather than being masked.
        with pytest.raises(RuntimeError, match="unreadable raw_label"):
            accessor._resolve_substring("definitely-not-a-label")
        with pytest.raises(RuntimeError, match="unreadable raw_label"):
            accessor._resolve_substring_by_scan("definitely-not-a-label")
    finally:
        tl.release_model(model)
        trace.cleanup()
