"""Hardening tests for loop/recurrence grouping (round-21, follow-up 2).

Locks the fixes for the two round-20 review findings the first r21 loop fixer did
not cover:

* Exact parameter identity must not override module identity or non-tensor call
  semantics (review cluster 2). Two DISTINCT modules deliberately sharing one
  weight tensor (tied encoder/decoder) and one kernel applied with different
  ``padding``/``stride``/``dilation`` are different semantic sites, never a
  false recurrent layer. Genuine reuse of ONE module (ALBERT-style) and genuine
  variable-length recurrence must keep grouping.
* Rolled variable-length recurrence must not silently project pass-1
  shape/memory/FLOPs onto every pass (review cluster 4). Divergent passes publish
  honest aggregates (per-dimension range shapes, per-pass maxima) plus the
  explicit machine-readable marker ``annotations["varying_across_passes"]``,
  and expanding every rolled Layer through ``Layer.ops`` reconstructs the exact
  unrolled record multiset (conservation).
"""

from collections import Counter

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from support.structural_grouper import legacy_structural_grouper  # noqa: F401

from torchlens import trace as trace_fn
from torchlens.validation._invariants import check_metadata_invariants

# This suite drives the structural grouper (grouper_v1), which serves captures
# without call-site evidence; torch captures group by call site
# (tests/test_grouping_call_sites.py).
pytestmark = pytest.mark.usefixtures("legacy_structural_grouper")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _module_addresses(op) -> tuple[str, ...]:
    """Return the op's containing-module address stack, addresses only.

    Finalized ops store module calls as ``"address:pass"`` strings (postprocess
    Step 7 sees the raw ``(address, pass)`` tuples); handle both forms.
    """
    addresses = []
    for module_pass in getattr(op, "modules", None) or ():
        if isinstance(module_pass, str):
            addresses.append(module_pass.rsplit(":", 1)[0])
        else:
            addresses.append(module_pass[0])
    return tuple(addresses)


def _op_record(op) -> tuple:
    """Structural record of one op for rolled/unrolled conservation checks."""
    return (
        op.label,
        tuple(op.parents),
        tuple(op.children),
        _module_addresses(op),
        tuple(op.shape) if isinstance(op.shape, (tuple, list)) else op.shape,
        str(op.dtype),
    )


# ---------------------------------------------------------------------------
# Fixtures: tied sites, arg-divergent convs, genuine recurrence controls
# ---------------------------------------------------------------------------


class _TiedEncoderDecoder(nn.Module):
    """Two DISTINCT modules deliberately sharing one weight tensor."""

    def __init__(self, direct_chain: bool = False) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 4, bias=False)
        self.decoder = nn.Linear(4, 4, bias=False)
        self.decoder.weight = self.encoder.weight
        self.direct_chain = direct_chain

    def forward(self, x):
        hidden = self.encoder(x)
        if not self.direct_chain:
            hidden = torch.relu(hidden)
        return self.decoder(hidden)


class _SharedKernelConvs(nn.Module):
    """One kernel applied with divergent non-tensor call structure."""

    def __init__(self, axis: str) -> None:
        super().__init__()
        self.kernel = nn.Parameter(torch.randn(1, 1, 3, 3))
        self.axis = axis

    def forward(self, x):
        if self.axis == "padding":
            a = F.conv2d(x, self.kernel, padding=0)
            b = F.conv2d(x, self.kernel, padding=1)
        elif self.axis == "stride":
            a = F.conv2d(x, self.kernel, stride=1)
            b = F.conv2d(x, self.kernel, stride=2)
        else:  # dilation
            a = F.conv2d(x, self.kernel, dilation=1)
            b = F.conv2d(x, self.kernel, dilation=2)
        return a.sum() + b.sum()


class _ShrinkingRecurrent(nn.Module):
    """Genuine variable-length recurrence: activations shrink each step."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        for _ in range(3):
            x = torch.relu(self.cell(x))
            x = x[:-1]
        return x


class _GrowingRecurrent(nn.Module):
    """Genuine variable-length recurrence where the MAX is not pass 1."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        for _ in range(3):
            x = torch.relu(self.cell(x))
            x = torch.cat([x, x[-1:]], dim=0)
        return x


class _FixedShapeLoop(nn.Module):
    """Uniform-pass control: reconciliation must leave it byte-identical."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        for _ in range(3):
            x = torch.relu(self.cell(x))
        return x


class _FusedLSTMLoop(nn.Module):
    """Multi-output fused module called repeatedly (conservation fixture)."""

    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.LSTM(4, 4, batch_first=True)

    def forward(self, x):
        out, _state = self.rnn(x)
        out, _state = self.rnn(out)
        out, _state = self.rnn(out)
        return out


# ---------------------------------------------------------------------------
# review test #2 -- weight-tied distinct modules retain distinct identity
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# review test #3 -- non-tensor call structure is part of recurrence identity
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/_layer_accessors.py.
def test_growing_recurrence_aggregate_is_not_call_1() -> None:
    """When passes GROW, the aggregate must differ from call 1's values."""
    torch.manual_seed(0)
    traced = trace_fn(_GrowingRecurrent(), torch.randn(2, 4))
    layer = traced._layer_logs["linear_1_1"]
    # OpAccessor get/[] are 0-based positional (C02 basis fix, lovely bug 27).
    first_pass = layer.ops.get(0)
    last_pass = layer.ops.get(layer.num_passes - 1)
    assert int(first_pass.cost.activation_memory) < int(last_pass.cost.activation_memory)
    assert int(layer.cost.activation_memory) == int(last_pass.cost.activation_memory)
    assert tuple(layer.shape) != tuple(first_pass.shape)
    assert check_metadata_invariants(traced)


def test_uniform_multi_pass_layer_left_untouched() -> None:
    """Fixed-shape loops keep plain pass values and carry NO varying marker."""
    torch.manual_seed(0)
    traced = trace_fn(_FixedShapeLoop(), torch.randn(4, 4))
    layer = traced._layer_logs["linear_1_1"]
    assert layer.num_passes == 3
    assert tuple(layer.shape) == (4, 4)
    assert "varying_across_passes" not in layer.annotations
    assert check_metadata_invariants(traced)


# ---------------------------------------------------------------------------
# review test #10 -- rolled/unrolled conservation and honest rolled rendering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_factory, model_input",
    [
        (_ShrinkingRecurrent, torch.randn(4, 4)),
        (_TiedEncoderDecoder, torch.randn(1, 4)),
        (_FixedShapeLoop, torch.randn(4, 4)),
        (lambda: _FusedLSTMLoop(), torch.randn(1, 5, 4)),
    ],
    ids=["ragged", "tied", "fixed", "fused_lstm"],
)
def test_rolled_unrolled_conservation(model_factory, model_input) -> None:
    """Expanding every rolled Layer through Layer.ops reconstructs the unrolled trace.

    The multiset of per-op structural records (label, parents, children, module
    call, shape, dtype) reached through ``Trace._layer_logs[*].ops`` must equal
    the unrolled ``layer_list`` exactly: no op dropped, none duplicated, and no
    rolled aggregate replacing per-pass truth.
    """
    torch.manual_seed(0)
    traced = trace_fn(model_factory(), model_input)

    unrolled_records = Counter(_op_record(op) for op in traced._layer_list)
    rolled_records: Counter = Counter()
    for layer in traced._layer_logs.values():
        for op in layer.ops.values():
            rolled_records[_op_record(op)] += 1

    assert rolled_records == unrolled_records
    assert check_metadata_invariants(traced)


# ---------------------------------------------------------------------------
# Deep-hunt L1: dict KEYS must get the same identity-repr guard as values
# ---------------------------------------------------------------------------


class _OpaqueKey:
    """Non-primitive dict key with the default address-bearing ``repr``."""


# Kept by shrink-r4 (per-id subset rule): sole coverage of _postprocess/loop_detection.py.
def test_arg_signature_ignores_object_key_identity() -> None:
    """Object-identity dict keys never split a genuine recurrence signature.

    Deep-hunt L1: ``_append_signature_tokens`` guarded VALUES against
    identity-based reprs but emitted dict KEYS verbatim (``key!r``, sorted by
    ``repr``), so a fresh non-primitive key per call (``cfg={SomeObject(): 1}``)
    gave every pass a different, address-bearing (ASLR-varying) signature and
    silently ungrouped real recurrence. Two structurally identical calls with
    distinct key objects must share one signature; primitive key CONTENT must
    still discriminate.
    """
    from types import SimpleNamespace

    from torchlens._postprocess.loop_detection import _structural_arg_signature

    def _op(key: object) -> SimpleNamespace:
        return SimpleNamespace(non_tensor_pos_args=(), non_tensor_kwargs={"cfg": {key: 1}})

    assert _structural_arg_signature(_op(_OpaqueKey())) == _structural_arg_signature(
        _op(_OpaqueKey())
    )
    assert _structural_arg_signature(_op("alpha")) != _structural_arg_signature(_op("beta"))


# ---------------------------------------------------------------------------
# Deep-hunt L2: the argsig split must propagate into equivalent_labels
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _postprocess/loop_detection.py.
def test_grouping_graph_equivalent_labels_respect_argsig_split(monkeypatch) -> None:
    """The neutral feed never hands a split node its unsplit class as seeds.

    Deep-hunt L2: ``equivalence_key`` was overridden with the argsig-split key
    while ``equivalent_labels`` still carried the UNSPLIT capture-time class,
    so ``_expand_isomorphic_subgraphs`` seeded every split key's expansion from
    ALL original members -- foreign-argsig subgraphs contributed adjacency and
    parameter evidence, and each split key re-ran a full expansion over the
    same mixed seed set. The neutral contract requires every member of a
    node's ``equivalent_labels`` to share that node's ``equivalence_key``.
    """
    import torchlens._postprocess.loop_detection as loop_detection

    captured: dict[str, object] = {}
    real_group = loop_detection.group_recurrent_nodes

    def _spy(graph):
        captured["graph"] = graph
        return real_group(graph)

    monkeypatch.setattr(loop_detection, "group_recurrent_nodes", _spy)
    torch.manual_seed(0)
    trace_fn(_SharedKernelConvs("padding"), torch.randn(1, 1, 8, 8))

    graph = captured["graph"]
    conv_nodes = [
        node for node in graph.nodes.values() if node.func_name == "conv2d" and node.uses_params
    ]
    assert len(conv_nodes) == 2
    assert conv_nodes[0].equivalence_key != conv_nodes[1].equivalence_key

    for node in graph.nodes.values():
        for member in node.equivalent_labels:
            assert graph.nodes[member].equivalence_key == node.equivalence_key, (
                f"{node.label} carries foreign-key seed {member}: "
                f"{graph.nodes[member].equivalence_key} != {node.equivalence_key}"
            )
