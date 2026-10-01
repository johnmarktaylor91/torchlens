"""Hardening tests for loop/recurrence grouping (round-21, follow-up 2).

Locks the fixes for the two round-20 Sol findings the first r21 loop fixer did
not cover:

* Exact parameter identity must not override module identity or non-tensor call
  semantics (Sol cluster 2). Two DISTINCT modules deliberately sharing one
  weight tensor (tied encoder/decoder) and one kernel applied with different
  ``padding``/``stride``/``dilation`` are different semantic sites, never a
  false recurrent layer. Genuine reuse of ONE module (ALBERT-style) and genuine
  variable-length recurrence must keep grouping.
* Rolled variable-length recurrence must not silently project pass-1
  shape/memory/FLOPs onto every pass (Sol cluster 4). Divergent passes publish
  honest aggregates (per-dimension range shapes, per-pass maxima) plus the
  explicit machine-readable marker ``annotations["varying_across_passes"]``,
  and expanding every rolled Layer through ``Layer.ops`` reconstructs the exact
  unrolled record multiset (conservation).
"""

from collections import Counter, OrderedDict

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchlens import trace as trace_fn
from torchlens.validation.invariants import check_metadata_invariants

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _layer_passes(traced) -> "OrderedDict[str, int]":
    """Return ordered mapping of layer_label -> number of grouped passes."""
    layers: OrderedDict[str, list] = OrderedDict()
    for op in traced:
        layers.setdefault(op.layer_label, []).append(op)
    return OrderedDict((label, len(ops)) for label, ops in layers.items())


def _layers_of_type(passes: "OrderedDict[str, int]", stem: str) -> dict[str, int]:
    """Filter layer pass counts whose label starts with ``stem``."""
    return {label: count for label, count in passes.items() if label.startswith(stem)}


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


def _rolled_node_label_line(traced, node_name: str, outpath: str) -> str:
    """Render the rolled graph and return the named node's label line."""
    graph = traced.draw(
        vis_mode="rolled", vis_save_only=True, return_graph=True, vis_outpath=outpath
    )
    for line in graph.source.splitlines():
        if line.strip().startswith(f"{node_name} [label"):
            return line.strip()
    raise AssertionError(f"node {node_name!r} not found in rolled render")


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


class _UntiedEncoderDecoder(nn.Module):
    """Structural control: identical topology, independent weights."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 4, bias=False)
        self.decoder = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.decoder(torch.relu(self.encoder(x)))


class _ReusedModuleLoop(nn.Module):
    """Genuine recurrence control: ONE module called n times (ALBERT-style)."""

    def __init__(self, n_steps: int = 3) -> None:
        super().__init__()
        self.cell = nn.Linear(4, 4, bias=False)
        self.n_steps = n_steps

    def forward(self, x):
        for _ in range(self.n_steps):
            x = torch.tanh(self.cell(x))
        return x


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


class _SharedKernelSameArgsLoop(nn.Module):
    """Genuine functional weight-tied loop: same kernel, SAME args, chained."""

    def __init__(self) -> None:
        super().__init__()
        self.kernel = nn.Parameter(torch.randn(1, 1, 3, 3))

    def forward(self, x):
        for _ in range(3):
            x = F.conv2d(x, self.kernel, padding=1)
        return x


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
# Sol test #2 -- weight-tied distinct modules retain distinct identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("direct_chain", [False, True])
def test_tied_distinct_modules_are_not_recurrent(direct_chain: bool) -> None:
    """Distinct tied encoder/decoder must never become one recurrent layer.

    Exact parameter identity used to override the module suffix: the union step
    keyed only on (func_name, param_barcodes), so the straight-line chain
    ``decoder(relu(encoder(x)))`` with a shared weight tensor collapsed into one
    false 2-pass layer (is_recurrent=True) in defiance of the documented
    invariant that identical ops in DIFFERENT modules are different sites.
    """
    torch.manual_seed(0)
    traced = trace_fn(_TiedEncoderDecoder(direct_chain=direct_chain), torch.randn(1, 4))

    assert traced.is_recurrent is False
    linear_layers = _layers_of_type(_layer_passes(traced), "linear")
    assert len(linear_layers) == 2, linear_layers
    assert set(linear_layers.values()) == {1}, linear_layers

    # Both semantic sites survive with their own module attribution; no layer's
    # pass ops disagree on module address.
    site_addresses = set()
    for layer in traced.layer_logs.values():
        addresses = {_module_addresses(op) for op in layer.ops.values()}
        assert len(addresses) <= 1, f"layer {layer.layer_label} spans module addresses {addresses}"
        site_addresses.update(addresses)
    assert ("encoder",) in site_addresses
    assert ("decoder",) in site_addresses

    assert check_metadata_invariants(traced)


def test_tied_distinct_modules_rolled_render_shows_both_sites(tmp_path) -> None:
    """The rolled render must contain BOTH tied sites, not one 2-pass node."""
    torch.manual_seed(0)
    traced = trace_fn(_TiedEncoderDecoder(), torch.randn(1, 4))
    graph = traced.draw(
        vis_mode="rolled",
        vis_save_only=True,
        return_graph=True,
        vis_outpath=str(tmp_path / "tied_rolled"),
    )
    label_lines = [
        line.strip()
        for line in graph.source.splitlines()
        if line.strip().startswith("linear_") and "[label" in line
    ]
    assert len(label_lines) == 2, label_lines
    joined = "\n".join(label_lines)
    assert "@encoder" in joined
    assert "@decoder" in joined
    assert "(x2)" not in joined
    assert check_metadata_invariants(traced)


def test_untied_control_and_reused_module_control() -> None:
    """Untied twin stays ungrouped; ONE reused module keeps genuine recurrence."""
    torch.manual_seed(0)
    untied = trace_fn(_UntiedEncoderDecoder(), torch.randn(1, 4))
    assert untied.is_recurrent is False
    assert set(_layers_of_type(_layer_passes(untied), "linear").values()) == {1}
    assert check_metadata_invariants(untied)

    reused = trace_fn(_ReusedModuleLoop(3), torch.randn(1, 4))
    assert reused.is_recurrent is True
    assert _layers_of_type(_layer_passes(reused), "linear") == {"linear_1_1": 3}
    assert check_metadata_invariants(reused)


# ---------------------------------------------------------------------------
# Sol test #3 -- non-tensor call structure is part of recurrence identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", ["padding", "stride", "dilation"])
def test_shared_kernel_divergent_args_stay_distinct(axis: str) -> None:
    """Same-kernel calls with different structural args are not recurrence.

    Parameter equivalence ignores non-tensor arguments, so two ``F.conv2d``
    calls sharing one kernel but differing in padding/stride/dilation (different
    output shapes, no loop) became one false recurrent layer whose aggregate
    Layer could not truthfully describe both calls.
    """
    torch.manual_seed(0)
    traced = trace_fn(_SharedKernelConvs(axis), torch.randn(1, 1, 5, 5))

    assert traced.is_recurrent is False
    conv_layers = _layers_of_type(_layer_passes(traced), "conv2d")
    assert len(conv_layers) == 2, conv_layers
    assert set(conv_layers.values()) == {1}, conv_layers

    # Each 1-pass layer keeps its own call's shape -- nothing is projected from
    # call 1 onto call 2.
    conv_shapes = {
        tuple(layer.shape)
        for layer in traced.layer_logs.values()
        if layer.layer_label.startswith("conv2d")
    }
    assert len(conv_shapes) == 2, conv_shapes

    assert check_metadata_invariants(traced)


def test_shared_kernel_same_args_loop_still_groups() -> None:
    """Control: a genuine functional weight-tied loop with uniform args groups."""
    torch.manual_seed(0)
    traced = trace_fn(_SharedKernelSameArgsLoop(), torch.randn(1, 1, 5, 5))
    assert traced.is_recurrent is True
    assert _layers_of_type(_layer_passes(traced), "conv2d") == {"conv2d_1_1": 3}
    assert check_metadata_invariants(traced)


def test_variable_length_recurrence_not_projected_from_pass_1() -> None:
    """Ragged recurrence groups, and its Layer aggregates are honest.

    The tensor-free arg signature must keep grouping a genuine loop whose
    activations change shape every step, and ``_build_layer_logs`` must not
    project pass-1 shape/memory/FLOPs onto the rolled aggregate.
    """
    torch.manual_seed(0)
    traced = trace_fn(_ShrinkingRecurrent(), torch.randn(4, 4))
    layer = traced.layer_logs["linear_1_1"]
    op_shapes = [tuple(op.shape) for op in layer.ops.values()]
    assert op_shapes == [(4, 4), (3, 4), (2, 4)]

    # Honest aggregate shape: constant dims stay ints, divergent dims become an
    # explicit range token -- never the silent pass-1 tuple.
    assert layer.shape == ("2..4", 4)

    # Honest aggregate memory/FLOPs: the per-pass maximum, with the explicit
    # machine-readable per-pass marker.
    op_memory = [int(op.activation_memory) for op in layer.ops.values()]
    assert int(layer.activation_memory) == max(op_memory)
    varying = layer.annotations["varying_across_passes"]
    assert varying["shape"] == [(4, 4), (3, 4), (2, 4)]
    assert [int(value) for value in varying["activation_memory"]] == op_memory
    assert "flops_forward" in varying

    assert check_metadata_invariants(traced)


def test_growing_recurrence_aggregate_is_not_call_1() -> None:
    """When passes GROW, the aggregate must differ from call 1's values."""
    torch.manual_seed(0)
    traced = trace_fn(_GrowingRecurrent(), torch.randn(2, 4))
    layer = traced.layer_logs["linear_1_1"]
    # OpAccessor get/[] are 0-based positional (C02 basis fix, lovely bug 27).
    first_pass = layer.ops.get(0)
    last_pass = layer.ops.get(layer.num_passes - 1)
    assert int(first_pass.activation_memory) < int(last_pass.activation_memory)
    assert int(layer.activation_memory) == int(last_pass.activation_memory)
    assert tuple(layer.shape) != tuple(first_pass.shape)
    assert check_metadata_invariants(traced)


def test_uniform_multi_pass_layer_left_untouched() -> None:
    """Fixed-shape loops keep plain pass values and carry NO varying marker."""
    torch.manual_seed(0)
    traced = trace_fn(_FixedShapeLoop(), torch.randn(4, 4))
    layer = traced.layer_logs["linear_1_1"]
    assert layer.num_passes == 3
    assert tuple(layer.shape) == (4, 4)
    assert "varying_across_passes" not in layer.annotations
    assert check_metadata_invariants(traced)


# ---------------------------------------------------------------------------
# Sol test #10 -- rolled/unrolled conservation and honest rolled rendering
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
    call, shape, dtype) reached through ``Trace.layer_logs[*].ops`` must equal
    the unrolled ``layer_list`` exactly: no op dropped, none duplicated, and no
    rolled aggregate replacing per-pass truth.
    """
    torch.manual_seed(0)
    traced = trace_fn(model_factory(), model_input)

    unrolled_records = Counter(_op_record(op) for op in traced.layer_list)
    rolled_records: Counter = Counter()
    for layer in traced.layer_logs.values():
        for op in layer.ops.values():
            rolled_records[_op_record(op)] += 1

    assert rolled_records == unrolled_records
    assert check_metadata_invariants(traced)


def test_rolled_render_displays_variation_not_pass_1(tmp_path) -> None:
    """The rolled node label must display shape variation, not pass-1 data."""
    torch.manual_seed(0)
    traced = trace_fn(_ShrinkingRecurrent(), torch.randn(4, 4))
    label_line = _rolled_node_label_line(traced, "linear_1_1", str(tmp_path / "ragged"))
    assert "(x3)" in label_line
    assert "(2..4, 4)" in label_line
    assert "(4, 4)," not in label_line
    assert check_metadata_invariants(traced)


# ---------------------------------------------------------------------------
# Deep-hunt L1: dict KEYS must get the same identity-repr guard as values
# ---------------------------------------------------------------------------


class _OpaqueKey:
    """Non-primitive dict key with the default address-bearing ``repr``."""


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

    from torchlens.postprocess.loop_detection import _structural_arg_signature

    def _op(key: object) -> SimpleNamespace:
        return SimpleNamespace(non_tensor_pos_args=(), non_tensor_kwargs={"cfg": {key: 1}})

    assert _structural_arg_signature(_op(_OpaqueKey())) == _structural_arg_signature(
        _op(_OpaqueKey())
    )
    assert _structural_arg_signature(_op("alpha")) != _structural_arg_signature(_op("beta"))


def test_arg_signature_mixed_key_dict_is_order_and_address_free() -> None:
    """Mixed primitive/object key dicts sign identically across constructions."""
    from types import SimpleNamespace

    from torchlens.postprocess.loop_detection import _structural_arg_signature

    first = SimpleNamespace(
        non_tensor_pos_args=({_OpaqueKey(): 1, "z": 2},),
        non_tensor_kwargs={},
    )
    second = SimpleNamespace(
        non_tensor_pos_args=({"z": 2, _OpaqueKey(): 1},),
        non_tensor_kwargs={},
    )

    assert _structural_arg_signature(first) == _structural_arg_signature(second)


# ---------------------------------------------------------------------------
# Deep-hunt L2: the argsig split must propagate into equivalent_labels
# ---------------------------------------------------------------------------


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
    import torchlens.postprocess.loop_detection as loop_detection

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
