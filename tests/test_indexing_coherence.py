"""Every TorchLens lookup door returns the kind of object its name promises.

The plain door ``trace[key]`` returns the kind the key names (a module address
its Module, ``address:N`` its ModuleCall, a parameter or buffer address its
record, a bare layer label its Layer, ``label:N`` its Op). A typed door (``trace.ops``, ``trace.layers``,
``trace.modules``, ``trace.module_calls``, ``trace.params``, ``trace.buffers``,
``trace.autograd.grad_fns``) always returns its own kind, converting a key
that names exactly one such object and refusing a key that names several with
the ``:N`` spellings. The same rule holds on Module, ModuleCall, Layer and
Bundle doors.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

import torchlens as tl
from torchlens._data_classes.buffer import Buffer
from torchlens._data_classes.grad_fn import GradFn
from torchlens._data_classes.layer import Layer
from torchlens._data_classes.module import Module, ModuleCall
from torchlens._data_classes.op import Op
from torchlens._data_classes.param import Param
from torchlens._errors import AmbiguousOpLookupError, InvalidArgumentError
from torchlens._selection.errors import AmbiguousSiteError

FC1 = "linear_1_4"
SHARED = "linear_2_6"
EMB = "embedding_1_2"
BN = "batchnorm_1_8"


class DoorModel(torch.nn.Module):
    """One module per lookup kind: a shared module called twice, buffers, params."""

    def __init__(self) -> None:
        """Build the layers."""

        super().__init__()
        self.fc1 = torch.nn.Linear(4, 4)
        self.shared = torch.nn.Linear(4, 4)
        self.bn = torch.nn.BatchNorm1d(4)
        self.fc2 = torch.nn.Linear(4, 2)
        self.emb = torch.nn.Embedding(3, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the shared linear twice around a relu."""

        x = x + self.emb(torch.tensor([0, 1, 2]))
        x = torch.relu(self.fc1(x))
        x = self.shared(x)
        x = torch.relu(self.shared(x))
        x = self.bn(x)
        return self.fc2(x)


@pytest.fixture(scope="module")
def trace() -> Iterator[tl.Trace]:
    """One capture with a backward pass, shared across the doors."""

    torch.manual_seed(0)
    captured = tl.trace(DoorModel().eval(), torch.randn(3, 4))
    captured.backward(captured.output.sum())
    try:
        yield captured
    finally:
        captured.cleanup()


@pytest.fixture(scope="module")
def bundle(trace: tl.Trace) -> tl.Bundle:
    """A two-member bundle over the same trace."""

    return tl.Bundle({"first": trace, "second": trace}, baseline="first")


# --------------------------------------------------------------------------- #
# The plain door: the kind the key names (RC-260).
# --------------------------------------------------------------------------- #


def test_plain_door_module_address_gives_the_module(trace: tl.Trace) -> None:
    """A module address names the Module whether it was called once or twice."""

    assert isinstance(trace["fc1"], Module)
    assert trace["fc1"] is trace.modules["fc1"]
    assert trace["shared"] is trace.modules["shared"]


def test_plain_door_module_out_is_the_output_tensor(trace: tl.Trace) -> None:
    """``trace["fc1"].out`` keeps working on the Module (the README draft line)."""

    assert torch.equal(trace["fc1"].out, trace[f"{FC1}:1"].out)


def test_plain_door_other_kinds(trace: tl.Trace) -> None:
    """Call label, parameter, buffer, op and grad_fn keys each give their own kind."""

    assert isinstance(trace["shared:1"], ModuleCall)
    assert trace["shared:1"] is trace.module_calls["shared:1"]
    assert isinstance(trace["fc1.weight"], Param)
    assert isinstance(trace["bn.running_mean"], Buffer)
    assert isinstance(trace[FC1], Layer)
    assert isinstance(trace[f"{SHARED}:2"], Op)
    grad_fn_label = trace[f"{FC1}:1"].grad_fn_label
    assert isinstance(trace[grad_fn_label], GradFn)


def test_plain_door_buffer_version_key_gives_the_version_op(trace: tl.Trace) -> None:
    """``buffer_address:N`` stays the documented spelling of one buffer version."""

    version = trace["bn.running_mean:1"]
    assert isinstance(version, Op)
    assert version is trace.buffers["bn.running_mean"].versions[0]


# --------------------------------------------------------------------------- #
# Layer labels at the plain door: a layer label names a Layer (lanes.1, 3 Nov 2026).
# --------------------------------------------------------------------------- #


def test_plain_door_bare_layer_label_gives_the_layer(trace: tl.Trace) -> None:
    """A bare layer label gives the Layer, single- or multi-pass."""

    assert trace[FC1] is trace.layers[FC1]
    assert trace[SHARED] is trace.layers[SHARED]
    assert trace[SHARED].num_passes == 2


def test_plain_door_pass_qualified_label_gives_the_op(trace: tl.Trace) -> None:
    """``label:N`` and ``label:last`` give one Op; ``label:all`` the Selection."""

    assert isinstance(trace[f"{FC1}:1"], Op)
    assert trace[f"{SHARED}:2"] is trace.ops[f"{SHARED}:2"]
    assert trace[f"{SHARED}:last"] is trace.ops[f"{SHARED}:2"]
    assert not isinstance(trace[f"{SHARED}:all"], Op | Layer)
    assert len(trace[f"{SHARED}:all"]) == 2


def test_single_pass_layer_passes_reads_through_to_its_op(trace: tl.Trace) -> None:
    """``trace[label].out`` and ``.shape`` on a single-pass Layer read its Op."""

    layer, op = trace[FC1], trace[f"{FC1}:1"]
    assert torch.equal(layer.out, op.out)
    assert layer.shape == op.shape
    assert layer.label == op.label


def test_multi_pass_layer_refuses_single_reads_with_pass_spellings(trace: tl.Trace) -> None:
    """``trace[label].out`` on a multi-pass Layer refuses naming ``label:1`` and ``label:last``."""

    with pytest.raises(InvalidArgumentError) as refusal:
        trace[SHARED].out
    assert refusal.value.fields["code"] == "layer_pass_ambiguous"
    assert f"trace['{SHARED}:1'].out" in str(refusal.value)
    assert f"trace['{SHARED}:last'].out" in str(refusal.value)


def test_ops_door_keeps_the_bare_multi_pass_refusal(trace: tl.Trace) -> None:
    """A door that must give one Op still refuses a bare multi-pass label."""

    with pytest.raises(AmbiguousOpLookupError):
        trace.ops[SHARED]
    with pytest.raises(AmbiguousSiteError) as refusal:
        trace[tl.label(SHARED)]
    assert refusal.value.fields["code"] == "multipass_bare_label_ambiguous"
    assert f"trace[{SHARED!r}]" in str(refusal.value)


def test_do_on_a_bare_multi_pass_label_edits_every_pass() -> None:
    """``trace[label].do(...)`` on a multi-pass Layer edits each pass."""

    torch.manual_seed(0)
    captured = tl.trace(DoorModel().eval(), torch.randn(3, 4))
    try:
        edited = captured[SHARED].do(
            lambda out, *, ctx: torch.zeros_like(out),
            intervention=tl.options.InterventionOptions(confirm_mutation=True),
        )
        for label in (f"{SHARED}:1", f"{SHARED}:2"):
            assert torch.count_nonzero(edited[label].out) == 0
        assert torch.count_nonzero(edited[f"{FC1}:1"].out) > 0
    finally:
        captured.cleanup()


# --------------------------------------------------------------------------- #
# trace.ops
# --------------------------------------------------------------------------- #


def test_ops_door_converts_module_keys_to_the_output_op(trace: tl.Trace) -> None:
    """A module address or call label gives that call's output Op."""

    assert trace.ops["fc1"] is trace[f"{FC1}:1"]
    assert trace.ops["shared:1"] is trace[f"{SHARED}:1"]
    assert trace.ops["shared:2"] is trace[f"{SHARED}:2"]


def test_ops_door_refuses_a_module_called_twice(trace: tl.Trace) -> None:
    """Two calls mean two output Ops: refuse and spell the call labels."""

    with pytest.raises(AmbiguousOpLookupError) as info:
        trace.ops["shared"]
    assert "'shared:1'" in str(info.value)
    assert "'shared:2'" in str(info.value)
    assert info.value.fields["code"] == "ambiguous_op_lookup"


def test_ops_door_buffer_address(trace: tl.Trace) -> None:
    """A buffer address gives its one version Op, or refuses naming the versions."""

    versions = list(trace.buffers["bn.running_mean"].versions)
    if len(versions) == 1:
        assert trace.ops["bn.running_mean"] is versions[0]
    else:
        with pytest.raises(AmbiguousOpLookupError) as info:
            trace.ops["bn.running_mean"]
        assert "'bn.running_mean:1'" in str(info.value)


def test_ops_door_still_refuses_unknown_keys(trace: tl.Trace) -> None:
    """A key naming nothing keeps the not-found error."""

    with pytest.raises(KeyError):
        trace.ops["no_such_module"]


# --------------------------------------------------------------------------- #
# trace.layers
# --------------------------------------------------------------------------- #


def test_layers_door_converts_module_keys(trace: tl.Trace) -> None:
    """A module address or call label gives the Layer of its output op."""

    assert trace.layers["fc1"] is trace.layers[FC1]
    assert trace.layers["shared:2"] is trace.layers[SHARED]
    # Both calls of ``shared`` land in one two-pass Layer, so the address is one Layer.
    assert trace.layers["shared"] is trace.layers[SHARED]
    # Two buffer versions sit in two Layers, so the bare address refuses; a version is one.
    with pytest.raises(AmbiguousOpLookupError) as info:
        trace.layers["bn.running_mean"]
    assert "'bn.running_mean:1'" in str(info.value)
    assert isinstance(trace.layers["bn.running_mean:1"], Layer)


# --------------------------------------------------------------------------- #
# trace.modules and trace.module_calls
# --------------------------------------------------------------------------- #


def test_modules_door_converts_op_param_and_buffer_keys(trace: tl.Trace) -> None:
    """An op label, parameter address or buffer address names the Module it belongs to."""

    assert trace.modules[FC1] is trace.modules["fc1"]
    assert trace.modules[f"{SHARED}:1"] is trace.modules["shared"]
    assert trace.modules[SHARED] is trace.modules["shared"]  # both passes ran in shared
    assert trace.modules["fc1.weight"] is trace.modules["fc1"]
    assert trace.modules["bn.running_mean"] is trace.modules["bn"]
    assert trace.modules["shared:1"] is trace.modules["shared"]  # existing: call label


def test_module_calls_door_converts_op_labels(trace: tl.Trace) -> None:
    """An op label names the call that produced it; a bare two-pass label refuses."""

    assert trace.module_calls[FC1] is trace.module_calls["fc1:1"]
    assert trace.module_calls[f"{SHARED}:2"] is trace.module_calls["shared:2"]
    with pytest.raises(AmbiguousOpLookupError) as info:
        trace.module_calls[SHARED]
    assert "'shared:1'" in str(info.value) and "'shared:2'" in str(info.value)
    with pytest.raises(AmbiguousOpLookupError):
        trace.module_calls["shared"]  # existing refusal


# --------------------------------------------------------------------------- #
# trace.params, trace.buffers, trace.autograd.grad_fns
# --------------------------------------------------------------------------- #


def test_params_door_converts_module_and_op_keys(trace: tl.Trace) -> None:
    """A module or op with one parameter gives it; several refuse by address."""

    assert trace.params["emb"] is trace.params["emb.weight"]
    assert trace.params[EMB] is trace.params["emb.weight"]
    with pytest.raises(AmbiguousOpLookupError) as info:
        trace.params["fc1"]
    assert "fc1.weight" in str(info.value) and "fc1.bias" in str(info.value)


def test_params_filtered_collection_does_not_leak(trace: tl.Trace) -> None:
    """A converted key must name a member of the filtered collection."""

    with pytest.raises(KeyError):
        trace.params.frozen["emb"]


def test_buffers_door_converts_module_and_version_keys(trace: tl.Trace) -> None:
    """A version op label names its Buffer; a module with several buffers refuses."""

    buffer = trace.buffers["bn.running_mean"]
    assert trace.buffers[buffer.versions[0].label] is buffer
    with pytest.raises(AmbiguousOpLookupError) as info:
        trace.buffers["bn"]
    assert "bn.running_mean" in str(info.value)


def test_grad_fns_door_converts_op_labels_and_refuses_several(trace: tl.Trace) -> None:
    """An op label names its grad_fn; a substring matching several refuses."""

    grad_fns = trace.autograd.grad_fns
    assert grad_fns[FC1] is grad_fns[trace[FC1].grad_fn_label]
    assert grad_fns["fc1"] is grad_fns[FC1]  # a module address names its output op's grad_fn
    with pytest.raises(AmbiguousOpLookupError):
        grad_fns["_back_"]


# --------------------------------------------------------------------------- #
# Scoped doors: Module, ModuleCall accessor, Layer.ops
# --------------------------------------------------------------------------- #


def test_module_door_gives_the_kind_the_key_names(trace: tl.Trace) -> None:
    """Inside a module: a layer label gives its Layer, a descendant address its Module."""

    root = trace.modules["self"]
    assert root[FC1] is trace[FC1]
    assert root["fc1"] is trace.modules["fc1"]
    assert root["shared:1"] is trace.module_calls["shared:1"]
    assert isinstance(root[0], Layer)
    assert trace.modules["shared"][f"{SHARED}:1"] is trace[f"{SHARED}:1"]
    with pytest.raises(KeyError):
        trace.modules["fc1"]["shared"]  # not a descendant of fc1


def test_module_calls_accessor_converts_op_labels(trace: tl.Trace) -> None:
    """``module.calls[op_label]`` names the call that ran it; a bare two-pass label refuses."""

    shared = trace.modules["shared"]
    assert shared.calls[f"{SHARED}:2"] is trace.module_calls["shared:2"]
    with pytest.raises(AmbiguousOpLookupError):
        shared.calls[SHARED]
    assert trace.modules["fc1"].calls[FC1] is trace.module_calls["fc1:1"]


def test_layer_ops_door_accepts_call_labels(trace: tl.Trace) -> None:
    """``layer.ops["shared:2"]`` is the op of that call."""

    assert trace.layers[SHARED].ops["shared:2"] is trace[f"{SHARED}:2"]
    assert trace.layers[FC1].ops["fc1:1"] is trace[f"{FC1}:1"]


# --------------------------------------------------------------------------- #
# Bundle doors
# --------------------------------------------------------------------------- #


def test_bundle_typed_doors_convert_module_keys(bundle: tl.Bundle) -> None:
    """Bundle doors follow the member doors, ambiguity included."""

    assert bundle.ops["fc1"].members["first"] is bundle["first"].ops["fc1"]
    assert bundle.ops["shared:1"].members["first"].label == f"{SHARED}:1"
    assert bundle.layers["fc1"].members["first"] is bundle["first"].layers[FC1]
    assert bundle.modules[FC1].members["first"] is bundle["first"].modules["fc1"]
    with pytest.raises(AmbiguousOpLookupError):
        bundle.ops["shared"]


# --------------------------------------------------------------------------- #
# Saved traces answer like live ones.
# --------------------------------------------------------------------------- #


def test_loaded_trace_doors_match_live(trace: tl.Trace, tmp_path) -> None:
    """A reloaded trace keeps the same kinds at every door."""

    path = tmp_path / "doors.tlspec"
    trace.save(path)
    loaded = tl.load(path)
    assert isinstance(loaded["fc1"], Module)
    assert isinstance(loaded["shared:1"], ModuleCall)
    assert loaded.ops["fc1"].label == f"{FC1}:1"
    assert loaded.modules[FC1].address == "fc1"
    assert loaded.module_calls[f"{SHARED}:2"].label == "shared:2"
