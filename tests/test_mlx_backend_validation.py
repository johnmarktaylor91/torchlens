"""MLX live replay-validation oracle tests."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.backend_mlx

mlx = pytest.importorskip("mlx")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

import torchlens as tl  # noqa: E402
from torchlens.backends.mlx import MLXBackend  # noqa: E402
from torchlens.validation.status import ValidationReplayStatus  # noqa: E402


class _TwoLayerMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(4, 3)
        self.l2 = nn.Linear(3, 2)

    def __call__(self, x: mx.array) -> mx.array:
        return self.l2(nn.relu(self.l1(x)))


def _healthy_trace() -> tl.Trace:
    return tl.trace(_TwoLayerMLP(), mx.ones((1, 4)), backend="mlx")


def test_mlx_validation_healthy_two_layer_mlp_passes() -> None:
    """Validate a healthy MLX MLP with per-op replay and perturbation."""

    trace = _healthy_trace()
    assert MLXBackend().validate_trace(trace) is True
    status = trace.validation_replay_status
    assert status.state == "passed"
    assert status.replayed_node_count >= 1


def test_mlx_validation_registered_spec_dispatches() -> None:
    """The registered spec routes to the real oracle, not the old refusal."""

    from torchlens.backends import get_backend_spec

    spec = get_backend_spec("mlx")
    assert spec.capabilities.validation_replay is True
    assert spec.validate_trace(_healthy_trace()) is True


def test_mlx_validation_fails_corrupted_saved_output() -> None:
    """Fail validation when a saved MLX op output payload is corrupted."""

    trace = _healthy_trace()
    victim = next(op for op in trace.layer_list if op.out is not None and op.uses_params)
    with pytest.warns(UserWarning):
        victim.out = victim.out + 1.0

    assert MLXBackend().validate_trace(trace) is False


def test_mlx_validation_fails_dropped_capture_material() -> None:
    """Fail (never vacuously pass) when replay material is missing on a live trace."""

    trace = _healthy_trace()
    trace._mlx_op_captures = []

    assert MLXBackend().validate_trace(trace) is False


def test_mlx_validation_fails_partial_capture_deletion() -> None:
    """Deleting a SUBSET of replay records must fail, never shrink the denominator."""

    for index in range(3):
        trace = _healthy_trace()
        assert len(trace._mlx_op_captures) == 3
        del trace._mlx_op_captures[index]

        assert MLXBackend().validate_trace(trace) is False


def test_mlx_validation_fails_inventory_tamper() -> None:
    """Shrinking the immutable inventory itself is caught by trace-op coverage."""

    trace = _healthy_trace()
    trace._mlx_replay_inventory = trace._mlx_replay_inventory[:-1]

    assert MLXBackend().validate_trace(trace) is False


def test_mlx_validation_fails_missing_inventory() -> None:
    """A live trace without the emit-time inventory can never pass."""

    trace = _healthy_trace()
    del trace._mlx_replay_inventory

    assert MLXBackend().validate_trace(trace) is False


class _BranchMergeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(4, 4)

    def __call__(self, x: mx.array) -> mx.array:
        hidden = self.l1(x)
        return mx.add(nn.relu(hidden), nn.sigmoid(hidden))


def _branch_trace_and_add_tamper_material() -> tuple[tl.Trace, int, str, str]:
    trace = tl.trace(_BranchMergeNet(), mx.ones((1, 4)), backend="mlx")
    captures = trace._mlx_op_captures
    add_index = next(i for i, c in enumerate(captures) if c.op_name == "add")
    relu_label = next(c.labels_raw[0] for c in captures if c.op_name == "relu")
    sigmoid_label = next(c.labels_raw[0] for c in captures if c.op_name == "sigmoid")
    return trace, add_index, relu_label, sigmoid_label


def test_mlx_validation_fails_incoherent_wrong_parent_attribution() -> None:
    """Rewiring only the trace's declared parents fails the structural cross-check."""

    trace, add_index, relu_label, sigmoid_label = _branch_trace_and_add_tamper_material()
    add_label = trace._mlx_op_captures[add_index].labels_raw[0]
    add_op = next(op for op in trace.layer_list if op._label_raw == add_label)
    # add_op.parents holds FINAL labels (bare layer_label for these
    # single-pass ops, torch parity); relu_label/sigmoid_label are RAW
    # capture labels, so resolve them to final form before rewiring.
    relu_final = trace[relu_label].layer_label
    sigmoid_final = trace[sigmoid_label].layer_label
    add_op.parents = [sigmoid_final if p == relu_final else p for p in add_op.parents]

    assert MLXBackend().validate_trace(trace) is False


def test_mlx_validation_fails_coherent_wrong_parent_attribution() -> None:
    """Rewiring trace parents AND capture leaf labels together fails numerically.

    Replay reconstructs arguments from the DECLARED parents' saved payloads, so
    a coherent wrong-parent story replays with the wrong branch's values and
    mismatches the saved output.
    """

    import dataclasses

    trace, add_index, relu_label, sigmoid_label = _branch_trace_and_add_tamper_material()
    captures = trace._mlx_op_captures
    capture = captures[add_index]
    captures[add_index] = dataclasses.replace(
        capture,
        arg_leaf_labels=tuple(
            tuple(sigmoid_label if label == relu_label else label for label in slot)
            for slot in capture.arg_leaf_labels
        ),
    )
    add_op = next(op for op in trace.layer_list if op._label_raw == capture.labels_raw[0])
    add_op.parents = [sigmoid_label if p == relu_label else p for p in add_op.parents]

    assert MLXBackend().validate_trace(trace) is False


def test_mlx_validation_branch_merge_healthy_passes() -> None:
    """The branch/merge oracle model itself validates cleanly untampered."""

    trace = tl.trace(_BranchMergeNet(), mx.ones((1, 4)), backend="mlx")

    assert MLXBackend().validate_trace(trace) is True


def test_mlx_end_to_end_bare_streaming_flip_refuses_typed() -> None:
    """Review probe, live: a gated flag flipped True in place on the registered MLX
    spec must refuse typed at trace() — never capture while silently ignoring
    the option. (Interventions are genuinely lifted on MLX now, so the probe
    uses streaming, which remains gated.)"""

    from torchlens.backends import BackendCapabilityConformanceError, get_backend_spec

    spec = get_backend_spec("mlx")
    object.__setattr__(spec.capabilities, "streaming", True)
    try:
        with pytest.raises(BackendCapabilityConformanceError):
            tl.trace(
                _TwoLayerMLP(),
                mx.ones((1, 4)),
                backend="mlx",
                storage=object(),
            )
    finally:
        object.__setattr__(spec.capabilities, "streaming", False)


def test_mlx_validation_loaded_payload_stripped_trace_is_unavailable() -> None:
    """Return unavailable status for loaded traces stripped of replay material."""

    trace = _healthy_trace()
    trace._loaded_from_bundle = True
    trace._mlx_op_captures = ()

    result = MLXBackend().validate_trace(trace)

    assert isinstance(result, ValidationReplayStatus)
    assert result.state == "unavailable"
    assert result.passed is False


class _SplitMergeNet(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        a, b = mx.split(x, 2, axis=1)
        return mx.add(a, b)


def test_mlx_split_container_outputs_wire_into_graph() -> None:
    """Container outputs (mx.split) materialize ops and parent the consumers.

    Review probe: at d3861eea the wrapped call's list output was dropped at emit,
    so the trace held only input+add with no split parents while validation
    still passed over the missing wiring.
    """

    trace = tl.trace(_SplitMergeNet(), mx.ones((1, 4)), backend="mlx")
    split_ops = [op for op in trace.layer_list if op._label_raw.startswith("split")]
    assert len(split_ops) == 2, "both split output arrays must materialize as ops"
    add_op = next(op for op in trace.layer_list if op._label_raw.startswith("add"))
    # add_op.parents holds FINAL labels (bare layer_label for these
    # single-pass split ops, torch parity), not the raw capture labels.
    assert set(add_op.parents) == {op.layer_label for op in split_ops}
    assert MLXBackend().validate_trace(trace) is True


def test_mlx_validation_fails_stripped_leaf_label_provenance() -> None:
    """Stripping a capture's recorded parent labels fails even coherently.

    Review probe: setting a captured parent label to None AND removing the
    declared parent passed at d3861eea with validate_metadata=False because
    replay silently reused the emit-time argument array. The emit-time
    inventory now fingerprints per-leaf labels, so the strip fails coverage.
    """

    import dataclasses

    trace = _healthy_trace()
    captures = trace._mlx_op_captures
    index = next(i for i, capture in enumerate(captures) if capture.op_name == "relu")
    capture = captures[index]
    captures[index] = dataclasses.replace(
        capture,
        arg_leaf_labels=tuple(tuple(None for _ in slot) for slot in capture.arg_leaf_labels),
    )
    relu_op = next(op for op in trace.layer_list if op._label_raw == capture.labels_raw[0])
    relu_op.parents = []

    assert MLXBackend().validate_trace(trace, validate_metadata=False) is False


class _ConstantProducerNet(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        constant = mx.add(1.0, 2.0)
        return mx.add(x, constant)


def test_mlx_constant_producer_reports_unverified_gap() -> None:
    """A call with no perturbable tensor argument is a recorded evidence gap
    surfacing as UNVERIFIED, never the silent vacuous pass shipped before."""

    trace = tl.trace(_ConstantProducerNet(), mx.ones((1, 4)), backend="mlx")
    result = MLXBackend().validate_trace(trace)

    assert isinstance(result, ValidationReplayStatus)
    assert result.state == "unverified"
    assert trace._mlx_perturbation_gaps
    assert all(label.startswith("add") for label in trace._mlx_perturbation_gaps)


def test_mlx_capture_records_slot_labeled_leaves_instead_of_retaining() -> None:
    """Replay records retain templates, not raw activations (opus F7).

    Every labeled argument leaf is a ``REPLAY_SLOT`` sentinel (replay sources
    it from the saved parent payload) and only unlabeled leaves — parameters
    and constants — keep live arrays. Validation still passes end-to-end, so
    the retention change provably feeds replay from the declared graph.
    """

    from torchlens.backends.mlx.validation import REPLAY_SLOT

    trace = _healthy_trace()

    def _leaves(node: object) -> list[object]:
        if isinstance(node, (list, tuple)):
            return [leaf for item in node for leaf in _leaves(item)]
        if isinstance(node, dict):
            return [leaf for item in node.values() for leaf in _leaves(item)]
        return [node]

    slotted = 0
    for capture in trace._mlx_op_captures:
        for index, value in enumerate(capture.args):
            labels = capture.arg_leaf_labels[index] if index < len(capture.arg_leaf_labels) else ()
            leaves = [
                leaf for leaf in _leaves(value) if isinstance(leaf, mx.array) or leaf is REPLAY_SLOT
            ]
            for leaf, label in zip(leaves, labels):
                if label is not None:
                    assert leaf is REPLAY_SLOT
                    slotted += 1
                else:
                    assert isinstance(leaf, mx.array)
    assert slotted >= 1, "at least one labeled intermediate must be slotted"
    assert MLXBackend().validate_trace(trace) is True


def test_mlx_perturbation_scan_includes_kwargs() -> None:
    """The perturbation tripwire perturbs keyword tensor arguments too."""

    from torchlens.backends.mlx.validation import (
        PERTURBATION_NO_PERTURBABLE_INPUT,
        PERTURBATION_PROVED,
        MLXOpCapture,
        _perturbation_evidence,
    )

    value = mx.ones((2, 2))

    def kwarg_only(*, a: mx.array) -> mx.array:
        return a * 2

    baseline = (kwarg_only(a=value),)
    mx.eval(*baseline)
    capture = MLXOpCapture(
        labels_raw=("kwarg_only_1_raw",),
        op_name="kwarg_only",
        func=kwarg_only,
        args=(),
        kwargs={"a": value},
    )
    assert _perturbation_evidence(capture, baseline) == PERTURBATION_PROVED

    def no_tensor() -> mx.array:
        return mx.ones((2, 2))

    constant_capture = MLXOpCapture(
        labels_raw=("no_tensor_1_raw",),
        op_name="no_tensor",
        func=no_tensor,
        args=(),
        kwargs={},
    )
    assert (
        _perturbation_evidence(constant_capture, (no_tensor(),))
        == PERTURBATION_NO_PERTURBABLE_INPUT
    )


class _DeadReluMLP(_TwoLayerMLP):
    """``_TwoLayerMLP`` pinned so every ReLU input sits below -0.5.

    About 1% of random ``_TwoLayerMLP`` inits land here; the healthy-trace
    tests above then failed nondeterministically because ``x + 0.5`` and
    ``2 * x`` both leave an all-negative ReLU output at zero.
    """

    def __init__(self) -> None:
        super().__init__()
        # Weights stay nonzero so l1's output still depends on its input
        # (zero weights would make UNPROVED the correct verdict); the bias
        # pushes every pre-activation to 4 * 0.1 - 2.0 = -1.6.
        self.l1.weight = mx.full(self.l1.weight.shape, 0.1)
        self.l1.bias = mx.full(self.l1.bias.shape, -2.0)


def test_mlx_validation_dead_relu_init_passes() -> None:
    """A healthy capture whose ReLU input is all strongly negative validates."""

    trace = tl.trace(_DeadReluMLP(), mx.ones((1, 4)), backend="mlx")
    assert MLXBackend().validate_trace(trace) is True


def test_mlx_perturbation_proves_saturated_dependency_and_rejects_vacuous() -> None:
    """Candidates must move a saturated call's output; a replay that ignores
    its argument must still classify UNPROVED under every candidate."""

    from torchlens.backends.mlx.validation import (
        PERTURBATION_PROVED,
        PERTURBATION_UNPROVED,
        MLXOpCapture,
        _perturbation_evidence,
    )

    for value in (mx.full((1, 3), -1.0), mx.zeros((1, 3)), mx.full((2,), -7.5)):
        baseline = (nn.relu(value),)
        mx.eval(*baseline)
        saturated = MLXOpCapture(
            labels_raw=("relu_1_raw",), op_name="relu", func=nn.relu, args=(value,)
        )
        assert _perturbation_evidence(saturated, baseline) == PERTURBATION_PROVED

        def ignores_argument(x: mx.array) -> mx.array:
            return mx.zeros((1, 3))

        vacuous = MLXOpCapture(
            labels_raw=("vacuous_1_raw",), op_name="vacuous", func=ignores_argument, args=(value,)
        )
        assert _perturbation_evidence(vacuous, (mx.zeros((1, 3)),)) == PERTURBATION_UNPROVED
