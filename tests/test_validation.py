"""Tests for the validation subpackage.

Covers: import paths, registry consistency, perturbation unit tests,
deep clone helpers, and integration tests through specific exemption paths.
"""

import pytest
import torch
import torch.nn as nn
from typing import cast

from torchlens import validate_forward_pass, validate_saved_outs, Trace, trace as trace_fn
from torchlens import check_metadata_invariants, MetadataInvariantError
from torchlens.validation import validate_saved_outs as validate_from_subpkg
from torchlens.validation.exemptions import (
    SKIP_VALIDATION_ENTIRELY,
    SKIP_PERTURBATION_ENTIRELY,
    STRUCTURAL_ARG_POSITIONS,
    CUSTOM_EXEMPTION_CHECKS,
)
from torchlens.validation.core import (
    _perturb_layer_outs,
    _deep_clone_tensors,
    _copy_validation_args,
    MAX_PERTURB_ATTEMPTS,
)
from torchlens.utils.tensor_utils import tensor_nanequal


# =============================================================================
# Import / binding tests
# =============================================================================


def test_validation_import_path():
    """from torchlens.validation import validate_saved_outs works."""
    assert callable(validate_from_subpkg)


@pytest.mark.smoke
def test_validate_forward_pass_importable():
    """validate_forward_pass is importable from torchlens top-level."""
    assert callable(validate_forward_pass)


def test_validate_forward_pass_output_aliasing_a_reassigned_buffer():
    """Regression: a model that RETURNS a registered buffer it reassigned must validate.

    Previously a false-negative: validate_forward_pass saved the ground-truth output by
    reference, then restored state_dict; load_state_dict writes buffers in-place, clobbering
    the saved ground-truth (which aliased the returned buffer) back to its initial value, so
    the (correct) traced output was compared against a corrupted zero tensor. Capture/replay
    were always correct; the validator now snapshots ground-truth outputs before the restore.
    """

    class RecurrentStateBuffer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("h", torch.zeros(3))
            self.lin = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for _ in range(4):
                h = cast(torch.Tensor, self._buffers["h"])
                self.h = torch.tanh(self.lin(x) + h)
            return cast(torch.Tensor, self._buffers["h"])

    torch.manual_seed(0)
    assert validate_forward_pass(RecurrentStateBuffer(), torch.randn(3)) is True

    class ReassignReturn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("b", torch.ones(3))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.b = cast(torch.Tensor, self._buffers["b"]) + x
            return cast(torch.Tensor, self._buffers["b"])

    assert validate_forward_pass(ReassignReturn(), torch.randn(3)) is True


def test_validate_forward_pass_plain_attribute_mutable_state_isolated() -> None:
    """Plain attribute-held mutable tensors should not poison the traced run."""

    class PlainMutableState(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.state = [torch.zeros(3)]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.state[0].add_(1)
            return x + self.state[0]

    assert validate_forward_pass(PlainMutableState(), torch.randn(3)) is True


def test_validate_forward_pass_deepcopy_fallback_warns_for_registered_state() -> None:
    """Un-deepcopyable models fall back to state_dict restore with warning."""

    class UndeepcopyableRegisteredState(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("buf", torch.zeros(3))
            self.handle = object()

        def __getstate__(self) -> dict[str, object]:
            """Make deepcopy fail like modules holding uncopyable resources."""

            raise TypeError("cannot deepcopy handle")

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            buf = cast(torch.Tensor, self.buf)
            buf.add_(1)
            return x + buf

    with pytest.warns(RuntimeWarning, match="could not deepcopy the model"):
        assert validate_forward_pass(UndeepcopyableRegisteredState(), torch.randn(3)) is True


def test_validation_tripwire_still_fails_on_wrong_output() -> None:
    """Trace validation still rejects genuinely wrong ground-truth outputs."""

    model = nn.Sequential(nn.Linear(3, 3), nn.ReLU()).eval()
    x = torch.randn(2, 3)
    trace = trace_fn(model, x)
    wrong_output = [trace[trace.output_layers[0]].out + 1]

    try:
        assert trace.validate_forward_pass(wrong_output, validate_metadata=False) is False
    finally:
        trace.cleanup()


def test_check_metadata_invariants_importable():
    """check_metadata_invariants and MetadataInvariantError importable from top-level."""
    assert callable(check_metadata_invariants)
    assert issubclass(MetadataInvariantError, ValueError)


def test_trace_validate_method_bound():
    """Trace.validate_saved_outs is callable."""
    assert hasattr(Trace, "validate_saved_outs")
    assert callable(Trace.validate_saved_outs)


def test_trace_check_metadata_method_bound():
    """Trace.check_metadata_invariants is callable."""
    assert hasattr(Trace, "check_metadata_invariants")
    assert callable(Trace.check_metadata_invariants)


def test_tensor_nanequal_uses_relative_tolerance_for_replay() -> None:
    """Validation replay should allow tiny relative floating-point drift."""
    saved = torch.tensor([1.0, 100.0], dtype=torch.float32)
    replayed = saved + torch.tensor([8e-6, 4e-3], dtype=torch.float32)
    mismatched = saved + torch.tensor([1e-3, 1e-1], dtype=torch.float32)

    assert not tensor_nanequal(saved, replayed, allow_tolerance=False)
    assert tensor_nanequal(saved, replayed, allow_tolerance=True)
    assert not tensor_nanequal(saved, mismatched, allow_tolerance=True)


# =============================================================================
# Registry consistency tests
# =============================================================================


def test_skip_validation_entirely_are_strings():
    assert len(SKIP_VALIDATION_ENTIRELY) > 0
    for entry in SKIP_VALIDATION_ENTIRELY:
        assert isinstance(entry, str) and len(entry) > 0


def test_skip_perturbation_entirely_are_strings():
    assert len(SKIP_PERTURBATION_ENTIRELY) > 0
    for entry in SKIP_PERTURBATION_ENTIRELY:
        assert isinstance(entry, str) and len(entry) > 0


def test_structural_arg_positions_values_are_sets_of_ints():
    for func_name, positions in STRUCTURAL_ARG_POSITIONS.items():
        assert isinstance(func_name, str) and len(func_name) > 0
        assert isinstance(positions, set)
        for pos in positions:
            assert isinstance(pos, int) and pos >= 0


def test_custom_exemption_checks_are_callable():
    for func_name, check_fn in CUSTOM_EXEMPTION_CHECKS.items():
        assert isinstance(func_name, str) and len(func_name) > 0
        assert callable(check_fn)


# =============================================================================
# Perturbation unit tests
# =============================================================================


@pytest.mark.smoke
def test_perturbation_changes_float_tensor():
    parent = torch.randn(10, 10)
    output = torch.randn(10, 10)
    perturbed = _perturb_layer_outs(parent, output)
    assert not torch.equal(perturbed, parent)
    assert perturbed.shape == parent.shape


def test_perturbation_changes_int_tensor():
    parent = torch.randint(0, 100, (10, 10))
    output = torch.randn(10, 10)
    perturbed = _perturb_layer_outs(parent, output)
    assert not torch.equal(perturbed, parent)
    assert perturbed.dtype == parent.dtype


def test_perturbation_changes_bool_tensor():
    parent = torch.ones(10, 10, dtype=torch.bool)
    output = torch.randn(10, 10)
    perturbed = _perturb_layer_outs(parent, output)
    # With 100 elements all True, random should differ
    assert not torch.equal(perturbed, parent)
    assert perturbed.dtype == torch.bool


def test_perturbation_changes_complex_tensor():
    parent = torch.complex(torch.randn(5, 5), torch.randn(5, 5))
    output = torch.randn(5, 5)
    perturbed = _perturb_layer_outs(parent, output)
    assert not torch.equal(perturbed, parent)
    assert perturbed.is_complex()


def test_perturbation_respects_dtype():
    for dtype in [torch.float32, torch.float64, torch.int32, torch.int64, torch.bool]:
        if dtype in (torch.int32, torch.int64):
            parent = torch.randint(0, 100, (5, 5), dtype=dtype)
        elif dtype == torch.bool:
            parent = torch.ones(5, 5, dtype=torch.bool)
        else:
            parent = torch.randn(5, 5, dtype=dtype)
        output = torch.randn(5, 5)
        perturbed = _perturb_layer_outs(parent, output)
        assert perturbed.dtype == dtype


def test_perturbation_handles_empty_tensor():
    parent = torch.tensor([])
    output = torch.tensor([])
    perturbed = _perturb_layer_outs(parent, output)
    assert perturbed.numel() == 0
    assert torch.equal(perturbed, parent)


def test_perturbation_terminates_on_scalar():
    """MAX_PERTURB_ATTEMPTS guard prevents infinite loop on single-element tensors."""
    # Single-element bool tensor: 50% chance each attempt matches original.
    # With MAX_PERTURB_ATTEMPTS=100, it should terminate regardless.
    parent = torch.tensor([True])
    output = torch.tensor([1.0])
    perturbed = _perturb_layer_outs(parent, output)
    assert perturbed.dtype == torch.bool
    assert perturbed.shape == parent.shape


# =============================================================================
# Deep clone tests
# =============================================================================


def test_deep_clone_nested_list_of_tensors():
    original = [torch.tensor([1.0, 2.0]), [torch.tensor([3.0]), torch.tensor([4.0])]]
    cloned = _deep_clone_tensors(original)
    assert isinstance(cloned, list)
    assert isinstance(cloned[1], list)
    assert torch.equal(cloned[0], original[0])
    assert torch.equal(cloned[1][0], original[1][0])


def test_deep_clone_nested_dict_of_tensors():
    original = {"a": torch.tensor([1.0]), "b": {"c": torch.tensor([2.0])}}
    cloned = _deep_clone_tensors(original)
    assert isinstance(cloned, dict)
    assert isinstance(cloned["b"], dict)
    assert torch.equal(cloned["a"], original["a"])
    assert torch.equal(cloned["b"]["c"], original["b"]["c"])


def test_deep_clone_independence():
    """Modifying clone doesn't affect original."""
    original = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    cloned = _deep_clone_tensors(original)
    cloned[0][0] = 999.0
    assert original[0][0].item() == 1.0


def test_deep_clone_preserves_non_tensors():
    original = [42, "hello", None, (1, 2)]
    cloned = _deep_clone_tensors(original)
    assert cloned == original


def test_copy_validation_args():
    """_copy_validation_args deep-clones tensors in args and kwargs."""
    t1 = torch.tensor([1.0, 2.0])
    t2 = torch.tensor([3.0])
    input_args = {
        "args": [t1, [t2, 42]],
        "kwargs": {"key": torch.tensor([5.0])},
    }
    copied = _copy_validation_args(input_args)

    # Independence
    copied["args"][0][0] = 999.0
    assert t1[0].item() == 1.0

    copied["kwargs"]["key"][0] = 999.0
    assert input_args["kwargs"]["key"][0].item() == 5.0


# =============================================================================
# Integration tests — validate full pipeline through specific exemption paths
# =============================================================================


class _GetItemTensorIndex(nn.Module):
    """Model that uses tensor indexing (__getitem__ with a tensor index)."""

    def forward(self, x):
        idx = torch.tensor([0, 2, 1])
        return x[idx]


class _ScatterModel(nn.Module):
    """Model that uses scatter_."""

    def forward(self, x):
        src = torch.ones(3, 5)
        index = torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2], [0, 0, 1, 2, 0]])
        out = torch.zeros(3, 5)
        out.scatter_(1, index, src)
        return x + out


class _MaskedFillModel(nn.Module):
    """Model that uses masked_fill_."""

    def forward(self, x):
        mask = x > 0.5
        return x.masked_fill_(mask, 0.0)


class _FunctionalMaskedFillModel(nn.Module):
    """Model that uses masked_fill."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a boolean mask through the non-in-place masked_fill method."""
        mask = x > 0.5
        return x.masked_fill(mask, 0.0)


class _ZerosLikeModel(nn.Module):
    """Model that uses zeros_like."""

    def forward(self, x):
        z = torch.zeros_like(x)
        return x + z


class _EmptyLikeModel(nn.Module):
    """Model that uses empty_like (tests SKIP_VALIDATION_ENTIRELY)."""

    def forward(self, x):
        # empty_like output is nondeterministic — don't use it in computation
        _ = torch.empty_like(x)
        return x * 2


def test_validation_with_getitem_tensor_index():
    model = _GetItemTensorIndex()
    x = torch.randn(5, 3)
    assert validate_forward_pass(model, x)


def test_validation_with_scatter():
    model = _ScatterModel()
    x = torch.randn(3, 5)
    assert validate_forward_pass(model, x)


def test_validation_with_masked_fill():
    model = _MaskedFillModel()
    x = torch.randn(4, 4)
    assert validate_forward_pass(model, x)


def test_validation_with_functional_masked_fill() -> None:
    """Validate non-in-place masked_fill boolean masks as structural args."""
    model = _FunctionalMaskedFillModel()
    x = torch.randn(4, 4)
    assert validate_forward_pass(model, x)


def test_validation_with_zeros_like():
    model = _ZerosLikeModel()
    x = torch.randn(3, 3)
    assert validate_forward_pass(model, x)


def test_validation_with_empty_like():
    model = _EmptyLikeModel()
    x = torch.randn(3, 3)
    assert validate_forward_pass(model, x)


# =============================================================================
# Metadata invariant tests — standalone + corruption
# =============================================================================


class _SimpleFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 3)

    def forward(self, x):
        return self.fc(x)


class _MidForwardGradModel(nn.Module):
    """Model that triggers ``autograd.grad`` before all forward ops exist."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.fc1 = nn.Linear(5, 4)
        self.fc2 = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a MAML-style forward with an inner differentiable grad."""

        hidden = torch.relu(self.fc1(x))
        inner_loss = hidden.square().mean()
        (weight_grad,) = torch.autograd.grad(
            inner_loss,
            self.fc1.weight,
            create_graph=True,
            retain_graph=True,
        )
        adapted_bias = weight_grad.mean()
        return self.fc2(hidden + adapted_bias)


def _make_clean_log():
    """Return a Trace with all outs and metadata for a simple FF model."""
    from torchlens import trace as trace_fn

    model = _SimpleFF()
    return trace_fn(model, torch.randn(2, 5), random_seed=42)


def _make_backward_log() -> Trace:
    """Return a Trace with backward metadata for a simple FF model."""
    from torchlens import trace as trace_fn

    model = _SimpleFF()
    x = torch.randn(2, 5, requires_grad=True)
    log = trace_fn(model, x, save_grads="all", random_seed=42)
    log.log_backward(log[log.output_layers[0]].out.sum())
    return log


def _make_mid_forward_backward_log() -> Trace:
    """Return a Trace with a backward trigger that occurs mid-forward."""
    from torchlens import trace as trace_fn

    model = _MidForwardGradModel()
    x = torch.randn(2, 5, requires_grad=True)
    return trace_fn(model, x, save_grads="all", random_seed=42)


def test_clean_log_ops_all_invariants():
    """An uncorrupted Trace ops all invariant checks."""
    log = _make_clean_log()
    assert check_metadata_invariants(log) is True
    log.cleanup()


def test_clean_log_ops_as_method():
    """check_metadata_invariants works as a bound method on Trace."""
    log = _make_clean_log()
    assert log.check_metadata_invariants() is True
    log.cleanup()


def test_backward_invariants_simple_mlp() -> None:
    """Backward metadata invariants pass on a clean backward trace."""

    log = _make_backward_log()
    assert check_metadata_invariants(log) is True
    log.cleanup()


def test_backward_invariants_with_intervening() -> None:
    """Backward metadata invariants allow intervening grad_fns."""

    log = _make_backward_log()
    assert any(not grad_fn_handle.has_op for grad_fn_handle in log.grad_fn_logs.values())
    assert check_metadata_invariants(log) is True
    log.cleanup()


def test_backward_invariants_allow_only_post_trigger_missing_backpointers() -> None:
    """Mid-forward backward triggers exempt only later-created forward layers."""

    log = _make_mid_forward_backward_log()
    try:
        trigger_positions = [
            event.forward_op_count_at_trigger
            for event in log._capture_events.backward_events
            if event.__class__.__name__ == "BackwardPassStart"
        ]
        assert trigger_positions
        last_trigger_position = max(
            position for position in trigger_positions if position is not None
        )
        assert any(
            layer.grad_fn_object_id is not None
            and layer.grad_fn is None
            and layer.step_index > last_trigger_position
            for layer in log.layer_list
        )

        assert check_metadata_invariants(log) is True
    finally:
        log.cleanup()


def test_bad_pre_trigger_layer_grad_fn_backpointer_raises() -> None:
    """A paired pre-trigger layer with a severed GradFn backpointer raises."""

    log = _make_backward_log()
    try:
        victim = next(
            layer
            for layer in log.layer_list
            if layer.grad_fn_object_id is not None and layer.grad_fn is not None
        )
        victim.grad_fn = None

        with pytest.raises(MetadataInvariantError, match="missing its GradFn backpointer"):
            check_metadata_invariants(log)
    finally:
        log.cleanup()


def test_bad_grad_fn_order_raises() -> None:
    """Unknown grad_fn_handle ids in grad_fn_order raise an invariant error."""

    log = _make_backward_log()
    log.grad_fn_order.append(-1)
    with pytest.raises(MetadataInvariantError, match="backward_graph_invariants"):
        check_metadata_invariants(log)
    log.cleanup()


def test_bad_backward_root_grad_fn_id_raises() -> None:
    """An unknown backward root id raises an invariant error."""

    log = _make_backward_log()
    log.backward_root_grad_fn_object_ids = -1  # type: ignore[assignment]
    with pytest.raises(MetadataInvariantError, match="backward_graph_invariants"):
        check_metadata_invariants(log)
    log.cleanup()


def test_bad_layer_grad_fn_backpointer_raises() -> None:
    """A mismatched layer-to-grad_fn_handle backpointer raises an invariant error."""

    log = _make_backward_log()
    for grad_fn_handle in log.grad_fn_logs.values():
        if grad_fn_handle.has_op:
            grad_fn_handle.op_label = "missing_layer_1"
            break
    with pytest.raises(MetadataInvariantError, match="backward_graph_invariants"):
        check_metadata_invariants(log)
    log.cleanup()


def test_bad_higher_order_creator_chain_order_raises() -> None:
    """A resolved higher-order creator chain with the wrong order raises."""

    class HigherOrderModel(nn.Module):
        """Tiny nonlinear model for higher-order backward metadata."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return a scalar nonlinear output."""

            return (torch.tanh(x) ** 3).sum()

    from torchlens import trace as trace_fn

    x = torch.randn(3, requires_grad=True)
    log = trace_fn(HigherOrderModel(), x, save_grads="all", random_seed=42)
    try:
        loss = log[log.output_layers[0]].out
        first_grad = torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)[0]
        torch.autograd.grad(first_grad.sum(), x, retain_graph=True)
        victim = next(
            grad_fn
            for grad_fn in log.grad_fns
            if grad_fn.creator_object_id is not None and grad_fn.order is not None
        )
        assert victim.order is not None
        victim.order = victim.order + 1

        with pytest.raises(MetadataInvariantError, match="creator order"):
            check_metadata_invariants(log)
    finally:
        log.cleanup()


def test_corruption_parent_child_link():
    """Breaking a parent→child link raises MetadataInvariantError."""
    log = _make_clean_log()
    # Find a layer with children and corrupt
    for lpl in log.layer_list:
        if lpl.children:
            child_label = lpl.children[0]
            child = log[child_label]
            # Remove the parent from the child's parents
            child.parents = [p for p in child.parents if p != lpl.layer_label]
            break
    with pytest.raises(MetadataInvariantError, match="graph_topology"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_num_ops():
    """Mismatched num_ops raises MetadataInvariantError."""
    log = _make_clean_log()
    log.num_ops = 9999
    with pytest.raises(MetadataInvariantError, match="trace_self_consistency"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_module_back_reference():
    """Removing a layer from its module's layers raises MetadataInvariantError."""
    log = _make_clean_log()
    # Find a layer with a containing module and corrupt the Module
    for lpl in log.layer_list:
        cmo = lpl.module
        if cmo:
            # module may include pass (e.g. 'fc:1'), strip it
            cmo_addr = cmo.split(":")[0] if ":" in cmo else cmo
            mod_log = log.modules._dict[cmo_addr]
            if lpl.layer_label in mod_log.layer_labels:
                mod_log.layer_labels = [x for x in mod_log.layer_labels if x != lpl.layer_label]
                # num_layers is a read-only property derived from layer_labels
                break
    with pytest.raises(MetadataInvariantError, match="module_layer_containment"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_layer_num_calls():
    """Wrong layer_num_calls raises MetadataInvariantError."""
    log = _make_clean_log()
    # Corrupt one entry
    first_key = list(log.layer_num_calls.keys())[0]
    log.layer_num_calls[first_key] = 999
    with pytest.raises(MetadataInvariantError, match="recurrence_invariants"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_output_layers_empty():
    """Emptying output_layers raises MetadataInvariantError."""
    log = _make_clean_log()
    log.output_layers = []
    with pytest.raises(MetadataInvariantError, match="trace_self_consistency"):
        check_metadata_invariants(log)
    log.cleanup()


# =============================================================================
# Phase 2: Complex semantic invariant corruption tests (M-R)
# =============================================================================


class _RecurrentFF(nn.Module):
    """Simple recurrent model for loop detection tests."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        for _ in range(3):
            x = self.relu(self.fc(x))
        return x


class _NestedModel(nn.Module):
    """Model with nested submodules for module containment tests."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(5, 4), nn.ReLU())
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        x = self.layer1(x)
        return self.fc(x)


def _make_recurrent_log():
    from torchlens import trace as trace_fn

    return trace_fn(_RecurrentFF(), torch.randn(2, 5), random_seed=42)


def _make_nested_log():
    from torchlens import trace as trace_fn

    return trace_fn(_NestedModel(), torch.randn(2, 5), random_seed=42)


# -- M. Graph ordering corruption --


def test_corruption_graph_ordering_duplicate_rt_num():
    """Duplicate raw_index triggers graph_ordering error."""
    log = _make_clean_log()
    # Set two layers to the same raw_index
    log.layer_list[0].raw_index = log.layer_list[1].raw_index
    with pytest.raises(MetadataInvariantError, match="graph_ordering"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_graph_ordering_topo_violation():
    """Parent with higher raw_index than child triggers error."""
    log = _make_clean_log()
    # Find a layer with parents and swap rt nums to break topo order
    for lpl in log.layer_list:
        if lpl.parents:
            parent = log[lpl.parents[0]]
            # Give parent a higher rt num than child
            parent.raw_index, lpl.raw_index = (
                lpl.raw_index,
                parent.raw_index,
            )
            break
    with pytest.raises(MetadataInvariantError, match="graph_ordering"):
        check_metadata_invariants(log)
    log.cleanup()


# -- N. Loop detection corruption --


def test_corruption_loop_detection_slo_empty():
    """Empty recurrent_ops triggers loop_detection error."""
    log = _make_clean_log()
    log.layer_list[0].recurrent_ops = []
    with pytest.raises(MetadataInvariantError, match="loop_detection"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_loop_detection_slo_asymmetry():
    """Asymmetric recurrent_ops triggers loop_detection error."""
    log = _make_recurrent_log()
    # Find a multi-pass layer and corrupt one member's slo list
    for lpl in log.layer_list:
        if lpl.num_passes > 1:
            # Remove one member from slo
            lpl.recurrent_ops = [lpl.layer_label]
            break
    with pytest.raises(MetadataInvariantError, match="loop_detection"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_loop_detection_ops_total():
    """Mismatched num_calls vs len(recurrent_ops) triggers error."""
    log = _make_clean_log()
    log.layer_list[0].num_passes = 99
    with pytest.raises(MetadataInvariantError, match="loop_detection"):
        check_metadata_invariants(log)
    log.cleanup()


# -- O. Distance / reachability corruption --


def test_corruption_distance_min_gt_max():
    """min_distance > max_distance triggers distance_invariants error."""
    log = _make_clean_log()
    # Find a non-input layer with distances set
    for lpl in log.layer_list:
        if (
            lpl.min_distance_from_input is not None
            and lpl.max_distance_from_input is not None
            and lpl.min_distance_from_input > 0
        ):
            lpl.min_distance_from_input = lpl.max_distance_from_input + 1
            break
    else:
        # If no layer has distances, skip (mark_layer_depths might be False)
        log.cleanup()
        return
    with pytest.raises(MetadataInvariantError, match="distance_invariants"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_distance_input_nonzero():
    """Input layer with nonzero distance_from_input triggers error."""
    log = _make_clean_log()
    if not log.mark_layer_depths:
        log.cleanup()
        return
    for label in log.input_layers:
        lpl = log[label]
        lpl.min_distance_from_input = 5
        lpl.max_distance_from_input = 5
        break
    with pytest.raises(MetadataInvariantError, match="distance_invariants"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_distance_ancestor_flag():
    """Mismatch between has_input_ancestor and input_ancestors triggers error."""
    log = _make_clean_log()
    if not log.mark_layer_depths:
        log.cleanup()
        return
    for lpl in log.layer_list:
        if lpl.has_input_ancestor and len(lpl.input_ancestors) > 0:
            lpl.has_input_ancestor = False
            break
    with pytest.raises(MetadataInvariantError, match="distance_invariants"):
        check_metadata_invariants(log)
    log.cleanup()


# -- P. Graph connectivity corruption --


def test_corruption_connectivity_parentless_layer():
    """Removing all parents from a computational layer triggers error."""
    log = _make_clean_log()
    for lpl in log.layer_list:
        if (
            not lpl.is_input
            and not lpl.is_buffer
            and not lpl.is_output
            and not lpl.is_internal_source
            and lpl.parents
        ):
            # Also fix the parent's child list to avoid graph_topology catching it first
            for p_label in lpl.parents:
                parent = log[p_label]
                parent.children = [c for c in parent.children if c != lpl.layer_label]
                parent.has_children = len(parent.children) > 0
            lpl.parents = []
            # has_parents is a read-only property derived from parents
            break
    with pytest.raises(MetadataInvariantError, match="graph_connectivity"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_connectivity_orphan_in_layer_list():
    """Adding a label to _orphan_labels that is also in layer_labels triggers error."""
    log = _make_clean_log()
    log._orphan_labels = [log.layer_labels[0]]
    with pytest.raises(MetadataInvariantError, match="graph_connectivity"):
        check_metadata_invariants(log)
    log.cleanup()


# -- Q. Module containment logic corruption --


def test_corruption_module_depth():
    """Wrong address_depth on a module triggers error."""
    log = _make_nested_log()
    for mod_log in log.modules:
        if mod_log.address != "self" and mod_log.address_depth > 0:
            mod_log.address_depth = 999
            break
    with pytest.raises(MetadataInvariantError, match="module_containment_logic"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_module_nested_path_leaf():
    """Last element of modules != module triggers error."""
    log = _make_nested_log()
    for lpl in log.layer_list:
        if len(lpl.modules) >= 2 and lpl.module:
            # Swap the last nested module to a different valid module so it
            # doesn't fail the module_layer_containment check but does fail
            # the leaf consistency check in module_containment_logic.
            # Use the first (parent) module as the last entry — valid module but wrong leaf
            lpl.modules[-1] = lpl.modules[0]
            break
    with pytest.raises(MetadataInvariantError, match="module_containment_logic"):
        check_metadata_invariants(log)
    log.cleanup()


# -- R. Lookup key consistency corruption --


def test_corruption_lookup_key_forward():
    """Adding a key to forward dict without reverse entry triggers error."""
    log = _make_clean_log()
    log._lookup_keys_to_layer_num_dict["bogus_key"] = 99999
    with pytest.raises(MetadataInvariantError, match="lookup_key_consistency"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_lookup_key_raw_to_final():
    """Adding a raw→final mapping that points to invalid label triggers error."""
    log = _make_clean_log()
    log._raw_to_final_layer_labels["bogus_raw"] = "bogus_final"
    log._final_to_raw_layer_labels["bogus_final"] = "bogus_raw"
    with pytest.raises(MetadataInvariantError, match="lookup_key_consistency"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_raw_label_asymmetry():
    """Mismatch between raw→final and final→raw triggers error."""
    log = _make_clean_log()
    if log._raw_to_final_layer_labels:
        first_raw = next(iter(log._raw_to_final_layer_labels))
        first_final = log._raw_to_final_layer_labels[first_raw]
        # Point the reverse to a different raw label
        log._final_to_raw_layer_labels[first_final] = "corrupted_raw"
        with pytest.raises(MetadataInvariantError, match="lookup_key_consistency"):
            check_metadata_invariants(log)
    log.cleanup()


# -- Clean recurrent and nested models pass all invariants --


def test_clean_recurrent_log_ops_all_invariants():
    """Recurrent model Trace ops all invariant checks."""
    log = _make_recurrent_log()
    assert check_metadata_invariants(log) is True
    log.cleanup()


def test_clean_nested_log_ops_all_invariants():
    """Nested model Trace ops all invariant checks."""
    log = _make_nested_log()
    assert check_metadata_invariants(log) is True
    log.cleanup()


# =============================================================================
# Bugfix regression tests
# =============================================================================


class _UnusedInputModel(nn.Module):
    """Model that ignores one of its keyword arguments."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x, unused_mask=None):
        return self.fc(x)


class _SharedParamDifferentOps(nn.Module):
    """Model where different operations consume the same parameter.

    The weight is used both in linear and also explicitly via torch.sum.
    """

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        out = self.fc(x)
        weight_sum = torch.sum(self.fc.weight)
        return out + weight_sum


class _SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


class _BatchNormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(10)
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(self.bn(x))


class TestValidationBugfixes:
    """Validation correctness."""

    def test_validation_basic(self):
        model = _SimpleLinear()
        x = torch.randn(2, 10)
        assert validate_forward_pass(model, x)

    def test_validation_batchnorm(self):
        """Validation with BatchNorm (has buffers) should work."""
        model = _BatchNormModel()
        x = torch.randn(4, 10)
        assert validate_forward_pass(model, x)

    def test_validation_unsaved_parent_no_crash(self):
        """Validation with layers_to_save subset should not crash on None parents."""
        from torchlens import trace as trace_fn

        model = _SimpleLinear()
        x = torch.randn(2, 10)
        log = trace_fn(model, x, layers_to_save="all")
        assert log is not None


class TestValidationNoSavedArgs:
    """Validation with save_arg_values=False."""

    def test_validation_no_args(self):
        """validate with save_arg_values=False should not crash."""
        from torchlens import trace as trace_fn

        model = _SimpleLinear()
        x = torch.randn(2, 10)
        log = trace_fn(model, x, save_arg_values=False)
        assert log is not None


class TestPosthocPerturbCheck:
    """posthoc_perturb_check correctly exempts layers with special-value args.

    Note: The original "return on first special arg" behavior is CORRECT —
    any single all-zeros/all-ones arg can explain output invariance.
    """

    def test_batchnorm_validation_with_buffers(self):
        model = _BatchNormModel()
        x = torch.randn(4, 10)
        result = validate_forward_pass(model, x)
        assert result is True


class TestUnusedInputValidation:
    """Regression: unused input kwargs should not crash validation.

    When a model ignores a kwarg (e.g. token_type_ids in DistilBert), the
    corresponding input layer has func=None and no children. Validation
    must skip replay for such layers instead of crashing on None().
    """

    def test_unused_kwarg_input_ops_validation(self):
        model = _UnusedInputModel()
        x = torch.randn(2, 10)
        mask = torch.ones(2, 10)
        assert validate_forward_pass(model, x, input_kwargs={"unused_mask": mask})

    def test_unused_kwarg_input_annotations_ops(self):
        """Metadata invariants pass even with unused input layers."""
        from torchlens import trace as trace_fn

        model = _UnusedInputModel()
        x = torch.randn(2, 10)
        mask = torch.ones(2, 10)
        log = trace_fn(model, x, input_kwargs={"unused_mask": mask})
        check_metadata_invariants(log)


class TestSharedParamDifferentOps:
    """Regression: different operations consuming the same parameter should not
    violate loop_detection invariants.

    The param sharing invariant must group by (func_name, param_barcodes),
    not just param_barcodes alone. Otherwise, e.g. isinf(weight) and expand(weight)
    would be falsely flagged as needing the same layer_label.
    """

    def test_shared_param_different_ops_ops_validation(self):
        model = _SharedParamDifferentOps()
        x = torch.randn(2, 10)
        assert validate_forward_pass(model, x)

    def test_shared_param_different_ops_metadata_ops(self):
        from torchlens import trace as trace_fn

        model = _SharedParamDifferentOps()
        x = torch.randn(2, 10)
        log = trace_fn(model, x)
        check_metadata_invariants(log)


# =============================================================================
# Tripwire: plain capture must NOT emit functionless intervention placeholders
# =============================================================================


class _VmapMaskConsumer(nn.Module):
    """Submodule that consumes an externally built mask tensor."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Add the mask to ``x``."""

        return x + mask


class _VmapMaskModel(nn.Module):
    """Build a fresh mask tensor via ``torch.vmap`` and feed it to a submodule.

    This mirrors how HuggingFace transformers (Mistral, VITS, etc.) construct
    the 4D causal/sliding-window attention mask: the mask is materialized inside
    a ``torch.vmap`` transform, whose internal operations TorchLens cannot trace.
    The fully-formed mask then enters a downstream module untagged. Plain capture
    must register it as a clean internal source -- NOT a functionless
    ``intervention_replacement`` placeholder (which would be a silent capture gap
    papered over by a validation exemption).
    """

    def __init__(self) -> None:
        """Initialize the mask-consuming submodule."""

        super().__init__()
        self.consumer = _VmapMaskConsumer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Build a vmap mask and route it through the consumer submodule."""

        q_idx = torch.arange(x.shape[-1])
        kv_idx = torch.arange(x.shape[-1])

        def cell(i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
            return (j <= i).float()

        mask = torch.vmap(torch.vmap(cell, in_dims=(None, 0)), in_dims=(0, None))(q_idx, kv_idx)
        return self.consumer(x, mask)


def _functionless_replacement_ops(log: "Trace") -> list:
    """Return ops that are auto-synthesized functionless intervention placeholders.

    A GENUINE raw forward-hook output replacement is legitimately functionless,
    but it carries ``func_name == "intervention_replacement"``. This helper
    targets exactly that placeholder shape so a capture gap (an untraced source
    surfacing as an intervention placeholder during plain tracing) is caught.
    """

    return [
        op
        for op in log.ops
        if getattr(op, "func_name", None) == "intervention_replacement"
        and getattr(op, "intervention_replaced", False)
    ]


def test_plain_trace_vmap_mask_has_no_functionless_replacement() -> None:
    """TRIPWIRE: an untraced vmap-built mask must not become a placeholder.

    Reintroducing the old behavior (logging the untagged mask as a functionless
    ``intervention_replacement`` op during plain capture) makes this fail loudly.
    """

    from torchlens import trace as trace_fn

    model = _VmapMaskModel().eval()
    x = torch.randn(4, 4)
    log = trace_fn(model, [x], {})

    # No genuine user intervention happened, so there must be zero functionless
    # intervention placeholders.
    assert _functionless_replacement_ops(log) == []
    assert [op for op in log.ops if getattr(op, "intervention_replaced", False)] == []

    # The mask is instead logged as a transform boundary node with a clean parent edge.
    transform_ops = [op for op in log.ops if getattr(op, "is_transform", False)]
    assert [op.type for op in transform_ops] == ["vmap"]
    assert transform_ops[0].transform_chain == ("vmap", "vmap")
    assert transform_ops[0].parents
    assert all(parent.startswith("arange_") for parent in transform_ops[0].parents)
    transform_label = transform_ops[0].label.split(":")[0]
    assert any(transform_label in op.parents for op in log.ops if op.type == "add")

    # Validation passes legitimately (not via an exemption hiding the gap).
    check_metadata_invariants(log)
    assert validate_forward_pass(model, [x], input_kwargs={})


def test_plain_trace_mistral_has_no_functionless_replacement():
    """TRIPWIRE on the real reproducer: tiny Mistral must trace cleanly.

    The HuggingFace Mistral attention mask is built inside ``torch.vmap``; plain
    tracing must surface it as an internal source, never a functionless
    intervention placeholder.
    """

    transformers = pytest.importorskip("transformers")
    from torchlens import trace as trace_fn

    cfg = transformers.MistralConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=100,
        max_position_embeddings=32,
        sliding_window=16,
    )
    model = transformers.MistralForCausalLM(cfg).eval()
    log = trace_fn(model, [], {"input_ids": torch.randint(0, 100, (1, 16))})

    assert _functionless_replacement_ops(log) == []
    assert [op for op in log.ops if getattr(op, "intervention_replaced", False)] == []
    check_metadata_invariants(log)
