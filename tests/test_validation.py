"""Tests for the validation subpackage.

Covers: import paths, registry consistency, perturbation unit tests,
deep clone helpers, and integration tests through specific exemption paths.
"""

import pytest
import torch
import torch.nn as nn

from torchlens import validate_forward_pass, validate_saved_activations, ModelLog
from torchlens import check_metadata_invariants, MetadataInvariantError
from torchlens.validation import validate_saved_activations as validate_from_subpkg
from torchlens.validation.exemptions import (
    SKIP_VALIDATION_ENTIRELY,
    SKIP_PERTURBATION_ENTIRELY,
    STRUCTURAL_ARG_POSITIONS,
    CUSTOM_EXEMPTION_CHECKS,
)
from torchlens.validation.core import (
    _perturb_layer_activations,
    _deep_clone_tensors,
    _copy_validation_args,
    MAX_PERTURB_ATTEMPTS,
)


# =============================================================================
# Import / binding tests
# =============================================================================


def test_validation_import_path():
    """from torchlens.validation import validate_saved_activations works."""
    assert callable(validate_from_subpkg)


def test_validate_forward_pass_importable():
    """validate_forward_pass is importable from torchlens top-level."""
    assert callable(validate_forward_pass)


def test_check_metadata_invariants_importable():
    """check_metadata_invariants and MetadataInvariantError importable from top-level."""
    assert callable(check_metadata_invariants)
    assert issubclass(MetadataInvariantError, ValueError)


def test_model_log_validate_method_bound():
    """ModelLog.validate_saved_activations is callable."""
    assert hasattr(ModelLog, "validate_saved_activations")
    assert callable(ModelLog.validate_saved_activations)


def test_model_log_check_metadata_method_bound():
    """ModelLog.check_metadata_invariants is callable."""
    assert hasattr(ModelLog, "check_metadata_invariants")
    assert callable(ModelLog.check_metadata_invariants)


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


def test_perturbation_changes_float_tensor():
    parent = torch.randn(10, 10)
    output = torch.randn(10, 10)
    perturbed = _perturb_layer_activations(parent, output)
    assert not torch.equal(perturbed, parent)
    assert perturbed.shape == parent.shape


def test_perturbation_changes_int_tensor():
    parent = torch.randint(0, 100, (10, 10))
    output = torch.randn(10, 10)
    perturbed = _perturb_layer_activations(parent, output)
    assert not torch.equal(perturbed, parent)
    assert perturbed.dtype == parent.dtype


def test_perturbation_changes_bool_tensor():
    parent = torch.ones(10, 10, dtype=torch.bool)
    output = torch.randn(10, 10)
    perturbed = _perturb_layer_activations(parent, output)
    # With 100 elements all True, random should differ
    assert not torch.equal(perturbed, parent)
    assert perturbed.dtype == torch.bool


def test_perturbation_changes_complex_tensor():
    parent = torch.complex(torch.randn(5, 5), torch.randn(5, 5))
    output = torch.randn(5, 5)
    perturbed = _perturb_layer_activations(parent, output)
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
        perturbed = _perturb_layer_activations(parent, output)
        assert perturbed.dtype == dtype


def test_perturbation_handles_empty_tensor():
    parent = torch.tensor([])
    output = torch.tensor([])
    perturbed = _perturb_layer_activations(parent, output)
    assert perturbed.numel() == 0
    assert torch.equal(perturbed, parent)


def test_perturbation_terminates_on_scalar():
    """MAX_PERTURB_ATTEMPTS guard prevents infinite loop on single-element tensors."""
    # Single-element bool tensor: 50% chance each attempt matches original.
    # With MAX_PERTURB_ATTEMPTS=100, it should terminate regardless.
    parent = torch.tensor([True])
    output = torch.tensor([1.0])
    perturbed = _perturb_layer_activations(parent, output)
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


def _make_clean_log():
    """Return a ModelLog with all activations and metadata for a simple FF model."""
    from torchlens import log_forward_pass

    model = _SimpleFF()
    return log_forward_pass(model, torch.randn(2, 5), random_seed=42)


def test_clean_log_passes_all_invariants():
    """An uncorrupted ModelLog passes all invariant checks."""
    log = _make_clean_log()
    assert check_metadata_invariants(log) is True
    log.cleanup()


def test_clean_log_passes_as_method():
    """check_metadata_invariants works as a bound method on ModelLog."""
    log = _make_clean_log()
    assert log.check_metadata_invariants() is True
    log.cleanup()


def test_corruption_parent_child_link():
    """Breaking a parent→child link raises MetadataInvariantError."""
    log = _make_clean_log()
    # Find a layer with children and corrupt
    for lpl in log.layer_list:
        if lpl.child_layers:
            child_label = lpl.child_layers[0]
            child = log[child_label]
            # Remove the parent from the child's parent_layers
            child.parent_layers = [p for p in child.parent_layers if p != lpl.layer_label]
            break
    with pytest.raises(MetadataInvariantError, match="graph_topology"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_num_operations():
    """Mismatched num_operations raises MetadataInvariantError."""
    log = _make_clean_log()
    log.num_operations = 9999
    with pytest.raises(MetadataInvariantError, match="model_log_self_consistency"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_module_back_reference():
    """Removing a layer from its module's all_layers raises MetadataInvariantError."""
    log = _make_clean_log()
    # Find a layer with a containing module and corrupt the ModuleLog
    for lpl in log.layer_list:
        cmo = lpl.containing_module_origin
        if cmo:
            # containing_module_origin may include pass (e.g. 'fc:1'), strip it
            cmo_addr = cmo.split(":")[0] if ":" in cmo else cmo
            mod_log = log.modules._dict[cmo_addr]
            if lpl.layer_label_no_pass in mod_log.all_layers:
                mod_log.all_layers = [x for x in mod_log.all_layers if x != lpl.layer_label_no_pass]
                mod_log.num_layers = len(mod_log.all_layers)
                break
    with pytest.raises(MetadataInvariantError, match="module_layer_containment"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_layer_num_passes():
    """Wrong layer_num_passes raises MetadataInvariantError."""
    log = _make_clean_log()
    # Corrupt one entry
    first_key = list(log.layer_num_passes.keys())[0]
    log.layer_num_passes[first_key] = 999
    with pytest.raises(MetadataInvariantError, match="recurrence_invariants"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_output_layers_empty():
    """Emptying output_layers raises MetadataInvariantError."""
    log = _make_clean_log()
    log.output_layers = []
    with pytest.raises(MetadataInvariantError, match="model_log_self_consistency"):
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
    from torchlens import log_forward_pass

    return log_forward_pass(_RecurrentFF(), torch.randn(2, 5), random_seed=42)


def _make_nested_log():
    from torchlens import log_forward_pass

    return log_forward_pass(_NestedModel(), torch.randn(2, 5), random_seed=42)


# -- M. Graph ordering corruption --


def test_corruption_graph_ordering_duplicate_rt_num():
    """Duplicate realtime_tensor_num triggers graph_ordering error."""
    log = _make_clean_log()
    # Set two layers to the same realtime_tensor_num
    log.layer_list[0].realtime_tensor_num = log.layer_list[1].realtime_tensor_num
    with pytest.raises(MetadataInvariantError, match="graph_ordering"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_graph_ordering_topo_violation():
    """Parent with higher realtime_tensor_num than child triggers error."""
    log = _make_clean_log()
    # Find a layer with parents and swap rt nums to break topo order
    for lpl in log.layer_list:
        if lpl.parent_layers:
            parent = log[lpl.parent_layers[0]]
            # Give parent a higher rt num than child
            parent.realtime_tensor_num, lpl.realtime_tensor_num = (
                lpl.realtime_tensor_num,
                parent.realtime_tensor_num,
            )
            break
    with pytest.raises(MetadataInvariantError, match="graph_ordering"):
        check_metadata_invariants(log)
    log.cleanup()


# -- N. Loop detection corruption --


def test_corruption_loop_detection_slo_empty():
    """Empty same_layer_operations triggers loop_detection error."""
    log = _make_clean_log()
    log.layer_list[0].same_layer_operations = []
    with pytest.raises(MetadataInvariantError, match="loop_detection"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_loop_detection_slo_asymmetry():
    """Asymmetric same_layer_operations triggers loop_detection error."""
    log = _make_recurrent_log()
    # Find a multi-pass layer and corrupt one member's slo list
    for lpl in log.layer_list:
        if lpl.layer_passes_total > 1:
            # Remove one member from slo
            lpl.same_layer_operations = [lpl.layer_label]
            break
    with pytest.raises(MetadataInvariantError, match="loop_detection"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_loop_detection_passes_total():
    """Mismatched layer_passes_total vs len(same_layer_operations) triggers error."""
    log = _make_clean_log()
    log.layer_list[0].layer_passes_total = 99
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
        # If no layer has distances, skip (mark_input_output_distances might be False)
        log.cleanup()
        return
    with pytest.raises(MetadataInvariantError, match="distance_invariants"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_distance_input_nonzero():
    """Input layer with nonzero distance_from_input triggers error."""
    log = _make_clean_log()
    if not log.mark_input_output_distances:
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
    if not log.mark_input_output_distances:
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
            not lpl.is_input_layer
            and not lpl.is_buffer_layer
            and not lpl.is_output_layer
            and not lpl.initialized_inside_model
            and lpl.parent_layers
        ):
            # Also fix the parent's child list to avoid graph_topology catching it first
            for p_label in lpl.parent_layers:
                parent = log[p_label]
                parent.child_layers = [c for c in parent.child_layers if c != lpl.layer_label]
                parent.has_children = len(parent.child_layers) > 0
            lpl.parent_layers = []
            lpl.has_parents = False
            break
    with pytest.raises(MetadataInvariantError, match="graph_connectivity"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_connectivity_orphan_in_layer_list():
    """Adding a label to orphan_layers that is also in layer_labels triggers error."""
    log = _make_clean_log()
    log.orphan_layers = [log.layer_labels[0]]
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
    """Last element of containing_modules_origin_nested != containing_module_origin triggers error."""
    log = _make_nested_log()
    for lpl in log.layer_list:
        if len(lpl.containing_modules_origin_nested) >= 2 and lpl.containing_module_origin:
            # Swap the last nested module to a different valid module so it
            # doesn't fail the module_layer_containment check but does fail
            # the leaf consistency check in module_containment_logic.
            # Use the first (parent) module as the last entry — valid module but wrong leaf
            lpl.containing_modules_origin_nested[-1] = lpl.containing_modules_origin_nested[0]
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


def test_clean_recurrent_log_passes_all_invariants():
    """Recurrent model ModelLog passes all invariant checks."""
    log = _make_recurrent_log()
    assert check_metadata_invariants(log) is True
    log.cleanup()


def test_clean_nested_log_passes_all_invariants():
    """Nested model ModelLog passes all invariant checks."""
    log = _make_nested_log()
    assert check_metadata_invariants(log) is True
    log.cleanup()
