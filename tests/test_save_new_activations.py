"""Tests for save_new_activations — the fast-path activation re-extraction.

Covers: successful re-extraction on simple models, multiple sequential calls,
activation value correctness, and the known failure mode on models with
identity-propagated operations.
"""

import warnings

import pytest
import torch
import torch.nn as nn

from torchlens import log_forward_pass, check_metadata_invariants


# =============================================================================
# Test models
# =============================================================================


class _SimpleFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 3)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class _RecurrentFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        for _ in range(3):
            x = self.relu(self.fc(x))
        return x


class _BranchingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 5)
        self.fc2 = nn.Linear(5, 3)

    def forward(self, x):
        h = self.fc1(x)
        return self.fc2(h) + h.sum()


# =============================================================================
# Positive tests — save_new_activations works correctly
# =============================================================================


@pytest.mark.smoke
def test_save_new_activations_basic():
    """save_new_activations replaces activations on a simple model."""
    model = _SimpleFF()
    x1 = torch.randn(2, 5)
    log = log_forward_pass(model, x1, random_seed=42)

    x2 = torch.randn(2, 5)
    log.save_new_activations(model, x2, random_seed=42)

    # Verify output matches direct model execution
    model.eval()
    with torch.no_grad():
        expected = model(x2)
    actual = log[log.output_layers[0]].tensor_contents
    assert torch.allclose(actual, expected, atol=1e-5)
    log.cleanup()


def test_save_new_activations_multiple_calls():
    """save_new_activations works correctly on repeated calls."""
    model = _SimpleFF()
    log = log_forward_pass(model, torch.randn(2, 5), random_seed=42)

    for i in range(5):
        x = torch.randn(2, 5)
        log.save_new_activations(model, x, random_seed=42)

    # Still valid after 5 calls
    assert len(log.layer_list) > 0
    assert log[log.output_layers[0]].has_saved_activations
    log.cleanup()


def test_save_new_activations_activations_change():
    """Activations actually change when new input is provided."""
    model = _SimpleFF()
    torch.manual_seed(0)
    x1 = torch.randn(2, 5)
    log = log_forward_pass(model, x1, random_seed=42)
    act1 = log[log.output_layers[0]].tensor_contents.clone()

    torch.manual_seed(99)
    x2 = torch.randn(2, 5)
    log.save_new_activations(model, x2, random_seed=42)
    act2 = log[log.output_layers[0]].tensor_contents.clone()

    assert not torch.equal(act1, act2), "Activations should differ for different inputs"
    log.cleanup()


def test_save_new_activations_metadata_preserved():
    """Metadata invariants hold after save_new_activations."""
    model = _SimpleFF()
    log = log_forward_pass(model, torch.randn(2, 5), random_seed=42)
    log.save_new_activations(model, torch.randn(2, 5), random_seed=42)

    assert check_metadata_invariants(log) is True
    log.cleanup()


def test_save_new_activations_recurrent():
    """save_new_activations works on recurrent models."""
    model = _RecurrentFF()
    log = log_forward_pass(model, torch.randn(2, 5), random_seed=42)
    assert log.model_is_recurrent

    log.save_new_activations(model, torch.randn(2, 5), random_seed=42)
    assert log[log.output_layers[0]].has_saved_activations
    log.cleanup()


def test_save_new_activations_branching():
    """save_new_activations works on branching models."""
    model = _BranchingModel()
    log = log_forward_pass(model, torch.randn(2, 5), random_seed=42)

    log.save_new_activations(model, torch.randn(2, 5), random_seed=42)
    assert log[log.output_layers[0]].has_saved_activations
    log.cleanup()


def test_save_new_activations_layers_to_save():
    """save_new_activations respects layers_to_save parameter."""
    model = _SimpleFF()
    log = log_forward_pass(model, torch.randn(2, 5), random_seed=42)

    # Only save the first layer
    first_label = log.layer_labels[0]
    log.save_new_activations(model, torch.randn(2, 5), random_seed=42, layers_to_save=[first_label])

    assert log[first_label].has_saved_activations
    log.cleanup()


# =============================================================================
# Known failure: torchvision models with identity-propagated ops
# =============================================================================


@pytest.mark.slow
def test_save_new_activations_alexnet_fails():
    """AlexNet save_new_activations fails due to identity op counter misalignment.

    Identity operations (where output tensor has same barcode as input)
    are detected and dropped in the exhaustive pass via label propagation
    (decorate_torch.py:98-100). The fast pass counter diverges because identity
    detection may not fire identically, producing raw labels not in the original
    _raw_to_final_layer_labels mapping.

    This is a known limitation — models with identity-propagated ops require
    a fresh log_forward_pass call instead of save_new_activations.
    """
    torchvision = pytest.importorskip("torchvision")
    model = torchvision.models.alexnet(weights=None)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    log = log_forward_pass(model, x, random_seed=42)

    with pytest.raises(ValueError, match="computational graph changed"):
        log.save_new_activations(model, torch.randn(1, 3, 224, 224), random_seed=42)
    log.cleanup()


@pytest.mark.slow
def test_save_new_activations_resnet_fails():
    """ResNet18 also fails save_new_activations (same identity op issue)."""
    torchvision = pytest.importorskip("torchvision")
    model = torchvision.models.resnet18(weights=None)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    log = log_forward_pass(model, x, random_seed=42)

    with pytest.raises(ValueError, match="computational graph changed"):
        log.save_new_activations(model, torch.randn(1, 3, 224, 224), random_seed=42)
    log.cleanup()


# =============================================================================
# Bugfix regression tests
# =============================================================================


class _SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


class _SharedBufferModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor([2.0]))
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        x = x * self.scale
        x = self.fc(x)
        x = x * self.scale
        return x


class TestSaveNewActivationsRegression:
    """Zombie LayerPassLogs on repeated calls."""

    def test_save_new_activations_3x(self):
        """3+ sequential save_new_activations calls should not crash."""
        model = _SimpleLinear()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x)
        for _ in range(3):
            log.save_new_activations(model, torch.randn(2, 10))

    def test_save_new_activations_different_values(self):
        """Activations should change with new inputs."""
        model = _SimpleLinear()
        x1 = torch.randn(2, 10)
        log = log_forward_pass(model, x1)
        first_output = log[log.output_layers[0]].tensor_contents.clone()
        x2 = torch.randn(2, 10) + 10
        log.save_new_activations(model, x2)
        second_output = log[log.output_layers[0]].tensor_contents
        assert not torch.equal(first_output, second_output)


class TestSaveNewActivationsStateReset:
    """Stale state in save_new_activations."""

    def test_timing_reset(self):
        """elapsed_time_function_calls should be fresh."""
        model = _SimpleLinear()
        log = log_forward_pass(model, torch.randn(2, 10), layers_to_save="all")
        log.save_new_activations(model, torch.randn(2, 10), layers_to_save="all")
        assert log.elapsed_time_function_calls >= 0

    def test_lookup_keys_clean(self):
        """Lookup caches should not have stale entries."""
        model = _SimpleLinear()
        log = log_forward_pass(model, torch.randn(2, 10), layers_to_save="all")
        labels_pass1 = set(log.layer_labels)
        log.save_new_activations(model, torch.randn(2, 10), layers_to_save="all")
        labels_pass2 = set(log.layer_labels)
        assert labels_pass1 == labels_pass2

    def test_5x_stress(self):
        """Stress test: 5 sequential save_new_activations calls."""
        model = _SimpleLinear()
        log = log_forward_pass(model, torch.randn(2, 10), layers_to_save="all")
        for i in range(5):
            log.save_new_activations(model, torch.randn(2, 10), layers_to_save="all")
            assert log.num_tensors_saved > 0

    def test_different_values(self):
        """Each pass should reflect new input values."""
        model = _SimpleLinear()
        log = log_forward_pass(model, torch.ones(2, 10), layers_to_save="all")
        input_val_1 = log["input_1"].tensor_contents.clone()
        log.save_new_activations(model, torch.zeros(2, 10), layers_to_save="all")
        input_val_2 = log["input_1"].tensor_contents
        assert not torch.equal(input_val_1, input_val_2)


class TestOutputTensorIndependence:
    """Fast-mode tensor_contents shared reference."""

    def test_output_independent_of_parent(self):
        model = _SimpleLinear()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x)
        log.save_new_activations(model, torch.randn(2, 10))
        for label in log.output_layers:
            output_entry = log[label]
            if output_entry.parent_layers and output_entry.tensor_contents is not None:
                parent_label = output_entry.parent_layers[0]
                parent_entry = log[parent_label]
                if parent_entry.tensor_contents is not None:
                    original_parent = parent_entry.tensor_contents.clone()
                    output_entry.tensor_contents.fill_(999)
                    assert torch.equal(parent_entry.tensor_contents, original_parent)
                    break


class TestFastPathModuleLogs:
    """postprocess_fast should preserve module logs from exhaustive pass."""

    def test_fast_path_preserves_module_logs(self):
        model = _SimpleLinear()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x)
        original_module_count = len(log.modules)
        original_addresses = [m.address for m in log.modules]
        assert original_module_count > 0
        log.save_new_activations(model, torch.randn(2, 10))
        assert len(log.modules) == original_module_count
        assert [m.address for m in log.modules] == original_addresses


class TestDescriptiveValueError:
    """log_source_tensor_fast should give descriptive error on graph change."""

    def test_dynamic_graph_descriptive_error(self):
        class DynamicModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 10)
                self.linear2 = nn.Linear(10, 10)
                self.call_count = 0

            def forward(self, x):
                self.call_count += 1
                x = self.linear1(x)
                if self.call_count > 1:
                    x = self.linear2(x)
                    x = torch.relu(x)
                    x = self.linear2(x)
                return x

        model = DynamicModel()
        log = log_forward_pass(model, torch.randn(2, 10))
        with pytest.raises(ValueError, match="computational graph changed"):
            log.save_new_activations(model, torch.randn(2, 10))


class TestFastPassBufferOrphan:
    """Fast-pass should not KeyError on models with shared buffers."""

    def test_shared_buffer_fast_path(self):
        model = _SharedBufferModel()
        log = log_forward_pass(model, torch.randn(2, 10), random_seed=42)
        # Should not raise KeyError on fast pass
        log.save_new_activations(model, torch.randn(2, 10), random_seed=42)
        assert log[log.output_layers[0]].has_saved_activations
        log.cleanup()

    def test_shared_buffer_fast_path_3x(self):
        model = _SharedBufferModel()
        log = log_forward_pass(model, torch.randn(2, 10), random_seed=42)
        for _ in range(3):
            log.save_new_activations(model, torch.randn(2, 10), random_seed=42)
        assert log[log.output_layers[0]].has_saved_activations
        log.cleanup()


class TestGraphConsistencyValidation:
    """log_source_tensor_fast warns on shape mismatch."""

    def test_shape_mismatch_warns(self):
        model = _SimpleLinear()
        log = log_forward_pass(model, torch.randn(2, 10))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            log.save_new_activations(model, torch.randn(4, 10))
            shape_warnings = [x for x in w if "shape changed" in str(x.message)]
            assert len(shape_warnings) > 0
