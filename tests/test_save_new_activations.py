"""Tests for save_new_activations — the fast-path activation re-extraction.

Covers: successful re-extraction on simple models, multiple sequential calls,
activation value correctness, and the known failure mode on models with
identity-propagated operations (Bug #39).
"""

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
# Known failure: torchvision models with identity-propagated ops (Bug #39)
# =============================================================================


@pytest.mark.slow
def test_save_new_activations_alexnet_fails():
    """AlexNet save_new_activations fails due to identity op counter misalignment.

    Bug #39: Identity operations (where output tensor has same barcode as input)
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
