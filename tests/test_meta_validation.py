"""Meta-validation tests: deliberately corrupt saved activations and verify
that validate_saved_activations() catches each corruption.
"""

import copy

import pytest
import torch

from example_models import SimpleFF
from torchlens import log_forward_pass


@pytest.fixture
def valid_mh_and_ground_truth():
    """Return a valid (ModelHistory, ground_truth_tensors) pair for SimpleFF."""
    model = SimpleFF()
    x = torch.rand(2, 3, 32, 32)
    mh = log_forward_pass(model, x, layers_to_save="all", save_function_args=True)
    ground_truth = [mh[label].tensor_contents.clone() for label in mh.output_layers]
    return mh, ground_truth


def test_uncorrupted_passes(valid_mh_and_ground_truth):
    """Sanity check: validation passes when nothing is corrupted."""
    mh, ground_truth = valid_mh_and_ground_truth
    assert mh.validate_saved_activations(ground_truth) is True


def test_corrupt_output_activations(valid_mh_and_ground_truth):
    """Replacing the output layer's tensor_contents with random data should fail."""
    mh, ground_truth = valid_mh_and_ground_truth
    output_label = mh.output_layers[0]
    original = mh[output_label].tensor_contents
    mh[output_label].tensor_contents = torch.randn_like(original)
    assert mh.validate_saved_activations(ground_truth) is False


def test_corrupt_intermediate_activations(valid_mh_and_ground_truth):
    """Replacing a non-output, non-input layer's tensor_contents should fail."""
    mh, ground_truth = valid_mh_and_ground_truth
    # Find an intermediate layer (not input, not output)
    intermediate = [
        label
        for label in mh.layer_labels
        if label not in mh.input_layers and label not in mh.output_layers
    ]
    assert len(intermediate) > 0, "No intermediate layers found"
    target = intermediate[0]
    original = mh[target].tensor_contents
    mh[target].tensor_contents = torch.randn_like(original)
    assert mh.validate_saved_activations(ground_truth) is False


def test_swap_two_layers_activations(valid_mh_and_ground_truth):
    """Swapping tensor_contents between two non-output layers should fail."""
    mh, ground_truth = valid_mh_and_ground_truth
    non_output = [label for label in mh.layer_labels if label not in mh.output_layers]
    assert len(non_output) >= 2, "Need at least 2 non-output layers to swap"
    a, b = non_output[0], non_output[1]
    ta = mh[a].tensor_contents.clone()
    tb = mh[b].tensor_contents.clone()
    # Only swap if they're different shapes or values — otherwise the swap is a no-op
    if ta.shape == tb.shape and torch.equal(ta, tb):
        pytest.skip("Layers have identical tensors; swap is invisible")
    mh[a].tensor_contents = tb
    mh[b].tensor_contents = ta
    assert mh.validate_saved_activations(ground_truth) is False


def test_zero_out_activations(valid_mh_and_ground_truth):
    """Zeroing out a layer's tensor_contents should fail."""
    mh, ground_truth = valid_mh_and_ground_truth
    non_output = [label for label in mh.layer_labels if label not in mh.output_layers]
    assert len(non_output) > 0
    target = non_output[0]
    mh[target].tensor_contents = torch.zeros_like(mh[target].tensor_contents)
    assert mh.validate_saved_activations(ground_truth) is False


def test_add_noise_to_activations(valid_mh_and_ground_truth):
    """Adding gaussian noise to a layer's tensor_contents should fail."""
    mh, ground_truth = valid_mh_and_ground_truth
    non_output = [label for label in mh.layer_labels if label not in mh.output_layers]
    assert len(non_output) > 0
    target = non_output[0]
    original = mh[target].tensor_contents
    mh[target].tensor_contents = original + torch.randn_like(original) * 0.1
    assert mh.validate_saved_activations(ground_truth) is False


def test_scale_activations(valid_mh_and_ground_truth):
    """Scaling a layer's tensor_contents by a large scalar should fail."""
    mh, ground_truth = valid_mh_and_ground_truth
    non_output = [label for label in mh.layer_labels if label not in mh.output_layers]
    assert len(non_output) > 0
    target = non_output[0]
    mh[target].tensor_contents = mh[target].tensor_contents * 100.0
    assert mh.validate_saved_activations(ground_truth) is False


def test_wrong_shape_activations(valid_mh_and_ground_truth):
    """Replacing tensor_contents with a wrong-shaped tensor should fail."""
    mh, ground_truth = valid_mh_and_ground_truth
    output_label = mh.output_layers[0]
    mh[output_label].tensor_contents = torch.randn(1, 1)
    assert mh.validate_saved_activations(ground_truth) is False


def test_corrupt_creation_args(valid_mh_and_ground_truth):
    """Modifying saved function arguments (creation_args) should fail."""
    mh, ground_truth = valid_mh_and_ground_truth
    # Find a non-input layer that has creation_args with tensors
    for label in mh.layer_labels:
        entry = mh[label]
        if entry.is_input_layer:
            continue
        if entry.creation_args and any(isinstance(a, torch.Tensor) for a in entry.creation_args):
            for i, arg in enumerate(entry.creation_args):
                if isinstance(arg, torch.Tensor):
                    corrupted_args = list(entry.creation_args)
                    corrupted_args[i] = torch.randn_like(arg)
                    entry.creation_args = tuple(corrupted_args)
                    assert mh.validate_saved_activations(ground_truth) is False
                    return
    pytest.skip("No layer with tensor creation_args found")
