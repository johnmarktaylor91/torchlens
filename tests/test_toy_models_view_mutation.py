"""View-mutation toy models: mutations through views log and validate cleanly.

Split from ``test_toy_models.py`` (its size ceiling); the models live in
``example_models``.
"""

import example_models

import torchlens as tl
from torchlens import trace
from torchlens.validation import validate_forward_pass


def test_view_mutation_unsqueeze(input_2d):
    """Mutation through unsqueeze view should be logged without error."""
    model = example_models.ViewMutationUnsqueeze()
    assert validate_forward_pass(model, input_2d)
    mh = trace(model, input_2d, capture=tl.options.CaptureOptions(save_args=True))
    assert mh is not None
    assert len(mh.layer_labels) > 0


def test_view_mutation_reshape(input_2d):
    """Mutation through reshape view should be logged without error."""
    model = example_models.ViewMutationReshape()
    assert validate_forward_pass(model, input_2d)
    mh = trace(model, input_2d, capture=tl.options.CaptureOptions(save_args=True))
    assert mh is not None
    assert len(mh.layer_labels) > 0


def test_view_mutation_transpose(input_2d):
    """Mutation through transpose view should be logged without error."""
    model = example_models.ViewMutationTranspose()
    assert validate_forward_pass(model, input_2d)
    mh = trace(model, input_2d, capture=tl.options.CaptureOptions(save_args=True))
    assert mh is not None
    assert len(mh.layer_labels) > 0


def test_multiple_view_mutations(input_2d):
    """Multiple views mutated independently should be logged without error."""
    model = example_models.MultipleViewMutations()
    assert validate_forward_pass(model, input_2d)
    mh = trace(model, input_2d, capture=tl.options.CaptureOptions(save_args=True))
    assert mh is not None
    assert len(mh.layer_labels) > 0


def test_chained_view_mutation(input_2d):
    """Mutation through chained views should be logged without error."""
    model = example_models.ChainedViewMutation()
    assert validate_forward_pass(model, input_2d)
    mh = trace(model, input_2d, capture=tl.options.CaptureOptions(save_args=True))
    assert mh is not None
    assert len(mh.layer_labels) > 0


def test_output_matches_parent_no_false_positive(input_2d):
    """No mutation model: verify no false-positive child tensor variations."""
    model = example_models.OutputMatchesParent()
    assert validate_forward_pass(model, input_2d)
    mh = trace(model, input_2d, capture=tl.options.CaptureOptions(save_args=True))
    assert mh is not None
    assert len(mh.layer_labels) > 0
    # No layer should have child tensor variations since nothing is mutated
    for label in mh.layer_labels:
        entry = mh[label]
        assert not entry.has_out_variations, (
            f"Layer {label} should not have child tensor variations in a "
            f"mutation-free model, but has_out_variations={entry.has_out_variations}"
        )
