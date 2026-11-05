"""View-mutation toy models: mutations through views log and validate cleanly.

Split from ``test_toy_models.py`` (its size ceiling); the models live in
``example_models``.
"""

import example_models

import torchlens as tl
from torchlens import trace
from torchlens.validation import validate_forward_pass


def test_multiple_view_mutations(input_2d):
    """Multiple views mutated independently should be logged without error."""
    model = example_models.MultipleViewMutations()
    assert validate_forward_pass(model, input_2d)
    mh = trace(model, input_2d, capture=tl.options.CaptureOptions(save_args=True))
    assert mh is not None
    assert len(mh.layer_labels) > 0
