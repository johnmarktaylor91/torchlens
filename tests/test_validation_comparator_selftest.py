"""Entry self-test for the shared replay comparator (R75-4).

``tensor_nanequal`` is the single judge behind every per-op replay verdict
AND capture-side alias/mutation bookkeeping, with no oracle above it: a
degradation making it vacuously true would blind the whole forward-replay
tripwire while the suite stays green (the b9-fable independence table's
"dominant hub" finding; kill margin measured at ONE test in R74-1).
``validate_saved_outs`` now proves the judge on known sentinel pairs before
trusting any verdict it produces.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.validation import core as validation_core

pytestmark = pytest.mark.smoke


def _traced_linear() -> tuple[tl.Trace, list[torch.Tensor], nn.Module]:
    """Capture a tiny linear model with full saves for replay validation.

    Returns
    -------
    tuple[tl.Trace, list[torch.Tensor], nn.Module]
        The trace, its ground-truth output tensors, AND the source model.
        The model rides along on purpose: capture releases the direct param
        references and rehydrates them through the source-model weakref, so a
        caller validating a trace must keep its model alive (dropping it here
        made the healthy-path test hostage to gc timing -- it raised
        PostTraceParamUnavailable whenever a collection happened to run
        between capture and validation).
    """

    model = nn.Sequential(nn.Linear(4, 3)).eval()
    x = torch.randn(2, 4)
    with torch.no_grad():
        ground_truth = model(x)
    trace = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )
    return trace, [ground_truth], model


def test_degraded_comparator_refuses_to_validate(monkeypatch: pytest.MonkeyPatch) -> None:
    """A vacuously-true comparator must abort validation, never bless it.

    Red-capable: before the entry self-test, this exact degradation made
    ``validate_saved_outs`` return a PASSING status (the comparator blesses
    every replay), which is the judge-corruption class the finding names.
    """

    trace, ground_truth, _model = _traced_linear()
    monkeypatch.setattr(validation_core, "tensor_nanequal", lambda *args, **kwargs: True)
    with pytest.raises(RuntimeError, match="comparator self-test failed"):
        validation_core.validate_saved_outs(trace, ground_truth)


def test_always_false_comparator_also_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """An always-False judge is equally untrustworthy and must abort."""

    trace, ground_truth, _model = _traced_linear()
    monkeypatch.setattr(validation_core, "tensor_nanequal", lambda *args, **kwargs: False)
    with pytest.raises(RuntimeError, match="comparator self-test failed"):
        validation_core.validate_saved_outs(trace, ground_truth)


def test_healthy_comparator_validates_normally() -> None:
    """The self-test is invisible on a healthy comparator."""

    trace, ground_truth, _model = _traced_linear()
    status = validation_core.validate_saved_outs(trace, ground_truth)
    assert bool(status)


def test_nan_doctrine_is_part_of_the_self_test(monkeypatch: pytest.MonkeyPatch) -> None:
    """A judge that loses the NaN-pattern doctrine must also abort.

    ``equal_nan`` semantics are load-bearing (identical NaN patterns are
    agreement; NaN-vs-number is a mismatch); a comparator that starts
    treating NaN-vs-number as equal is exactly the vacuous-under-NaN
    degradation the backward degeneracy guard exists to catch.
    """

    real = validation_core.tensor_nanequal

    def _nan_blind(a: torch.Tensor, b: torch.Tensor, **kwargs: object) -> bool:
        if bool(torch.isnan(a).any()) or bool(torch.isnan(b).any()):
            return True
        return bool(real(a, b, **kwargs))

    trace, ground_truth, _model = _traced_linear()
    monkeypatch.setattr(validation_core, "tensor_nanequal", _nan_blind)
    with pytest.raises(RuntimeError, match="comparator self-test failed"):
        validation_core.validate_saved_outs(trace, ground_truth)


class TestSignedZeroDoctrine:
    """sol+fable r4 probes: ``torch.equal`` reads ``-0.0 == +0.0`` as True,
    so the comparator certified a sign-flipped-zero replay as EXACT even
    though the payloads are bit-distinct and diverge through ``1/x``. A flip
    is now refused at the exact tier and admitted only by the tolerance
    band; NaN sign stays out of scope (kernels legitimately differ on it)."""

    def test_signed_zero_flip_is_not_exact(self) -> None:
        from torchlens.utils.tensor_utils import tensor_nanequal

        neg = torch.tensor([-0.0, 1.0])
        pos = torch.tensor([0.0, 1.0])
        assert not tensor_nanequal(neg, pos)
        assert not tensor_nanequal(pos, neg)

    def test_signed_zero_flip_is_within_tolerance(self) -> None:
        from torchlens.utils.tensor_utils import tensor_nanequal

        neg = torch.tensor([-0.0, 1.0])
        pos = torch.tensor([0.0, 1.0])
        assert tensor_nanequal(neg, pos, allow_tolerance=True)

    def test_matching_negative_zeros_stay_exact(self) -> None:
        from torchlens.utils.tensor_utils import tensor_nanequal

        neg = torch.tensor([-0.0, 0.0, 1.0])
        assert tensor_nanequal(neg, neg.clone())

    def test_complex_component_signed_zero_flip_is_not_exact(self) -> None:
        from torchlens.utils.tensor_utils import tensor_nanequal

        a = torch.tensor([complex(0.0, 0.0), 1 + 2j])
        b = torch.tensor([complex(-0.0, 0.0), 1 + 2j])
        assert not tensor_nanequal(a, b)
        assert tensor_nanequal(a, a.clone())

    def test_nan_sign_is_out_of_scope(self) -> None:
        from torchlens.utils.tensor_utils import tensor_nanequal

        plus_nan = torch.tensor([float("nan"), 1.0])
        minus_nan = torch.tensor([-float("nan"), 1.0])
        assert tensor_nanequal(plus_nan, minus_nan)

    def test_self_test_carries_the_signed_zero_sentinel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A regression re-blessing signed-zero flips as exact must be
        caught at validation ENTRY by the comparator self-test."""

        from torchlens.utils import tensor_utils

        real = tensor_utils.tensor_nanequal

        def reblessed(a: torch.Tensor, b: torch.Tensor, allow_tolerance: bool = False) -> bool:
            if real(a, b, allow_tolerance=allow_tolerance):
                return True
            # Simulate the pre-fix comparator: IEEE equality certifies the
            # flip as exact again.
            return bool(
                a.shape == b.shape
                and a.dtype == b.dtype
                and a.dtype.is_floating_point
                and torch.equal(a, b)
            )

        monkeypatch.setattr(validation_core, "tensor_nanequal", reblessed)
        with pytest.raises(RuntimeError, match="self-test failed"):
            validation_core._comparator_self_test()
