"""Regression tests for capture-time short/friendly label selector matching.

These lock the fix for the class of capture-time ``save=`` selectors that resolve
through the capture label universe (``tl.label``, ``tl.label_contains``, ``tl.label_matches``). At capture
time the base ``RecordContext`` only carries the RAW label (e.g. ``"conv2d_2_4_raw"``);
the short/friendly ``"{layer_type}_{type_index}"`` label (e.g. ``"conv2d_2"``) is
synthesized only by the alias retry in
``torchlens._capture.predicates._evaluate_keep_op``. Before the fix,
``_keep_op_needs_alias_retry`` excluded label-matching structured selectors, so a
short-label ``save=`` selector matched zero sites and warned.
"""

from __future__ import annotations

import warnings

import torch
from torch import nn

import torchlens as tl
from torchlens._capture.predicates import _keep_op_needs_alias_retry


def _build_two_conv_chain() -> tuple[nn.Module, torch.Tensor]:
    """Return a positive two-convolution chain and a unit input.

    Returns
    -------
    tuple[nn.Module, torch.Tensor]
        A ``Sequential`` with two ``conv2d`` ops (short labels ``conv2d_1`` and
        ``conv2d_2``) plus a deterministic input.
    """

    model = nn.Sequential(
        nn.Conv2d(1, 1, 3, padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(1, 1, 3, padding=1, bias=False),
    )
    return model, torch.ones(1, 1, 5, 5)


def _trace_capturing_warnings(save: object) -> tuple[object, list[str]]:
    """Trace the two-conv chain with a save selector and collect zero-match warnings.

    Parameters
    ----------
    save:
        Capture-time ``save=`` selector under test.

    Returns
    -------
    tuple[object, list[str]]
        The captured trace and any "matched zero sites" warning messages.
    """

    model, inputs = _build_two_conv_chain()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(model, inputs, save=save)
    zero_match = [
        str(record.message) for record in caught if "matched zero sites" in str(record.message)
    ]
    return trace, zero_match


def test_short_label_selector_matches_intended_conv() -> None:
    """``tl.label("conv2d_2")`` retains only the second conv and never zero-matches."""

    trace, zero_match = _trace_capturing_warnings(tl.label("conv2d_2"))

    assert zero_match == []
    assert trace["conv2d_2"].has_saved_activation is True
    assert trace["conv2d_1"].has_saved_activation is False
    assert trace["relu_1"].has_saved_activation is False


def test_anchored_regex_selector_matches_short_label() -> None:
    """An anchored regex that only fits the short label still matches at capture time.

    ``^conv2d_2$`` cannot match the raw label ``"conv2d_2_4_raw"``; it fits only the
    synthesized short label ``"conv2d_2"``, so this exercises the alias retry directly.
    """

    trace, zero_match = _trace_capturing_warnings(tl.label_matches(r"^conv2d_2$"))

    assert zero_match == []
    assert trace["conv2d_2"].has_saved_activation is True
    assert trace["conv2d_1"].has_saved_activation is False


def test_contains_selector_matches_short_label() -> None:
    """``tl.label_contains("conv2d_2")`` selects the second conv without a zero-match warning."""

    trace, zero_match = _trace_capturing_warnings(tl.label_contains("conv2d_2"))

    assert zero_match == []
    assert trace["conv2d_2"].has_saved_activation is True


def test_nonexistent_short_label_still_zero_matches() -> None:
    """The alias retry must not over-broaden: a truly absent label still warns."""

    trace, zero_match = _trace_capturing_warnings(tl.label("conv2d_9"))

    assert len(zero_match) == 1
    assert trace["conv2d_2"].has_saved_activation is False
    assert trace["conv2d_1"].has_saved_activation is False


def test_func_selector_control_is_unaffected() -> None:
    """A non-label selector still matches on the base context (both convs saved)."""

    trace, zero_match = _trace_capturing_warnings(tl.func("conv2d"))

    assert zero_match == []
    assert trace["conv2d_1"].has_saved_activation is True
    assert trace["conv2d_2"].has_saved_activation is True


def test_alias_retry_covers_label_matching_selector_kinds() -> None:
    """Lock the precise contract: label-matching kinds need the alias retry.

    ``label``/``contains``/``regex`` resolve through the capture label universe and can target
    the short label only visible in the alias context, so they need the retry; selectors
    that match non-label fields (``func``/``module``/``in_module``/``output``) do not.
    """

    assert _keep_op_needs_alias_retry(tl.label("conv2d_2")) is True
    assert _keep_op_needs_alias_retry(tl.label_contains("conv2d")) is True
    assert _keep_op_needs_alias_retry(tl.label_matches(r"^conv2d_2$")) is True
    assert _keep_op_needs_alias_retry(tl.where(lambda ctx: True)) is True

    assert _keep_op_needs_alias_retry(tl.func("conv2d")) is False
    assert _keep_op_needs_alias_retry(tl.in_module("encoder")) is False
    assert _keep_op_needs_alias_retry(tl.module("encoder")) is False

    # Non-selector callables (legacy predicates) always keep the retry.
    assert _keep_op_needs_alias_retry(lambda ctx: True) is True


def test_composite_selector_with_label_child_keeps_alias_retry() -> None:
    """A composite carrying a label child still needs the alias retry (tree recursion)."""

    composite = tl.func("conv2d") | tl.label("conv2d_2")
    assert _keep_op_needs_alias_retry(composite) is True
