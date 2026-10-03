"""Observe items 7-8: saved-band decomposition + trace-global first-save identity.

The parameter phantom: most of the bytes autograd retains on real models are
PARAMETER storages already counted in the parameter band, so a naive
parameter+autograd_saved stack double-plots them. The storage-class
decomposition (saved_parameter / saved_buffer / saved_activation, keyed on
``untyped_storage().data_ptr()`` on BOTH sides) and the trace-global
first-save counter (``newly_saved_bytes``) make the stack honest. The
replacement oracles here compute truth a DIFFERENT way: torch's own
``saved_tensors_hooks`` is the independent storage-identity oracle.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class _MLP(nn.Module):
    """Two linears around a tanh: params + activations both get saved."""

    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(8, 8)
        self.l2 = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear -> tanh -> linear."""

        return self.l2(torch.tanh(self.l1(x)))


class _BufferScale(nn.Module):
    """Multiplies by a registered buffer so autograd saves BUFFER storage."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.full((4,), 2.0))
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The mul's backward saves ``scale`` (a buffer storage)."""

        return self.linear(x) * self.scale


def _traced_bands(model: nn.Module, x: torch.Tensor) -> tuple[tl.Trace, dict[str, dict[str, int]]]:
    """Capture with backward_ready and return (trace, per-raw-label bands)."""

    captured = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(backward_ready=True),
        save_mode="reference",
    )
    return captured, dict(captured.__dict__.get("_autograd_saved_bands", {}))


def test_classes_sum_to_gross_band_per_op_and_total() -> None:
    """Oracle (i): the three storage classes sum EXACTLY to the gross band."""

    model = _MLP()
    captured, bands = _traced_bands(model, torch.randn(2, 8, requires_grad=True))
    try:
        assert bands, "backward-ready capture must produce saved-band rows"
        per_op_gross = {}
        for op in captured:
            raw = getattr(op, "raw_label", None)
            gross = getattr(op, "autograd_memory", None)
            if gross:
                per_op_gross[str(raw)] = int(gross)
        total_gross = sum(per_op_gross.values())
        class_total = 0
        for band in bands.values():
            class_sum = band["saved_parameter"] + band["saved_buffer"] + band["saved_activation"]
            class_total += class_sum
        assert class_total == total_gross == int(captured.total_autograd_memory)
    finally:
        captured.cleanup()


def test_param_slice_is_pointerwise_subset_of_parameter_band() -> None:
    """Oracle (ii): saved_parameter bytes come from REAL parameter storages.

    Storage identity, never byte comparison: a transposed weight VIEW inside
    the saved set must classify as saved_parameter (linear's AddmmBackward
    saves ``weight.t()``, an offset view -- keying on ``tensor.data_ptr()``
    would have misclassified it as an activation).
    """

    model = _MLP()
    captured, bands = _traced_bands(model, torch.randn(2, 8, requires_grad=True))
    try:
        param_bytes = sum(
            parameter.numel() * parameter.element_size() for parameter in model.parameters()
        )
        weight_bytes = sum(
            parameter.numel() * parameter.element_size()
            for name, parameter in model.named_parameters()
            if name.endswith("weight")
        )
        saved_param_total = sum(band["saved_parameter"] for band in bands.values())
        # Linear backwards save exactly the weights (views), never the biases.
        assert saved_param_total == weight_bytes
        assert saved_param_total <= param_bytes
    finally:
        captured.cleanup()


def test_cumulative_newly_saved_matches_saved_tensors_hooks_oracle() -> None:
    """Oracle (iii): first-save totals equal torch's own saved-storage census.

    The independent oracle runs the SAME model outside TorchLens under
    ``torch.autograd.graph.saved_tensors_hooks`` and sums each unique saved
    storage once -- computing the truth a different way.
    """

    torch.manual_seed(0)
    model = _MLP()
    x = torch.randn(2, 8, requires_grad=True)

    captured, bands = _traced_bands(model, x)
    try:
        cumulative_newly_saved = sum(band["newly_saved_bytes"] for band in bands.values())
    finally:
        captured.cleanup()

    seen: dict[int, int] = {}

    def _pack(tensor: torch.Tensor) -> torch.Tensor:
        storage = tensor.untyped_storage()
        seen.setdefault(storage.data_ptr(), int(storage.nbytes()))
        return tensor

    def _unpack(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(_pack, _unpack):
        model(x.detach().clone().requires_grad_(True))
    oracle_unique_total = sum(seen.values())
    assert cumulative_newly_saved == oracle_unique_total


def test_buffer_storage_classifies_as_saved_buffer() -> None:
    """A registered buffer inside the saved set lands in saved_buffer."""

    model = _BufferScale()
    captured, bands = _traced_bands(model, torch.randn(2, 4, requires_grad=True))
    try:
        buffer_total = sum(band["saved_buffer"] for band in bands.values())
        assert buffer_total == model.scale.numel() * model.scale.element_size()
    finally:
        captured.cleanup()


def test_param_saved_fixed_while_activation_saved_scales_with_batch() -> None:
    """The decomposition is USEFUL, not merely correct: param slice is fixed.

    'N MB held for backward, of which M MB scales with your batch' is the
    sentence the split buys: the param-saved slice must be byte-identical
    across batch sizes while activation-saved grows.
    """

    def _totals(batch: int) -> tuple[int, int]:
        model = _MLP()
        captured, bands = _traced_bands(model, torch.randn(batch, 8, requires_grad=True))
        try:
            return (
                sum(band["saved_parameter"] for band in bands.values()),
                sum(band["saved_activation"] for band in bands.values()),
            )
        finally:
            captured.cleanup()

    small_param, small_act = _totals(2)
    large_param, large_act = _totals(8)
    assert small_param == large_param
    assert large_act > small_act


def test_phantom_stack_regression_params_never_double_stack() -> None:
    """The naive parameter+gross-saved stack overstates; the split does not.

    The honest stackable saved series is ``newly_saved_bytes`` (plus the
    saved_parameter ANNOTATION on the parameter band); gross-band stacking
    double-plots every saved parameter byte.
    """

    model = _MLP()
    captured, bands = _traced_bands(model, torch.randn(2, 8, requires_grad=True))
    try:
        param_band = sum(
            parameter.numel() * parameter.element_size() for parameter in model.parameters()
        )
        gross = int(captured.total_autograd_memory)
        newly_param = sum(band["newly_saved_parameter"] for band in bands.values())
        newly_activation = sum(band["newly_saved_activation"] for band in bands.values())
        newly_buffer = sum(band["newly_saved_buffer"] for band in bands.values())
        newly_total = sum(band["newly_saved_bytes"] for band in bands.values())
        assert newly_param + newly_activation + newly_buffer == newly_total
        assert newly_param > 0, "the phantom requires saved parameter storages"
        naive_stack = param_band + gross
        honest_stack = param_band + newly_activation
        # The naive stack double-plots exactly the parameter storages autograd
        # re-holds (plus re-saved activations); the honest stack never does.
        assert naive_stack > honest_stack
        assert naive_stack - honest_stack >= newly_param
    finally:
        captured.cleanup()
