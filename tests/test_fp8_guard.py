"""fp8 (``float8_*``) dtypes: real comparisons, not crashes and not exemptions.

fp8 tensors report ``dtype.is_floating_point == True`` while torch implements no
``isinf`` / ``isfinite`` / ``nan_to_num`` / ``allclose`` / reduction kernels for them.
Every TorchLens numeric path that branched on ``is_floating_point`` therefore walked
into a raw ``NotImplementedError: "isinf" not implemented for 'Float8_e4m3fn'`` --
or, where the caller swallowed ``RuntimeError`` (which ``NotImplementedError``
subclasses), silently declined to run a tripwire at all.

The fix widens those comparisons to float32, which is EXACT: every one of the 256 bit
patterns of every fp8 variant round-trips bit-identically, and NaN patterns stay NaN.
``test_every_fp8_bit_pattern_survives_the_widening_exactly`` is the load-bearing proof
of that claim -- if it ever fails, the widening became an exemption and the tripwire
tests below are no longer trustworthy.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.data_classes._nonfinite import nonfinite_layers, uncheckable_payload_count
from torchlens.errors import CaptureError
from torchlens.utils._torch_compat import get_fp8_dtypes, get_torch_capability_snapshot
from torchlens.utils.tensor_utils import (
    fp8_safe_comparison_pair,
    fp8_widen_for_numeric_ops,
    tensor_nanequal,
)


def _fp8_dtypes() -> tuple[torch.dtype, ...]:
    """Return this build's fp8 dtypes, skipping the test when there are none.

    Returns
    -------
    tuple[torch.dtype, ...]
        Available ``float8_*`` dtypes.
    """

    dtypes = tuple(get_fp8_dtypes(force_probe=True))
    if not dtypes:
        pytest.skip("this torch build exposes no float8 dtypes")
    return dtypes


class _Fp8CastModel(nn.Module):
    """Model that casts an activation to fp8 mid-forward and back.

    This is the realistic shape: the fp8 tensor is produced *inside* the forward, so
    nothing inspectable before capture can see it.
    """

    def __init__(self, dtype: torch.dtype) -> None:
        """Store the fp8 dtype to cast through.

        Parameters
        ----------
        dtype:
            fp8 dtype for the intermediate activation.
        """

        super().__init__()
        self.fc = nn.Linear(4, 4)
        self._fp8_dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a linear layer through an fp8 round trip.

        Parameters
        ----------
        x:
            Input batch.

        Returns
        -------
        torch.Tensor
            float32 output.
        """

        quantized = self.fc(x).to(self._fp8_dtype)
        return quantized.abs().float() + 1.0


# ---------------------------------------------------------------------------
# Capability probe
# ---------------------------------------------------------------------------


def test_fp8_dtypes_are_feature_probed_not_version_parsed() -> None:
    """The dtype set comes from ``torch`` attributes, never a version string."""

    dtypes = _fp8_dtypes()
    assert all(isinstance(dtype, torch.dtype) for dtype in dtypes)
    assert torch.float8_e4m3fn in dtypes


def test_fp8_capability_flag_is_in_the_snapshot() -> None:
    """The degradation point is visible through the capability snapshot."""

    snapshot = get_torch_capability_snapshot()
    assert "HAS_FP8_DTYPES" in snapshot
    assert isinstance(snapshot["HAS_FP8_DTYPES"], bool)


# ---------------------------------------------------------------------------
# The widening is exact -- the premise every test below depends on
# ---------------------------------------------------------------------------


def test_every_fp8_bit_pattern_survives_the_widening_exactly() -> None:
    """All 256 fp8 bit patterns round-trip through float32 bit-identically.

    This is what makes the widening a faithful comparison rather than a loosened
    one. NaN patterns are checked as "still NaN" because ``torch.equal`` is IEEE and
    fp8 NaN encodings are not unique.
    """

    raw = torch.arange(256, dtype=torch.uint8)
    for dtype in _fp8_dtypes():
        original = raw.view(dtype)
        widened = fp8_widen_for_numeric_ops(original)
        assert widened.dtype is torch.float32
        narrowed = widened.to(dtype)
        nan_mask = original.isnan()
        assert torch.equal(narrowed.view(torch.uint8)[~nan_mask], raw[~nan_mask]), (
            f"{dtype} lost a finite value in the float32 round trip"
        )
        assert bool(narrowed.isnan()[nan_mask].all()), f"{dtype} lost a NaN in the round trip"


def test_widening_leaves_non_fp8_tensors_untouched() -> None:
    """Ordinary dtypes must pass through as the same object."""

    for dtype in (torch.float32, torch.float64, torch.bfloat16, torch.int64, torch.bool):
        tensor = torch.ones(3, dtype=dtype)
        assert fp8_widen_for_numeric_ops(tensor) is tensor
    left = torch.ones(3)
    right = torch.zeros(3)
    assert fp8_safe_comparison_pair(left, right) == (left, right)


# ---------------------------------------------------------------------------
# tensor_nanequal: answers instead of NotImplementedError, and still discriminates
# ---------------------------------------------------------------------------


def test_tensor_nanequal_matches_identical_fp8_payloads() -> None:
    """Equal fp8 tensors compare equal, NaN positions included."""

    for dtype in _fp8_dtypes():
        values = torch.tensor([1.0, 2.0, float("nan")]).to(dtype)
        assert tensor_nanequal(values, values.clone()) is True


def test_tensor_nanequal_rejects_differing_fp8_payloads() -> None:
    """A genuine value difference must still FAIL -- with and without tolerance."""

    dtype = torch.float8_e4m3fn
    left = torch.tensor([1.0, 2.0, float("nan")]).to(dtype)
    right = torch.tensor([1.0, 3.0, float("nan")]).to(dtype)
    assert tensor_nanequal(left, right) is False
    assert tensor_nanequal(left, right, allow_tolerance=True) is False


def test_tensor_nanequal_rejects_a_shifted_fp8_nan_mask() -> None:
    """A NaN facing a real value must never read equal."""

    dtype = torch.float8_e4m3fn
    with_nan = torch.tensor([1.0, 2.0, float("nan")]).to(dtype)
    without_nan = torch.tensor([1.0, 2.0, 4.0]).to(dtype)
    assert tensor_nanequal(with_nan, without_nan) is False
    assert tensor_nanequal(without_nan, with_nan) is False


def test_tensor_nanequal_rejects_a_single_ulp_fp8_difference_even_with_tolerance() -> None:
    """fp8's coarse epsilon must not become the comparison tolerance.

    One fp8 ULP is a ~12.5% relative step for e4m3. Widening deliberately keeps the
    float32-grade tolerance so a real neighbouring-value difference fails loudly
    instead of being absorbed.
    """

    dtype = torch.float8_e4m3fn
    base = torch.tensor([1.0, 1.0], dtype=torch.float32).to(dtype)
    neighbour_bits = base.view(torch.uint8).clone()
    neighbour_bits[0] += 1  # next representable fp8 value
    neighbour = neighbour_bits.view(dtype)

    assert not torch.equal(base.float(), neighbour.float())
    assert tensor_nanequal(base, neighbour, allow_tolerance=True) is False


@pytest.mark.parametrize("dtype_name", ["float8_e5m2fnuz", "float8_e8m0fnu"])
def test_tensor_nanequal_rejects_adjacent_tiny_fp8_values_with_tolerance(
    dtype_name: str,
) -> None:
    """Absolute float32 tolerance must not absorb tiny adjacent fp8 values.

    Parameters
    ----------
    dtype_name:
        Narrow fp8 variant whose first two encodings differ by less than ``1e-5``.
    """

    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        pytest.skip(f"{dtype_name} is unavailable")
    left = torch.tensor([0], dtype=torch.uint8).view(dtype)
    right = torch.tensor([1], dtype=torch.uint8).view(dtype)
    assert not torch.equal(left.float(), right.float())
    assert tensor_nanequal(left, right, allow_tolerance=True) is False


# ---------------------------------------------------------------------------
# Capture + validation on a model that really produces fp8 activations
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_fp8_capture_records_the_cast_activation() -> None:
    """Capture logs the fp8 op and reports its real dtype."""

    trace = tl.trace(_Fp8CastModel(torch.float8_e4m3fn), torch.randn(2, 4))
    labels = [label for label in trace.layer_labels if label.startswith("to_")]
    assert labels, f"expected an fp8 cast op, saw {trace.layer_labels}"
    assert trace[labels[0]].out.dtype is torch.float8_e4m3fn


@pytest.mark.heavy
@pytest.mark.parametrize("scope", ["forward", "saved"])
def test_fp8_validation_replay_passes_instead_of_raising(scope: str) -> None:
    """Validation used to die with NotImplementedError from inside replay."""

    assert tl.validate(_Fp8CastModel(torch.float8_e4m3fn), torch.randn(2, 4), scope=scope) is True


@pytest.mark.heavy
def test_raise_on_nan_fires_on_a_non_finite_fp8_activation() -> None:
    """The NaN tripwire must not silently skip fp8 tensors.

    ``torch.isfinite`` has no fp8 kernel and the caller swallows ``RuntimeError``,
    which ``NotImplementedError`` subclasses -- so before the widening this check
    quietly returned without inspecting a single fp8 activation.
    """

    class Fp8NonFinite(nn.Module):
        """Model whose fp8 cast is the FIRST non-finite tensor in the graph.

        e5m2 tops out around 57344, so scaling a finite float32 activation past that
        overflows to inf *at the cast*. Every earlier tensor stays finite, which is
        what makes this a test of the fp8 check rather than of the float32 one.
        """

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Overflow a finite float32 activation into fp8 inf.

            Parameters
            ----------
            x:
                Input batch.

            Returns
            -------
            torch.Tensor
                float32 view of the non-finite fp8 activation.
            """

            return (x * 1e5).to(torch.float8_e5m2).float()

    with pytest.raises(CaptureError) as excinfo:
        tl.trace(
            Fp8NonFinite(),
            torch.ones(1, 2),
            capture=tl.options.CaptureOptions(raise_on_nan=True),
        )
    message = str(excinfo.value)
    assert "non-finite" in message
    assert "float8_e5m2" in message, f"the fp8 cast should be the flagged op, got: {message}"


@pytest.mark.heavy
def test_unrunnable_nonfinite_check_warns_instead_of_reading_as_clean(monkeypatch) -> None:
    """An unrunnable check is not a clean tensor.

    fp8 was the known dtype without an ``isfinite`` kernel and is now widened, but the
    call site still swallows ``RuntimeError`` (which ``NotImplementedError``
    subclasses). Any future such dtype must be DISCLOSED rather than silently skipped:
    the user explicitly asked for NaN checking, so silence would let them read an
    unchecked forward as a checked one.

    Parameters
    ----------
    monkeypatch:
        pytest monkeypatch fixture, used to make ``torch.isfinite`` unavailable for one
        dtype so the real production fallback runs.
    """

    real_isfinite = torch.isfinite

    def refusing_isfinite(tensor: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        """Reject bfloat16 the way a missing kernel would.

        Parameters
        ----------
        tensor:
            Tensor to test for finiteness.
        *args:
            Passed through.
        **kwargs:
            Passed through.

        Returns
        -------
        torch.Tensor
            Boolean finiteness mask for every other dtype.
        """

        if tensor.dtype is torch.bfloat16:
            raise NotImplementedError("\"isfinite\" not implemented for 'BFloat16'")
        return real_isfinite(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "isfinite", refusing_isfinite)

    class BFloatModel(nn.Module):
        """Model producing several bfloat16 activations."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Cast through bfloat16 with a real NaN in the activation.

            The two-stage check (snoop D4) screens every tensor through a
            float32 accumulator, so a kernel-less dtype only reaches its own
            ``isfinite`` call on a TRIPPED screen -- the NaN makes the screen
            trip and the exact stage-2 check hit the refusing kernel.

            Parameters
            ----------
            x:
                Input batch.

            Returns
            -------
            torch.Tensor
                float32 output.
            """

            reduced = (x.abs() + 1).to(torch.bfloat16) / 0.0
            # clamp() maps the +inf back to a finite value so the only
            # nonfinite activation on the forward is the bfloat16 one --
            # every float32 op stays finite and the real raise_on_nan abort
            # never fires.
            return reduced.clamp(-1.0, 1.0).to(torch.float32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.trace(
            BFloatModel(),
            torch.randn(2, 4),
            capture=tl.options.CaptureOptions(raise_on_nan=True),
        )

    skipped = [item for item in caught if "UNCHECKED for NaN/Inf" in str(item.message)]
    assert len(skipped) == 1, f"expected exactly one disclosure, got {len(skipped)}"
    text = str(skipped[0].message)
    assert "raise_on_nan could not check" in text
    assert "bfloat16" in text


@pytest.mark.heavy
def test_clean_fp8_capture_does_not_trip_raise_on_nan() -> None:
    """A finite fp8 activation must not be reported as non-finite."""

    trace = tl.trace(
        _Fp8CastModel(torch.float8_e4m3fn),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(raise_on_nan=True),
    )
    assert trace.num_params > 0


# ---------------------------------------------------------------------------
# compat.report tells the user before they capture
# ---------------------------------------------------------------------------


def test_compat_report_flags_fp8_state() -> None:
    """An fp8 buffer is reported, with the .tlspec consequence named."""

    class Fp8Buffer(nn.Module):
        """Model holding an fp8 buffer."""

        def __init__(self) -> None:
            """Register a linear layer and an fp8 scale buffer."""

            super().__init__()
            self.fc = nn.Linear(4, 4)
            self.register_buffer("scale", torch.ones(4).to(torch.float8_e4m3fn))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply the linear layer.

            Parameters
            ----------
            x:
                Input batch.

            Returns
            -------
            torch.Tensor
                Layer output.
            """

            return self.fc(x)

    row = tl.compat.report(Fp8Buffer(), torch.randn(2, 4)).row("fp8_dtype")
    assert row.detected is True
    assert row.status == "scope"
    assert "float32" in row.details
    assert ".tlspec" in row.details


def test_compat_report_fp8_row_is_quiet_and_honest_when_clean() -> None:
    """No fp8 state reports pass -- while disclosing what it could not see."""

    row = tl.compat.report(nn.Linear(4, 4), torch.randn(2, 4)).row("fp8_dtype")
    assert row.detected is False
    assert row.status == "pass"
    assert "inside the forward" in row.details


def test_compat_report_flags_an_fp8_input() -> None:
    """An fp8 model input is detected too."""

    fp8_input = torch.randn(2, 4).to(torch.float8_e4m3fn)
    row = tl.compat.report(nn.Linear(4, 4), fp8_input).row("fp8_dtype")
    assert row.detected is True
    assert "inputs" in row.details


# ---------------------------------------------------------------------------
# Portable save stays a typed refusal, not a silent drop
# ---------------------------------------------------------------------------


@pytest.mark.heavy
def test_saving_an_fp8_activation_refuses_typed(tmp_path) -> None:
    """safetensors has no fp8 transport; the refusal must name the dtype.

    Parameters
    ----------
    tmp_path:
        pytest temporary directory.
    """

    from torchlens.errors import TorchLensIOError

    trace = tl.trace(_Fp8CastModel(torch.float8_e4m3fn), torch.randn(2, 4))
    with pytest.raises(TorchLensIOError) as excinfo:
        tl.save(trace, str(tmp_path / "fp8.tlspec"))
    assert "float8_e4m3fn" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The reporting-side non-finite scan: a dtype nobody can check is NOT "clean"
# ---------------------------------------------------------------------------


def _first_nonfinite_bit_pattern(dtype: torch.dtype) -> int:
    """Return a byte value that decodes to a non-finite value in ``dtype``.

    Derived by widening all 256 patterns rather than hardcoding per-variant NaN
    encodings (e4m3fn NaN is ``0x7F``, e4m3fnuz's is ``0x80``, e8m0fnu's is ``0xFF``),
    so the test cannot drift from the dtype it claims to cover.

    Parameters
    ----------
    dtype:
        fp8 dtype to search.

    Returns
    -------
    int
        Byte value whose fp8 interpretation is NaN or Inf.
    """

    widened = torch.arange(256, dtype=torch.uint8).view(dtype).to(torch.float32)
    nonfinite = torch.nonzero(~torch.isfinite(widened)).flatten()
    if nonfinite.numel() == 0:  # pragma: no cover - every shipped variant has one
        pytest.skip(f"{dtype} encodes no non-finite value")
    return int(nonfinite[0].item())


class _Fp8NanBitPattern(nn.Module):
    """Bit-view a finite integer payload into an fp8 non-finite value.

    The fp8 activation is the ONLY non-finite payload in the capture: the input, the
    integer arithmetic, and the float32 tail are all finite, so a scan that cannot
    check fp8 has nothing else to trip on and answers "clean". That is what makes
    this a test of the fp8 check and not of the float32 one.
    """

    def __init__(self, dtype: torch.dtype, bit_pattern: int) -> None:
        """Store the fp8 dtype and the byte value to reinterpret.

        Parameters
        ----------
        dtype:
            fp8 dtype to view the bytes as.
        bit_pattern:
            Byte value that decodes to NaN or Inf in ``dtype``.
        """

        super().__init__()
        self.dtype = dtype
        self.bit_pattern = bit_pattern

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a finite float32 tensor via a non-finite fp8 intermediate.

        Parameters
        ----------
        x:
            Input batch.

        Returns
        -------
        torch.Tensor
            Finite float32 activation.
        """

        bits = (x * 0 + float(self.bit_pattern)).to(torch.uint8)
        return bits.view(self.dtype).to(torch.uint8).float() + 1.0


@pytest.mark.heavy
def test_report_scan_finds_an_fp8_nan_the_native_kernel_cannot_see() -> None:
    """``print(trace)``'s verdict must not read "clean" over an unchecked fp8 payload.

    The scan behind ``Trace.first_nonfinite`` / ``_repr_html_`` / ``report.explain``
    swallowed ``RuntimeError`` and answered ``False``, so an all-NaN fp8 activation
    produced the whole-capture verdict "No non-finite tensor values found in saved
    outs" -- a false clean, the same disarmed tripwire as the ``raise_on_nan``
    swallow one layer down.
    """

    for dtype in _fp8_dtypes():
        pattern = _first_nonfinite_bit_pattern(dtype)
        trace = tl.trace(_Fp8NanBitPattern(dtype, pattern), torch.ones(4))
        answer = trace.first_nonfinite(link_format="text")
        assert "First non-finite saved out" in answer, f"{dtype} read as clean: {answer}"
        assert str(dtype) in answer, f"{dtype} was not named as the culprit: {answer}"
        assert len(nonfinite_layers(trace, kind="saved")) == 1, (
            f"{dtype}: exactly the fp8 activation should be flagged"
        )
        assert uncheckable_payload_count(trace, kind="saved") == 0, (
            f"{dtype} is checked by widening, so it must not be reported as unchecked"
        )


def test_no_fp8_variant_can_be_trusted_to_its_native_finiteness_kernel() -> None:
    """Pin the premise the scan fix rests on, per variant.

    For every fp8 dtype torch exposes, the native ``isfinite`` either raises
    ``NotImplementedError`` or -- ``float8_e8m0fnu`` -- returns ``True`` for a NaN
    pattern, i.e. answers WRONGLY. The exact float32 widening is right for all of
    them. If a future torch ships correct fp8 kernels this test fails loudly, which
    is the intended signal to revisit the widening rather than to keep it by inertia.
    """

    trustworthy: list[str] = []
    for dtype in _fp8_dtypes():
        pattern = _first_nonfinite_bit_pattern(dtype)
        payload = torch.tensor([pattern], dtype=torch.uint8).view(dtype)
        assert bool((~torch.isfinite(fp8_widen_for_numeric_ops(payload))).all()), (
            f"the widened check must see {dtype}'s non-finite pattern"
        )
        try:
            native_ok = bool((~torch.isfinite(payload)).all())
        except NotImplementedError:
            continue
        if native_ok:
            trustworthy.append(str(dtype))
    assert trustworthy == ["torch.float8_e5m2"], (
        "native fp8 finiteness support changed; revisit the widening deliberately, "
        f"trustworthy variants now: {trustworthy}"
    )


@pytest.mark.heavy
def test_a_dtype_with_no_finiteness_kernel_is_disclosed_not_called_finite() -> None:
    """A quantized payload yields no evidence, and the verdict must say so.

    Widening fixes fp8, but quantized and sparse payloads still have no runnable
    ``isfinite``. Silently folding "could not look" into a clean answer is exactly
    the dishonesty this phase removes, so the count is disclosed instead.
    """

    class Quantized(nn.Module):
        """Model that produces a qint8 activation mid-forward."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return a float32 activation via a quantized intermediate.

            Parameters
            ----------
            x:
                Input batch.

            Returns
            -------
            torch.Tensor
                Dequantized float32 activation.
            """

            quantized = torch.quantize_per_tensor(x, 0.1, 0, torch.qint8)
            return quantized.dequantize() + 1.0

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="torch")
        trace = tl.trace(Quantized(), torch.ones(4))
    assert uncheckable_payload_count(trace, kind="saved") == 1
    answer = trace.first_nonfinite()
    assert "no runnable finiteness check" in answer, answer
    assert answer != "No non-finite tensor values found in saved outs.", (
        "an unhedged clean verdict must not cover a payload that was never checked"
    )
    # ``report.explain``'s anomaly bullet publishes the same verdict standalone.
    anomaly = [
        line
        for line in tl.report.explain(trace).splitlines()
        if "NaN or Inf" in line or "no runnable finiteness check" in line
    ]
    assert anomaly, "the report must still carry an anomaly verdict"
    assert all("no runnable finiteness check" in line for line in anomaly), (
        f"report.explain published an unhedged clean bill of health: {anomaly}"
    )


@pytest.mark.heavy
def test_both_coverage_gaps_are_disclosed_together() -> None:
    """An unsaved op and an uncheckable dtype in one capture are both named."""

    class Quantized(nn.Module):
        """Model with a qint8 activation and several float32 ops around it."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return a float32 activation via a quantized intermediate.

            Parameters
            ----------
            x:
                Input batch.

            Returns
            -------
            torch.Tensor
                Dequantized float32 activation.
            """

            quantized = torch.quantize_per_tensor(torch.relu(x), 0.1, 0, torch.qint8)
            return quantized.dequantize() + 1.0

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="torch")
        trace = tl.trace(
            Quantized(),
            torch.ones(4),
            save=tl.func("quantize_per_tensor") | tl.func("relu"),
        )
    answer = trace.first_nonfinite()
    assert "retained no payload" in answer, answer
    assert "no runnable finiteness check" in answer, answer


@pytest.mark.heavy
def test_a_clean_fp8_capture_keeps_the_unhedged_clean_answer() -> None:
    """fp8 is really checked, so it adds no hedge to an otherwise clean capture.

    This is the other half of honesty: the disclosure must not fire for payloads the
    scan genuinely examined, or every fp8 capture would read as partially unchecked.
    """

    trace = tl.trace(_Fp8CastModel(torch.float8_e4m3fn), torch.randn(2, 4))
    assert uncheckable_payload_count(trace, kind="saved") == 0
    assert trace.first_nonfinite() == "No non-finite tensor values found in saved outs."


@pytest.mark.heavy
def test_the_memo_never_serves_a_stale_clean_verdict_for_fp8() -> None:
    """The scan is memoized per log; the second question must agree with the first."""

    dtype = torch.float8_e4m3fn
    trace = tl.trace(_Fp8NanBitPattern(dtype, _first_nonfinite_bit_pattern(dtype)), torch.ones(4))
    first = trace.first_nonfinite(link_format="text")
    assert first == trace.first_nonfinite(link_format="text")
    assert "First non-finite saved out" in first


# --------------------------------------------------------------------------- #
# Sparse layouts (R65): the layout sibling of the fp8 raw-kernel-escape class  #
# --------------------------------------------------------------------------- #


class _SparseOutputModel(nn.Module):
    """Model whose forward ends in a sparse COO output."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x)).to_sparse()


def test_tensor_nanequal_is_sparse_coo_safe() -> None:
    """Sparse COO pairs compare structurally instead of hitting aten::equal."""

    dense = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
    other = torch.tensor([[0.0, 1.0], [3.0, 0.0]])
    assert tensor_nanequal(dense.to_sparse(), dense.clone().to_sparse())
    assert not tensor_nanequal(dense.to_sparse(), other.to_sparse())
    nan_a = torch.tensor([[float("nan"), 1.0]]).to_sparse()
    nan_b = torch.tensor([[float("nan"), 1.0]]).to_sparse()
    assert tensor_nanequal(nan_a, nan_b)


def test_tensor_nanequal_is_sparse_csr_safe() -> None:
    """Compressed layouts compare via their canonical index/value components."""

    dense = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # CSR beta-state advisory
        assert tensor_nanequal(dense.to_sparse_csr(), dense.clone().to_sparse_csr())
        assert not tensor_nanequal(
            dense.to_sparse_csr(), torch.tensor([[0.0, 1.0], [3.0, 0.0]]).to_sparse_csr()
        )


def test_sparse_coo_model_output_traces_complete() -> None:
    """A forward ending in .to_sparse() completes instead of crashing raw.

    Fail-before: tl.trace() escaped with a raw torch-internal
    ``NotImplementedError: Could not run 'aten::equal' ... SparseCPU`` out of
    tensor_nanequal via the output-attribution comparison.
    """

    log = tl.trace(_SparseOutputModel(), torch.ones(2, 3))
    assert log.outcome.status.name == "COMPLETE"
