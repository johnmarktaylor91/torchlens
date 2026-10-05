"""Tensor utilities: NaN-aware comparison, memory calculation, safe_copy, safe device transfer.

Many functions in this module use ``pause_logging()`` to temporarily disable
the torchlens logging toggle before calling tensor custom_methods.  This is
necessary because tensor custom_methods like ``.clone()``, ``.to()``,
``.nelement()``, and ``.element_size()`` are all decorated at import time
(see ``decoration/torch_funcs.py``).  Without pausing, these internal calls
would be logged as user operations, creating spurious entries and, in the
case of ``safe_copy`` called from *inside* the logging pipeline, infinite
recursion.

The ``_clean_*`` function imports (e.g. ``_clean_clone``) MUST be resolved
before decoration runs, since after decoration the module-level names point
to wrapped versions.
"""

import copy
import threading
import warnings
import weakref
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from math import prod
from typing import Any, Literal, cast, get_args

import torch

from ..backends.torch._tl import get_tensor_label, set_tensor_label
from ._torch_compat import (
    HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE,
    get_fp8_dtypes,
    get_functorch_wrapped_tensor_checker,
)
from ._torch_symbols import torch_attr
from .env_flags import closed_bool_env

SaveMode = Literal["copy", "reference", "view", "cpu_async"]

#: Runtime authority for SaveMode membership checks: derived from the Literal
#: (typing.get_args) so a vocabulary change cannot drift from the validators.
SAVE_MODES: frozenset[str] = frozenset(get_args(SaveMode))

# Replay comparison tolerances are DERIVED from each dtype's machine epsilon
# rather than spelled as decimal literals, so every dtype gets the same
# strictness measured in its own ULPs.  Error model for a faithful replay of
# one op (same kernel family, same device, possibly different accumulation
# order / thread count):
#
# * fp32 / fp64 payloads accumulate in their own precision; reduction-order
#   round-off for the shallow (< band-C-depth) ops this tolerance covers is a
#   small multiple of eps, so the headroom is 512 ULP (~6e-5 relative for
#   fp32, a mild tightening of the former 1e-4 literal's ~840 ULP) -- far
#   below any real corruption (a sign flip, a zeroed value, a stale buffer
#   all read as many thousands of ULPs) while comfortably above observed
#   reorder noise (~4 ULP on eval MHA, see _runnable_path_faithfulness.py).
# * fp16 / bf16 payloads accumulate in fp32 and round ONCE to storage, so the
#   replay difference is storage-rounding dominated: a few ULPs of the
#   storage dtype. Headroom 4 ULP -- a MODEL bound (unmeasured), unlike the
#   fp32 row's measured ~4-ULP eval-MHA reorder observation cited above.
#
# The absolute term exists ONLY to absorb jitter at the very bottom of the
# representable range (denormal quanta): it is the same ULP headroom applied
# to the smallest subnormal step (finfo.tiny * eps).  The former decimal
# atol floors (1e-3 fp16 / 1e-2 bf16 / 1e-5 fp32+fp64) silently blessed
# TOTAL corruption of every element below the floor -- post-softmax and
# post-norm bf16 activations live almost entirely below 1e-2 -- and are gone.
_LOW_PRECISION_REPLAY_ULP_HEADROOM = 4.0
_ACCUMULATING_REPLAY_ULP_HEADROOM = 512.0

_REPLAY_ULP_HEADROOM: dict[torch.dtype, float] = {
    torch.float16: _LOW_PRECISION_REPLAY_ULP_HEADROOM,
    torch.bfloat16: _LOW_PRECISION_REPLAY_ULP_HEADROOM,
    torch.float32: _ACCUMULATING_REPLAY_ULP_HEADROOM,
    torch.float64: _ACCUMULATING_REPLAY_ULP_HEADROOM,
}


def derive_float_tolerances(dtype: torch.dtype, ulp_headroom: float) -> tuple[float, float]:
    """Derive an ``(rtol, atol)`` pair from a dtype's finfo and a ULP budget.

    ``rtol`` is ``ulp_headroom`` machine epsilons; ``atol`` is the same
    headroom applied to the dtype's smallest subnormal step
    (``finfo.tiny * finfo.eps``), i.e. it forgives jitter only at the very
    bottom of the representable range and never blesses corruption of small
    normal values.  Complex dtypes derive from their component real dtype
    (``torch.finfo`` already reports component precision for complex).
    """

    finfo = torch.finfo(dtype)
    rtol = ulp_headroom * float(finfo.eps)
    atol = ulp_headroom * float(finfo.tiny) * float(finfo.eps)
    return rtol, atol


_DTYPE_FLOAT_TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    dtype: derive_float_tolerances(dtype, headroom)
    for dtype, headroom in _REPLAY_ULP_HEADROOM.items()
}

# Legacy names, kept because they are exported through the torchlens.utils
# facade.  They now expose the DERIVED fp32 replay row instead of the former
# hand-picked literals (rtol 1e-4 was ~840 fp32 ULP; atol 1e-5 blessed total
# corruption of every element below 1e-5).
REL_FLOATING_POINT_TOLERANCE, MAX_FLOATING_POINT_TOLERANCE = _DTYPE_FLOAT_TOLERANCES[torch.float32]

# Gradient-validation tolerance pairs, spelled ONCE here (formerly bare
# literals repeated across validation/backward.py, validation/consolidated.py,
# validation/_layer_grad_report.py, and receptive_field/__init__.py, where two
# backward checks of the same capture disagreed 10x with no error model).
#
# Error model (fp32 gradients, the overwhelmingly common case):
# * PARAMETER grads are REDUCTIONS -- autograd sums each parameter's
#   contribution over the batch and every spatial/sequence position, so the
#   candidate-vs-stock difference carries accumulation-order round-off
#   proportional to that depth. rtol 1e-4 (~840 fp32 ULP) with a small
#   absolute floor for near-zero grads.
# * LAYER (module-output) grads and receptive-field empirical-adjoint probes
#   are compared ELEMENTWISE -- each element is one chain-rule product with no
#   cross-element reduction between the two pipelines under comparison, so
#   they earn a 10x tighter pair: rtol 1e-5, atol 1e-6.
# NaN handling at every consumer follows tensor_nanequal's doctrine: identical
# NaN patterns compare EQUAL (``equal_nan=True``), so a correct NaN-bearing
# gradient can never false-FAIL, while NaN-vs-number still fails.
#
# These four constants are the FP32 ROW of that error model.  Applying them to
# other dtypes is wrong in both directions: fp64 gradients get an rtol worth
# ~4.5e11 of their own ULPs (masking corruption far above fp64 round-off),
# while fp16 gradients get an rtol two orders BELOW their own eps (false-
# failing every non-bitwise agreement).  Dtype-resolving consumers call
# ``param_grad_tolerances_for_dtype`` / ``layer_grad_tolerances_for_dtype``,
# whose fp32 rows are bit-identical to these constants.
PARAM_GRAD_VALIDATION_RTOL = 1e-4
PARAM_GRAD_VALIDATION_ATOL = 1e-5
LAYER_GRAD_VALIDATION_RTOL = 1e-5
LAYER_GRAD_VALIDATION_ATOL = 1e-6

# Storage-rounding ULP headroom for low-precision (eps > fp32 eps) gradient
# comparisons.  Both pipelines under comparison compute the same op sequence
# in the same dtype (kernels widen internally and round ONCE to storage), so
# the legitimate difference is a few storage ULPs; reductions (param grads)
# get double the elementwise (layer grad) budget for reorder noise.
_PARAM_GRAD_LOW_PRECISION_ULP_HEADROOM = 8.0
_LAYER_GRAD_LOW_PRECISION_ULP_HEADROOM = 4.0


def _grad_tolerances_for_dtype(
    dtype: torch.dtype,
    fp32_rtol: float,
    fp32_atol: float,
    low_precision_ulp_headroom: float,
) -> tuple[float, float]:
    """Derive a gradient-validation ``(rtol, atol)`` row for one dtype.

    Accumulating dtypes (eps <= fp32's) rescale the legacy fp32 decimal
    budget by the eps ratio, so every dtype gets the SAME strictness measured
    in its own ULPs -- fp32 reproduces the legacy constants exactly, fp64
    tightens by ~9 orders of magnitude.  Storage-rounding dtypes (fp16/bf16)
    get a few-ULP budget in their own eps.  The absolute floor scales with
    the same ratio, keeping the legacy fp32 rtol/atol proportion: gradient
    reductions carry absolute cancellation noise near zero, so a
    subnormal-scale atol would false-fail legitimate near-zero grads.
    Complex dtypes derive from their component real dtype (``torch.finfo``
    reports component precision).

    Parameters
    ----------
    dtype:
        Gradient dtype being compared.
    fp32_rtol:
        Legacy fp32 relative tolerance anchoring the accumulating budget.
    fp32_atol:
        Legacy fp32 absolute floor anchoring the accumulating budget.
    low_precision_ulp_headroom:
        ULP budget for storage-rounding (low-precision) dtypes.

    Returns
    -------
    tuple[float, float]
        ``(rtol, atol)`` pair for ``torch.allclose``.
    """

    eps32 = float(torch.finfo(torch.float32).eps)
    try:
        eps = float(torch.finfo(dtype).eps)
    except (TypeError, ValueError):
        # Non-float dtype (no finfo): exact comparison paths handle these;
        # return the strictest float row so a misrouted call stays strict.
        eps = float(torch.finfo(torch.float64).eps)
    if eps > eps32:
        rtol = low_precision_ulp_headroom * eps
        return rtol, rtol / 10.0
    scale = eps / eps32
    return fp32_rtol * scale, fp32_atol * scale


def param_grad_tolerances_for_dtype(dtype: torch.dtype) -> tuple[float, float]:
    """Return the parameter-gradient validation tolerances for ``dtype``.

    Parameter grads are REDUCTIONS (summed over batch and spatial/sequence
    positions), so they carry accumulation-order round-off proportional to
    that depth; the fp32 row is exactly the legacy
    ``PARAM_GRAD_VALIDATION_RTOL`` / ``PARAM_GRAD_VALIDATION_ATOL`` pair.

    Parameters
    ----------
    dtype:
        Gradient dtype being compared.

    Returns
    -------
    tuple[float, float]
        ``(rtol, atol)`` pair for ``torch.allclose``.
    """

    return _grad_tolerances_for_dtype(
        dtype,
        PARAM_GRAD_VALIDATION_RTOL,
        PARAM_GRAD_VALIDATION_ATOL,
        _PARAM_GRAD_LOW_PRECISION_ULP_HEADROOM,
    )


def layer_grad_tolerances_for_dtype(dtype: torch.dtype) -> tuple[float, float]:
    """Return the layer-gradient validation tolerances for ``dtype``.

    Layer (module-output) grads and receptive-field empirical-adjoint probes
    compare ELEMENTWISE with no cross-element reduction between the two
    pipelines, so they earn a 10x tighter budget than parameter grads; the
    fp32 row is exactly the legacy ``LAYER_GRAD_VALIDATION_RTOL`` /
    ``LAYER_GRAD_VALIDATION_ATOL`` pair.

    Parameters
    ----------
    dtype:
        Gradient dtype being compared.

    Returns
    -------
    tuple[float, float]
        ``(rtol, atol)`` pair for ``torch.allclose``.
    """

    return _grad_tolerances_for_dtype(
        dtype,
        LAYER_GRAD_VALIDATION_RTOL,
        LAYER_GRAD_VALIDATION_ATOL,
        _LAYER_GRAD_LOW_PRECISION_ULP_HEADROOM,
    )


# Cached result of torch.cuda.is_available().  Evaluated once per process
# because CUDA availability cannot change at runtime.  Avoids repeated
# calls into the CUDA runtime (which involve driver queries).
_cuda_available: bool | None = None

_TensorSizeMethod = Callable[[torch.Tensor], int]


def _is_cuda_available() -> bool:
    """Return True if CUDA is available (cached after first call).

    The result is cached in a module-level global because CUDA availability
    is fixed for the lifetime of the process, and ``torch.cuda.is_available()``
    involves a non-trivial driver query.

    ``torch.cuda.is_available()`` normally swallows driver failures and returns
    False, but a visible-but-unusable CUDA stack (stale driver, mismatched
    build) can make the probe itself raise.  A failed *probe* is treated as
    "no CUDA": TorchLens' CUDA uses are all opportunistic (cache release,
    device-side RNG snapshots), so a broken accelerator must degrade a CPU
    capture, never abort it.  The failure is surfaced as a warning, once per
    process, rather than silently.
    """
    global _cuda_available
    if _cuda_available is None:
        try:
            _cuda_available = bool(torch.cuda.is_available())
        except Exception as exc:  # noqa: BLE001 - any driver/runtime probe failure
            # Cache BEFORE warning: under a caller's warnings-as-errors policy the
            # warn() itself raises, and the answer must still be latched so the
            # broken probe is not repeated on the next call.
            _cuda_available = False
            warnings.warn(
                "torch.cuda.is_available() raised "
                f"{type(exc).__name__}: {exc}. Treating CUDA as unavailable for "
                "this process; CPU capture continues unaffected.",
                stacklevel=2,
            )
    return _cuda_available


def _is_cuda_initialized() -> bool:
    """Return True if this process has already initialized the CUDA runtime.

    Unlike :func:`_is_cuda_available` this is a pure read of torch's own
    module-level init flag: it never queries the driver, never initializes a
    device, and is therefore NOT cached (it flips from False to True the first
    time anything in the process touches CUDA).

    Callers use it to distinguish "CUDA state exists and may have been
    consumed" from "nothing in this process has ever touched CUDA", so that
    opportunistic CUDA bookkeeping can be skipped instead of force-initializing
    every visible device.
    """
    try:
        return bool(torch.cuda.is_initialized())
    except Exception:  # noqa: BLE001 - torch build without CUDA support
        return False


def _tolerances_for_dtype(dtype: torch.dtype) -> tuple[float, float]:
    """Return replay comparison tolerances for ``dtype``.

    Rows are derived from ``torch.finfo(dtype).eps`` (see the error model on
    ``_REPLAY_ULP_HEADROOM``).  A float or complex dtype outside the
    precomputed table (e.g. ``complex64``, or a future torch float format)
    derives its own row instead of inheriting another dtype's literals --
    inheriting fp32's decimal row is exactly how float64 used to get an rtol
    worth 4.5e11 of its own ULPs.  The headroom class follows the eps class,
    mirroring ``_grad_tolerances_for_dtype``: a storage-rounding dtype
    (``eps`` above fp32's, e.g. ``complex32`` with component eps ~9.8e-4)
    gets the few-ULP low-precision budget -- deriving it at the accumulating
    512-ULP headroom produced rtol 0.5, a row that would bless 40%%
    corruption the day torch lands the missing comparison kernels.

    Parameters
    ----------
    dtype:
        Tensor dtype being compared.

    Returns
    -------
    tuple[float, float]
        ``(rtol, atol)`` pair for ``torch.allclose``.
    """

    cached = _DTYPE_FLOAT_TOLERANCES.get(dtype)
    if cached is not None:
        return cached
    try:
        eps = float(torch.finfo(dtype).eps)
        headroom = (
            _LOW_PRECISION_REPLAY_ULP_HEADROOM
            if eps > float(torch.finfo(torch.float32).eps)
            else _ACCUMULATING_REPLAY_ULP_HEADROOM
        )
        derived = derive_float_tolerances(dtype, headroom)
    except (TypeError, ValueError):
        # Non-float dtype (no finfo): exact comparison paths handle these;
        # return the strictest float row so a misrouted call stays strict.
        derived = _DTYPE_FLOAT_TOLERANCES[torch.float64]
    _DTYPE_FLOAT_TOLERANCES[dtype] = derived
    return derived


def _is_fp8_tensor(tensor: torch.Tensor) -> bool:
    """Return True when ``tensor`` has one of this build's fp8 dtypes.

    Parameters
    ----------
    tensor:
        Tensor to classify.

    Returns
    -------
    bool
        True for ``float8_*`` payloads, False on builds with no fp8 dtypes.
    """

    fp8_dtypes = get_fp8_dtypes()
    return bool(fp8_dtypes) and tensor.dtype in fp8_dtypes


def fp8_safe_comparison_pair(
    tensor_a: torch.Tensor, tensor_b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Widen an fp8 tensor pair to float32 so comparison kernels exist.

    fp8 payloads report ``dtype.is_floating_point == True`` yet torch ships no
    ``isinf`` / ``nan_to_num`` / ``allclose`` / reduction kernels for them, so every
    numeric comparison helper raised a raw ``NotImplementedError: "isinf" not
    implemented for 'Float8_e4m3fn'`` out of validation replay.

    Widening is EXACT, not a relaxation: all 256 bit patterns of every fp8 variant
    torch exposes round-trip bit-identically through float32, and NaN patterns stay
    NaN (verified exhaustively per variant). The comparison that follows is
    therefore the same comparison native fp8 kernels would perform, so the
    validation tripwire keeps its full strength. Callers deliberately keep their
    float32-grade tolerances afterwards rather than fp8's coarse 2^-3 / 2^-2
    epsilon, which would let a genuine one-ULP fp8 difference read as equal.

    Parameters
    ----------
    tensor_a:
        First tensor of the pair.
    tensor_b:
        Second tensor of the pair.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        float32 copies when the pair is fp8, otherwise the inputs unchanged.

    Notes
    -----
    Callers must have already established that both tensors share a dtype, and must
    call this inside ``pause_logging()`` -- ``.to()`` is a decorated method.
    """

    if not _is_fp8_tensor(tensor_a):
        return tensor_a, tensor_b
    return tensor_a.to(torch.float32), tensor_b.to(torch.float32)


def fp8_widen_for_numeric_ops(tensor: torch.Tensor) -> torch.Tensor:
    """Return a float32 view of an fp8 tensor, or ``tensor`` unchanged.

    The single-tensor form of :func:`fp8_safe_comparison_pair`, for predicates such
    as ``torch.isfinite`` that torch does not implement for fp8. The widening is
    exact (see that function), so the predicate's verdict is unchanged.

    Parameters
    ----------
    tensor:
        Tensor to widen when its dtype is fp8.

    Returns
    -------
    torch.Tensor
        float32 copy for fp8 payloads, otherwise the input unchanged.

    Notes
    -----
    Call inside ``pause_logging()`` -- ``.to()`` is a decorated method.
    """

    if not _is_fp8_tensor(tensor):
        return tensor
    return tensor.to(torch.float32)


def tensor_all_nan(tensor: torch.Tensor) -> bool:
    """Return True if every element in the tensor is NaN."""
    # bool(): the comparison yields a 0-d TENSOR, and this function is declared
    # (and consumed) as a plain bool. SIM103 collapsed the original
    # if/True/else/False into a bare return, which silently changed the return
    # TYPE; the explicit cast keeps the collapse and the contract.
    return bool(torch.isnan(tensor).int().sum() == tensor.numel())


def _quantized_tensor_equal(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> bool:
    """Return exact equality for quantized tensors without floating ops.

    Parameters
    ----------
    tensor_a:
        First quantized tensor.
    tensor_b:
        Second quantized tensor.

    Returns
    -------
    bool
        True if quantization metadata and integer payloads match.
    """

    if not (tensor_a.is_quantized and tensor_b.is_quantized):
        return False
    if tensor_a.qscheme() != tensor_b.qscheme():
        return False
    if not torch.equal(tensor_a.int_repr(), tensor_b.int_repr()):
        return False
    if tensor_a.qscheme() in (torch.per_tensor_affine, torch.per_tensor_symmetric):
        return tensor_a.q_scale() == tensor_b.q_scale() and (
            tensor_a.q_zero_point() == tensor_b.q_zero_point()
        )
    return (
        tensor_a.q_per_channel_axis() == tensor_b.q_per_channel_axis()
        and torch.equal(tensor_a.q_per_channel_scales(), tensor_b.q_per_channel_scales())
        and torch.equal(
            tensor_a.q_per_channel_zero_points(),
            tensor_b.q_per_channel_zero_points(),
        )
    )


def is_functorch_wrapped_tensor(value: Any) -> bool:
    """Return whether ``value`` is a functorch wrapper tensor.

    Parameters
    ----------
    value:
        Object to inspect.

    Returns
    -------
    bool
        True when PyTorch reports a functorch wrapper tensor.
    """

    if not isinstance(value, torch.Tensor):
        return False
    checker = get_functorch_wrapped_tensor_checker()
    if checker is None:
        return False
    try:
        return bool(checker(value))
    except RuntimeError:
        return False


def _signed_zeros_match(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> bool:
    """Return whether zero elements carry the same sign bit on both sides.

    IEEE equality reads ``-0.0 == +0.0`` as True, so ``torch.equal`` alone
    certifies a sign-flipped-zero replay as EXACT (sol+fable r4 probes) even
    though the payloads are bit-distinct and diverge downstream
    (``1/x`` -> opposite infinities). Equal NON-zero finite floats share one
    representation and equal NaN masks are enforced separately, so the zero
    positions are the only place bitwise identity can hide behind IEEE
    equality. NaN sign stays out of scope: kernels legitimately differ on it.

    Parameters
    ----------
    tensor_a:
        First tensor (floating or complex; fp8 callers widen first).
    tensor_b:
        Second tensor, already known elementwise-equal to ``tensor_a``.

    Returns
    -------
    bool
        True when every zero element has the same sign bit on both sides.
    """

    if tensor_a.is_complex():
        tensor_a = torch.view_as_real(tensor_a.resolve_conj())
        tensor_b = torch.view_as_real(tensor_b.resolve_conj())
    zeros = tensor_a == 0
    if not bool(zeros.any()):
        return True
    return bool(torch.equal(tensor_a.signbit() & zeros, tensor_b.signbit() & zeros))


def tensor_nanequal(
    tensor_a: torch.Tensor, tensor_b: torch.Tensor, allow_tolerance: bool = False
) -> bool:
    """NaN-aware tensor equality check, used by validation replay.

    NaN positions are treated as equal (NaN == NaN is True here), which
    differs from IEEE 754 semantics.  This is intentional: validation
    needs to confirm that the replay produced the same NaN pattern, not
    that NaN != NaN.

    ``pause_logging()`` is required because this function is called during
    active logging (from ``_tag_tensor_and_track_variations``) and uses
    decorated tensor custom_methods like ``.resolve_conj()``, ``.isinf()``, etc.
    Without pausing, these calls re-enter the logging pipeline and cause
    infinite recursion.

    Args:
        tensor_a: First tensor.
        tensor_b: Second tensor.
        allow_tolerance: If True, allow element-wise differences within the
            dtype-derived ULP band from :func:`_tolerances_for_dtype` (for
            floating-point non-determinism on GPU).

    Returns:
        True if the tensors are considered equal.
    """
    from .._state import pause_logging

    if is_functorch_wrapped_tensor(tensor_a) or is_functorch_wrapped_tensor(tensor_b):
        return False

    if tensor_a.shape != tensor_b.shape:
        return False

    if tensor_a.dtype != tensor_b.dtype:
        return False
    original_dtype = tensor_a.dtype

    # Meta tensors carry no data: with shape and dtype already matched there
    # is nothing left to compare, and any content op (torch.equal, .isinf())
    # raises "Cannot copy out of meta tensor" on them.
    if tensor_a.is_meta or tensor_b.is_meta:
        return tensor_a.is_meta and tensor_b.is_meta

    with pause_logging():
        if tensor_a.is_quantized or tensor_b.is_quantized:
            return _quantized_tensor_equal(tensor_a, tensor_b)

        # Sparse layouts have no aten::equal / isinf / nan_to_num kernels: every
        # comparison below used to escape tl.trace() as a raw torch-internal
        # NotImplementedError naming the SparseCPU dispatcher (R65; the layout
        # sibling of the fp8 class documented below). Compare the canonical
        # structure exactly and recurse on the strided values tensor so
        # NaN/Inf/tolerance semantics match the dense path.
        if tensor_a.layout != tensor_b.layout:
            return False
        if tensor_a.layout == torch.sparse_coo:
            tensor_a = tensor_a.coalesce()
            tensor_b = tensor_b.coalesce()
            if not torch.equal(tensor_a.indices(), tensor_b.indices()):
                return False
            return tensor_nanequal(
                tensor_a.values(), tensor_b.values(), allow_tolerance=allow_tolerance
            )
        if tensor_a.layout in (
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        ):
            if tensor_a.layout in (torch.sparse_csr, torch.sparse_bsr):
                index_pairs = (
                    (tensor_a.crow_indices(), tensor_b.crow_indices()),
                    (tensor_a.col_indices(), tensor_b.col_indices()),
                )
            else:
                index_pairs = (
                    (tensor_a.ccol_indices(), tensor_b.ccol_indices()),
                    (tensor_a.row_indices(), tensor_b.row_indices()),
                )
            for index_a, index_b in index_pairs:
                if not torch.equal(index_a, index_b):
                    return False
            return tensor_nanequal(
                tensor_a.values(), tensor_b.values(), allow_tolerance=allow_tolerance
            )

        # Validation overwhelmingly compares identical ordinary floating-point
        # payloads. Avoid constructing the Inf/NaN masks and substituted tensors
        # in that common case; non-exact comparisons and non-floating dtypes
        # retain the full comparison below.
        # IEEE equality hides -0.0 vs +0.0; only certify EXACT when zero sign
        # bits agree too (fp8 widens first: no signbit kernel). A flip falls
        # through -- the tolerance band below may still legitimately accept it.
        # fp8 tensors must skip this fast path entirely: torch has no
        # equal_cpu kernel for them at all (raises regardless of whether the
        # payloads actually match), so the raw torch.equal call below would
        # crash before ever reaching fp8_safe_comparison_pair's widening --
        # exactly the class of raw NotImplementedError/RuntimeError the
        # widened comparison further below exists to avoid.
        if (
            tensor_a.layout == torch.strided
            and tensor_a.dtype.is_floating_point
            and not _is_fp8_tensor(tensor_a)
            and torch.equal(tensor_a, tensor_b)
            and _signed_zeros_match(tensor_a, tensor_b)
        ):
            return True

        # fp8 has no isinf/nan_to_num/allclose kernel, so every line below used to
        # raise a raw NotImplementedError out of validation replay. The exact-equality
        # fast path above only hides that while the tensors match bit-for-bit, and one
        # NaN element defeats it (torch.equal is IEEE, so NaN != NaN). The widening is
        # exact; see fp8_safe_comparison_pair.
        tensor_a, tensor_b = fp8_safe_comparison_pair(tensor_a, tensor_b)

        # Inf positions must match exactly (inf != -inf).
        if not torch.equal(tensor_a.isinf(), tensor_b.isinf()):
            return False

        # NaN positions must match exactly BEFORE the sentinel substitution
        # below.  ``nan_to_num`` rewrites every NaN to the finite sentinel
        # 0.7234691827346; without this mask check a real finite value that
        # happens to equal the sentinel would read EQUAL to a NaN (in either
        # direction), silently defeating the validation tripwire.  ``isnan`` on
        # a complex tensor is True whenever either component is NaN, matching the
        # ``view_as_real`` substitution used for the complex branch below.
        if not torch.equal(tensor_a.isnan(), tensor_b.isnan()):
            return False

        # Replace NaNs with a sentinel value so torch.equal treats NaN positions
        # as equal.  The NaN masks are already confirmed identical above, so the
        # sentinel (0.7234691827346) never collides with a real finite value on
        # one side against a NaN on the other.  Complex tensors need
        # view_as_real/view_as_complex because torch.nan_to_num doesn't support
        # complex dtypes directly.
        if tensor_a.is_complex():
            tensor_a_nonan = torch.view_as_complex(
                torch.nan_to_num(torch.view_as_real(tensor_a.resolve_conj()), 0.7234691827346)
            )
            tensor_b_nonan = torch.view_as_complex(
                torch.nan_to_num(torch.view_as_real(tensor_b.resolve_conj()), 0.7234691827346)
            )
        else:
            tensor_a_nonan = torch.nan_to_num(tensor_a, 0.7234691827346)
            tensor_b_nonan = torch.nan_to_num(tensor_b, 0.7234691827346)

        if torch.equal(tensor_a_nonan, tensor_b_nonan):
            payload_dtype = tensor_a_nonan.dtype
            if not (payload_dtype.is_floating_point or payload_dtype.is_complex):
                return True
            if _signed_zeros_match(tensor_a_nonan, tensor_b_nonan):
                return True
            # Signed-zero flip: not EXACT; the tolerance band below may
            # still accept it when the caller allows tolerance.

        # Tolerance path: allow small floating-point differences (e.g. from
        # convolution replay order, non-deterministic GPU reductions, or
        # mixed-precision rounding).  It applies ONLY to inexact (floating-point
        # / complex) dtypes.  Integer and boolean tensors are exact and are
        # handled entirely by the torch.equal check above; applying a float
        # allclose tolerance to integers would let genuinely different values
        # (e.g. 1_000_000 vs 1_000_001) read EQUAL, defeating the tripwire.
        # (dtypes are already confirmed identical above, so one side suffices.)
        payload_dtype = tensor_a_nonan.dtype
        if allow_tolerance and (payload_dtype.is_floating_point or payload_dtype.is_complex):
            rtol, atol = _tolerances_for_dtype(payload_dtype)
            if original_dtype in get_fp8_dtypes():
                # Widening is exact, but even a denormal-scale float32 absolute
                # term is measured against the WRONG dtype here: adjacent
                # subnormal values in e5m2fnuz/e8m0fnu sit far above float32's
                # bottom-of-range quanta, so keep the fp8 comparison rtol-only.
                atol = 0.0
            if torch.allclose(tensor_a_nonan, tensor_b_nonan, rtol=rtol, atol=atol):
                return True

    return False


def safe_to(obj: Any, device: str) -> Any:
    """Move a tensor to ``device`` without triggering torchlens logging.

    Non-tensor objects are returned unchanged.  ``pause_logging()`` is
    required because ``.to()`` is a decorated tensor method — calling it
    while logging is active would create a spurious log entry.

    Args:
        obj: A tensor or arbitrary object.
        device: Target device string (e.g. ``"cpu"``, ``"cuda:0"``).

    Returns:
        The tensor on the target device, or the original object if not a tensor.
    """
    from .._state import pause_logging

    if isinstance(obj, torch.Tensor):
        with pause_logging():
            return obj.to(device)
    else:
        return obj


def _unwrapped_tensor_size_method(method: _TensorSizeMethod) -> _TensorSizeMethod:
    """Return the undecorated implementation for a Tensor size method.

    Parameters
    ----------
    method:
        Tensor method descriptor or TorchLens wrapper to resolve.

    Returns
    -------
    _TensorSizeMethod
        Original method when TorchLens has decorated it, otherwise ``method``.
    """

    from .. import _state

    return cast(_TensorSizeMethod, _state._decorated_to_orig.get(id(method), method))


def _dense_tensor_memory_amount(t: torch.Tensor) -> int:
    """Return dense tensor bytes without entering the logging toggle machinery.

    Parameters
    ----------
    t:
        Dense tensor to measure.

    Returns
    -------
    int
        Number of bytes represented by ``t.nelement() * t.element_size()``.
    """

    nelement = _unwrapped_tensor_size_method(torch.Tensor.nelement)
    element_size = _unwrapped_tensor_size_method(torch.Tensor.element_size)
    return int(nelement(t) * element_size(t))


_SPARSE_COMPONENT_ACCESSORS: dict[Any, tuple[str, ...]] = {
    torch.sparse_coo: ("_indices", "_values"),
    torch.sparse_csr: ("crow_indices", "col_indices", "values"),
    torch.sparse_csc: ("ccol_indices", "row_indices", "values"),
    torch.sparse_bsr: ("crow_indices", "col_indices", "values"),
    torch.sparse_bsc: ("ccol_indices", "row_indices", "values"),
}
"""Physical component tensors per sparse layout: every index tensor AND values."""


def sparse_component_tensors(t: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return the physical component tensors of a sparse tensor.

    Parameters
    ----------
    t:
        Sparse tensor (COO or any compressed layout).

    Returns
    -------
    tuple[torch.Tensor, ...]
        Index tensor(s) and values tensor backing ``t``.

    Raises
    ------
    ValueError
        For non-strided layouts without a known component decomposition
        (callers treat that as unmeasurable, never as zero-index-bytes).
    """

    accessors = _SPARSE_COMPONENT_ACCESSORS.get(t.layout)
    if accessors is None:
        raise ValueError(f"no known component decomposition for layout {t.layout}")
    return tuple(getattr(t, name)() for name in accessors)


def get_memory_amount(t: torch.Tensor) -> int:
    """Return the memory footprint of a tensor in bytes.

    Tensor size methods are called through their unwrapped implementations when
    TorchLens has decorated them, avoiding logging recursion without toggling
    global logging state for each tensor.

    Meta tensors have no storage and return 0. Sparse tensors (COO and the
    compressed layouts) report their physical components: index storage AND
    values storage. Counting only values ledgered a 1-nnz float32 COO cell as
    4 bytes when its int64 indices alone hold 8 bytes per sparse dim, and the
    dense fallback billed compressed layouts at logical-shape bytes.

    Args:
        t: Tensor to measure.

    Returns:
        Size in bytes, or 0 on failure / meta tensors.
    """

    try:
        if t.device.type == "meta":
            return 0
        if t.layout is not torch.strided:
            return sum(
                _dense_tensor_memory_amount(component) for component in sparse_component_tensors(t)
            )
        return _dense_tensor_memory_amount(t)
    except Exception:
        return 0


def get_memory_amount_from_metadata(
    t: torch.Tensor,
    shape: tuple[int, ...] | torch.Size,
    dtype: torch.dtype,
) -> int:
    """Return tensor memory bytes using already-captured dense metadata.

    Parameters
    ----------
    t:
        Tensor being measured.
    shape:
        Already-captured tensor shape.
    dtype:
        Already-captured tensor dtype.

    Returns
    -------
    int
        Size in bytes, or the guarded tensor-method fallback for layouts whose
        storage size is not represented by ``shape * dtype.itemsize``.
    """

    try:
        if t.device.type == "meta":
            return 0
        if t.layout is not torch.strided:
            # Sparse layouts (COO and compressed): physical component bytes,
            # never logical shape * itemsize.
            return get_memory_amount(t)
        return int(prod(shape) * dtype.itemsize)
    except Exception:
        return get_memory_amount(t)


def concatenate_batch_tensors(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Concatenate two tensors along the leading batch dimension.

    Parameters
    ----------
    left:
        Existing accumulated tensor.
    right:
        New chunk tensor.

    Returns
    -------
    torch.Tensor
        Tensor containing ``left`` followed by ``right`` on dimension 0.
    """

    from .._state import pause_logging

    with pause_logging():
        return torch.cat([left, right], dim=0)


@contextmanager
def _pause_dispatch_modes_for_subclass_clone(x: Any) -> Iterator[None]:
    """Pause TorchLens dispatch modes around a strict-subclass ``clone()``, when needed.

    Torch's default ``__torch_function__`` subclass-return conversion calls
    ``ret.as_subclass(cls)`` INSIDE ``x.clone()`` to reconstruct the subclass
    type (R16-5, see ``torchlens/backends/torch/wrappers.py``). On torch 2.1
    and 2.2 that reconstruction raises "Creating a new Tensor subclass ...
    already associated to a python object" whenever a TorchLens dispatch mode
    (the completeness witness, armed by default; an intervention-ready
    capture's mode) is active -- reproduced on stock torch with a no-op mode,
    not something TorchLens's own wrapping causes. Pausing ONCE here, around
    the whole ``clone()`` call, covers that nested reconstruction in a single
    non-reentrant bracket (see ``HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE``); a
    plain ``torch.Tensor``/``Parameter`` receiver, or a modern torch build
    that tolerates the reconstruction, pays nothing.

    Parameters
    ----------
    x:
        Tensor about to be cloned with its own type preserved.

    Yields
    ------
    None
        The ``clone()`` call this brackets.
    """

    if (
        HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE
        or type(x) is torch.Tensor
        or type(x) is torch.nn.Parameter
    ):
        yield
        return
    # Deferred: backends.torch is upward of utils in the layer map (C01 item
    # 1); this branch is the floor-only (torch 2.1/2.2) fallback, so a plain
    # `import torchlens` never pays for it.
    from ..backends.torch._modes import pause_own_dispatch_modes

    with pause_own_dispatch_modes():
        yield


def _safe_get_memory_format(t: torch.Tensor) -> torch.memory_format:
    """Best-effort memory format probe — returns ``preserve_format`` on any error.

    ``is_contiguous(memory_format=...)`` is the recommended query; it is
    undefined for some exotic layouts (sparse, meta), so we wrap in a
    try/except and fall back to ``preserve_format`` (clone's default).

    Standard (torch.contiguous_format) is checked FIRST and wins on ties.
    For tensors with a size-1 dimension (most commonly ``C=1``, e.g. a mono
    spectrogram or single-channel image), ``is_contiguous(memory_format=
    torch.channels_last)`` is degenerately also ``True`` even though the
    tensor is genuinely NCHW-contiguous — the collapsed size-1 axis makes
    both stride orderings equally valid descriptions of the same bytes.
    Checking ``channels_last`` first (the prior behavior) would then force
    ``.clone(memory_format=torch.channels_last)`` on an already-standard
    tensor, physically rewriting its strides to the channels-last layout.
    That silently corrupts any downstream ``.view()`` call the traced model
    makes under the (correct, for its real input) assumption of standard
    contiguity — a real capture bug, not a model bug. See
    ``torchlens/utils/tensor_utils.py`` history / BC-ResNet capture repro.
    """
    try:
        if t.is_contiguous(memory_format=torch.contiguous_format):
            return torch.contiguous_format
        if t.is_contiguous(memory_format=torch.channels_last):
            return torch.channels_last
        if t.is_contiguous(memory_format=torch.channels_last_3d):
            return torch.channels_last_3d
    except (RuntimeError, TypeError, AttributeError):
        pass
    return torch.preserve_format


# ---------------------------------------------------------------------------
# Deferred payload clones (clone-on-write)
# ---------------------------------------------------------------------------
# Eagerly cloning every captured activation payload is a large slice of plain
# capture wall time, and most of those clones are never needed: the source
# storage is never written again. When the wrapper arms the payload window
# (plain torch exhaustive captures only), ``_clone_tensor_payload`` returns a
# detached ALIAS of the source instead of a clone and registers it here, keyed
# by storage identity. The alias stays zero-copy FOREVER unless something is
# about to write its storage. Byte-identity of the saved value is guaranteed
# by two cooperating mechanisms:
#
#   1. INTERCEPTION (authoritative, permanent): torch wrappers stay installed
#      for the process lifetime, so EVERY wrapped call — during the capture
#      and after it — is seen BEFORE execution. Calls that can write through
#      tensor arguments (in-place signatures, ``out=``, ``inplace=True``,
#      mutating property setters, ``__setitem__``) first materialize pending
#      aliases sharing those storages via :func:`materialize_deferred_for_call`.
#      This also reproduces eager isolation for the user's own post-hoc edits:
#      ``log[...].out.add_(1)`` rebinds every co-resident saved alias onto
#      exclusive fresh storage before the write lands.
#   2. VERSION BELT (redundant tripwire): each pending alias records its
#      autograd ``_version`` at defer time (detached aliases share the source's
#      version counter). Materialization refuses — loudly — if the version
#      moved without interception, so an unforeseen torch-side mutation path
#      becomes a hard error instead of a silently corrupted saved activation.
#
# Known residual (documented; the same class as untraced ops): a host-level
# write that bypasses torch dispatch entirely (raw ``data_ptr()``/numpy buffer
# writes, ctypes) neither triggers interception nor bumps the version counter.
# Eager cloning was immune to that case; deferral is therefore gated to plain
# captures where none of the honesty machinery (runnable witnesses, backward
# capture, transforms) is armed.
#
# GRAPH-CONNECTED PAYLOADS (the default ``tl.trace(model, x)`` regime)
# --------------------------------------------------------------------
# ``detach_saved_activations`` defaults to False, so in a plain grad-enabled
# capture the eager clone is ``x.clone()``: graph-connected, ``requires_grad``
# True, ``grad_fn`` ``CloneBackward0``. A ``detach()``-flavored alias cannot
# stand in for that, which is why deferral was originally gated to the
# detached/no-grad cases. Standing an ALIAS in for such a clone needs three
# things to hold, each of which is a measured hazard rather than a worry:
#
#   H1 The alias must not be an autograd VIEW of the source. ``aten.alias``
#      keeps ``requires_grad`` but registers a differentiable view, so a later
#      in-place write to the SOURCE rebases the payload's ``grad_fn``
#      (``AliasBackward0`` -> ``AsStridedBackward0``) and silently re-routes
#      the gradient through ops that ran AFTER the capture point. In-place
#      ``ReLU`` makes that the common path, not an exotic one. The mint here
#      therefore grafts the graph edge onto a plain ``detach()`` alias through
#      :class:`_DeferredPayloadCloneFn` (identity backward), whose output is
#      NOT a view; the alias is passed in a holder so autograd cannot see it
#      as an input and wrap it into one.
#   H2 Materialization must not go through ``Tensor.set_``. ``set_`` has no
#      derivative, so rebinding a graph-connected alias poisons the graph:
#      backward then dies with "derivative for set_ is not implemented".
#      ``set_`` also BUMPS the version counter every alias shares with its
#      live source, so a read-only source argument of a mutating call looked
#      modified in place: a custom autograd Function's view output then
#      raised "a view created inside a custom Function ... modified inplace"
#      at its next use, and any ``save_for_backward`` holder of the source
#      would fail autograd's version guard. :func:`_rebind_alias_to_fresh_clone`
#      therefore uses ``.data =`` for every alias, which swaps storage
#      without touching autograd metadata (``grad_fn`` and the version
#      counter both survive, unbumped).
#   H3 RESIDUAL, and the reason grad-connected deferral stays opt-in:
#      autograd's saved-tensor machinery is a SECOND holder of the alias that
#      interception cannot reach. If the user builds a differentiable graph on
#      a saved payload while it is still pending, the ``SavedVariable`` inside
#      that graph aliases the source storage; a later intercepted in-place
#      write rebinds the payload's Python object but NOT the saved copy, and
#      backward then fails on autograd's version guard where an eager clone
#      would have succeeded. It fails LOUD (never a silent wrong gradient,
#      because the alias deliberately keeps sharing the source's version
#      counter) and the documented way to build losses from saved outs,
#      ``backward_ready=True``, already keeps eager clones. Closing it needs a
#      read barrier on the wrapper's pre-call path (materialize a pending
#      graph-connected alias when it appears as an argument to ANY wrapped
#      call, not only a mutating one), which is out of this module's scope.

# Kill switch: TORCHLENS_EAGER_PAYLOAD_CLONE=1 restores unconditional eager
# clones (also used by the perf harness for A/B runs).
# IMPORT-TIME LATCH (R47-4): read once here and value-copied into
# ``backends/torch/wrappers.py`` at ITS import; a runtime ``setenv`` is a
# silent no-op. Set the variable BEFORE the process imports torchlens (user
# guidance must never recommend the runtime spelling). Promotion to a
# session-time CaptureOptions knob spans options.py + wrappers.py and ships
# with their owning lanes.
_DEFER_ENABLED: bool = not closed_bool_env("TORCHLENS_EAGER_PAYLOAD_CLONE")

# Opt-in: TORCHLENS_DEFER_GRAD_PAYLOADS=1 extends deferral to graph-connected
# payloads (the default grad-enabled capture regime). OFF by default because of
# residual H3 above; H1/H2 are closed unconditionally by the mint and rebind.
# Same import-time latch caveat as above (R47-4).
_DEFER_GRAD_ENABLED: bool = closed_bool_env("TORCHLENS_DEFER_GRAD_PAYLOADS")

# storage key -> list of pending aliases. NEVER rebound (only mutated), so the
# wrapper can bind the dict object once and use plain truthiness on its hot
# path. Keys are (storage_data_ptr, storage_nbytes, device_str): unique among
# live storages, and every pending alias keeps its storage alive. Dead entries
# (payloads the capture discarded, dropped traces) are pruned lazily on lookup
# and at every window arm.
_DEFER_PENDING: dict[tuple[int, int, str], list["_PendingPayloadAlias"]] = {}

# Window state, armed by the torch wrapper strictly around the payload-saving
# call for eligible captures. Single-threaded by design, like all capture
# state; the arming side stores the excluded state-storage pointers.
_DEFER_WINDOW_DEPTH: int = 0
_DEFER_STATE_PTRS: frozenset[int] | None = None
_DEFER_BUSY: bool = False


class _PendingPayloadAlias:
    """One deferred payload copy: a weakly-referenced alias plus its belt state."""

    __slots__ = ("ref", "version")

    def __init__(self, ref: "weakref.ref[torch.Tensor]", version: int) -> None:
        self.ref = ref
        self.version = version


@contextmanager
def _paused_internal_reads() -> Iterator[None]:
    """Pause logging and mark storage-identity reads as TorchLens bookkeeping.

    Mirrors the sanctioned ``set_tensor_label`` pattern: ``untyped_storage()``
    / ``data_ptr()`` are witnessed host-escape surfaces, so bookkeeping reads
    must run under ``pause_logging`` plus ``internal_scalar_read`` or they
    would register as user raw-pointer escapes on witness-armed captures.
    """
    from .._state import pause_logging
    from ..backends.torch.completeness_witness import internal_scalar_read

    with pause_logging(), internal_scalar_read():
        yield


def _deferred_storage_key(x: torch.Tensor) -> tuple[int, int, str] | None:
    """Return the pending-registry key for ``x``'s storage, or ``None``.

    Callers must hold ``_paused_internal_reads()``. Any failure (exotic layout,
    storageless tensor) reads as ineligible rather than raising. Tracing
    tensor variants (FakeTensor/FunctionalTensor) are ineligible WITHOUT
    touching storage: reading a FakeTensor's data pointer trips torch's
    "almost definitely a bug" warning before the entry guard's typed refusal
    fires, and the entry guard's own contract is that FakeTensors never reach
    pointer-reading metadata (grind-r5 b6 R16, order-dependent repro).
    """
    from ._torch_compat import get_tracing_tensor_types

    tracing_types = get_tracing_tensor_types()
    if tracing_types and isinstance(x, tracing_types):
        return None
    try:
        storage = x.untyped_storage()
        ptr = storage.data_ptr()
        nbytes = storage.nbytes()
    except Exception:
        return None
    if ptr == 0 or nbytes == 0:
        return None
    return (ptr, nbytes, str(x.device))


class _DeferredPayloadCloneFn(torch.autograd.Function):
    """Identity autograd node standing in for ``CloneBackward0`` on an alias.

    ``clone`` and ``alias`` both have identity gradients, so grafting this node
    onto a ``detach()`` alias reproduces the eager clone's gradient exactly.
    The alias arrives inside ``holder`` rather than as a tensor argument on
    purpose: autograd wraps any output that IS one of its inputs into a
    differentiable view (``var.view_as(var)``), which is precisely the view
    relationship hazard H1 above. Passing it in a list keeps the output a
    plain non-view tensor whose ``grad_fn`` is this node.
    """

    @staticmethod
    def forward(ctx: Any, source: torch.Tensor, holder: list[torch.Tensor]) -> torch.Tensor:
        """Return the aliased tensor held in ``holder``, unchanged.

        The alias arrives inside ``holder`` rather than as a tensor argument so the
        output is not one of autograd's inputs; otherwise autograd would return a
        differentiable view instead of a plain tensor.
        """

        return holder[0]

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Pass the gradient straight through: ``clone`` and ``alias`` are identities."""

        return grad_output, None


def _mint_graph_connected_alias(x: torch.Tensor) -> torch.Tensor:
    """Return a non-view alias of ``x`` carrying an identity gradient edge.

    The result matches what ``x.clone()`` would have produced on every field
    the capture records — dtype, shape, strides, ``requires_grad`` — plus a
    live gradient path back to ``x``. It deliberately keeps sharing ``x``'s
    autograd version counter so the belt (and autograd's own guard) still see
    unintercepted writes; see H1-H3 in the module notes above.
    """
    return cast(torch.Tensor, _DeferredPayloadCloneFn.apply(x, [x.detach()]))


def _try_defer_payload_alias(
    x: torch.Tensor, *, graph_connected: bool = False
) -> torch.Tensor | None:
    """Return a registered clone-on-write alias for ``x``, or ``None``.

    Only called from ``_clone_tensor_payload`` (already under
    ``pause_logging``) for the plain ``save_mode="copy"`` path while the
    wrapper's payload window is armed. Ineligible tensors fall back to the
    historical eager clone.

    Parameters
    ----------
    x
        Source payload tensor.
    graph_connected
        Whether the eager clone this alias replaces would have stayed attached
        to the autograd graph. ``False`` mints the historical ``detach()``
        alias; ``True`` mints the identity-grafted alias described above.
    """
    if isinstance(x, torch.nn.Parameter):
        return None
    if x.layout is not torch.strided or x.is_quantized:
        return None
    if x.device.type == "meta" or x.numel() == 0:
        return None
    try:
        if x.is_conj() or x.is_neg() or x.is_inference():
            return None
    except Exception:
        return None
    state_ptrs = _DEFER_STATE_PTRS
    if state_ptrs is None:
        return None
    from ..backends.torch.completeness_witness import internal_scalar_read

    with internal_scalar_read():
        key = _deferred_storage_key(x)
    if key is None or key[0] in state_ptrs:
        # Model param/buffer storages (and views of them) stay eager: their
        # bytes can move through C++ side effects (train-mode batch_norm
        # running stats) that no wrapped-call signature announces.
        return None
    if not _alias_covers_whole_storage(x, key[1]):
        # Partial-coverage outputs (slices, chunks) stay eager: an alias would
        # pin the WHOLE backing storage, where the eager clone compacts.
        return None
    try:
        alias = _mint_graph_connected_alias(x) if graph_connected else x.detach()
        version = int(alias._version)
    except Exception:
        return None
    _DEFER_PENDING.setdefault(key, []).append(
        _PendingPayloadAlias(weakref.ref(alias, _note_dead_deferred_alias), version)
    )
    return alias


def _belt_check_pending_alias(entry: _PendingPayloadAlias, alias: torch.Tensor) -> None:
    """Refuse — loudly — if a pending alias was mutated without interception."""
    if int(alias._version) != entry.version:
        raise RuntimeError(
            "torchlens deferred-clone tripwire: a captured activation's source "
            "storage was mutated through a path the capture wrapper did not "
            "intercept (autograd version moved between defer and materialize). "
            "The saved payload bytes can no longer be proven identical to the "
            "capture-time value. Relaunch with TORCHLENS_EAGER_PAYLOAD_CLONE=1 "
            "set in the environment BEFORE importing torchlens (the flag is "
            "read once at import) to restore eager payload clones, and please "
            "report the model/op that triggered this."
        )


def _rebind_alias_to_fresh_clone(alias: torch.Tensor) -> None:
    """Copy a pending alias's bytes into fresh exclusive storage, in place.

    Callers must hold ``pause_logging``. The alias keeps its Python identity
    (it is already stored in capture fields); ``set_`` rebinds it onto the
    fresh clone, which carries exactly the metadata the historical eager
    clone would have had (same clone call on identical layout/bytes).

    The rebind goes through ``.data =`` for every alias (hazard H2 in the
    module notes), never ``Tensor.set_``: ``set_`` has no derivative, so a
    graph-connected payload would raise "derivative for set_ is not
    implemented" on backward, and it bumps the version counter that every
    alias SHARES with its live source. Materialization runs for every tensor
    argument of a mutating call, read-only sources included, so that bump
    made an unmodified live tensor look modified in place (a custom autograd
    Function's view output then refused its next use). ``.data =`` swaps
    storage without entering autograd: ``grad_fn``, ``requires_grad`` and the
    version counter all survive untouched.
    """
    fmt = _safe_get_memory_format(alias)
    with torch.no_grad():
        # The throwaway clone contributes nothing but storage; taking it
        # under no_grad keeps a dead CloneBackward node out of the graph.
        try:
            fresh = alias.clone(memory_format=fmt)
        except (TypeError, RuntimeError):
            fresh = alias.clone()
    alias.data = fresh


def materialize_deferred_for_call(tensors: Iterable[Any]) -> None:
    """Materialize pending payload aliases before a mutating wrapped call.

    Called by the torch wrapper pre-execution — during capture AND on the
    post-capture fast path — for any call that can write through its tensor
    arguments. For each argument whose storage has pending aliases, every
    pending alias is copied out onto exclusive fresh storage BEFORE the
    mutation runs.
    """
    global _DEFER_BUSY
    if not _DEFER_PENDING or _DEFER_BUSY:
        return
    from .. import _state

    if (
        _state._active_trace is not None
        and _state._active_owner_thread_id is not None
        and threading.get_ident() != _state._active_owner_thread_id
    ):
        # Never toggle the global logging pause from a non-owner thread while
        # a capture is live (r43: it blinds owner op capture). Cross-thread
        # mutation of a pending storage is outside the single-threaded capture
        # claim; the version belt still reports it loudly at the next touch.
        return
    _DEFER_BUSY = True
    try:
        with _paused_internal_reads():
            for t in tensors:
                if not isinstance(t, torch.Tensor):
                    continue
                key = _deferred_storage_key(t)
                if key is None:
                    continue
                entries = _DEFER_PENDING.pop(key, None)
                if not entries:
                    continue
                group = [(e, e.ref()) for e in entries]
                # All belt checks BEFORE the first rebind, so the belt reads
                # every group member's version before any storage moves.
                for entry, alias in group:
                    if alias is not None:
                        _belt_check_pending_alias(entry, alias)
                for entry, alias in group:
                    if alias is not None:
                        _rebind_alias_to_fresh_clone(alias)
    finally:
        _DEFER_BUSY = False


def _alias_covers_whole_storage(alias: torch.Tensor, storage_nbytes: int) -> bool:
    """Return whether ``alias`` spans its storage end to end (offset 0)."""
    try:
        if alias.storage_offset() != 0:
            return False
        span_elems = 1
        # len(shape) == len(stride) is a torch invariant (both are the rank).
        for size, stride in zip(alias.shape, alias.stride(), strict=True):
            if size == 0:
                return False
            span_elems += (size - 1) * abs(stride)
        return span_elems * alias.element_size() == storage_nbytes
    except Exception:
        return False


# Dead registry entries are FUNCTIONALLY harmless — materialization skips
# dead weakrefs, and a reused (ptr, nbytes, device) key simply appends fresh
# entries after the dead ones — so pruning is memory hygiene only, gated on
# this threshold to keep window arming O(1) per op.
_DEFER_PRUNE_THRESHOLD = 2048

# Next-prune size: doubles away from the live population after each sweep.
# A fixed threshold alone is quadratic on large captures: once the LIVE
# pending population crosses it, every per-op window arming re-swept the
# whole registry and removed nothing (measured O(n^2), the dominant term at
# 4k ops). Doubling makes total prune work linear in total insertions.
_defer_prune_watermark = _DEFER_PRUNE_THRESHOLD

# Registered-alias deaths since the last sweep, counted O(1) by weakref
# callback. The doubling watermark alone never DECAYS: after one large
# capture pushed it up, a registry that then went mostly DEAD but sat below
# the doubled key-count watermark was never re-swept, pinning the dead
# entries (and their key tuples) indefinitely (r3 bounds-gap finding on the
# wave's own O(n^2)-kill). Crossing _DEFER_PRUNE_THRESHOLD dead entries
# forces a sweep regardless of the watermark; the post-sweep watermark
# reset (2x the now-live population) then decays back down. Amortization
# holds: each forced sweep requires THRESHOLD fresh deaths.
_defer_dead_alias_count = 0


def _note_dead_deferred_alias(_ref: "weakref.ref[torch.Tensor]") -> None:
    """Weakref callback: count one registered alias death (O(1))."""
    global _defer_dead_alias_count
    _defer_dead_alias_count += 1


def prune_dead_deferred_entries() -> None:
    """Drop registry entries whose aliases were garbage-collected.

    Discarded payload copies and dropped traces leave dead weakrefs behind;
    this bounded sweep keeps the registry sized to the live pending
    population.
    """
    global _defer_dead_alias_count
    _defer_dead_alias_count = 0
    for key in list(_DEFER_PENDING.keys()):
        entries = _DEFER_PENDING.get(key)
        if not entries:
            _DEFER_PENDING.pop(key, None)
            continue
        live = [e for e in entries if e.ref() is not None]
        if len(live) != len(entries):
            if live:
                _DEFER_PENDING[key] = live
            else:
                _DEFER_PENDING.pop(key, None)


def arm_deferred_payload_window(state_storage_ptrs: frozenset[int]) -> None:
    """Arm the clone-on-write payload window (wrapper-managed, nestable)."""
    global _DEFER_WINDOW_DEPTH, _DEFER_STATE_PTRS, _defer_prune_watermark
    if _DEFER_WINDOW_DEPTH == 0 and (
        len(_DEFER_PENDING) > _defer_prune_watermark
        or _defer_dead_alias_count > _DEFER_PRUNE_THRESHOLD
    ):
        prune_dead_deferred_entries()
        _defer_prune_watermark = max(_DEFER_PRUNE_THRESHOLD, 2 * len(_DEFER_PENDING))
    _DEFER_WINDOW_DEPTH += 1
    _DEFER_STATE_PTRS = state_storage_ptrs


def disarm_deferred_payload_window() -> None:
    """Disarm one nesting level of the clone-on-write payload window."""
    global _DEFER_WINDOW_DEPTH, _DEFER_STATE_PTRS
    _DEFER_WINDOW_DEPTH = max(0, _DEFER_WINDOW_DEPTH - 1)
    if _DEFER_WINDOW_DEPTH == 0:
        _DEFER_STATE_PTRS = None


#: Pending fence events for in-flight ``cpu_async`` D2H copies (R36-1).
#: Capture-scoped accumulate/drain state: each async pinned-buffer copy
#: records one event on its source device's current stream, and
#: ``synchronize_pending_cpu_async_copies()`` drains the list at the capture
#: finalize seam (and on the failure-scrub arms), so no host-side read
#: (``op.out``, ``tl.save`` serialization, dedup/attestation digests) can
#: observe partial bytes from an unfinished ``non_blocking=True`` copy.
_CPU_ASYNC_PENDING_EVENTS: list[Any] = []

#: Hard bound on accumulated fence events (R36): a capture that never reaches
#: a drain seam (or an exotic failure path) must not grow the list without
#: limit across captures. Crossing it drains inline — a fence, so strictly
#: correctness-neutral; it only reduces async overlap for that one copy.
_CPU_ASYNC_PENDING_EVENTS_MAX = 512


def _record_cpu_async_copy_event(device: torch.device) -> None:
    """Record a stream event fencing one ``cpu_async`` D2H copy (R36-1).

    Parameters
    ----------
    device:
        Source (non-CPU) device of the asynchronous copy. Only CUDA streams
        expose event fencing; other accelerators' ``non_blocking`` copies
        fall back to the conservative device synchronize at drain time.
    """

    if device.type == "cuda":
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device))
        _CPU_ASYNC_PENDING_EVENTS.append(event)
    else:
        _CPU_ASYNC_PENDING_EVENTS.append(device)
    if len(_CPU_ASYNC_PENDING_EVENTS) > _CPU_ASYNC_PENDING_EVENTS_MAX:
        synchronize_pending_cpu_async_copies()


def synchronize_pending_cpu_async_copies() -> None:
    """Fence every pending ``cpu_async`` D2H copy recorded this capture (R36-1).

    Called at the capture finalize seam and on the failure-scrub arms.
    Idempotent and cheap when nothing is pending; a completed copy's event
    synchronizes immediately.

    A synchronize failure never loses the unfenced tail: entries are retired
    one at a time as they are successfully fenced, and any failure restores
    the not-yet-fenced remainder (the failing entry included) to the pending
    registry ahead of copies recorded since the drain began, so a later drain
    retries them instead of silently no-opping while un-fenced
    ``non_blocking=True`` copies stay in flight.
    """

    if not _CPU_ASYNC_PENDING_EVENTS:
        return
    pending = list(_CPU_ASYNC_PENDING_EVENTS)
    _CPU_ASYNC_PENDING_EVENTS.clear()
    synced_devices: set[str] = set()
    index = 0
    try:
        while index < len(pending):
            entry = pending[index]
            if isinstance(entry, torch.device):
                key = str(entry)
                if key not in synced_devices:
                    torch_module = torch_attr(entry.type)
                    sync = getattr(torch_module, "synchronize", None)
                    if sync is not None:
                        try:
                            sync(entry)
                        except TypeError:
                            # torch.mps.synchronize() (and kin) take no device
                            # argument. The unguarded call raised TypeError from
                            # the drain — on the failure-scrub arms that masked
                            # the ORIGINAL capture exception with a drain
                            # traceback.
                            sync()
                    # Marked fenced only AFTER the synchronize succeeded, so a
                    # failed device sync is retried for the device's later
                    # entries on the retry drain.
                    synced_devices.add(key)
            else:
                entry.synchronize()
            index += 1
    finally:
        if index < len(pending):
            # A failed fence must not lose the copies behind it: restore the
            # unfenced tail (failing entry included) so a later drain retries
            # instead of returning at the empty-list guard while
            # ``non_blocking=True`` copies are still in flight.
            _CPU_ASYNC_PENDING_EVENTS[:0] = pending[index:]


#: Backend labels that positively identify a non-CUDA capture home (R36).
#: Only these skip the allocator flush when the op-device scan yields no
#: evidence; any label OUTSIDE this closed set (including ``"unknown"``/
#: missing) fails toward the historical flush, so a future accelerator label
#: can never silently skip it.
_KNOWN_NON_CUDA_MEMORY_BACKENDS = frozenset({"cpu", "mps", "xpu", "hpu"})


def _entry_device_is_cuda(entry: Any) -> bool | None:
    """Classify one record's ``device_ref`` fact: CUDA, non-CUDA, or absent.

    Accepts both recorded shapes: the backend-neutral ``DeviceRef`` (hardware
    device class on ``.backend``) and a plain device string (``"cuda:0"``).
    """

    ref = getattr(entry, "device_ref", None)
    if ref is None:
        return None
    if isinstance(ref, str):
        return ref.startswith("cuda")
    backend = getattr(ref, "backend", None)
    if backend is None:
        return None
    return backend == "cuda"


def _capture_observed_cuda_device(trace: Any) -> bool | None:
    """Best-effort scan of recorded op device facts for a CUDA device (R36-3).

    Parameters
    ----------
    trace:
        Captured (possibly mid-postprocess) Trace. Both record sources are
        consulted: iterating the trace itself (finished layer records) and
        its ``ops`` sequence (raw/mid-postprocess records).

    Returns
    -------
    bool | None
        ``True`` when any recorded op ran on a CUDA device, ``False`` when at
        least one op carried a device fact and none was CUDA, and ``None``
        when the scan yields no evidence (non-iterable object, zero recorded
        ops, no op with a device fact, or any scan failure) — the caller then
        falls back to the stamped backend fact.
    """

    saw_device_fact = False
    for source in (trace, getattr(trace, "ops", None)):
        if source is None:
            continue
        source_saw_fact = False
        try:
            for entry in source:
                verdict = _entry_device_is_cuda(entry)
                if verdict is None:
                    continue
                source_saw_fact = True
                if verdict:
                    return True
        except Exception:
            # An interrupted scan is NO evidence: partial negative facts must
            # not settle "provably non-CUDA" (the scan only ever ADDS
            # flushes, never removes one).
            continue
        saw_device_fact = saw_device_fact or source_saw_fact
    return False if saw_device_fact else None


def capture_touched_cuda(trace: Any) -> bool:
    """Return whether this capture's forward plausibly touched CUDA (R36-3).

    Gates the capture-lifecycle ``torch.cuda.empty_cache()`` calls on the
    CAPTURE having used CUDA, not on process-wide availability: a CPU-only
    trace inside a GPU training loop must not flush the caller's allocator.

    Keying decision (R36 b6 pair — one fix for both directions): the stamped
    ``forward_memory_backend`` fact reflects the MODEL device only, so it is
    a false negative for a CPU-homed model that moves tensors to CUDA inside
    ``forward`` and over-broad for non-CUDA accelerator captures whose fact
    reads ``"unknown"``. The recorded op ``device_ref`` facts are the
    authoritative key: any CUDA-deviced op means the capture touched CUDA,
    and a completed scan with none means it provably did not. When no op
    facts are scannable, an unknown or missing backend fact fails toward the
    historical flush, never toward skipping it.

    A non-CUDA-homed model can still move tensors to CUDA inside ``forward``,
    so the recorded op devices are always consulted: any recorded ``cuda``
    output flips the verdict to flush. A failure while consulting keeps the
    stamped fact's verdict — the scan only ever ADDS flushes, never removes
    one.

    Parameters
    ----------
    trace:
        Captured (possibly mid-postprocess) Trace.

    Returns
    -------
    bool
        False only when the capture provably ran on non-CUDA devices
        (completed op-device scan with no CUDA fact) or, absent any op
        evidence, was stamped with a KNOWN non-CUDA backend.
    """

    backend = getattr(trace, "forward_memory_backend", None)
    if backend == "cuda":
        return True
    observed = _capture_observed_cuda_device(trace)
    if observed is not None:
        return observed
    return backend not in _KNOWN_NON_CUDA_MEMORY_BACKENDS


def _copy_tensor_payload(
    x: torch.Tensor | torch.nn.Parameter,
    *,
    detach_tensor: bool,
    save_mode: SaveMode,
    target_device: str | None = None,
) -> torch.Tensor:
    """Return a tensor payload according to the requested save mode.

    Parameters
    ----------
    x:
        Tensor or parameter to materialize.
    detach_tensor:
        Whether the saved payload should be detached from autograd.
    save_mode:
        Payload retention mode. ``"copy"`` safely clones; ``"reference"`` safely
        preserves the original value by relying on capture-time in-place handling;
        ``"view"`` stores a live alias that downstream in-place operations can mutate;
        and ``"cpu_async"`` clones to CPU with ``non_blocking=True``.
    target_device:
        Optional retention device for ``"copy"`` mode: the clone materializes
        DIRECTLY on this device in one copy (r8 b5 R35: the serial
        ``safe_copy(...)`` then ``safe_to(...)`` idiom paid a same-device
        clone AND a cross-device move per saved activation -- a 2x transient
        peak on the source device for ``output_device=``-offloading
        captures). Ignored for the other save modes (the caller's move
        handles those), and any transport failure falls back to the
        same-device clone ladder so exotic tensors keep their historical
        behavior (the caller's follow-up move still runs).

    Returns
    -------
    torch.Tensor
        Tensor payload for storage.
    """

    if save_mode == "reference":
        return x.detach() if detach_tensor else x
    if save_mode == "view":
        return x
    if save_mode == "cpu_async":
        payload = x.detach() if detach_tensor else x
        try:
            if payload.device.type != "cpu":
                cpu_payload = torch.empty_like(
                    payload,
                    device="cpu",
                    memory_format=_safe_get_memory_format(payload),
                    pin_memory=True,
                )
                cpu_payload.copy_(payload, non_blocking=True)
                _record_cpu_async_copy_event(payload.device)
                return cpu_payload
        except (TypeError, RuntimeError):
            pass
        result = payload.to(device="cpu", non_blocking=True, copy=True)
        if payload.device.type != "cpu":
            _record_cpu_async_copy_event(payload.device)
        return result

    mem_fmt = _safe_get_memory_format(x)
    if target_device is not None:
        # Single-transport clone: ``.to(device, copy=True)`` materializes the
        # retained payload directly on the retention device (one allocation,
        # one copy) with the same autograd semantics as clone-then-move (the
        # move is differentiable when not detached).
        source = x.detach() if detach_tensor else x
        try:
            return source.to(device=target_device, memory_format=mem_fmt, copy=True)
        except (TypeError, RuntimeError):
            try:
                return source.to(device=target_device, copy=True)
            except (TypeError, RuntimeError):
                # Fall through to the same-device ladder; the caller's
                # follow-up move preserves the historical behavior.
                pass
    if not detach_tensor:
        with _pause_dispatch_modes_for_subclass_clone(x):
            try:
                return x.clone(memory_format=mem_fmt)
            except (TypeError, RuntimeError):
                return x.clone()
    try:
        return x.detach().clone(memory_format=mem_fmt)
    except (TypeError, RuntimeError):
        try:
            return x.detach().clone()
        except Exception:
            try:
                return x.data.cpu().clone()
            except Exception as exc:
                # Fail loud rather than fabricate a payload. The former
                # ``torch.zeros(x.shape, dtype=torch.float32)`` last resort
                # silently returned a WRONG value AND a WRONG dtype (float32
                # regardless of the source) with no marker, corrupting the
                # captured activation invisibly. A tensor that survives none of
                # the three clone strategies cannot be copied; surfacing that is
                # the only honest outcome, and it mirrors the non-detached path
                # above, which already propagates a clone failure.
                raise RuntimeError(
                    "torchlens could not copy a tensor payload: every clone "
                    "strategy failed. Refusing to fabricate a placeholder tensor "
                    "(which would silently corrupt the captured activation). "
                    f"Source tensor: shape={tuple(x.shape)}, dtype={x.dtype}."
                ) from exc


def _clone_tensor_payload(
    x: torch.Tensor | torch.nn.Parameter,
    *,
    detach_tensor: bool,
    save_mode: SaveMode,
    target_device: str | None = None,
) -> torch.Tensor | torch.nn.Parameter:
    """Clone or retain one tensor payload without triggering TorchLens logging.

    Parameters
    ----------
    x
        Tensor or parameter to clone or retain.
    detach_tensor
        Whether to detach copied payloads from the autograd graph.
    save_mode
        Tensor retention mode. ``"copy"`` preserves historical clone behavior,
        ``"reference"`` stores the source tensor, ``"view"`` stores the
        graph-connected source tensor, and ``"cpu_async"`` copies to CPU.
    target_device
        Optional device for the stored copy; only ``save_mode="copy"`` moves,
        and ``None``/``"same"``/the source device leave the payload in place.

    Returns
    -------
    torch.Tensor | torch.nn.Parameter
        Tensor payload with TorchLens raw label preserved, or a rewrapped
        parameter payload for parameter inputs.
    """
    from .._state import pause_logging

    with pause_logging():
        if save_mode not in SAVE_MODES:
            raise ValueError(
                "save_mode must be one of " + ", ".join(repr(m) for m in sorted(SAVE_MODES))
            )
        move_target: str | None = None
        if (
            target_device is not None
            and save_mode == "copy"
            and target_device not in ("same", str(x.device))
        ):
            move_target = target_device
        vals_tensor = None
        if _DEFER_WINDOW_DEPTH and save_mode == "copy" and move_target is None:
            # A plain ``detach()`` alias carries NO autograd state, so it may
            # only stand in for a clone taken with ``detach_tensor=True``, from
            # a ``requires_grad=False`` source, or under disabled grad mode
            # (where ``clone`` outputs are detached too, e.g. the common
            # ``torch.no_grad()`` activation-extraction pattern).
            if detach_tensor or not x.requires_grad or not torch.is_grad_enabled():
                vals_tensor = _try_defer_payload_alias(x)
            elif _DEFER_GRAD_ENABLED:
                # Default grad-enabled regime: the eager clone stays attached,
                # so the alias needs a grafted identity gradient edge. Opt-in
                # while residual H3 is open; see the module notes.
                vals_tensor = _try_defer_payload_alias(x, graph_connected=True)
        if vals_tensor is None:
            vals_tensor = _copy_tensor_payload(
                x,
                detach_tensor=detach_tensor,
                save_mode=save_mode,
                target_device=move_target,
            )
        label = None if isinstance(x, torch.nn.Parameter) else get_tensor_label(x)
        if label is not None:
            set_tensor_label(vals_tensor, label)
        if isinstance(x, torch.nn.Parameter):
            # Preserve the source parameter's requires_grad. torch.nn.Parameter
            # defaults requires_grad=True, so a frozen (requires_grad=False)
            # parameter would otherwise yield a copy that falsely claims grad --
            # misrepresenting the captured parameter in every save mode.
            return torch.nn.Parameter(vals_tensor, requires_grad=x.requires_grad)
        return vals_tensor


def copy_tensor_payload(
    x: Any,
    *,
    save_mode: SaveMode = "copy",
    detach_tensor: bool = False,
    target_device: str | None = None,
) -> Any:
    """Copy an output payload with tensor-clone and shallow non-tensor semantics.

    Uses ``pause_logging()`` so that ``.clone()``, ``.detach()``,
    ``.cpu()`` etc. don't get logged — these are all decorated tensor
    custom_methods, and calling them during active logging would create spurious
    entries or infinite recursion.

    For non-tensor inputs, falls back to ``copy.copy()`` (shallow copy),
    which is safe because non-tensor objects don't have circular-reference
    issues the way tensor wrappers do (see :func:`_safe_copy_arg` for the
    deeper discussion on why ``deepcopy`` is avoided).

    Parameters
    ----------
    x
        Input value, tensor, parameter, or arbitrary object.
    save_mode
        Tensor retention mode. ``"copy"`` preserves historical clone behavior.
        ``"reference"`` stores the detached source tensor. ``"view"`` stores
        the graph-connected source tensor. ``"cpu_async"`` copies to CPU using
        ``non_blocking=True``.
    detach_tensor
        If True, detach the saved payload from the autograd graph. This is used
        when saving outs to avoid retaining the full computational graph in
        memory.
    target_device
        Optional device for the stored copy, forwarded to the tensor-clone
        path; ``None``/``"same"``/the source device leave the payload in place.

    Returns
    -------
    Any
        Tensor payload copy/retention result, or a shallow copy for non-tensors.
    """

    if isinstance(x, (torch.Tensor, torch.nn.Parameter)):
        if _is_uninitialized_lazy_tensor(x):
            # Lazy (uninitialized) parameters/buffers have no payload to copy
            # -- every tensor accessor raises until the first forward
            # materializes them in place. Retain the object reference itself.
            return x
        return _clone_tensor_payload(
            x,
            detach_tensor=detach_tensor,
            save_mode=save_mode,
            target_device=target_device,
        )
    else:
        # Non-tensor: shallow copy is sufficient and avoids deepcopy's
        # circular-reference pitfalls.
        return copy.copy(x)


def _is_uninitialized_lazy_tensor(value: Any) -> bool:
    """Return whether ``value`` is a lazy uninitialized parameter or buffer."""

    is_lazy = getattr(torch.nn.parameter, "is_lazy", None)
    if is_lazy is not None:
        try:
            return bool(is_lazy(value))
        except (TypeError, ValueError, RuntimeError):
            return False
    return isinstance(value, torch.nn.parameter.UninitializedParameter)


def safe_copy(
    x: Any,
    detach_tensor: bool = False,
    save_mode: SaveMode = "copy",
    target_device: str | None = None,
) -> Any:
    """Compatibility alias for :func:`copy_tensor_payload`.

    Parameters
    ----------
    x
        Input value, tensor, parameter, or arbitrary object.
    detach_tensor
        Whether tensor payloads should detach from autograd.
    save_mode
        Tensor retention mode.
    target_device
        Optional single-transport retention device for ``"copy"`` mode.

    Returns
    -------
    Any
        Output-payload copy result.
    """

    return copy_tensor_payload(
        x,
        save_mode=save_mode,
        detach_tensor=detach_tensor,
        target_device=target_device,
    )


def print_override(t: torch.Tensor, func_name: str) -> str:
    """Safe ``__str__``/``__repr__`` for tensors during active logging.

    The default ``Tensor.__repr__`` calls decorated custom_methods internally,
    which would re-enter the logging pipeline and cause infinite recursion.
    This override pauses logging, converts to a numpy array for formatting,
    and appends autograd metadata (``grad_fn_handle`` / ``requires_grad``) to
    match the standard PyTorch repr style.

    Falls back to a shape/dtype summary for tensors that can't be converted
    to numpy (sparse, quantized, meta, float8, etc.).

    Args:
        t: Tensor to format.
        func_name: Either ``"__str__"`` or ``"__repr__"``.

    Returns:
        Human-readable string representation of the tensor.
    """
    from .._state import pause_logging

    try:
        with pause_logging():
            cpu_data = t.data.cpu()
            # numpy() doesn't support bfloat16 — upcast first.
            if cpu_data.dtype == torch.bfloat16:
                cpu_data = cpu_data.to(torch.float32)
            # ``.detach()`` is a decorated torch method like any other; calling
            # it outside pause_logging (while a trace is actively logging)
            # would log a real "detach" op and consume a raw-op-counter slot,
            # leaving a graph orphan and staling any raw labels recorded just
            # before this repr fired. Keep it inside the paused block.
            n = cpu_data.detach().numpy()
        np_str = getattr(n, func_name)()
        # Cosmetic: replace "array" with "tensor" to match PyTorch style.
        np_str = np_str.replace("array", "tensor")
        np_str = np_str.replace("\n", "\n ")
    except Exception:
        # Fallback for sparse, quantized, meta, float8, etc.
        np_str = f"tensor(shape={list(t.shape)}, dtype={t.dtype})"
    # Append autograd info to mimic standard PyTorch repr.
    if t.grad_fn is not None:
        grad_fn_str = f", grad_fn_handle={type(t.grad_fn).__name__})"
        np_str = np_str[0:-1] + grad_fn_str
    elif t.requires_grad:
        np_str = np_str[0:-1] + ", requires_grad=True)"
    return cast(str, np_str)
