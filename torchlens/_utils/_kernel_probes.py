"""Capability probes that run one real tensor op each (a torch-only leaf).

The ``HAS_*`` flags these answer, their lazy latches and their capability-snapshot
rows stay in ``torchlens._utils._torch_compat``, which imports this module at probe
time through its one ``_real_op_probe_leaf`` site, so its eager import closure does
not grow. Every probe catches only the error torch raises when the capability is
missing, so any other failure still surfaces.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from ._cuda_probe import _is_cuda_available

#: Substring of torch's refusal when a CUDA build was compiled without the cuDSS
#: sparse direct solver (``_apply_sparse_csr_linear_solve`` in
#: ``aten/src/ATen/native/sparse/cuda/SparseCsrTensorMath.cu``).
_NO_CUDSS_MARKER = "cuDSS"


def probe_cached_untyped_storage_wrapper() -> bool:
    """Return whether a tensor retains one stable untyped-storage Python wrapper.

    Torch 2.1 creates a fresh ``UntypedStorage`` wrapper on every
    ``Tensor.untyped_storage()`` call. Its wrapper destructor can leave a
    ``weakref.ref`` pointing at freed memory, so placing an ephemeral handle in a
    ``WeakKeyDictionary`` can later segfault CPython while clearing the weakref.
    Newer torch retains one wrapper on the tensor, which makes weak-key storage
    registries safe for the tensor's lifetime.

    The probe deliberately compares two live handles instead of constructing a
    weakref: exercising the broken weakref destructor would itself corrupt the
    interpreter on an unsupported runtime.

    Returns
    -------
    bool
        ``True`` when repeated calls return the same retained wrapper.
    """

    try:
        tensor = torch.empty(0)
        first = tensor.untyped_storage()
        second = tensor.untyped_storage()
    except (AttributeError, RuntimeError, TypeError):
        return False
    return first is second


def probe_cpu_half_kernels() -> bool:
    """Return whether common CPU kernels accept Half-dtype operands.

    Returns
    -------
    bool
        ``True`` when CPU ``addmm``, ``layer_norm``, ``nextafter``, and
        ``aminmax`` all accept Half operands. Older torch CPU backends lack these
        kernels for Half tensors (``addmm_impl_cpu_``, ``LayerNormKernelImpl``,
        ``nextafter_cpu``, and ``aminmax_cpu`` all raise ``"... not implemented
        for 'Half'"``); later torch ships them. Absence is a genuine torch CPU
        limitation on the floor, not a TorchLens degradation: tests that exercise
        a half-precision CPU forward, TorchLens's own ULP-step validation
        machinery (``torch.nextafter`` in ``validation/core.py`` and
        ``validation/exemptions.py``) or its dense tensor-stats kernel
        (``torch.aminmax`` in ``stats/_stats_kernel.py``) on a Half output skip
        on it.
    """

    try:
        half = torch.randn(2, 2, dtype=torch.float16)
        torch.addmm(half, half, half)
        torch.nn.functional.layer_norm(half, (2,))
        torch.nextafter(half, half)
        torch.aminmax(half)
    except RuntimeError:
        return False
    return True


def probe_eval_batch_norm_mean_residual() -> bool:
    """Return whether eval batch_norm at a non-zero running mean still depends on running_var.

    Returns
    -------
    bool
        ``True`` when an eval-mode, affine-free ``torch.batch_norm`` whose input
        equals a non-zero ``running_mean`` exactly changes its output when
        ``running_var`` changes. The x86 CPU kernel folds the statistics into
        ``input * alpha + (bias - running_mean * alpha)`` and the two
        ``running_mean * alpha`` terms cancel only to within a few ulps, so a
        live running_var edge moves the output; the Apple ARM kernel cancels
        exactly and the output stays all-zero whatever ``running_var`` is. A
        kernel property, not a TorchLens degradation: the running_var value
        proof's tests read it to know whether a live running_var edge is
        observable at a non-zero mean on this build.
    """

    generator = torch.Generator().manual_seed(2)
    mean = torch.randn(64, generator=generator) * 3
    var = torch.rand(64, generator=generator) + 1.5
    x = mean.view(1, 64, 1, 1).expand(2, 64, 3, 3).contiguous()
    try:
        base = torch.batch_norm(x, None, None, mean, var, False, 0.1, 1e-5, False)
        moved = torch.batch_norm(x, None, None, mean, var + 1.0, False, 0.1, 1e-5, False)
    except RuntimeError:
        return False
    return not torch.equal(base, moved)


def probe_cpu_float8_deterministic_fill() -> bool:
    """Return whether CPU tensor allocation fills Float8 dtypes under determinism.

    Returns
    -------
    bool
        ``True`` when ``torch.empty`` of the ``float8_e4m3fn`` dtype succeeds with
        ``torch.use_deterministic_algorithms(True)`` active. Older torch's
        ``fill_empty_deterministic_`` CPU kernel does not cover Float8 dtypes, so
        any empty-tensor allocation under deterministic mode raises
        ``RuntimeError: "fill_empty_deterministic_" not implemented``. The global
        determinism setting is restored before returning.
    """

    float8_dtype = getattr(torch, "float8_e4m3fn", None)
    if float8_dtype is None:
        return False
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        torch.empty((1,), dtype=float8_dtype)
    except RuntimeError:
        return False
    finally:
        torch.use_deterministic_algorithms(was_deterministic, warn_only=was_warn_only)
    return True


def probe_sparse_spsolve_cuda() -> bool:
    """Return whether ``torch.sparse.spsolve`` runs on CUDA in this torch build.

    Returns
    -------
    bool
        ``False`` without the op (it postdates the torch floor), without a usable
        CUDA device (the op has only a ``SparseCsrCUDA`` kernel), or when the
        CUDA build lacks the cuDSS solver, in which case torch refuses every call
        with a ``RuntimeError`` naming cuDSS (official wheels ship without it).
        Otherwise the probe solves the identity system and returns ``True``.
        Any other error propagates: it is a fault, not an absent capability.
    """

    spsolve = getattr(torch.sparse, "spsolve", None)
    if spsolve is None or not _is_cuda_available():
        return False
    return _solves_identity_system(spsolve, "cuda")


def _solves_identity_system(spsolve: Callable[..., torch.Tensor], device: str) -> bool:
    """Solve a tiny CSR identity system; ``False`` only on torch's no-cuDSS refusal.

    Parameters
    ----------
    spsolve:
        The ``torch.sparse.spsolve`` callable to exercise.
    device:
        Device the operands are built on.

    Returns
    -------
    bool
        ``True`` when the solve runs, ``False`` when torch refuses it for want of
        cuDSS. Every other error, the dispatcher's ``NotImplementedError`` for a
        device with no kernel included, propagates.
    """

    matrix = torch.eye(2, device=device).to_sparse_csr()
    try:
        spsolve(matrix, torch.ones(2, device=device))
    except RuntimeError as error:
        if _NO_CUDSS_MARKER in str(error):
            return False
        raise
    return True
