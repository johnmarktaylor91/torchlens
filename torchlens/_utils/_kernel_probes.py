"""Capability probes that run one real tensor op (or one non-importing lookup) each (a torch-only leaf).

The ``HAS_*`` flags these answer, their lazy latches and their capability-snapshot
rows stay in ``torchlens._utils._torch_compat``, which imports this module at probe
time through its one ``_real_op_probe_leaf`` site, so its eager import closure does
not grow. Every probe catches only the error torch raises when the capability is
missing, so any other failure still surfaces.
"""

from __future__ import annotations

import contextlib
import importlib.util
import inspect
import sys
import types
from collections.abc import Callable
from typing import Any

import torch

from ._cuda_probe import _is_cuda_available

#: Substring of torch's refusal when a CUDA build was compiled without the cuDSS
#: sparse direct solver (``_apply_sparse_csr_linear_solve`` in
#: ``aten/src/ATen/native/sparse/cuda/SparseCsrTensorMath.cu``).
_NO_CUDSS_MARKER = "cuDSS"

#: Substrings of torch's refusal when an active ``sdpa_kernel`` restriction leaves
#: no SDPA backend for a call (``select_sdp_backend`` / ``_fused_sdp_choice``).
_SDPA_NO_BACKEND_MARKERS = (
    "No viable backend for scaled_dot_product_attention",
    "No available kernel",
)


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


def probe_node_sequence_nr() -> bool:
    """Return whether autograd nodes expose their creation order (``Node._sequence_nr()``).

    The in-place view-write rebind proves that a ``CopySlices`` node belongs to the
    write that just ran by comparing creation order with every recorded node.

    Returns
    -------
    bool
        ``True`` when a fresh node answers ``_sequence_nr()`` with an ``int``.
    """

    try:
        with torch.enable_grad():
            node = (torch.ones(1, requires_grad=True) * 2.0).grad_fn
        read = getattr(node, "_sequence_nr", None)
        return callable(read) and isinstance(read(), int)
    except (AttributeError, RuntimeError, TypeError):
        return False


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
        machinery (``torch.nextafter`` in ``validation/_core.py`` and
        ``validation/_exemptions.py``) or its dense tensor-stats kernel
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


def probe_reduce_tuple_dim() -> bool:
    """Return whether ``Tensor.any``/``Tensor.all`` accept a tuple ``dim``.

    Returns
    -------
    bool
        ``True`` when a boolean tensor's ``.any(dim=(0, 1))`` accepts a tuple
        of axes directly (added torch 2.2; torch 2.1 only accepts a single
        int). Absence is a healthy old install: :func:`tensor_any_over_dims`
        falls back to sequential single-axis reduction.
    """

    probe = torch.zeros((2, 2), dtype=torch.bool)
    try:
        probe.any(dim=(0, 1))
    except TypeError:
        return False
    return True


def probe_torch_cpp_extension() -> bool:
    """Return whether ``torch.utils.cpp_extension`` imports cleanly.

    Torch 2.1's extension builder imports ``pkg_resources`` when it loads, and current
    setuptools no longer ships that module, so on such an install inductor's CPU backend and
    ``torch.utils.benchmark`` fail at import. Later torch dropped the import (torch 2.2
    through 2.9 still need setuptools itself to be installed).

    Returns
    -------
    bool
        ``True`` when the import succeeds.
    """

    try:
        import torch.utils.cpp_extension  # noqa: F401
    except ImportError:
        return False
    return True


def probe_meta_item_guard() -> bool:
    """Return whether ``Tensor.item()`` raises its own guard on meta tensors.

    Returns
    -------
    bool
        ``True`` when calling ``.item()`` (or any other scalar extraction,
        e.g. ``bool()``) on a meta tensor raises a plain ``RuntimeError``
        carrying torch's own "Tensor.item() cannot be called on meta
        tensors" guard. torch 2.1-2.2 have no such guard: the call instead
        falls all the way through to the aten dispatcher's generic
        ``NotImplementedError`` ("... not implemented for this backend").
        TorchLens's structure-only forward-boundary backstop
        (``backends/torch/structure_only_belt.py``) classifies a caught
        ``NotImplementedError`` as ``meta_kernel_unavailable`` and a generic
        ``RuntimeError`` as ``value_dependent_branch_unsupported`` by
        raising-frame PROVENANCE, never message text -- correctly, since
        the two exception FAMILIES genuinely differ here. Absence is a
        healthy old install surfacing the honest alternate typed refusal,
        not a TorchLens degradation.
    """

    try:
        torch.ones((), device="meta").item()
    except NotImplementedError:
        return False
    except RuntimeError:
        return True
    return False


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


def probe_numpy_bridge() -> bool:
    """Return whether torch's C-level NumPy bridge works in this process.

    Returns
    -------
    bool
        ``True`` when ``torch.from_numpy`` and ``Tensor.numpy()`` both round-trip
        a one-element array. torch wheels compiled against NumPy 1.x (torch 2.1
        and 2.2) import under NumPy 2 with a "Failed to initialize NumPy"
        warning, after which every bridge call raises ``RuntimeError: Numpy is
        not available``. Measured 2026-10-31: torch 2.1.2 + NumPy 2.2.6 fails;
        torch 2.3.1 and 2.4.1 + NumPy 2.2.6 pass. Feature-detected, never parsed
        from version strings.
    """

    try:
        import numpy as np

        torch.from_numpy(np.zeros(1, dtype=np.float32))
        torch.zeros(1).numpy()
    except (ImportError, RuntimeError):
        return False
    return True


def probe_gradient_edge() -> bool:
    """Return whether torch's GradientEdge surface actually WORKS end to end.

    Returns
    -------
    bool
        ``True`` when ``torch.autograd.graph`` exposes both ``GradientEdge``
        and ``get_gradient_edge`` -- the ``(node, output slot)`` addressing
        pair the one-backward read engine seeds ``autograd.grad`` with -- AND
        a real ``autograd.grad`` call seeding an OUTPUT ``GradientEdge`` with
        an explicit cotangent succeeds. Attribute presence alone is
        insufficient: torch 2.2-2.3 ships the ``GradientEdge`` class, but its
        own internal ``torch.autograd._make_grads`` calls ``out.is_nested`` on
        every output -- including a bare ``GradientEdge``, which has no such
        attribute -- and raises ``AttributeError`` (fixed upstream by 2.4,
        matching this read's long-documented "2.4+" remedy text). Absent on
        older torch or broken on this intermediate band is a healthy old
        install either way, not a TorchLens degradation: the read refuses
        typed.
    """

    from torch.autograd import graph

    gradient_edge = getattr(graph, "GradientEdge", None)
    if gradient_edge is None or getattr(graph, "get_gradient_edge", None) is None:
        return False

    try:
        x = torch.randn(1, requires_grad=True)
        h = x * 2
        if h.grad_fn is None:
            # Never happens for a requires_grad leaf's product, but the
            # static annotation is ``Node | None``; treat an impossible-in-
            # practice None the same as any other probe failure below.
            return False
        torch.autograd.grad(
            [gradient_edge(h.grad_fn, h.output_nr)], [x], grad_outputs=[torch.ones(1)]
        )
    except Exception:
        return False
    return True


def _redispatch_probe_mode(mode_base: type[Any]) -> Any:
    """Return a dispatch mode built on ``mode_base`` that redispatches every op unchanged.

    Parameters
    ----------
    mode_base:
        Torch's dispatch-mode base class. ``_torch_compat`` (the one module that imports
        it) hands it in, so this leaf touches no private torch module.

    Returns
    -------
    Any
        A fresh mode instance, entered with ``with``.
    """

    class _RedispatchProbeMode(mode_base):
        """Redispatch operations unchanged for the Parameter subclass capability probe."""

        def __torch_dispatch__(
            self,
            func: Any,
            types: tuple[type[Any], ...],
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            """Redispatch one probe operation without changing its result.

            Parameters
            ----------
            func:
                Dispatcher operator invoked by the probe.
            types:
                Participating tensor subclass types.
            args:
                Positional operator arguments.
            kwargs:
                Keyword operator arguments.

            Returns
            -------
            Any
                Unmodified operator result.
            """

            del types
            return func(*args, **(kwargs or {}))

    return _RedispatchProbeMode()


def probe_parameter_as_subclass_in_dispatch_mode(mode_base: type[Any]) -> bool:
    """Return whether Parameter-to-Tensor subclass conversion works in a dispatch mode.

    Torch 2.1 rejects ``Parameter.as_subclass(torch.Tensor)`` while a
    :class:`TorchDispatchMode` is active because the returned raw tensor is already
    associated with a base ``Tensor`` Python object. Newer torch releases permit the
    conversion. TorchLens' completeness witness is itself a dispatch mode, so this
    behavioral probe mirrors the dynamic-parameter logging context without parsing a
    version string.

    Parameters
    ----------
    mode_base:
        Torch's dispatch-mode base class, handed in by ``_torch_compat``.

    Returns
    -------
    bool
        ``True`` when the conversion succeeds inside a redispatching mode.
    """

    parameter = torch.nn.Parameter(torch.empty(0))
    try:
        with _redispatch_probe_mode(mode_base):
            plain_tensor = parameter.as_subclass(torch.Tensor)
    except (RuntimeError, TypeError):
        return False
    return type(plain_tensor) is torch.Tensor


class _SubclassCtorProbe(torch.Tensor):
    """Plain strict Tensor subclass used only by the ctor-in-mode probe below."""


def probe_subclass_ctor_in_dispatch_mode(mode_base: type[Any]) -> bool:
    """Return whether constructing a STRICT Tensor subclass works in a dispatch mode.

    Torch 2.1 and 2.2 reject ``Tensor.as_subclass``/``_make_subclass`` (and the
    equivalent ``__new__`` path) into a custom (non-``torch.Tensor``) subclass
    while a :class:`TorchDispatchMode` is active: "Creating a new Tensor
    subclass X but the raw Tensor object is already associated to a python
    object of type Tensor." This reproduces on stock torch with a no-op mode
    (TorchLens's own wrapping is not the cause); newer torch permits the
    conversion. TorchLens's completeness witness and intervention-ready
    captures are themselves dispatch modes, so call sites that construct or
    convert into a strict subclass while one is active need this behavioral
    probe rather than a parsed version string. Sibling probe (Parameter-to-
    Tensor, the opposite direction, also version-gated but independently):
    :func:`probe_parameter_as_subclass_in_dispatch_mode`.

    Parameters
    ----------
    mode_base:
        Torch's dispatch-mode base class, handed in by ``_torch_compat``.

    Returns
    -------
    bool
        ``True`` when the conversion succeeds inside a redispatching mode.
    """

    base = torch.empty(0)
    try:
        with _redispatch_probe_mode(mode_base):
            converted = base.as_subclass(_SubclassCtorProbe)
    except (RuntimeError, TypeError):
        return False
    return type(converted) is _SubclassCtorProbe


def probe_roll_tensor_shifts() -> bool:
    """Return whether ``torch.roll`` accepts a bare 0-dim tensor ``shifts``.

    Older torch (verified: 2.8 rejects with ``RuntimeError: shifts required``;
    2.13 accepts) does not unpack a 0-dim tensor passed as the ``shifts``
    argument itself; the tuple spelling ``shifts=(tensor,)`` is accepted
    everywhere. Behavioral probe on a 1-element tensor, never a version parse.

    Returns
    -------
    bool
        ``True`` when the bare tensor ``shifts`` spelling executes.
    """

    try:
        torch.roll(torch.zeros(2), shifts=torch.zeros((), dtype=torch.long))  # type: ignore[arg-type]
    except (RuntimeError, TypeError):
        return False
    return True


def _sdpa_math_backend() -> contextlib.AbstractContextManager[Any]:
    """Pin SDPA to its math backend for one probe call, whatever backend the caller forced.

    The SDPA probes run lazily, inside the first capture, so they can run under a
    user's ``sdpa_kernel([...])`` restriction; a restriction to a backend with no CPU
    kernel (memory-efficient, cuDNN) would refuse the probe's CPU call. Torch before
    ``torch.nn.attention.sdpa_kernel`` (2.1-2.2) gets no pin; its refusal is handled
    by ``_is_sdpa_backend_refusal`` at the call site.

    Returns
    -------
    contextlib.AbstractContextManager[Any]
        ``sdpa_kernel([SDPBackend.MATH])`` when available, else a null context.
    """

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        return contextlib.nullcontext()
    return sdpa_kernel([SDPBackend.MATH])


def _is_sdpa_backend_refusal(error: RuntimeError) -> bool:
    """Return whether ``error`` is torch refusing SDPA because no backend is enabled."""

    return any(marker in str(error) for marker in _SDPA_NO_BACKEND_MARKERS)


def _sdpa_signature_names_enable_gqa() -> bool:
    """Feature-detect ``enable_gqa`` from SDPA's documented signature (no call)."""

    doc = getattr(torch.nn.functional.scaled_dot_product_attention, "__doc__", None) or ""
    return "enable_gqa" in doc.split("->", 1)[0]


def probe_sdpa_gqa() -> bool:
    """Return whether ``scaled_dot_product_attention`` accepts ``enable_gqa`` (torch 2.5+).

    Older torch rejects the keyword with ``TypeError``; the probe runs one call with
    two query heads over one key/value head, pinned to the math backend so a
    caller's ``sdpa_kernel`` restriction cannot refuse it. If torch still refuses
    for want of a backend, the answer comes from SDPA's documented signature.

    Returns
    -------
    bool
        ``True`` when the grouped-query call executes.
    """

    query, key_value = torch.ones(1, 2, 1, 1), torch.ones(1, 1, 1, 1)
    sdpa: Any = torch.nn.functional.scaled_dot_product_attention
    try:
        with _sdpa_math_backend():
            sdpa(query, key_value, key_value, enable_gqa=True)
    except TypeError:
        return False
    except RuntimeError as error:
        if not _is_sdpa_backend_refusal(error):
            raise
        return _sdpa_signature_names_enable_gqa()
    return True


def probe_sdpa_empty_row_zero() -> bool:
    """Return whether SDPA writes zero, not NaN, for a query row with no admissible key.

    Torch 2.5+ softmax leaves such a row all zero, so the output row is zero; older
    torch returns NaN. Both spellings of an empty row are probed: a fully masked row
    and a row whose only key scores ``-inf``. The calls are pinned to the math
    backend so a caller's ``sdpa_kernel`` restriction cannot refuse them; if torch
    still refuses for want of a backend, the capability is unknown and reported
    absent (the NaN-writing older behaviour is the conservative assumption).

    Returns
    -------
    bool
        ``True`` when both empty rows come out zero over a zero value.
    """

    sdpa = torch.nn.functional.scaled_dot_product_attention
    query, value = torch.ones(1, 1, 1, 1), torch.zeros(1, 1, 1, 1)
    try:
        with _sdpa_math_backend():
            masked = sdpa(query, query, value, attn_mask=torch.zeros(1, 1, dtype=torch.bool))
            non_finite = sdpa(query, torch.full((1, 1, 1, 1), float("-inf")), value)
    except RuntimeError as error:
        if not _is_sdpa_backend_refusal(error):
            raise
        return False
    return bool((masked == 0).all()) and bool((non_finite == 0).all())


def probe_attention_causal_bias() -> bool:
    """Return whether ``torch.nn.attention.bias.CausalBias`` is discoverable.

    NEVER executes ``torch.nn.attention.bias`` (lazy-import belt): that
    module's top level calls ``torch._dynamo.allow_in_graph(...)``, dragging the
    whole ``_dynamo``/``_inductor`` tree into every ``import torchlens``. When
    the USER has already imported the module, the class is inspected exactly;
    otherwise ``find_spec`` proves existence without execution (the parent
    ``torch.nn.attention`` package body is dynamo-free) and the exact
    ``__torch_function__`` site check is deferred to causal-bias shim install,
    which resolves only through ``sys.modules``.

    Returns
    -------
    bool
        ``True`` when the module is already imported and the class defines its
        own ``__torch_function__`` (the identity-dispatch site the causal-bias
        identity shim normalizes), or when the module exists but has not been
        imported yet (the shim installer re-reads ``sys.modules`` at each wrap).
        Absent on torch builds predating the ``torch.nn.attention`` namespace.

    Notes
    -----
    This probe NEVER imports ``torch.nn.attention.bias``: on torch 2.13 its
    module body reaches ``torch._dynamo``, whose import tree also drags in
    ``torch.distributed.fsdp`` -- breaking the cold-start guarantee that a
    plain eager capture pays neither import. ``find_spec`` imports only the
    parent ``torch.nn.attention`` package (verified dynamo-free), and a process
    that never imported the module cannot hold a ``CausalBias`` instance, so
    the deferred read is exact, not heuristic.
    """

    module = sys.modules.get("torch.nn.attention.bias")
    if module is not None:
        causal_bias = getattr(module, "CausalBias", None)
        return causal_bias is not None and "__torch_function__" in vars(causal_bias)
    try:
        return importlib.util.find_spec("torch.nn.attention.bias") is not None
    except (ImportError, AttributeError, ValueError):
        return False


def probe_safe_weights_only_load(globals_in_pkl: object | None) -> bool:
    """Return whether ``torch.load(..., weights_only=True)`` is CVE-2025-32434-safe.

    CVE-2025-32434 (critical, CVSS 9.3; affects torch <= 2.5.1, fixed in torch
    2.6.0) is a remote-code-execution reachable through ``torch.load`` *itself*
    even with ``weights_only=True``: the pre-2.6 weights-only unpickler could be
    bypassed. TorchLens no longer hands file bytes to ``torch.load`` (the metadata
    unpickler refuses ``torch.storage._load_from_bytes`` by name; plain metadata
    tensors use its own inline reduction); ``.pt`` extraction shards and pyreft weight
    files, the last artifact bytes handed to ``torch.load``, refuse when it is false.

    The patched behavior is internal to the unpickler and has NO version-string-
    free *behavioral* signature we can cheaply exercise, so -- consistent with this
    module's feature-detect convention (never parse ``torch.__version__`` for a
    behavioral branch) -- we probe the 2.6.0 serialization-hardening SURFACE that
    shipped *in* the fix release: the public
    ``torch.serialization.get_unsafe_globals_in_checkpoint`` inspection API plus the
    rewritten ``torch._weights_only_unpickler.get_globals_in_pkl`` GLOBAL-opcode
    handler. Both are ABSENT on every affected version (<= 2.5.1) and PRESENT on
    2.6.0+ (verified against the v2.5.1 and v2.6.0 sources). Requiring BOTH is the
    conservative, fail-closed direction: a false negative merely refuses an
    embedded load on a safe torch, whereas admitting it on a vulnerable torch would
    reopen the RCE.

    Parameters
    ----------
    globals_in_pkl:
        ``torch._weights_only_unpickler.get_globals_in_pkl``, or ``None`` when absent.
        ``_torch_compat`` resolves it (the private import stays in that module) and
        passes it in; this leaf checks only the public half.

    Returns
    -------
    bool
        ``True`` only when the running torch carries the CVE-2025-32434 fix.
    """

    try:
        import torch.serialization as _serialization
    except Exception:  # pragma: no cover - defensive; imports on supported torch.
        return False
    return globals_in_pkl is not None and hasattr(
        _serialization, "get_unsafe_globals_in_checkpoint"
    )


def probe_code_positions() -> bool:
    """Return whether code objects expose PEP 657 fine-grained positions.

    ``co_positions()`` (CPython 3.11+) yields per-instruction column offsets.
    Conditional-branch attribution uses them to tell same-line ternary /
    ``IfExp`` arms apart; without them the branch indexer degrades to
    line-only matching, which deliberately FAILS CLOSED (un-attributed, never
    mis-attributed) on same-line arms. Probe the capability behaviorally.

    Returns
    -------
    bool
        ``True`` when per-instruction position introspection is available.
    """

    positions = getattr(probe_code_positions.__code__, "co_positions", None)
    if not callable(positions):
        return False
    try:
        next(iter(positions()), None)
    except (TypeError, ValueError):
        return False
    return True


def probe_code_qualname() -> bool:
    """Return whether code objects expose ``co_qualname`` (CPython 3.11+).

    Frame-to-scope resolution for branch attribution prefers qualified names;
    without them, same-named nested functions resolve by name plus first line
    only and ambiguous matches fail closed.

    Returns
    -------
    bool
        ``True`` when ``co_qualname`` is available on code objects.
    """

    return isinstance(getattr(probe_code_qualname.__code__, "co_qualname", None), str)


def probe_saved_tensors_hooks_patchable() -> bool:
    """Return whether the saved-tensors-hooks context class accepts an init patch.

    ``torch.autograd.graph.saved_tensors_hooks`` is the public context manager that installs user
    pack/unpack hooks (``save_on_cpu`` and the non-reentrant checkpoint hook subclass it).
    TorchLens scopes user hook bodies as autograd-internal during an active capture by patching the
    class ``__init__`` (construction-time scoping) AND ``__enter__`` (use-time re-scoping, covering
    contexts constructed before the first capture) at wrap time, so the patch needs a plain-Python
    ``__init__`` with the ``(self, pack_hook, unpack_hook)`` shape and a plain-Python ``__enter__``
    taking only ``self``. Probe the shape behaviorally; never parse ``torch.__version__``.

    Returns
    -------
    bool
        ``True`` when the class exists and both ``__init__`` and ``__enter__``
        are patchable Python functions with the expected shapes.
    """

    hooks_cls = getattr(getattr(torch.autograd, "graph", None), "saved_tensors_hooks", None)
    if not isinstance(hooks_cls, type):
        return False
    init = hooks_cls.__dict__.get("__init__")
    enter = hooks_cls.__dict__.get("__enter__")
    if not isinstance(init, types.FunctionType) or not isinstance(enter, types.FunctionType):
        return False
    try:
        init_params = list(inspect.signature(init).parameters)
        enter_params = list(inspect.signature(enter).parameters)
    except (TypeError, ValueError):
        return False
    return init_params == ["self", "pack_hook", "unpack_hook"] and enter_params == ["self"]
