"""Tensor-variant detection and pre-flight guards for ``trace``.

TorchLens was designed around standard dense ``torch.Tensor`` /
``torch.nn.Parameter`` objects on real (CPU/CUDA/MPS) devices.  A number of
tensor variants break the logging pipeline in different ways:

================  =================================================  =================
Variant           Why TorchLens cannot handle it today                Detection outcome
================  =================================================  =================
Meta tensor       No storage, so out saving returns garbage;  raise RuntimeError
                  ``.clone()`` yields another meta tensor, etc.
Sparse tensor     ``safe_copy``/print-override paths assume dense    raise RuntimeError
                  layouts; postprocess indexing uses ``.numel()``
                  which double-counts sparse entries.
Symbolic shape   Dimensions that are ``torch.SymInt`` /              raise RuntimeError
                  ``torch.SymFloat`` break shape-dependent metadata
                  (flops, tensor memory, counter alignment).
Tracing tensor    ``FakeTensor`` / ``FunctionalTensor`` carry no      raise RuntimeError
                  data, so every value-reading step (``safe_copy``,
                  ``torch.equal``, ``.item()``, ``data_ptr()``) is
                  meaningless; torch's own fake machinery aborts
                  mid-forward with a bare ``AssertionError``.
Quantized model   Partial support: logging works but FLOPs are        warn (keep going)
                  computed as zero/wrong for quantized ops.
================  =================================================  =================

This module centralises detection.  Callers (``trace``,
``log_model_metadata``, ``validate_forward_pass``) invoke
:func:`check_model_and_input_variants` near entry, *before* decoration or
session setup, so failures happen up front with a clear error message
instead of partway through an 18-step pipeline.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import Any

import torch
from torch import nn

from ._distributed import check_distributed_capture
from ._input_walk import INPUT_TREE_MAX_DEPTH
from .errors._base import CompatibilityError, TorchLensWarning
from .utils._torch_compat import get_tracing_tensor_types

# ---------------------------------------------------------------------------
# Per-tensor detectors
# ---------------------------------------------------------------------------


def _is_meta_tensor(t: torch.Tensor) -> bool:
    """True if ``t`` lives on the meta device (no backing storage)."""
    try:
        return t.device.type == "meta"
    except Exception:
        return False


def _is_sparse_tensor(t: torch.Tensor) -> bool:
    """True for any sparse layout (COO, CSR, CSC, BSR, BSC)."""
    # ``layout`` exists on every torch.Tensor; sparse variants are not ``strided``.
    try:
        layout = t.layout
    except Exception:
        return False
    return layout is not torch.strided


def _tracing_tensor_kind(t: torch.Tensor) -> str | None:
    """Return the data-free tracing-subclass name for ``t``, if it is one.

    ``FakeTensor`` and ``FunctionalTensor`` are tensor subclasses used by Dynamo,
    AOTAutograd, ``torch.export``, and functionalization. They carry shape and
    dtype but no storage, so every TorchLens step that reads a value is either
    meaningless or fatal: ``data_ptr()`` on a FakeTensor is a torch-flagged bug,
    and torch's own fake machinery aborts the forward with a bare
    ``AssertionError`` ("Please convert all Tensors to FakeTensors first") the
    moment a real parameter meets a fake activation.

    Parameters
    ----------
    t:
        Tensor to classify.

    Returns
    -------
    str | None
        Class name of the tracing tensor, or ``None`` for an ordinary tensor.
    """
    functional_predicate = getattr(torch, "_is_functional_tensor", None)
    if callable(functional_predicate):
        try:
            if bool(functional_predicate(t)):
                return "FunctionalTensor"
        except (RuntimeError, TypeError):
            pass
    if type(t) is torch.Tensor:
        return None
    tracing_types = get_tracing_tensor_types()
    if tracing_types and isinstance(t, tracing_types):
        return type(t).__name__
    # Structural fallback for builds where the exact classes could not be probed.
    type_name = type(t).__name__
    if type_name in {"FakeTensor", "FunctionalTensor"}:
        return type_name
    return None


def _has_symbolic_shape(t: torch.Tensor) -> bool:
    """True if any dimension is a ``torch.SymInt`` / ``torch.SymFloat``.

    Concrete ``int`` dims are safe.  Symbolic dims arise under
    ``torch._dynamo.mark_dynamic`` / ``torch.export`` traces and break
    metadata collection.
    """
    SymInt = getattr(torch, "SymInt", None)
    SymFloat = getattr(torch, "SymFloat", None)
    if SymInt is None and SymFloat is None:
        return False
    try:
        shape = t.shape
    except Exception:
        return False
    for dim in shape:
        if SymInt is not None and isinstance(dim, SymInt):
            return True
        if SymFloat is not None and isinstance(dim, SymFloat):
            return True
    return False


# ---------------------------------------------------------------------------
# Model-level detectors
# ---------------------------------------------------------------------------


# Quantized module class names — string-match to avoid importing
# ``torch.ao.quantization`` modules when the user doesn't have them compiled in.
_QUANTIZED_MODULE_NAME_PREFIXES: tuple[str, ...] = (
    "torch.ao.nn.quantized",
    "torch.nn.quantized",
    "torch.ao.nn.intrinsic.quantized",
    "torch.ao.nn.qat",
    "torch.nn.qat",
    # bitsandbytes 8/4-bit modules (lane F37): same disclosed-degradation
    # class -- capture works with value parity, but quantization-state reads
    # (state.CB/SCB, packed uint8 payloads) sit outside the dense-tensor
    # contract and FLOPs/out-dtype handling is best-effort.
    "bitsandbytes.nn",
)


def _is_quantized_module(module: nn.Module) -> bool:
    """True if ``module``'s class lives in a quantization namespace."""
    mod_name = type(module).__module__ or ""
    return any(mod_name.startswith(prefix) for prefix in _QUANTIZED_MODULE_NAME_PREFIXES)


def _model_has_quantized_modules(model: nn.Module) -> bool:
    """True if any submodule is a quantized ``nn`` module."""
    return any(_is_quantized_module(sub) for sub in model.modules())


# ---------------------------------------------------------------------------
# Input-tree walk
# ---------------------------------------------------------------------------


class VariantScanTruncationWarning(TorchLensWarning):
    """Emitted when the bounded entry-time tensor scan is truncated.

    The input-tree walk in :func:`_iter_tensors` is bounded (depth and total
    node count) so a pathological or adversarial container cannot stall
    capture entry. When either bound truncates the scan, tensors beyond the
    bound were NOT inspected: an unsupported variant hiding there will not
    receive the typed entry refusal and will instead fail later, mid-capture,
    with a raw error. This category discloses that honestly instead of
    silently narrowing the guarantee.
    """


_ITER_TENSORS_MAX_DEPTH = INPUT_TREE_MAX_DEPTH + 56
"""Maximum container-nesting depth inspected by :func:`_iter_tensors`.

Derived ABOVE the declared input-container contract (``INPUT_TREE_MAX_DEPTH``,
200) with headroom: a bound below the contract (the original 128) truncated the
scan -- and emitted the spurious truncation disclosure -- on fully SUPPORTED
deep inputs, and preempted the device-move walker's typed over-depth refusal
on unsupported ones. Trees within the contract now scan completely; over-deep
input trees reach the walker's ``InvalidArgumentError``; only genuinely deeper
non-input attribute graphs still truncate with the disclosure.
"""

_ITER_TENSORS_MAX_NODES = 4096
"""Maximum total objects inspected by one :func:`_iter_tensors` traversal."""


def _iter_tensors(
    obj: Any,
    _seen: set[int] | None = None,
) -> Iterator[torch.Tensor]:
    """Yield tensors through builtin and inspectable user containers.

    Thin adapter over :func:`_iter_tensors_with_paths` for callers that do not
    need the input-tree location (e.g. ``compat/_report.py``).
    """

    for _path, tensor in _iter_tensors_with_paths(obj, _seen=_seen):
        yield tensor


def _path_key(key: Any) -> str:
    """Render one dict key for an input-tree path, bounded against hostile reprs."""

    try:
        rendered = repr(key)
    except Exception:  # noqa: BLE001 - hostile __repr__ must not break the refusal
        rendered = f"<{type(key).__name__}>"
    if len(rendered) > 40:
        rendered = rendered[:37] + "..."
    return rendered


def _iter_tensors_with_paths(
    obj: Any,
    _seen: set[int] | None = None,
    root_path: str = "",
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(input-tree path, tensor)`` through builtin and user containers.

    Parameters
    ----------
    obj:
        Root object to inspect.
    _seen:
        Shared object-identity set for cycle prevention.
    root_path:
        Path prefix naming the root (e.g. ``"args"`` / ``"kwargs"``).

    Yields
    ------
    tuple[str, torch.Tensor]
        Reachable tensor values with the path that reaches them, so refusals
        can name WHERE the offending tensor sits (R67).

    Notes
    -----
    ``nn.Module`` instances are not descended into because registered state is
    handled separately. Instance ``__dict__`` is read directly, so properties and
    descriptors never execute. Traversal is iterative (an explicit worklist, so
    the depth bound is decoupled from Python's recursion limit) and capped at
    ``_ITER_TENSORS_MAX_DEPTH`` levels / 4096 objects; when either bound truncates the scan, a one-shot
    :class:`VariantScanTruncationWarning` disclosure is emitted because
    unsupported variants beyond the bound would fail undetected later. Opaque
    slots-only objects and tensors created later inside ``forward`` remain
    outside entry-time detection and are disclosed in the compatibility report.
    """
    if _seen is None:
        _seen = set()
    nodes = 0
    truncated_by: str | None = None
    stack: list[tuple[Any, int, str]] = [(obj, 0, root_path)]
    while stack:
        current, depth, path = stack.pop()
        if depth > _ITER_TENSORS_MAX_DEPTH:
            if truncated_by is None:
                truncated_by = f"depth bound ({_ITER_TENSORS_MAX_DEPTH} nesting levels)"
            continue
        if nodes >= _ITER_TENSORS_MAX_NODES:
            if truncated_by is None:
                truncated_by = f"node bound ({_ITER_TENSORS_MAX_NODES} objects)"
            # The counter never decreases, so every remaining item would be
            # skipped identically — stop instead of draining the worklist.
            break
        obj_id = id(current)
        if obj_id in _seen:
            continue
        _seen.add(obj_id)
        nodes += 1
        if isinstance(current, torch.Tensor):
            yield path, current
            continue
        if isinstance(current, nn.Module):
            continue
        if isinstance(current, (list, tuple)):
            children = [(child, f"{path}[{index}]") for index, child in enumerate(current)]
        elif isinstance(current, (set, frozenset)):
            # Set members have no stable position; the braces still say "inside
            # this set" without claiming an ordering.
            children = [(child, f"{path}{{...}}") for child in current]
        elif isinstance(current, dict):
            children = [(child, f"{path}[{_path_key(key)}]") for key, child in current.items()]
        else:
            try:
                attributes = vars(current)
            except (TypeError, AttributeError):
                continue
            children = [(child, f"{path}.{name}") for name, child in attributes.items()]
        # Reverse so the stack pops children in original order (DFS preorder,
        # matching the recursive traversal this replaced).
        for child, child_path in reversed(children):
            stack.append((child, depth + 1, child_path))
    if truncated_by is not None:
        warnings.warn(
            "TorchLens entry-time tensor-variant scan was truncated at its "
            f"{truncated_by}: tensors beyond the bound were not inspected, so "
            "unsupported tensor variants hiding there will not be refused up "
            "front and may fail later during capture with a raw error.",
            VariantScanTruncationWarning,
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


class UnsupportedTensorVariantError(CompatibilityError, RuntimeError):
    """Raised when ``trace`` is called on a model/input combination
    that TorchLens cannot reliably log (see module docstring for the matrix).

    Every raise site attaches structured context on ``fields`` so callers
    branch without parsing message text: ``code`` is always
    ``"unsupported_tensor_variant"``, ``remedy`` names the fix, and
    ``offenses`` is a tuple of
    ``{"name", "reason", "path", "shape", "dtype"}`` dicts, one per detected
    variant class, where ``path`` is the input-tree location of the first
    offending tensor (``"args[0]"``, ``"kwargs['x'].deep"``, ``"model.<param>"``,
    or a mid-forward op label). ``shape`` is ``None`` for shapeless variants.
    """


def _offense_entry(name: str, reason: str, path: str, tensor: torch.Tensor) -> dict[str, Any]:
    """Build one structured offense record (R67: WHERE, not just WHAT).

    Parameters
    ----------
    name:
        Variant-class label (e.g. ``"meta tensor in input"``).
    reason:
        Why the variant is unsupported (may be empty for compact channels).
    path:
        Input-tree location that reaches the offending tensor.
    tensor:
        The offending tensor; shape/dtype reads are guarded because shapeless
        variants raise on ``.shape`` and hostile subclasses may raise anywhere.

    Returns
    -------
    dict[str, Any]
        ``{"name", "reason", "path", "shape", "dtype"}``.
    """

    try:
        shape: tuple[int, ...] | None = tuple(tensor.shape)
    except Exception:  # noqa: BLE001 - shapeless variants raise internal errors here
        shape = None
    try:
        dtype = str(tensor.dtype)
    except Exception:  # noqa: BLE001
        dtype = None
    return {"name": name, "reason": reason, "path": path, "shape": shape, "dtype": dtype}


def _docs_pointer(section: str | None = None) -> str:
    """Human-readable pointer to the limitations documentation.

    Parameters
    ----------
    section:
        Optional exact section heading of ``docs/reference/limitations.md``
        to cite; ``None`` points at the catalog as a whole.

    Returns
    -------
    str
        Pointer sentence naming a real documentation location.
    """
    if section is None:
        return "See docs/reference/limitations.md for supported alternatives."
    return (
        f"See the {section!r} section of docs/reference/limitations.md for supported alternatives."
    )


def _first_user_frame() -> tuple[str | None, int | None]:
    """Return the first non-torchlens frame (the user's capture callsite).

    Teaching-enrichment helper (L7a): names WHERE the refused capture was
    requested. Best-effort; ``(None, None)`` when no external frame is
    visible.
    """

    import inspect
    from pathlib import Path

    torchlens_root = Path(__file__).resolve().parent
    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame is not None else None
        while frame is not None:
            filename = Path(frame.f_code.co_filename).resolve()
            try:
                filename.relative_to(torchlens_root)
            except ValueError:
                return str(filename), frame.f_lineno
            frame = frame.f_back
    finally:
        del frame
    return None, None


def check_model_and_input_variants(
    model: nn.Module,
    input_args: Any = None,
    input_kwargs: dict[str, Any] | None = None,
    *,
    admit_meta: bool = False,
) -> Any:
    """Pre-flight check for ``trace``.

    Raises :class:`UnsupportedTensorVariantError` when a fundamentally
    incompatible tensor variant is detected on the model or its inputs, and
    :class:`torchlens._distributed.DistributedCaptureUnsupportedError` when the
    model holds distributed/sharded state that capture would record incorrectly
    rather than fail on.
    Emits :class:`UserWarning` for variants with partial / degraded support
    (quantization) so the user knows what to treat with skepticism in the log.

    Args:
        model: The ``nn.Module`` about to be logged.
        input_args: Positional arguments that will be passed to
            ``model.forward`` (may contain nested containers of tensors).
        input_kwargs: Keyword arguments to ``model.forward``.
        admit_meta: Scoped weights-free admission (W2, weightsfree memo D2):
            ``True`` exactly when resolved ``CaptureOptions.structure_only``
            is in force at a CAPTURE entry. Meta tensors are then admitted
            IFF the capability table's ``meta_admission`` row is flipped
            (D8) AND the substrate is uniform (all-meta inputs and state;
            mixed cells refuse typed with
            ``structure_only_substrate_mismatch``). The rerun, backward, and
            fastlog gate sites thread the default — a permanently closed
            regime. Every other variant refusal (sparse, symbolic, fake,
            distributed) is unchanged by admission.

    Returns:
        ``torchlens.capture._weightsfree_admission.MetaAdmissionRecord`` when
        a meta substrate was admitted, else ``None``. The capture entry
        registers the record against the Trace; settlement consumes it (an
        admitted meta offense without the final marker fails closed).
    """
    if input_kwargs is None:
        input_kwargs = {}

    # Assignment-redirecting wrappers (transformer_lens TransformerBridge) are
    # refused before anything else: instrumentation assignments would silently
    # land on the wrapped components and capture would die with an internal
    # AttributeError (mikit F10).
    from ._model_wrappers import check_model_wrapper

    check_model_wrapper(model)

    # Un-materialized lazy state does not refuse capture entry. Lazy
    # PARAMETERS flipped off with the numbers-truth completion unit
    # (quickstart memo wave 1c); lazy BUFFERS flipped off with the F20
    # buffer-side completion (A10-fix2 remainder): the buffer-write tracker
    # skips storage-less pending buffers at index time, torch's lazy
    # pre-hook materialization plumbing passes through the wrapper unlogged,
    # and the materialized buffer registers at the module-entry gate. The
    # typed ``lazy_uninitialized`` teach still fires where the request is
    # genuinely unanswerable: the armed-lane state baseline
    # (``state_baseline_unavailable`` in ``snapshot_capture_state`` -- a
    # pending slot has no bytes to witness) and zero-input shape inference
    # (refuse before probing; a lazy module accepts any width).

    # Distributed/sharded state is checked next: DTensor parameters otherwise
    # sail past every dense-tensor check below (a DTensor reports a real device
    # and a strided layout) and capture then silently reports zero parameters.
    check_distributed_capture(model, input_args, input_kwargs)

    # SPMD processes that first-capture with distributed already initialized
    # arm collective-boundary capture lazily here (restricted registry seeding,
    # design-merge-ranks-c v5 rule 1.3.2). Explicit torchlens.distributed.arm()
    # at process start remains the required spelling for MPMD programs.
    from .distributed._lifecycle import maybe_auto_arm

    maybe_auto_arm()

    meta_admission = _maybe_admit_meta(model, input_args, input_kwargs) if admit_meta else None

    offenses: list[dict[str, Any]] = []

    # Treat a bare tensor and a container of tensors identically — ``_iter_tensors``
    # yields tensors directly for a tensor, or recurses into list/tuple/dict.
    if input_args is None:
        args_payload: Any = []
    elif isinstance(input_args, torch.Tensor):
        args_payload = input_args
    else:
        args_payload = input_args

    # Input-side tensors.
    for path, t in _iter_tensors_with_paths(args_payload, root_path="args"):
        if _is_meta_tensor(t) and meta_admission is None:
            offenses.append(
                _offense_entry(
                    "meta tensor in input",
                    "Meta tensors have no backing storage, so out saving "
                    "cannot produce usable values.",
                    path,
                    t,
                )
            )
        if _is_sparse_tensor(t):
            offenses.append(
                _offense_entry(
                    f"sparse tensor ({t.layout}) in input",
                    "TorchLens' copy/print/FLOPs paths assume dense strided layouts.",
                    path,
                    t,
                )
            )
        if _has_symbolic_shape(t):
            offenses.append(
                _offense_entry(
                    "symbolic (SymInt/SymFloat) tensor shape in input",
                    "TorchLens requires concrete integer shapes for metadata and "
                    "counter alignment.",
                    path,
                    t,
                )
            )
        tracing_kind = _tracing_tensor_kind(t)
        if tracing_kind is not None:
            offenses.append(
                _offense_entry(
                    f"{tracing_kind} in input",
                    "Tracing tensors carry shape and dtype but no data, so saved "
                    "activations would be empty and torch's own fake-tensor machinery "
                    "aborts the forward as soon as a real parameter meets a fake "
                    "activation. Capture the eager forward on real tensors instead.",
                    path,
                    t,
                )
            )
    for path, t in _iter_tensors_with_paths(dict(input_kwargs), root_path="kwargs"):
        if _is_meta_tensor(t) and meta_admission is None:
            offenses.append(_offense_entry("meta tensor in keyword input", "", path, t))
        if _is_sparse_tensor(t):
            offenses.append(
                _offense_entry(f"sparse tensor ({t.layout}) in keyword input", "", path, t)
            )
        if _has_symbolic_shape(t):
            offenses.append(_offense_entry("symbolic tensor shape in keyword input", "", path, t))
        tracing_kind = _tracing_tensor_kind(t)
        if tracing_kind is not None:
            offenses.append(_offense_entry(f"{tracing_kind} in keyword input", "", path, t))

    # Model params + buffers (dedupe across both generators).
    # Accelerate-offload-backed meta state is admitted with evidence (lane
    # F37 / R5 deployment envelope): the hook's weights_map materializes the
    # real values onto the execution device for the duration of each module
    # call, so the captured forward sees real tensors. The set is computed
    # lazily on the FIRST meta tensor found -- zero cost on the default path.
    offload_backed: frozenset[str] | None = None
    seen_ids: set[int] = set()
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        if id(t) in seen_ids:
            continue
        seen_ids.add(id(t))
        # Tracing kinds are checked BEFORE the meta cell: a FakeTensor may sit
        # on the meta device, and admission must never launder a fake/functional
        # variant through the meta carve-out (weightsfree memo sec 4.2).
        tracing_kind = _tracing_tensor_kind(t)
        if _is_meta_tensor(t) and tracing_kind is None:
            if meta_admission is None:
                if offload_backed is None:
                    from ._deploy_env import offload_backed_state_paths

                    offload_backed = offload_backed_state_paths(model)
                if name in offload_backed:
                    # Offload-hook-backed meta state: real values arrive at
                    # forward time from the hook's weights_map. Admitted.
                    continue
                offenses.append(
                    _offense_entry(
                        "meta tensor among model parameters/buffers",
                        "Meta-init models (e.g. HuggingFace device_map='meta') must be "
                        "materialized on a real device before logging.",
                        f"model.{name}",
                        t,
                    )
                )
                break  # one message is enough — don't list every param.
            continue  # admitted uniform-meta state; variant checks above still ran
        if tracing_kind is not None:
            offenses.append(
                _offense_entry(
                    f"{tracing_kind} among model parameters/buffers",
                    "The model was constructed under a fake/functional tracing mode and "
                    "holds no real weights. Build it on a real device before logging.",
                    f"model.{name}",
                    t,
                )
            )
            break

    if offenses:
        # Dedupe by variant class while preserving order of first appearance;
        # the surviving entry keeps the first offending tensor's path/shape/dtype.
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for offense in offenses:
            if offense["name"] in seen:
                continue
            seen.add(offense["name"])
            unique.append(offense)
        bullet_list = "\n".join(
            f"  - {offense['name']} (at {offense['path']})"
            + (f": {offense['reason']}" if offense["reason"] else "")
            for offense in unique
        )
        # L7a default-path teaching enrichment (memo 1.4-B item 3, ships
        # regardless of D8): name the USER callsite and point meta-init
        # holders at the structure-only mode's contract. No new code is
        # minted and the offense payload shape is unchanged.
        caller_file, caller_line = _first_user_frame()
        callsite_note = ""
        if caller_file is not None and caller_line is not None:
            from ._source_links import file_line_text

            callsite_note = (
                f"\nCapture was requested at {file_line_text(caller_file, caller_line)}."
            )
        meta_pointer = ""
        if any("meta tensor" in offense["name"] for offense in unique):
            if _meta_admission_row_open():
                meta_pointer = (
                    "\nMeta-initialized models are admitted ONLY under the "
                    "structure-only contract (decision point D8): pass "
                    "capture=CaptureOptions(structure_only=True) with an "
                    "all-meta model and all-meta inputs to record the op "
                    "graph, module nesting, and shape/dtype HYPOTHESES with "
                    "no tensor values -- see "
                    "docs/reference/structure_only_capabilities.md."
                )
            else:
                meta_pointer = (
                    "\nMeta-initialized models stay refused at this gate "
                    "(decision point D8); structure-only capture "
                    "(structure_only=True) records graph structure and shape "
                    "hypotheses for supported substrates -- see "
                    "docs/reference/structure_only_capabilities.md."
                )
        raise UnsupportedTensorVariantError(
            "torchlens.trace cannot run on this model/input "
            "combination. Detected unsupported tensor variant(s):\n"
            f"{bullet_list}\n"
            f"{callsite_note}{meta_pointer}"
            f"\n{_docs_pointer('Capture entry and execution contexts')}",
            code="unsupported_tensor_variant",
            file_path=caller_file,
            line_no=caller_line,
            remedy=(
                "materialize dense, strided tensors with concrete integer "
                "shapes on a real device before capture"
            ),
            offenses=tuple(unique),
        )

    # Warnings (non-fatal).
    if _model_has_quantized_modules(model):
        warnings.warn(
            "TorchLens detected quantized submodules. Activation capture "
            "generally works, but FLOPs counts are estimated only for common "
            "quantized Linear/Conv module outputs and out dtype handling is best-effort. "
            f"{_docs_pointer()}",
            UserWarning,
            stacklevel=3,
        )

    return meta_admission


def _meta_admission_row_open() -> bool:
    """Whether the capability table's ``meta_admission`` row is D8-flipped.

    The capability table is the ONE code authority for the flip (weightsfree
    memo build item 16): admission plumbing lands first with the row still
    refusing, and THE FLIP is the last merge — the row state is read through
    the table module's own accessor, never duplicated.
    """

    from .capture.structure_only import meta_admission_open

    return meta_admission_open()


def _maybe_admit_meta(model: nn.Module, input_args: Any, input_kwargs: dict[str, Any]) -> Any:
    """Classify substrates and admit a uniform meta capture entry (W2/D2).

    Returns ``None`` when no meta tensor is present (the ordinary real-path
    scan proceeds) or when the D8 flip has not landed (the historical meta
    refusal proceeds unchanged). Raises typed on a mixed substrate. On
    admission, runs the D20 identity self-test and returns the
    ``MetaAdmissionRecord`` consumed at settlement.

    Substrate uniformity is judged over the complete registered
    parameter/buffer scan (tied objects deduplicated by identity) plus the
    normalized input tensor leaves — never a walk of arbitrary Python object
    graphs (memo D11). Tracing variants (fake/functional) are never
    classified as meta; their own refusals run in the main scan.
    """

    meta_inputs, real_inputs = _classify_input_substrates(input_args, input_kwargs)
    meta_state, real_state = _classify_state_substrates(model)

    if not meta_inputs and not meta_state:
        return None
    if not _meta_admission_row_open():
        return None  # pre-flip: the historical refusal in the main scan governs

    if real_inputs or real_state:
        from ._errors import SubstrateMismatchError

        caller_file, caller_line = _first_user_frame()
        meta_side = tuple(meta_inputs + meta_state)
        real_side = tuple(real_inputs + real_state)
        raise SubstrateMismatchError(
            "Weights-free capture requires a UNIFORM meta substrate: every "
            "input tensor leaf meta AND every registered parameter/buffer "
            f"meta. This call mixes substrates — meta side: "
            f"{', '.join(meta_side[:3])}{'...' if len(meta_side) > 3 else ''} "
            f"({len(meta_side)} tensor(s)); real side: "
            f"{', '.join(real_side[:3])}{'...' if len(real_side) > 3 else ''} "
            f"({len(real_side)} tensor(s)). A mixed capture would record a "
            "graph that is neither the real model's nor a coherent "
            "hypothesis. Remedy: construct the WHOLE model under "
            "torch.device('meta') and pass meta inputs "
            "(torch.zeros(..., device='meta')), or materialize everything "
            "on a real device.",
            code="structure_only_substrate_mismatch",
            file_path=caller_file,
            line_no=caller_line,
            meta_side=meta_side,
            real_side=real_side,
        )

    from . import _state
    from .capture._weightsfree_admission import MetaAdmissionRecord, self_test_meta_identity

    self_test_meta_identity()

    return MetaAdmissionRecord(
        substrate="meta",
        factory_device_policy="torchlens_owned",
        ambient_mode_present=_ambient_device_context_present(),
        wrap_generation=int(getattr(_state, "_wrap_epoch", 0)),
        meta_input_paths=tuple(meta_inputs),
        meta_state_names=tuple(meta_state),
    )


def _classify_input_substrates(
    input_args: Any, input_kwargs: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Partition input tensor leaves into (meta paths, real paths).

    Tracing variants (fake/functional) are never classified as meta; their
    own refusals run in the main scan (weightsfree memo sec 4.2).
    """

    meta_inputs: list[str] = []
    real_inputs: list[str] = []
    args_payload = [] if input_args is None else input_args
    for path, tensor in _iter_tensors_with_paths(args_payload, root_path="args"):
        if _tracing_tensor_kind(tensor) is not None:
            continue
        (meta_inputs if _is_meta_tensor(tensor) else real_inputs).append(path)
    for path, tensor in _iter_tensors_with_paths(dict(input_kwargs), root_path="kwargs"):
        if _tracing_tensor_kind(tensor) is not None:
            continue
        (meta_inputs if _is_meta_tensor(tensor) else real_inputs).append(path)
    return meta_inputs, real_inputs


def _classify_state_substrates(model: nn.Module) -> tuple[list[str], list[str]]:
    """Partition registered parameters/buffers into (meta names, real names).

    The complete registered scan with tied objects deduplicated by identity
    (weightsfree memo D11) — never a walk of arbitrary Python object graphs.
    """

    meta_state: list[str] = []
    real_state: list[str] = []
    seen_state_ids: set[int] = set()
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if id(tensor) in seen_state_ids:
            continue
        seen_state_ids.add(id(tensor))
        if _tracing_tensor_kind(tensor) is not None:
            continue
        (meta_state if _is_meta_tensor(tensor) else real_state).append(f"model.{name}")
    return meta_state, real_state


def _ambient_device_context_present() -> bool:
    """Whether a caller-active torch ``DeviceContext`` mode is on the stack.

    Recorded as evidence-envelope provenance (D19). The admitted capture
    scope absorbs or refuses the mode at forward time; entry only observes.
    """

    try:
        from .backends.torch.wrappers import _get_active_device

        return _get_active_device() is not None
    except Exception:  # noqa: BLE001 — provenance observation, never authority
        return False
