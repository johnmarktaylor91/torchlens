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
from ._errors import LazyStateUnsupportedError
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


def check_lazy_state(model: nn.Module) -> None:
    """Refuse capture entry on a model carrying un-materialized lazy BUFFERS.

    Pending lazy PARAMETERS are tolerated: the lazy completion unit
    (quickstart memo wave 1c, landed by the numbers-truth lane) materializes
    executed lazy modules during the ONE captured forward and keeps
    never-run ones at zero geometry in the inventory. Pending lazy BUFFERS
    (``LazyBatchNorm*`` running stats) remain a genuine blocker: the
    capture-boundary buffer-write tracker must index every buffer's physical
    storage BEFORE the forward runs, and a pending buffer has no storage yet
    (measured: ``untyped_storage()`` on it raises torch's raw
    ``load_state_dict``-flavored ``ValueError`` inside model preparation).
    For that case the typed ``lazy_uninitialized`` teach names the first
    pending module and enumerates the pending set by name and ``id()`` on
    ``exc.fields`` (``pending_modules`` / ``pending_parameters`` /
    ``pending_buffers``). The model is left untouched -- detection never
    probes a lazy module.

    Parameters
    ----------
    model:
        The ``nn.Module`` about to be captured.

    Raises
    ------
    torchlens._errors.LazyStateUnsupportedError
        When any lazy BUFFER is still pending. Pending parameters alone
        never refuse.
    """

    from .utils.lazy_state import has_uninitialized_lazy_state, pending_lazy_state

    if not has_uninitialized_lazy_state(model):
        return
    pending = pending_lazy_state(model)
    if not pending.buffers:
        return
    if pending.modules:
        address, type_name, _ = pending.modules[0]
        first = f"model.{address} ({type_name})" if address else f"the root module ({type_name})"
    else:
        first = f"buffer {pending.buffers[0][0]!r}"
    caller_file, caller_line = _first_user_frame()
    callsite_note = ""
    if caller_file is not None and caller_line is not None:
        from ._source_links import file_line_text

        callsite_note = f" Capture was requested at {file_line_text(caller_file, caller_line)}."
    raise LazyStateUnsupportedError(
        f"The model contains un-materialized lazy BUFFERS whose storage does "
        f"not exist until a real forward pass runs -- the first pending "
        f"module is {first} ({len(pending.modules)} pending module(s), "
        f"{len(pending.parameters)} pending parameter(s), "
        f"{len(pending.buffers)} pending buffer(s); the full set rides "
        f"exc.fields). Capture must index every buffer's physical storage "
        f"before the forward runs, so this capture would fail inside model "
        f"preparation with torch's raw uninitialized-parameter ValueError. "
        f"(Un-materialized lazy PARAMETERS alone are fine: they materialize "
        f"during the captured forward.) The model was left "
        f"untouched.{callsite_note}",
        code="lazy_uninitialized",
        remedy=(
            "materialize the lazy modules with one real forward pass outside "
            "capture -- `with torch.no_grad(): model(x)` -- then retry the capture"
        ),
        file_path=caller_file,
        line_no=caller_line,
        pending_modules=pending.modules,
        pending_parameters=pending.parameters,
        pending_buffers=pending.buffers,
    )


def check_model_and_input_variants(
    model: nn.Module,
    input_args: Any = None,
    input_kwargs: dict[str, Any] | None = None,
) -> None:
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
    """
    if input_kwargs is None:
        input_kwargs = {}

    # Assignment-redirecting wrappers (transformer_lens TransformerBridge) are
    # refused before anything else: instrumentation assignments would silently
    # land on the wrapped components and capture would die with an internal
    # AttributeError (mikit F10).
    from ._model_wrappers import check_model_wrapper

    check_model_wrapper(model)

    # Un-materialized lazy PARAMETERS do not refuse capture entry: the lazy
    # completion unit (quickstart memo wave 1c, landed by the numbers-truth
    # lane) materializes executed lazy modules during the ONE captured
    # forward and tolerates never-run ones at zero geometry, so the wave-1a
    # entry teach flipped off for them on the memo's own signal (4.4: the
    # refusal holds only until the metadata invariants pass on the lazy-head
    # fixture). The typed ``lazy_uninitialized`` teach still fires exactly
    # where the request is genuinely unanswerable today: pending lazy
    # BUFFERS (the buffer-write tracker cannot index storage that does not
    # exist yet -- checked here), the armed-lane state baseline
    # (``state_baseline_unavailable`` in ``snapshot_capture_state`` -- a
    # pending slot has no bytes to witness), and zero-input shape inference
    # (refuse before probing; a lazy module accepts any width).
    check_lazy_state(model)

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
        if _is_meta_tensor(t):
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
        if _is_meta_tensor(t):
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
    seen_ids: set[int] = set()
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        if id(t) in seen_ids:
            continue
        seen_ids.add(id(t))
        if _is_meta_tensor(t):
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
        tracing_kind = _tracing_tensor_kind(t)
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
