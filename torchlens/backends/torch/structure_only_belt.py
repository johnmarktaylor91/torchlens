"""Structure-only teaching-refusal belts (L7a memo sec 2, D8-default branch).

Two layers over the torch capture path, both active ONLY when the session
trace carries ``structure_only=True`` (the default path is zero-diff):

LAYER 1 — MODE BELT: the escalated escape-belt configuration. Its surface is
:data:`STRUCTURE_ONLY_ESCAPE_SURFACE`, DEFINED AS THE UNION of the five
existing closed escape frozensets in ``completeness_witness`` (a union
expression, never a re-typed literal, so growth in any constituent is
inherited and one-sided drift is impossible; the belt-census meta-test pins
it against all five). Each patched site RAISES the teaching refusal
DEVICE-NEUTRALLY — meta or real tensor, both admission forms — whenever the
CALLING frame is user code: under structure-only the recorded graph must
never be VALUE-SELECTED, so a real input driving ``if x.sum():`` refuses
exactly like a meta one. Scoping is PROVENANCE, not device: torchlens
internals legitimately touch real-tensor storage in form (b)
(aliasing/dedup/hashing), so internal-frame calls retain today's behavior.
The plain belt's dtype-is-bool gate is LIFTED here by construction (the
escalated wrapper raises before recording), while the default path keeps its
documented false negative untouched.

LAYER 2 — FORWARD-BOUNDARY BACKSTOP: wraps the user forward invocation and
classifies escaping exceptions by raising-frame PROVENANCE (never by message
text as the decision): a non-user-raised ``NotImplementedError`` from the op
machinery is re-raised typed as ``meta_kernel_unavailable`` (original chained
via ``raise ... from``); a non-user-raised ``RuntimeError`` whose innermost
frame is an external CALL site (a C-level death of a direct call from user
code — the unenumerated-escape shape, e.g. ``tobytes`` on meta) is re-raised
typed as ``value_dependent_branch_unsupported`` with
``consumer_kind="unclassified_escape"``; exceptions raised BY user code
propagate UNCHANGED with the guarded ``add_note`` annotation (C-HONESTY —
user errors stay user errors; on 3.10 the degradation is a RuntimeWarning).
Unclassifiable exceptions propagate annotated: the fail direction is "don't
claim what we can't classify", never "wrap everything".

NAMING: all spellings DOCUMENTED-UNSTABLE pending naming-session/S2
ratification.
"""

from __future__ import annotations

import dis
import functools
import inspect
import threading
import types
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

import torch

from ... import _state
from ...errors._base import TorchLensError
from ...utils._torch_symbols import torch_attr
from ._tl import DescriptorCompatProperty, get_tensor_label
from .completeness_witness import (
    _TORCHLENS_ROOT,
    HOST_VALUE_ESCAPE_METHODS,
    HOST_VALUE_ESCAPE_MODULE_FUNCS,
    INVISIBLE_HOST_ESCAPE_FUNCS,
    INVISIBLE_HOST_ESCAPE_PROPERTIES,
    STORAGE_BRIDGE_ESCAPE_FUNCS,
    _internal_read_active,
)

_TORCH_ROOT: Final[Path] = Path(torch.__file__).resolve().parent

# The belt surface IS the union of the five existing closed sets (memo sec
# 2.2; opus r2 M2): computed from the constituents at import so a new member
# in any constituent set is inherited automatically. Pinned against all five
# by the belt-census meta-test.
STRUCTURE_ONLY_ESCAPE_SURFACE: Final[frozenset[str]] = (
    HOST_VALUE_ESCAPE_METHODS
    | HOST_VALUE_ESCAPE_MODULE_FUNCS
    | INVISIBLE_HOST_ESCAPE_FUNCS
    | STORAGE_BRIDGE_ESCAPE_FUNCS
    | INVISIBLE_HOST_ESCAPE_PROPERTIES
)

#: The Tensor-METHOD slice of the surface (module funcs and the property
#: install through their own mechanisms below).
_TENSOR_METHOD_SURFACE: Final[frozenset[str]] = (
    HOST_VALUE_ESCAPE_METHODS | INVISIBLE_HOST_ESCAPE_FUNCS | STORAGE_BRIDGE_ESCAPE_FUNCS
)


class _StructureOnlyBeltState:
    """Per-capture belt state (owner thread + trace identity)."""

    __slots__ = ("trace", "owner_thread_id")

    def __init__(self, trace: Any) -> None:
        self.trace = trace
        self.owner_thread_id = threading.get_ident()


_BELT_MODULE_FILE: Final[str] = str(Path(__file__).resolve())


def _first_non_torchlens_caller() -> tuple[str, int, bool] | None:
    """Return ``(file, line, is_user)`` for the escape's IMMEDIATE caller.

    Skips only this belt module's own frames (the wrapper plus its helpers)
    and classifies the very next frame: a torchlens-internal or torch-internal
    caller retains today's behavior (``is_user`` False) — torchlens internals
    legitimately touch real-tensor storage in form (b) and torch's own
    formatting reads scalars — while a user-code caller refuses (memo sec 2.2:
    the scoping is PROVENANCE, not device). Classifying the immediate caller,
    not "the first user frame anywhere below", is load-bearing: every capture
    ultimately has user frames below it.
    """

    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame is not None else None
        while (
            frame is not None and str(Path(frame.f_code.co_filename).resolve()) == _BELT_MODULE_FILE
        ):
            frame = frame.f_back
        if frame is None:
            return None
        filename = Path(frame.f_code.co_filename).resolve()
        try:
            filename.relative_to(_TORCHLENS_ROOT)
        except ValueError:
            pass
        else:
            return str(filename), frame.f_lineno, False
        try:
            filename.relative_to(_TORCH_ROOT)
        except ValueError:
            return str(filename), frame.f_lineno, True
        return str(filename), frame.f_lineno, False
    finally:
        del frame


def _belt_should_consider(state: _StructureOnlyBeltState) -> bool:
    """Cheap hot-path guard shared by every patched site."""

    active_trace, logging_enabled = _state.active_capture()
    return (
        logging_enabled
        and active_trace is state.trace
        and threading.get_ident() == state.owner_thread_id
        and not _internal_read_active()
    )


def _classify_consumer_kind(filename: str, line: int) -> str:
    """Raise-time branch-kind classification (memo sec 2.3).

    Uses the standalone ``ast_branches.classify_bool`` lookup — no
    postprocess dependency; its ``"unknown"`` maps to ``"scalar_escape"`` (an
    escape the classifier cannot tie to a branch context is by definition a
    bare escape). The vocabulary of record is ``BoolConsumer.kind``.
    """

    try:
        from ...postprocess.ast_branches import classify_bool

        kind = classify_bool(filename, line).kind
    except Exception:  # noqa: BLE001 — classification must never mask the refusal
        return "scalar_escape"
    return "scalar_escape" if kind == "unknown" else kind


def _tensor_substrate(tensor: Any) -> str:
    """Return ``"meta"`` or ``"real"`` (guarded: hostile subclasses)."""

    try:
        return "meta" if bool(tensor.is_meta) else "real"
    except Exception:  # noqa: BLE001 — substrate is diagnostic, never authority
        return "real"


def _raise_teaching_refusal(
    *,
    escape_method: str,
    tensor: Any,
    tensor_label: str,
    filename: str,
    line: int,
) -> None:
    """Raise the sec-2.3 typed teaching refusal for one user-frame escape."""

    from ...capture.structure_only import ValueDependentBranchError

    consumer_kind = _classify_consumer_kind(filename, line)
    substrate = _tensor_substrate(tensor)
    if substrate == "meta":
        substrate_note = "[meta: this value does not exist]"
    else:
        substrate_note = (
            "[real: consuming this value would select the graph by values "
            "the capture does not record]"
        )
    offense = {
        "file": filename,
        "line": line,
        "consumer_kind": consumer_kind,
        "tensor_label": tensor_label,
        "escape_method": escape_method,
        "substrate": substrate,
    }
    raise ValueDependentBranchError(
        "Structure-only capture refused: your model consumed a tensor VALUE. "
        "Structure-only capture must never record a value-selected graph.\n"
        f"  - {filename}:{line} ({consumer_kind}): tensor {tensor_label} via "
        f"{escape_method} {substrate_note}\n"
        "A value branch would make every claim downstream of this line a "
        "guess about WHICH graph exists. Remedy: this branch needs a real "
        "value — either run a real capture (tl.trace without structure_only) "
        "to resolve it, or restructure the branch to be shape-derived. "
        "Structure-only capture can only follow value-free control flow. See "
        "the 'Structure-only capture' section of docs/reference/limitations.md.",
        code="value_dependent_branch_unsupported",
        file_path=filename,
        line_no=line,
        consumer_kind=consumer_kind,
        offenses=(offense,),
        remedy=(
            "run a real capture (tl.trace without structure_only) or "
            "restructure the branch to be shape-derived"
        ),
    )


def _maybe_refuse_escape(state: _StructureOnlyBeltState, escape_method: str, tensor: Any) -> None:
    """Refuse one escape when it is a labeled-tensor read from a USER frame."""

    if not _belt_should_consider(state):
        return
    label = get_tensor_label(tensor)
    if label is None:
        return
    caller = _first_non_torchlens_caller()
    if caller is None:
        return
    filename, line, is_user = caller
    if not is_user:
        return
    _raise_teaching_refusal(
        escape_method=escape_method,
        tensor=tensor,
        tensor_label=label,
        filename=filename,
        line=line,
    )


def _make_escalated_method(original: Any, state: _StructureOnlyBeltState, name: str) -> Any:
    """Wrap one Tensor method with the escalated (raising) belt."""

    @functools.wraps(original)
    def wrapper(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        """Refuse on hypothesis tensors, then defer to the original method."""

        _maybe_refuse_escape(state, name, self)
        return original(self, *args, **kwargs)

    return wrapper


def _make_escalated_module_func(original: Any, state: _StructureOnlyBeltState, name: str) -> Any:
    """Wrap one ``torch.*`` module predicate with the escalated belt."""

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Refuse on hypothesis tensor operands, then defer to the original."""

        for operand in args:
            if isinstance(operand, torch.Tensor):
                _maybe_refuse_escape(state, f"torch.{name}", operand)
        return original(*args, **kwargs)

    return wrapper


def _make_escalated_property(descriptor: Any, state: _StructureOnlyBeltState, name: str) -> Any:
    """Wrap one getset-descriptor property with the escalated belt."""

    def getter(self: torch.Tensor) -> Any:
        """Refuse on hypothesis tensors, then read the original descriptor."""

        _maybe_refuse_escape(state, name, self)
        return descriptor.__get__(self, type(self))

    replacement = DescriptorCompatProperty(getter)
    replacement.__objclass__ = getattr(descriptor, "__objclass__", torch.Tensor)
    return replacement


def _install_method_belt(
    state: _StructureOnlyBeltState, method_restores: dict[str, tuple[bool, Any]]
) -> None:
    """Wrap the Tensor method escape surface, recording shadow-aware restores."""

    for name in sorted(_TENSOR_METHOD_SURFACE):
        original = getattr(torch.Tensor, name, None)
        if original is None or not callable(original):
            continue
        shadowed = name in torch.Tensor.__dict__
        try:
            setattr(torch.Tensor, name, _make_escalated_method(original, state, name))
        except (TypeError, AttributeError):
            continue
        method_restores[name] = (shadowed, original)


def _install_module_func_belt(
    state: _StructureOnlyBeltState, module_restores: list[tuple[Any, str, Any]]
) -> None:
    """Wrap the ``torch.*`` module predicate escape surface."""

    for name in sorted(HOST_VALUE_ESCAPE_MODULE_FUNCS):
        original = torch_attr(name)
        if original is None or not callable(original):
            continue
        try:
            setattr(torch, name, _make_escalated_module_func(original, state, name))
        except (TypeError, AttributeError):
            continue
        module_restores.append((torch, name, original))


def _install_property_belt(
    state: _StructureOnlyBeltState, property_restores: dict[str, tuple[bool, Any]]
) -> None:
    """Wrap the invisible getset-descriptor escape surface."""

    for name in sorted(INVISIBLE_HOST_ESCAPE_PROPERTIES):
        descriptor = inspect.getattr_static(torch.Tensor, name, None)
        if descriptor is None or not hasattr(descriptor, "__get__"):
            continue
        shadowed = name in torch.Tensor.__dict__
        try:
            setattr(torch.Tensor, name, _make_escalated_property(descriptor, state, name))
        except (TypeError, AttributeError):
            continue
        property_restores[name] = (shadowed, descriptor)


@contextmanager
def structure_only_escape_belt(trace: Any) -> Iterator[None]:
    """LAYER 1: install the escalated escape belt for one capture.

    No-op unless ``trace.structure_only`` (default path zero-diff). Install
    and unwind are BaseException-safe with shadow-aware restore (the R07
    unwind standard): a name set on ``torch.Tensor`` that shadowed a C-level
    slot member must be DELETED on restore, never written back.
    """

    if not bool(getattr(trace, "structure_only", False)):
        yield
        return

    state = _StructureOnlyBeltState(trace)
    method_restores: dict[str, tuple[bool, Any]] = {}
    module_restores: list[tuple[Any, str, Any]] = []
    property_restores: dict[str, tuple[bool, Any]] = {}

    def _restore() -> None:
        """Unwind every belt patch, shadow-aware (delete unshadowed names)."""

        for name, (shadowed, original) in method_restores.items():
            if shadowed:
                setattr(torch.Tensor, name, original)
            else:
                delattr(torch.Tensor, name)
        for module, name, original in module_restores:
            setattr(module, name, original)
        for name, (shadowed, descriptor) in property_restores.items():
            if shadowed:
                setattr(torch.Tensor, name, descriptor)
            else:
                delattr(torch.Tensor, name)

    try:
        _install_method_belt(state, method_restores)
        _install_module_func_belt(state, module_restores)
        _install_property_belt(state, property_restores)
    except BaseException:
        _restore()
        raise

    try:
        yield
    finally:
        _restore()


# ---------------------------------------------------------------------------
# LAYER 2 — forward-boundary backstop
# ---------------------------------------------------------------------------


def _innermost_traceback_frame(exc: BaseException) -> types.TracebackType | None:
    """Return the LAST traceback entry (the raising frame's tb node)."""

    tb = exc.__traceback__
    if tb is None:
        return None
    while tb.tb_next is not None:
        tb = tb.tb_next
    return tb


def _frame_location(tb: types.TracebackType) -> tuple[Path, int]:
    """Return the resolved (file, line) of one traceback frame."""

    return Path(tb.tb_frame.f_code.co_filename).resolve(), tb.tb_lineno


def _is_external(path: Path) -> bool:
    """Whether a frame path is outside both torchlens and torch."""

    for root in (_TORCHLENS_ROOT, _TORCH_ROOT):
        try:
            path.relative_to(root)
        except ValueError:
            continue
        return False
    return True


def _external_frames(exc: BaseException) -> list[tuple[Path, int]]:
    """External (non-torchlens/non-torch) frames in tb_next order."""

    frames: list[tuple[Path, int]] = []
    tb = exc.__traceback__
    while tb is not None:
        path, line = _frame_location(tb)
        if _is_external(path):
            frames.append((path, line))
        tb = tb.tb_next
    return frames


def _raised_by_raise_statement(tb: types.TracebackType) -> bool:
    """Whether the frame's active instruction is a RAISE (user ``raise``),
    as opposed to a call whose C-level callee raised. Structural, never
    message-based."""

    try:
        for instruction in dis.get_instructions(tb.tb_frame.f_code):
            if instruction.offset == tb.tb_lasti:
                return instruction.opname.startswith("RAISE") or (instruction.opname == "RERAISE")
    except Exception:  # noqa: BLE001 — classification must never mask the exception
        return False
    return False


def _annotate_guarded(exc: BaseException, note: str) -> None:
    """The guarded PEP-678 house idiom (3.10 degrades to a RuntimeWarning)."""

    add_note = getattr(exc, "add_note", None)
    if add_note is not None:
        add_note(note)
    else:  # Python 3.10: no PEP 678 notes — surface via warning.
        warnings.warn(note, RuntimeWarning, stacklevel=3)


_USER_ANNOTATION: Final[str] = (
    "torchlens: raised during structure-only capture; tensors here may be meta (no values)"
)


def _wrapper_op_name(exc: BaseException) -> str | None:
    """Best-effort op name from the innermost torchlens wrapper frame locals.

    Annotation only — never part of the classification decision."""

    tb = exc.__traceback__
    candidate: str | None = None
    while tb is not None:
        path, _ = _frame_location(tb)
        try:
            path.relative_to(_TORCHLENS_ROOT)
        except ValueError:
            tb = tb.tb_next
            continue
        for variable in ("func", "orig_func", "original", "callable_obj"):
            value = tb.tb_frame.f_locals.get(variable)
            name = getattr(value, "__name__", None)
            if isinstance(name, str):
                candidate = name
        tb = tb.tb_next
    return candidate


def _substrate_mismatch_kind(exc: BaseException) -> str | None:
    """Classify a RuntimeError as a substrate mismatch, or ``None`` (W1-CLS).

    Two provenance-based mechanisms, never message text: (a) the innermost
    TorchLens wrapper frame's failing call OBSERVABLY mixed meta and real
    tensor operands; (b) the raising frame sits inside ``torch/amp`` (the
    autocast device-validation family — measured: torch validates the device
    string even for ``enabled=True`` on meta and dies there).
    """

    tb = exc.__traceback__
    amp_root = _TORCH_ROOT / "amp"
    saw_meta = False
    saw_real = False
    while tb is not None:
        path, _ = _frame_location(tb)
        try:
            path.relative_to(amp_root)
            return "torch.amp device validation (autocast family)"
        except ValueError:
            pass
        try:
            path.relative_to(_TORCHLENS_ROOT)
        except ValueError:
            tb = tb.tb_next
            continue
        frame_meta, frame_real = _frame_operand_substrates(tb.tb_frame)
        saw_meta = saw_meta or frame_meta
        saw_real = saw_real or frame_real
        tb = tb.tb_next
    if saw_meta and saw_real:
        return "mixed meta/real operands in the failing wrapped call"
    return None


def _frame_operand_substrates(frame: types.FrameType) -> tuple[bool, bool]:
    """Whether one TorchLens wrapper frame holds (meta, real) tensor operands."""

    saw_meta = False
    saw_real = False
    for variable in ("args", "call_args", "kwargs", "call_kwargs"):
        value = frame.f_locals.get(variable)
        if isinstance(value, dict):
            values: tuple[Any, ...] = tuple(value.values())
        elif isinstance(value, (list, tuple)):
            values = tuple(value)
        else:
            continue
        for item in values:
            if isinstance(item, torch.Tensor):
                if item.is_meta:
                    saw_meta = True
                else:
                    saw_real = True
    return saw_meta, saw_real


@contextmanager
def structure_only_forward_boundary(trace: Any) -> Iterator[None]:
    """LAYER 2: classify exceptions escaping the user forward (memo 2.2/2.4).

    No-op unless ``trace.structure_only``. TorchLens-typed exceptions
    (including Layer-1 refusals and halt/stop signals) always pass through
    byte-untouched.
    """

    if not bool(getattr(trace, "structure_only", False)):
        yield
        return
    try:
        yield
    except TorchLensError:
        raise
    except Exception as exc:
        innermost = _innermost_traceback_frame(exc)
        if innermost is None:
            raise
        innermost_path, _ = _frame_location(innermost)
        innermost_external = _is_external(innermost_path)
        user_raised = innermost_external and _raised_by_raise_statement(innermost)
        if user_raised:
            _annotate_guarded(exc, _USER_ANNOTATION)
            raise
        externals = _external_frames(exc)
        # The INNERMOST external frame is the failing callsite (a traceback
        # enumerates OUTER frames first, so "last in tb order" is the frame
        # adjacent to the transition into torch — sol r2 M2); the outermost
        # is recorded for orientation only.
        failing_file, failing_line = (
            (str(externals[-1][0]), externals[-1][1]) if externals else (None, None)
        )
        entry_frame = f"{externals[0][0]}:{externals[0][1]}" if externals else None
        if isinstance(exc, NotImplementedError):
            from ...capture.structure_only import MetaKernelUnavailableError

            op_name = _wrapper_op_name(exc)
            location = f"{failing_file}:{failing_line}" if failing_file else "<unknown>"
            raise MetaKernelUnavailableError(
                "Structure-only capture refused: an operation has no meta "
                f"kernel (op {op_name or 'unknown'}, invoked at {location}). "
                "The meta substrate can only propagate shapes through ops "
                "torch implements for the meta device. Remedy: run a real "
                "capture (tl.trace without structure_only), or upgrade torch "
                "for broader meta-kernel coverage. Original error annotates, "
                f"never decides: {exc}",
                code="meta_kernel_unavailable",
                file_path=failing_file,
                line_no=failing_line,
                op_name=op_name,
                entry_frame=entry_frame,
                remedy=(
                    "run a real capture (tl.trace without structure_only) or "
                    "upgrade torch for broader meta-kernel coverage"
                ),
            ) from exc
        mismatch_kind = (
            _substrate_mismatch_kind(exc)
            if isinstance(exc, RuntimeError) and not user_raised
            else None
        )
        if mismatch_kind is not None:
            # W1-CLS (weightsfree memo D11): device-mismatch RuntimeErrors and
            # the autocast/unsupported-scalarType family classify into the
            # SUBSTRATE family BEFORE the value-escape fallback — both were
            # measured to mis-teach the user to hunt a value branch that does
            # not exist. Classification is by observed wrapper operands
            # (mixed meta/real tensors in the failing wrapped call) or
            # raising-frame provenance (torch/amp), never message text.
            from ..._errors import SubstrateMismatchError

            kind = mismatch_kind
            location = f"{failing_file}:{failing_line}" if failing_file else "<unknown>"
            raise SubstrateMismatchError(
                "Structure-only capture refused: a REAL tensor met the meta "
                f"substrate mid-forward at {location} ({kind}). A stale "
                "pre-wrap factory reference or a device='cpu' literal mints "
                "real tensors inside an admitted weights-free forward; the "
                "recorded graph would be neither the real model's nor a "
                "coherent hypothesis. Remedy: pass device= through the "
                "module's factory kwargs (or drop the pinned device so the "
                "owned factory scope places it), and construct every tensor "
                "the forward mints on the model's substrate. Original error "
                f"annotates, never decides: {exc}",
                code="structure_only_substrate_mismatch",
                file_path=failing_file,
                line_no=failing_line,
                mismatch_kind=kind,
                entry_frame=entry_frame,
            ) from exc
        if isinstance(exc, RuntimeError) and not user_raised:
            # Memo 2.2 Layer 2: a RuntimeError that was NOT raised by a user
            # `raise` statement — either the raising frame sits inside torch
            # (meta registrations raise RuntimeError from torch Python
            # frames: "Tensor.item() cannot be called on meta tensors") or a
            # C-level callee died at an external CALL site (the
            # unenumerated-escape shape, e.g. tensor.tobytes() on meta). The
            # exception family + provenance decide; message text never does.
            from ...capture.structure_only import ValueDependentBranchError

            location = f"{failing_file}:{failing_line}" if failing_file else "<unknown>"
            raise ValueDependentBranchError(
                "Structure-only capture refused: a value-consuming call died "
                f"at {location} (unenumerated escape; typed via the "
                "forward-boundary backstop, without branch classification). "
                f"Original error annotates, never decides: {exc}",
                code="value_dependent_branch_unsupported",
                file_path=failing_file,
                line_no=failing_line,
                consumer_kind="unclassified_escape",
                offenses=(
                    {
                        "file": failing_file,
                        "line": failing_line,
                        "consumer_kind": "unclassified_escape",
                        "tensor_label": None,
                        "escape_method": None,
                        "substrate": "meta",
                    },
                ),
                entry_frame=entry_frame,
                remedy=(
                    "run a real capture (tl.trace without structure_only) to consume this value"
                ),
            ) from exc
        # Unclassifiable: propagate unchanged with the guarded annotation.
        _annotate_guarded(exc, _USER_ANNOTATION)
        raise
