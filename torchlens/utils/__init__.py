"""Focused utility modules: RNG, tensor ops, argument handling, introspection, collections, hashing, display."""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import re
import subprocess
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import torch
from packaging.requirements import Requirement
from torch import nn

from .. import _state

if TYPE_CHECKING:
    from ..capture.flops import register_op_rule
    from .arg_handling import (
        _model_expects_single_arg,
        _safe_copy_arg,
        copy_arg_tree,
        normalize_input_args,
        safe_copy_args,
        safe_copy_kwargs,
    )
    from .collections import (
        assign_to_sequence_or_dict,
        ensure_iterable,
        index_nested,
        is_iterable,
        remove_entry_from_list,
    )
    from .display import (
        format_flops,
        format_size,
        human_readable_size,
        identity,
        in_notebook,
        int_list_to_compact_str,
        progress_bar,
        tensor_stats_summary,
        warn_parallel,
    )
    from .hashing import make_random_barcode, make_short_barcode_from_input
    from .introspection import (
        _ATTR_SKIP_SET,
        _get_code_context,
        get_attr_values_from_tensor_list,
        get_vars_of_type_from_obj,
        iter_accessible_attributes,
        nested_assign,
        nested_getattr,
        remove_attributes_with_prefix,
    )
    from .rng import (
        _AUTOCAST_DEVICES,
        AutocastRestore,
        log_current_autocast_state,
        log_current_rng_states,
        set_random_seed,
        set_rng_from_saved_states,
    )
    from .tensor_utils import (
        MAX_FLOATING_POINT_TOLERANCE,
        _cuda_available,
        _is_cuda_available,
        copy_tensor_payload,
        get_memory_amount,
        print_override,
        safe_copy,
        safe_to,
        tensor_all_nan,
        tensor_nanequal,
    )


@dataclass(frozen=True)
class DoctorCheck:
    """One TorchLens environment health-check row.

    Parameters
    ----------
    name:
        Human-readable check name.
    status:
        ``"PASS"``, ``"FAIL"``, ``"SKIP"``, or ``"WARN"``.
    detail:
        Short diagnostic detail.
    """

    name: str
    status: Literal["PASS", "FAIL", "SKIP", "WARN"]
    detail: str


@dataclass(frozen=True)
class DoctorReport:
    """Structured report returned by :func:`doctor`.

    Parameters
    ----------
    checks:
        Ordered health-check rows.
    """

    checks: tuple[DoctorCheck, ...]

    def show(self) -> str:
        """Render the doctor report as a text table.

        Returns
        -------
        str
            Human-readable report.
        """

        rows = ["TorchLens doctor report:"]
        for check in self.checks:
            rows.append(f"- {check.status:<4} {check.name}: {check.detail}")
        return "\n".join(rows)

    def __str__(self) -> str:
        """Return the rendered report.

        Returns
        -------
        str
            Human-readable report.
        """

        return self.show()

    def __repr__(self) -> str:
        """Return the designed bounded render, never a dataclass wall.

        The auto-generated dataclass repr was a 3,000+-character one-line
        wall (sumfam wave-0 item 2). Every emitted line is width-capped with
        a disclosed truncation; the full row text stays on ``.checks``.

        Returns
        -------
        str
            Bounded multi-line report.
        """

        lines = []
        for line in self.show().splitlines():
            if len(line) > _REPORT_REPR_LINE_CAP:
                line = line[: _REPORT_REPR_LINE_CAP - 3] + "..."
            lines.append(line)
        lines.append("(full row text: report.checks; capability dump: capability_snapshot())")
        return "\n".join(lines)

    def capability_snapshot(self) -> dict[str, bool]:
        """Return the full runtime capability snapshot (the detail accessor).

        The doctor row shows the grouped absences-first summary; this is the
        complete ``flag -> available`` mapping the summary was computed from.

        Returns
        -------
        dict[str, bool]
            Capability flag availability.
        """

        return _runtime_capability_snapshot()


#: Width cap for report ``__repr__`` lines (bounded-reprs doctrine).
_REPORT_REPR_LINE_CAP = 200


def capability_snapshot() -> dict[str, bool]:
    """Return the full runtime capability snapshot (module-level accessor).

    Returns
    -------
    dict[str, bool]
        Capability flag availability, torch plus optional preview backends.
    """

    return _runtime_capability_snapshot()


def format_capability_summary(snapshot: dict[str, bool]) -> str:
    """Return the grouped absences-first summary of a capability snapshot.

    ONE shared renderer (sumfam wave-0 item 2): the full ``name=value`` dump
    used to sit in a report data cell, so every renderer inherited a
    ~1,900-character line. Absences are the signal and are named first; the
    full dump moves to the detail accessors.

    Parameters
    ----------
    snapshot:
        Capability flag availability.

    Returns
    -------
    str
        ``present/total capabilities present`` plus the sorted absent names.
    """

    absent = sorted(name for name, available in snapshot.items() if not available)
    present = len(snapshot) - len(absent)
    summary = f"{present}/{len(snapshot)} capabilities present"
    if absent:
        summary += "; absent: " + ", ".join(absent)
    return summary


# Extras the user-facing doctor never probes: the rollups and the contributor/test
# extras. deploy / tlens-oracle / trackers-relay-test are TEST-FACING (pyproject: the
# skip-audit ledger's optional-preview rows, the TransformerLens oracle rows, the
# tracker relay-fidelity pins) and never advertised to users; probing them also
# imports peft/transformer_lens/wandb/clearml for real (the probe IS an import),
# which pushed the doctor past the MCP doctor tool's 12 s budget (D05, 2026-09-02).
_DOCTOR_EXCLUDED_EXTRAS = frozenset(
    {"all", "all-stretch", "dev", "test", "deploy", "tlens-oracle", "trackers-relay-test"}
)
_EXTRA_MARKER_RE = re.compile(r"""extra\s*==\s*['"](?P<extra>[^'"]+)['"]""")
_REQUIREMENT_IMPORT_NAME_OVERRIDES: dict[str, tuple[str, ...]] = {
    "brain-score": ("brainscore_core",),
    "jupyter-client": ("jupyter_client",),
    "lit-nlp": ("lit_nlp",),
    "lovely-tensors": ("lovely_tensors",),
    "paddlepaddle": ("paddle",),
    "pytorch-grad-cam": ("pytorch_grad_cam",),
    "sae-lens": ("sae_lens",),
    "sentence-transformers": ("sentence_transformers",),
    "steering-vectors": ("steering_vectors",),
}


def _extras_from_requirement_marker(requirement: Requirement) -> tuple[str, ...]:
    """Return every extra referenced by a requirement marker.

    Parameters
    ----------
    requirement:
        Parsed requirement line from package metadata.

    Returns
    -------
    tuple[str, ...]
        Extras referenced in the requirement marker, preserving first-seen
        order.
    """

    marker = requirement.marker
    if marker is None:
        return ()
    extras = [
        match.group("extra")
        for match in _EXTRA_MARKER_RE.finditer(str(marker))
        if match.group("extra") not in _DOCTOR_EXCLUDED_EXTRAS
    ]
    return tuple(dict.fromkeys(extras))


def _probe_modules_for_requirement(requirement: Requirement) -> tuple[str, ...]:
    """Return import-module probes for one optional requirement.

    Parameters
    ----------
    requirement:
        Parsed requirement line from package metadata.

    Returns
    -------
    tuple[str, ...]
        Module names whose importability best approximates whether the
        requirement is installed for diagnostic reporting.
    """

    normalized_name = requirement.name.lower()
    override = _REQUIREMENT_IMPORT_NAME_OVERRIDES.get(normalized_name)
    if override is not None:
        return override
    return (requirement.name.replace("-", "_"),)


def _declared_extra_probes() -> dict[str, tuple[str, ...]]:
    """Return doctor extra probes derived from installed package metadata.

    Returns
    -------
    dict[str, tuple[str, ...]]
        Optional extra names mapped to representative import-module probes.

    Raises
    ------
    importlib.metadata.PackageNotFoundError
        If the installed ``torchlens`` distribution metadata is unavailable.
    """

    distribution = importlib.metadata.distribution("torchlens")
    extra_names = sorted(
        extra
        for extra in (distribution.metadata.get_all("Provides-Extra") or [])
        if extra not in _DOCTOR_EXCLUDED_EXTRAS
    )
    probes: dict[str, list[str]] = {extra: [] for extra in extra_names}
    for requirement_line in distribution.requires or ():
        requirement = Requirement(requirement_line)
        for extra in _extras_from_requirement_marker(requirement):
            if extra not in probes:
                continue
            probes[extra].extend(_probe_modules_for_requirement(requirement))
    return {extra: tuple(dict.fromkeys(module_names)) for extra, module_names in probes.items()}


def _module_is_installed(module_name: str) -> bool:
    """Return whether a module can be imported.

    Parameters
    ----------
    module_name:
        Importable module name.

    Returns
    -------
    bool
        ``True`` when import succeeds.
    """

    try:
        importlib.import_module(module_name)
    except Exception:
        return False
    return True


def _probe_graphviz() -> DoctorCheck:
    """Probe Python and system Graphviz availability.

    Returns
    -------
    DoctorCheck
        Graphviz health-check row.
    """

    python_graphviz = _module_is_installed("graphviz")
    try:
        # R40: routed through the ONE bounded spawn seam. subprocess.run's
        # timeout killed only the direct child, so a wedged ``dot`` wrapper's
        # grandchild survived the "bounded" probe for the life of the box;
        # the shared runner tears down the whole process group.
        from ._subprocess import run_bounded_subprocess

        completed = run_bounded_subprocess(
            ["dot", "-V"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return DoctorCheck(
            "graphviz",
            "FAIL",
            f"python graphviz={python_graphviz}; dot unavailable ({exc})",
        )
    version_text = (completed.stderr or completed.stdout).strip()
    status: Literal["PASS", "FAIL"] = (
        "PASS" if python_graphviz and completed.returncode == 0 else "FAIL"
    )
    return DoctorCheck(
        "graphviz",
        status,
        f"python graphviz={python_graphviz}; dot={version_text or completed.returncode}",
    )


def _probe_extras() -> DoctorCheck:
    """Probe declared optional extras by importing their representative modules.

    Returns
    -------
    DoctorCheck
        Optional-extras health-check row.
    """

    try:
        extra_probes = _declared_extra_probes()
    except importlib.metadata.PackageNotFoundError as exc:
        return DoctorCheck("extras", "FAIL", f"package metadata unavailable ({exc})")

    installed = []
    missing = []
    no_python_probes = []
    for extra, modules in extra_probes.items():
        if not modules:
            no_python_probes.append(extra)
            continue
        if all(_module_is_installed(module_name) for module_name in modules):
            installed.append(extra)
        else:
            missing.append(extra)
    detail = (
        f"installed={installed or 'none'}; missing={missing or 'none'}; "
        f"no_python_probes={no_python_probes or 'none'}"
    )
    return DoctorCheck("extras", "PASS", detail)


def _probe_fingerprint() -> DoctorCheck:
    """Probe model weight fingerprinting on a tiny module.

    Returns
    -------
    DoctorCheck
        Fingerprint utility health-check row.
    """

    try:
        from torchlens.user_funcs import _fingerprint_model_weights

        model = nn.Linear(2, 1)
        fingerprint = _fingerprint_model_weights(model)
    except Exception as exc:
        return DoctorCheck("model fingerprint", "FAIL", repr(exc))
    status: Literal["PASS", "FAIL"] = "PASS" if fingerprint else "FAIL"
    return DoctorCheck("model fingerprint", status, fingerprint[:16] if fingerprint else "empty")


def _probe_torch_capabilities() -> DoctorCheck:
    """Return the runtime capability snapshot doctor row.

    Returns
    -------
    DoctorCheck
        Snapshot of probed private integration capabilities.
    """

    from . import _torch_compat as _tc
    from ._torch_compat import OPTIONAL_CAPABILITY_FLAGS

    snapshot = _runtime_capability_snapshot()
    absent = [name for name, available in snapshot.items() if not available]

    def _probe_latched(name: str) -> bool:
        """Return whether a flag's lazy probe has actually run.

        Lazily probed flags (kineto event fields, memory profile, ...) default
        ``False`` with a ``_<NAME>_PROBED`` latch flipped at first use; a flag
        with no latch is probed eagerly at import and always reads as latched.
        """

        latch = "_" + name.removeprefix("HAS_") + "_PROBED"
        return bool(getattr(_tc, latch, True))

    # r-b4 R26-4: only genuine DEGRADATIONS drive WARN. An absent optional
    # feature (interpreter-version surface, upstream-removed API, an optional
    # backend that is not installed) is reported with its true value but keeps
    # a healthy install at PASS -- a permanent false alarm trains users to
    # ignore the row. A lazily probed flag whose probe has never run is
    # UNPROBED, not missing: nothing has consumed the capability yet, so
    # nothing has degraded (D04 integration fix; the doctor previously read
    # every unprobed-lazy default-False flag as a degradation WARN).
    missing = [
        name for name in absent if name not in OPTIONAL_CAPABILITY_FLAGS and _probe_latched(name)
    ]
    unprobed = [
        name
        for name in absent
        if name not in OPTIONAL_CAPABILITY_FLAGS and not _probe_latched(name)
    ]
    optional_absent = [name for name in absent if name in OPTIONAL_CAPABILITY_FLAGS]
    # Grouped absences-first summary; the full name=value dump lives on the
    # detail accessor (tl.utils.capability_snapshot()), not in the row cell.
    detail = format_capability_summary(snapshot)
    if missing:
        detail += "; missing=" + ",".join(missing)
    if unprobed:
        detail += "; unprobed=" + ",".join(unprobed)
    if optional_absent:
        detail += "; optional_absent=" + ",".join(optional_absent)
    detail += "; full dump: tl.utils.capability_snapshot()"
    # Report the true state: a missing private-integration capability is a
    # degraded (WARN) row, not a "PASS". These flags are feature-detected and may
    # be legitimately absent across torch versions, so WARN (not FAIL) is honest.
    status: Literal["PASS", "WARN"] = "PASS" if not missing else "WARN"
    return DoctorCheck("runtime capabilities", status, detail)


def _probe_torch_wrapper_bindings() -> DoctorCheck:
    """Check torch namespace attributes against the installed wrapper registry.

    Returns
    -------
    DoctorCheck
        Warning-only detector for stale torch module attributes. This cannot
        inspect arbitrary local aliases such as closure-bound ``from torch
        import relu`` references, but it catches the cheap process-global case
        where torch itself is no longer pointing at registered wrappers.
    """

    from .introspection import nested_getattr

    if not _state._orig_to_decorated:
        return DoctorCheck("torch wrapper bindings", "SKIP", "wrappers not installed yet")
    if not _state._is_decorated:
        return DoctorCheck("torch wrapper bindings", "SKIP", "torch is currently unwrapped")

    from ..constants import get_orig_torch_funcs

    stale: list[str] = []
    checked = 0
    for namespace_name, func_name in get_orig_torch_funcs():
        namespace_key = namespace_name.replace("torch.", "")
        try:
            namespace = nested_getattr(torch, namespace_key)
        except (AttributeError, TypeError):
            continue
        if not hasattr(namespace, func_name):
            continue
        current = getattr(namespace, func_name)
        checked += 1
        if id(current) in _state._decorated_to_orig:
            continue
        if id(current) in _state._orig_to_decorated:
            stale.append(f"{namespace_name}.{func_name}")

    if stale:
        examples = ", ".join(stale[:5])
        return DoctorCheck(
            "torch wrapper bindings",
            "WARN",
            f"{len(stale)}/{checked} torch attrs still point at original callables; "
            f"examples={examples}",
        )
    return DoctorCheck(
        "torch wrapper bindings",
        "PASS",
        f"checked={checked}; no stale torch namespace bindings detected",
    )


def _probe_mechanical_belt() -> DoctorCheck:
    """Report protocol-invisible belt coverage gaps.

    Returns
    -------
    DoctorCheck
        Disclosure row for belt probe failures and unprobed candidates. A
        candidate whose mode visibility could not be MEASURED (probe raised,
        or no probe recipe exists) is neither belt-patched nor proven
        protocol-visible, so a stale pre-wrap reference to it can drop ops
        with zero signal while the capture still reports
        ``capture_verified=True`` (grind-r6 b3 R02, sol MED).
    """

    from ..backends.torch.belt import belt_report

    if not _state._is_decorated:
        return DoctorCheck(
            "mechanical belt",
            "SKIP",
            "belt not derived yet (torch wrapping is lazy; run a capture first)",
        )
    report = belt_report()
    if report is None:
        return DoctorCheck("mechanical belt", "SKIP", "belt derivation unavailable")
    # Unprobed candidates are a STANDING recipe-coverage limitation (hundreds
    # of in-place variants have no probe recipe on every healthy build), so
    # they are disclosed as a count with examples but never flip the status —
    # a permanent false alarm trains users to ignore the row (r-b4 R26-4).
    # A probe FAILURE is unexpected breakage on this build and drives WARN.
    unprobed_examples = ", ".join(f"{ns}.{fn}" for ns, fn in report.unprobed_candidates[:5])
    unprobed_detail = f"unprobed_candidates={report.unprobed_candidate_count}"
    if unprobed_examples:
        unprobed_detail += f" (e.g. {unprobed_examples})"
    if not report.probe_failures:
        return DoctorCheck(
            "mechanical belt",
            "PASS",
            f"members={len(report.members)}; probe_failures=none; {unprobed_detail}",
        )
    failure_names = ", ".join(f"{ns}.{fn}" for ns, fn in report.probe_failures)
    detail = (
        f"members={len(report.members)}; probe FAILURES (visibility unmeasured; a stale "
        f"pre-wrap reference to these can silently drop ops): {failure_names}; "
        f"{unprobed_detail}"
    )
    if report.probe_failure_details:
        detail += "; failure_details=" + "; ".join(
            f"{ns}.{fn}: {reason}" for ns, fn, reason in report.probe_failure_details
        )
    return DoctorCheck("mechanical belt", "WARN", detail)


def _runtime_capability_snapshot() -> dict[str, bool]:
    """Return all runtime compatibility capability flags.

    Returns
    -------
    dict[str, bool]
        Mapping from capability flag names to availability.
    """

    from ._torch_compat import get_torch_capability_snapshot

    snapshot = get_torch_capability_snapshot()
    try:
        from torchlens.backends.tf._tf_compat import get_tf_capability_snapshot
    except ImportError:
        return snapshot
    snapshot.update(get_tf_capability_snapshot())
    return snapshot


def doctor() -> DoctorReport:
    """Run a TorchLens startup health check.

    Returns
    -------
    DoctorReport
        Structured report with PyTorch, CUDA, Graphviz, safetensors, extras,
        and model-fingerprint checks.
    """

    checks: list[DoctorCheck] = [
        DoctorCheck("pytorch", "PASS", torch.__version__),
        _probe_torch_capabilities(),
        _probe_torch_wrapper_bindings(),
        _probe_mechanical_belt(),
        DoctorCheck(
            "cuda",
            "PASS" if torch.cuda.is_available() else "SKIP",
            f"available={torch.cuda.is_available()}; devices={torch.cuda.device_count()}",
        ),
        _probe_graphviz(),
        DoctorCheck(
            "safetensors",
            "PASS" if _module_is_installed("safetensors") else "FAIL",
            "installed" if _module_is_installed("safetensors") else "missing",
        ),
        _probe_extras(),
        _probe_fingerprint(),
    ]
    return DoctorReport(tuple(checks))


def list_modules(model: nn.Module) -> list[tuple[str, type[nn.Module]]]:
    """List every module registered on a model.

    Parameters
    ----------
    model:
        PyTorch module to inspect.

    Returns
    -------
    list[tuple[str, type[nn.Module]]]
        Module addresses paired with concrete module classes. The root module is
        reported as ``"self"``.
    """

    return [(address or "self", type(module)) for address, module in model.named_modules()]


def _ops_from_log(trace: Any) -> list[tuple[str, int]]:
    """Summarize operator counts from a Trace.

    Parameters
    ----------
    trace:
        TorchLens log produced by ``trace``.

    Returns
    -------
    list[tuple[str, int]]
        Operator names and counts sorted by first observed name.
    """

    counts: Counter[str] = Counter()
    for layer in trace.layer_list:
        op_name = getattr(layer, "func_name", None) or getattr(layer, "layer_type", "unknown")
        counts[str(op_name)] += 1
    return sorted(counts.items())


def _log_ops_for_mode(
    model: nn.Module,
    x: Any,
    mode: Literal["current", "eval", "train"],
) -> list[tuple[str, int]]:
    """Run a metadata-only capture and return operator counts.

    Parameters
    ----------
    model:
        PyTorch model to run.
    x:
        Input passed to ``trace``.
    mode:
        Module training mode to use for this one capture.

    Returns
    -------
    list[tuple[str, int]]
        Operator names and counts.
    """

    from torchlens import trace as trace_fn
    from torchlens.options import CaptureOptions

    # Snapshot every submodule's training flag, not just the root's. A recursive
    # ``model.train(root_mode)`` restore would clobber mixed child states (e.g. a
    # frozen ``bn.eval()`` under a training root). For ``mode="current"`` no mode
    # change is applied at all, so the model is left byte-for-byte as found.
    original_modes = {submodule: submodule.training for submodule in model.modules()}
    if mode == "eval":
        model.eval()
    elif mode == "train":
        model.train()
    try:
        trace = cast(Callable[..., Any], trace_fn)(
            model,
            x,
            capture=CaptureOptions(layers_to_save=None),
        )
    finally:
        for submodule, was_training in original_modes.items():
            submodule.training = was_training
    return _ops_from_log(trace)


def list_ops(
    model: nn.Module,
    x: Any,
    mode: Literal["current", "eval", "train", "both"] = "current",
) -> list[tuple[str, int]] | dict[str, list[tuple[str, int]]]:
    """List operators that run in a model forward pass.

    Parameters
    ----------
    model:
        PyTorch model to run.
    x:
        Input passed to ``trace``.
    mode:
        ``"current"``, ``"eval"``, ``"train"``, or ``"both"``. ``"both"``
        returns separate eval/train summaries.

    Returns
    -------
    list[tuple[str, int]] | dict[str, list[tuple[str, int]]]
        Operator count table, or eval/train tables for ``mode="both"``.
    """

    if mode == "both":
        return {
            "eval": _log_ops_for_mode(model, x, "eval"),
            "train": _log_ops_for_mode(model, x, "train"),
        }
    if mode not in {"current", "eval", "train"}:
        raise ValueError("mode must be 'current', 'eval', 'train', or 'both'.")
    return _log_ops_for_mode(model, x, mode)


def _refuse_flop_count_options(flop_convention: object, unsupported: dict[str, Any]) -> None:
    """Refuse a removed or unknown ``flop_count`` option before any capture runs."""

    from torchlens._errors import InvalidArgumentError

    if "count_fma_as_two" in unsupported:
        problem = (
            "flop_count() no longer accepts count_fma_as_two= (removed); use "
            "flop_convention='fma2' (was True) or flop_convention='fma1' (was False)."
        )
    elif unsupported:
        problem = (
            f"flop_count() got unknown option(s) {sorted(unsupported)}; its only "
            "option is flop_convention ('fma2' or 'fma1')."
        )
    elif flop_convention not in ("fma2", "fma1"):
        problem = (
            f"flop_count() got invalid flop_convention={flop_convention!r}; "
            "valid choices: 'fma2', 'fma1'."
        )
    else:
        return
    raise InvalidArgumentError(
        problem,
        code="flop_count_option_invalid",
        remedy="pass flop_convention='fma2' or flop_convention='fma1'",
    )


def flop_count(
    model: nn.Module,
    x: Any,
    *,
    flop_convention: Literal["fma2", "fma1"] = "fma2",
    **unsupported: Any,
) -> int:
    """Return a lightweight forward FLOP count from TorchLens metadata.

    Parameters
    ----------
    model:
        PyTorch model to run.
    x:
        Input passed to ``trace``.
    flop_convention:
        ``"fma2"`` (default; one multiply-accumulate = 2 FLOPs, the stored
        convention) or ``"fma1"`` (recount from each op's two-term compute
        record). Ops with no derivable MAC split make the fma=1 conversion
        impossible and refuse typed (``flop_convention_unavailable``) -- the
        request is NEVER accepted-and-ignored. Same vocabulary as
        ``summary(flop_convention=...)``.
    **unsupported:
        Refused typed (``flop_count_option_invalid``). The removed
        ``count_fma_as_two`` spelling names ``flop_convention`` as its
        replacement.

    Raises
    ------
    InvalidArgumentError
        ``flop_count_option_invalid`` for an unknown or removed option name
        or a ``flop_convention`` outside ``"fma2"``/``"fma1"``.

    Returns
    -------
    int
        Sum of available forward FLOP estimates under the requested
        convention. Operators without a built-in estimate contribute zero
        (disclosed by ``Trace.summary``'s unknown-op count).
    """

    from torchlens import trace as trace_fn
    from torchlens.options import CaptureOptions
    from torchlens.report._compute_truth import forward_flops_total

    _refuse_flop_count_options(flop_convention, unsupported)
    trace = cast(Callable[..., Any], trace_fn)(
        model,
        x,
        capture=CaptureOptions(layers_to_save=None),
    )
    fma = 1 if flop_convention == "fma1" else 2
    return int(forward_flops_total(trace, fma=fma))


def peek_graph(
    model: nn.Module,
    x: Any,
    view: Literal["unrolled", "rolled", "none"] = "unrolled",
    container_path: str | Path = "modelgraph",
    file_format: str = "pdf",
) -> None:
    """Capture and render a model graph with quickstart defaults.

    Parameters
    ----------
    model:
        PyTorch model to run.
    x:
        Input passed to ``trace``.
    view:
        Visualization view vocabulary passed as ``vis_mode``.
    container_path:
        Output path stem for the renderer.
    file_format:
        Renderer output format.

    Returns
    -------
    None
        The graph renderer writes its output as a side effect.
    """

    from torchlens import trace as trace_fn
    from torchlens.options import CaptureOptions

    trace = cast(Callable[..., Any], trace_fn)(
        model,
        x,
        capture=CaptureOptions(layers_to_save=None),
    )
    trace.draw(
        vis_mode=view,
        vis_outpath=str(container_path),
        vis_fileformat=file_format,
        vis_save_only=True,
    )
    return None


def _shape_from_annotation(annotation: Any) -> tuple[int, ...] | None:
    """Extract a tensor shape from a forward-parameter annotation.

    Parameters
    ----------
    annotation:
        Raw annotation from ``inspect.signature``.

    Returns
    -------
    tuple[int, ...] | None
        Shape tuple when inferable.
    """

    if annotation is inspect.Signature.empty:
        return None
    if isinstance(annotation, (tuple, list)) and all(isinstance(dim, int) for dim in annotation):
        return tuple(annotation)
    if get_origin(annotation) is Annotated:
        for item in get_args(annotation)[1:]:
            if isinstance(item, (tuple, list)) and all(isinstance(dim, int) for dim in item):
                return tuple(item)
    return None


def _synthetic_arg_for_parameter(parameter: inspect.Parameter) -> torch.Tensor:
    """Build one synthetic tensor for a forward parameter.

    Parameters
    ----------
    parameter:
        Parameter from ``model.forward``.

    Returns
    -------
    torch.Tensor
        Zero tensor matching the inferred shape.

    Raises
    ------
    ValueError
        If no shape can be inferred.
    """

    if isinstance(parameter.default, torch.Tensor):
        return torch.zeros_like(parameter.default)
    if isinstance(parameter.default, (tuple, list)) and all(
        isinstance(dim, int) for dim in parameter.default
    ):
        return torch.zeros(tuple(parameter.default))
    shape = _shape_from_annotation(parameter.annotation)
    if shape is None:
        raise ValueError(
            "Cannot infer synthetic input shape for forward parameter "
            f"{parameter.name!r}. Annotate it as Annotated[torch.Tensor, (..shape..)] "
            "or provide a tensor/shape default."
        )
    return torch.zeros(shape)


def synthetic_input(model: nn.Module) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Generate a dummy input from a model's ``forward`` signature.

    Parameters
    ----------
    model:
        PyTorch model whose ``forward`` signature should be inspected.

    Returns
    -------
    torch.Tensor | tuple[torch.Tensor, ...]
        One tensor for single-input models, otherwise a tuple of positional tensors.

    Raises
    ------
    ValueError
        If any required input shape cannot be inferred from the signature alone.
    """

    signature = inspect.signature(model.forward)
    annotations = get_type_hints(model.forward, include_extras=True)
    args: list[torch.Tensor] = []
    for parameter in signature.parameters.values():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            # The public return is positional-only, so a keyword-only argument
            # can never be delivered through it. An optional keyword-only param
            # is safely omitted (forward uses its default); a *required* one
            # cannot be represented and must fail loudly here rather than emit a
            # positional tuple that raises a confusing TypeError at forward call.
            if parameter.default is not inspect.Signature.empty:
                continue
            raise ValueError(
                "Cannot build a positional synthetic input for required "
                f"keyword-only forward parameter {parameter.name!r}. "
                "synthetic_input only returns positional tensors; pass this "
                "input explicitly."
            )
        if parameter.default is not inspect.Signature.empty and not isinstance(
            parameter.default, (torch.Tensor, tuple, list)
        ):
            continue
        if parameter.name in annotations:
            parameter = parameter.replace(annotation=annotations[parameter.name])
        args.append(_synthetic_arg_for_parameter(parameter))
    if not args:
        raise ValueError("Cannot infer any tensor inputs from model.forward signature.")
    if len(args) == 1:
        return args[0]
    return tuple(args)


def _memory_budget_to_bytes(memory_budget: int | str) -> int:
    """Convert an integer or simple memory string to bytes.

    Parameters
    ----------
    memory_budget:
        Byte count or string ending in KB, MB, or GB.

    Returns
    -------
    int
        Memory budget in bytes.
    """

    if isinstance(memory_budget, int):
        return memory_budget
    text = memory_budget.strip().lower()
    multipliers = {"kb": 1024, "mb": 1024**2, "gb": 1024**3}
    for suffix, multiplier in multipliers.items():
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)].strip()) * multiplier)
    return int(text)


def find_executable_save_set(
    trace: Any,
    layers: Iterable[str],
    memory_budget: int | str,
) -> list[str]:
    """Choose the largest heuristic subset of layers that fits a memory budget.

    Parameters
    ----------
    trace:
        Trace containing layer memory metadata.
    layers:
        Candidate layer labels or lookup strings.
    memory_budget:
        Byte budget, or a simple string such as ``"64 MB"``.

    Returns
    -------
    list[str]
        Selected layer labels. The heuristic sorts candidates by out
        memory ascending to maximize count.
    """

    budget = _memory_budget_to_bytes(memory_budget)
    candidates: list[tuple[int, str]] = []
    for layer in layers:
        entry = trace[layer]
        label = str(getattr(entry, "layer_label", layer))
        memory = int(
            getattr(entry, "transformed_activation_memory", None)
            or getattr(entry, "activation_memory", None)
            or 0
        )
        candidates.append((memory, label))

    selected: list[str] = []
    total = 0
    for memory, label in sorted(candidates):
        if total + memory <= budget:
            selected.append(label)
            total += memory
    return selected


def trace_streaming(model: nn.Module, inputs_iter: Iterable[Any], **kwargs: Any) -> Any:
    """Capture an iterable of inputs as a stacked multi-pass log.

    Parameters
    ----------
    model:
        Model to capture.
    inputs_iter:
        Iterable producing model inputs.
    **kwargs:
        Keyword arguments forwarded to ``torchlens.trace``.

    Returns
    -------
    tuple[Any, ...]
        Captured traces, one per input item.
    """

    import torchlens

    trace_fn = cast(Callable[..., Any], torchlens.trace)
    logs = [trace_fn(model, inputs, **kwargs) for inputs in inputs_iter]
    if not logs:
        raise ValueError("inputs_iter must yield at least one input.")
    return tuple(logs)


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AutocastRestore": ("torchlens.utils.rng", "AutocastRestore"),
    "MAX_FLOATING_POINT_TOLERANCE": (
        "torchlens.utils.tensor_utils",
        "MAX_FLOATING_POINT_TOLERANCE",
    ),
    "_ATTR_SKIP_SET": ("torchlens.utils.introspection", "_ATTR_SKIP_SET"),
    "_AUTOCAST_DEVICES": ("torchlens.utils.rng", "_AUTOCAST_DEVICES"),
    "_cuda_available": ("torchlens.utils.tensor_utils", "_cuda_available"),
    "_get_code_context": ("torchlens.utils.introspection", "_get_code_context"),
    "_is_cuda_available": ("torchlens.utils.tensor_utils", "_is_cuda_available"),
    "_model_expects_single_arg": (
        "torchlens.utils.arg_handling",
        "_model_expects_single_arg",
    ),
    "_safe_copy_arg": ("torchlens.utils.arg_handling", "_safe_copy_arg"),
    "assign_to_sequence_or_dict": (
        "torchlens.utils.collections",
        "assign_to_sequence_or_dict",
    ),
    "copy_arg_tree": ("torchlens.utils.arg_handling", "copy_arg_tree"),
    "copy_tensor_payload": ("torchlens.utils.tensor_utils", "copy_tensor_payload"),
    "ensure_iterable": ("torchlens.utils.collections", "ensure_iterable"),
    "format_flops": ("torchlens.utils.display", "format_flops"),
    "format_size": ("torchlens.utils.display", "format_size"),
    "get_attr_values_from_tensor_list": (
        "torchlens.utils.introspection",
        "get_attr_values_from_tensor_list",
    ),
    "get_memory_amount": ("torchlens.utils.tensor_utils", "get_memory_amount"),
    "get_torch_capability_snapshot": (
        "torchlens.utils._torch_compat",
        "get_torch_capability_snapshot",
    ),
    "get_vars_of_type_from_obj": (
        "torchlens.utils.introspection",
        "get_vars_of_type_from_obj",
    ),
    "human_readable_size": ("torchlens.utils.display", "human_readable_size"),
    "identity": ("torchlens.utils.display", "identity"),
    "in_notebook": ("torchlens.utils.display", "in_notebook"),
    "index_nested": ("torchlens.utils.collections", "index_nested"),
    "int_list_to_compact_str": ("torchlens.utils.display", "int_list_to_compact_str"),
    "is_iterable": ("torchlens.utils.collections", "is_iterable"),
    "iter_accessible_attributes": (
        "torchlens.utils.introspection",
        "iter_accessible_attributes",
    ),
    "log_current_autocast_state": ("torchlens.utils.rng", "log_current_autocast_state"),
    "log_current_rng_states": ("torchlens.utils.rng", "log_current_rng_states"),
    "make_random_barcode": ("torchlens.utils.hashing", "make_random_barcode"),
    "make_short_barcode_from_input": (
        "torchlens.utils.hashing",
        "make_short_barcode_from_input",
    ),
    "nested_assign": ("torchlens.utils.introspection", "nested_assign"),
    "nested_getattr": ("torchlens.utils.introspection", "nested_getattr"),
    "normalize_input_args": ("torchlens.utils.arg_handling", "normalize_input_args"),
    "print_override": ("torchlens.utils.tensor_utils", "print_override"),
    "progress_bar": ("torchlens.utils.display", "progress_bar"),
    "register_op_rule": ("torchlens.capture.flops", "register_op_rule"),
    "remove_attributes_with_prefix": (
        "torchlens.utils.introspection",
        "remove_attributes_with_prefix",
    ),
    "remove_entry_from_list": (
        "torchlens.utils.collections",
        "remove_entry_from_list",
    ),
    "safe_copy": ("torchlens.utils.tensor_utils", "safe_copy"),
    "safe_copy_args": ("torchlens.utils.arg_handling", "safe_copy_args"),
    "safe_copy_kwargs": ("torchlens.utils.arg_handling", "safe_copy_kwargs"),
    "safe_to": ("torchlens.utils.tensor_utils", "safe_to"),
    "set_random_seed": ("torchlens.utils.rng", "set_random_seed"),
    "set_rng_from_saved_states": ("torchlens.utils.rng", "set_rng_from_saved_states"),
    "tensor_all_nan": ("torchlens.utils.tensor_utils", "tensor_all_nan"),
    "tensor_nanequal": ("torchlens.utils.tensor_utils", "tensor_nanequal"),
    "tensor_stats_summary": ("torchlens.utils.display", "tensor_stats_summary"),
    "warn_parallel": ("torchlens.utils.display", "warn_parallel"),
}


def __getattr__(name: str) -> Any:
    """Resolve one legacy utility facade export on first access.

    Parameters
    ----------
    name:
        Module attribute requested by Python's PEP 562 lookup.

    Returns
    -------
    Any
        The original object from its defining module.

    Raises
    ------
    AttributeError
        If ``name`` is not a utility facade export.
    """

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(importlib.import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return eager and lazy utility facade attributes.

    Returns
    -------
    list[str]
        Sorted module attribute names, including unresolved lazy exports.
    """

    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "AutocastRestore",
    "DoctorCheck",
    "DoctorReport",
    "MAX_FLOATING_POINT_TOLERANCE",
    "_ATTR_SKIP_SET",
    "_AUTOCAST_DEVICES",
    "_cuda_available",
    "_get_code_context",
    "_is_cuda_available",
    "_model_expects_single_arg",
    "_safe_copy_arg",
    "assign_to_sequence_or_dict",
    "capability_snapshot",
    "copy_arg_tree",
    "copy_tensor_payload",
    "doctor",
    "format_capability_summary",
    "ensure_iterable",
    "get_attr_values_from_tensor_list",
    "get_memory_amount",
    "get_vars_of_type_from_obj",
    "human_readable_size",
    "identity",
    "in_notebook",
    "index_nested",
    "int_list_to_compact_str",
    "is_iterable",
    "iter_accessible_attributes",
    "find_executable_save_set",
    "flop_count",
    "format_flops",
    "format_size",
    "list_modules",
    "list_ops",
    "trace_streaming",
    "log_current_autocast_state",
    "log_current_rng_states",
    "make_random_barcode",
    "make_short_barcode_from_input",
    "nested_assign",
    "nested_getattr",
    "normalize_input_args",
    "print_override",
    "peek_graph",
    "progress_bar",
    "register_op_rule",
    "remove_attributes_with_prefix",
    "remove_entry_from_list",
    "safe_copy",
    "safe_copy_args",
    "safe_copy_kwargs",
    "safe_to",
    "set_random_seed",
    "synthetic_input",
    "set_rng_from_saved_states",
    "tensor_all_nan",
    "tensor_nanequal",
    "tensor_stats_summary",
    "warn_parallel",
]
