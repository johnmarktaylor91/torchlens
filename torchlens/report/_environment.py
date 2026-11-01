"""Environment health checks behind ``tl.report.doctor()``.

Each probe returns one :class:`_EnvironmentCheck` row; ``_environment_checks``
collects them in display order. ``capability_snapshot`` is the module-level
accessor for the full runtime capability mapping, which
``DoctorReport.capability_snapshot()`` returns.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import re
import subprocess
from dataclasses import dataclass
from typing import Literal

import torch
from packaging.requirements import Requirement
from torch import nn

from .. import _state
from .._constants import get_orig_torch_funcs
from .._utils import _torch_compat as _tc
from .._utils._subprocess import run_bounded_subprocess
from .._utils._torch_compat import OPTIONAL_CAPABILITY_FLAGS, get_torch_capability_snapshot
from .._utils.introspection import nested_getattr


@dataclass(frozen=True)
class _EnvironmentCheck:
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


def capability_snapshot() -> dict[str, bool]:
    """Return the full runtime capability snapshot (module-level accessor).

    Returns
    -------
    dict[str, bool]
        Capability flag availability: the torch snapshot from
        ``_torch_compat.get_torch_capability_snapshot`` plus the tf backend's
        own snapshot when that backend imports.
    """

    snapshot = get_torch_capability_snapshot()
    try:
        from torchlens.backends.tf._tf_compat import get_tf_capability_snapshot
    except ImportError:
        return snapshot
    snapshot.update(get_tf_capability_snapshot())
    return snapshot


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


def _probe_graphviz() -> _EnvironmentCheck:
    """Probe Python and system Graphviz availability.

    Returns
    -------
    _EnvironmentCheck
        Graphviz health-check row.
    """

    python_graphviz = _module_is_installed("graphviz")
    try:
        # R40: routed through the ONE bounded spawn seam. subprocess.run's
        # timeout killed only the direct child, so a wedged ``dot`` wrapper's
        # grandchild survived the "bounded" probe for the life of the box;
        # the shared runner tears down the whole process group.

        completed = run_bounded_subprocess(
            ["dot", "-V"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _EnvironmentCheck(
            "graphviz",
            "FAIL",
            f"python graphviz={python_graphviz}; dot unavailable ({exc})",
        )
    version_text = (completed.stderr or completed.stdout).strip()
    status: Literal["PASS", "FAIL"] = (
        "PASS" if python_graphviz and completed.returncode == 0 else "FAIL"
    )
    return _EnvironmentCheck(
        "graphviz",
        status,
        f"python graphviz={python_graphviz}; dot={version_text or completed.returncode}",
    )


def _probe_extras() -> _EnvironmentCheck:
    """Probe declared optional extras by importing their representative modules.

    Returns
    -------
    _EnvironmentCheck
        Optional-extras health-check row.
    """

    try:
        extra_probes = _declared_extra_probes()
    except importlib.metadata.PackageNotFoundError as exc:
        return _EnvironmentCheck("extras", "FAIL", f"package metadata unavailable ({exc})")

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
    return _EnvironmentCheck("extras", "PASS", detail)


def _probe_fingerprint() -> _EnvironmentCheck:
    """Probe model weight fingerprinting on a tiny module.

    Returns
    -------
    _EnvironmentCheck
        Fingerprint utility health-check row.
    """

    try:
        from torchlens._capture_state_helpers import _fingerprint_model_weights

        model = nn.Linear(2, 1)
        fingerprint = _fingerprint_model_weights(model)
    except Exception as exc:
        return _EnvironmentCheck("model fingerprint", "FAIL", repr(exc))
    status: Literal["PASS", "FAIL"] = "PASS" if fingerprint else "FAIL"
    return _EnvironmentCheck(
        "model fingerprint", status, fingerprint[:16] if fingerprint else "empty"
    )


def _classify_capability_absences(
    snapshot: dict[str, bool],
) -> tuple[list[str], list[str], list[str]]:
    """Split a capability snapshot's absent flags into degradation groups.

    ONE shared classifier for ``tl.report.doctor()`` and ``tl.compat.report()``
    so both rows read the same snapshot the same way.

    Parameters
    ----------
    snapshot:
        Capability flag availability, as returned by the runtime snapshot.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        ``(missing, unprobed, optional_absent)``. Only ``missing`` is a genuine
        degradation (r-b4 R26-4): an absent OPTIONAL feature (interpreter-version
        surface, upstream-removed API, an optional backend that is not
        installed) keeps a healthy install at PASS, and a lazily probed flag
        whose probe has never run (kineto event fields, the memory profile, ...)
        is UNPROBED, not missing: nothing has consumed the capability yet, so
        nothing has degraded (D04 integration fix).
    """

    def _probe_latched(name: str) -> bool:
        """Return whether a flag's lazy probe has run (eager flags always have)."""

        latch = "_" + name.removeprefix("HAS_") + "_PROBED"
        return bool(getattr(_tc, latch, True))

    absent = [name for name, available in snapshot.items() if not available]
    required = [name for name in absent if name not in OPTIONAL_CAPABILITY_FLAGS]
    missing = [name for name in required if _probe_latched(name)]
    unprobed = [name for name in required if not _probe_latched(name)]
    optional_absent = [name for name in absent if name in OPTIONAL_CAPABILITY_FLAGS]
    return missing, unprobed, optional_absent


def _probe_torch_capabilities() -> _EnvironmentCheck:
    """Return the runtime capability snapshot doctor row.

    Returns
    -------
    _EnvironmentCheck
        Snapshot of probed private integration capabilities.
    """

    snapshot = capability_snapshot()
    missing, unprobed, optional_absent = _classify_capability_absences(snapshot)
    # Grouped absences-first summary; the full name=value dump lives on the
    # detail accessor (report.capability_snapshot()), not in the row cell.
    detail = format_capability_summary(snapshot)
    if missing:
        detail += "; missing=" + ",".join(missing)
    if unprobed:
        detail += "; unprobed=" + ",".join(unprobed)
    if optional_absent:
        detail += "; optional_absent=" + ",".join(optional_absent)
    detail += "; full dump: report.capability_snapshot()"
    # Report the true state: a missing private-integration capability is a
    # degraded (WARN) row, not a "PASS". These flags are feature-detected and may
    # be legitimately absent across torch versions, so WARN (not FAIL) is honest.
    status: Literal["PASS", "WARN"] = "PASS" if not missing else "WARN"
    return _EnvironmentCheck("runtime capabilities", status, detail)


def _probe_torch_wrapper_bindings() -> _EnvironmentCheck:
    """Check torch namespace attributes against the installed wrapper registry.

    Returns
    -------
    _EnvironmentCheck
        Warning-only detector for stale torch module attributes. This cannot
        inspect arbitrary local aliases such as closure-bound ``from torch
        import relu`` references, but it catches the cheap process-global case
        where torch itself is no longer pointing at registered wrappers.
    """

    if not _state._orig_to_decorated:
        return _EnvironmentCheck("torch wrapper bindings", "SKIP", "wrappers not installed yet")
    if not _state._is_decorated:
        return _EnvironmentCheck("torch wrapper bindings", "SKIP", "torch is currently unwrapped")

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
        return _EnvironmentCheck(
            "torch wrapper bindings",
            "WARN",
            f"{len(stale)}/{checked} torch attrs still point at original callables; "
            f"examples={examples}",
        )
    return _EnvironmentCheck(
        "torch wrapper bindings",
        "PASS",
        f"checked={checked}; no stale torch namespace bindings detected",
    )


def _probe_mechanical_belt() -> _EnvironmentCheck:
    """Report protocol-invisible belt coverage gaps.

    Returns
    -------
    _EnvironmentCheck
        Disclosure row for belt probe failures and unprobed candidates. A
        candidate whose mode visibility could not be MEASURED (probe raised,
        or no probe recipe exists) is neither belt-patched nor proven
        protocol-visible, so a stale pre-wrap reference to it can drop ops
        with zero signal while the capture still reports
        ``capture_verified=True`` (grind-r6 b3 R02, sol MED).
    """

    from ..backends.torch.belt import belt_report

    if not _state._is_decorated:
        return _EnvironmentCheck(
            "mechanical belt",
            "SKIP",
            "belt not derived yet (torch wrapping is lazy; run a capture first)",
        )
    report = belt_report()
    if report is None:
        return _EnvironmentCheck("mechanical belt", "SKIP", "belt derivation unavailable")
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
        return _EnvironmentCheck(
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
    return _EnvironmentCheck("mechanical belt", "WARN", detail)


def _environment_checks() -> tuple[_EnvironmentCheck, ...]:
    """Run a TorchLens startup health check.

    Returns
    -------
    tuple[_EnvironmentCheck, ...]
        Private probes for PyTorch, CUDA, Graphviz, safetensors, extras,
        and model-fingerprint checks.
    """

    checks: list[_EnvironmentCheck] = [
        _EnvironmentCheck("pytorch", "PASS", torch.__version__),
        _probe_torch_capabilities(),
        _probe_torch_wrapper_bindings(),
        _probe_mechanical_belt(),
        _EnvironmentCheck(
            "cuda",
            "PASS" if torch.cuda.is_available() else "SKIP",
            f"available={torch.cuda.is_available()}; devices={torch.cuda.device_count()}",
        ),
        _probe_graphviz(),
        _EnvironmentCheck(
            "safetensors",
            "PASS" if _module_is_installed("safetensors") else "FAIL",
            "installed" if _module_is_installed("safetensors") else "missing",
        ),
        _probe_extras(),
        _probe_fingerprint(),
    ]
    return tuple(checks)
