"""check_determinism: one controlled repeatability verdict, cost never hidden.

Observe memo item 9: the DEFAULT answers exactly one question -- do N isolated
same-seed runs agree? -- with a three-state verdict that never generalizes N
runs into "deterministic". Same-state repeatability and different-seed
sensitivity are DIFFERENT questions with opposite RNG protocols, so the second
question is an explicit opt-in (``seed_sensitivity=True``: exactly ONE extra
differently-seeded run, its cost disclosed as runs+1) landing in a SECOND
named verdict slot reserved from day one. Every spelling here is
DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import torch

from ..utils._torch_compat import read_legacy_fp32_controls, snapshot_fp32_precision_controls
from ._common import _compute_ops, _op_label, _safe_out, _source_line
from ._first_bad import FirstBadThing

if TYPE_CHECKING:
    from torchlens.data_classes.trace import Trace

__all__ = ["DeterminismReport", "check_determinism"]

#: Closed verdict vocabulary. "repeatable_under_test" deliberately never says
#: "deterministic": N agreeing runs support only a repeatability claim under
#: the declared conditions.
VERDICTS = ("repeatable_under_test", "nondeterminism_observed", "inconclusive")


@dataclass(frozen=True)
class DeterminismReport:
    """Typed result of :func:`check_determinism`.

    Parameters
    ----------
    verdict:
        ``repeatable_under_test`` / ``nondeterminism_observed`` /
        ``inconclusive`` -- the SAME-SEED repeatability question only.
    runs:
        Number of same-seed isolated executions compared.
    seed:
        The seed every compared run used.
    policy:
        ``"exact"`` (bitwise agreement; matching NaN masks agree while
        non-finites are separately recorded) or ``"tolerance"`` (the claim
        weakens to NUMERICAL repeatability).
    rtol:
        Relative tolerance under the tolerance policy, else ``None``.
    atol:
        Absolute tolerance under the tolerance policy, else ``None``.
    isolation:
        How runs were isolated (fresh deepcopy + released copy + cloned
        inputs + preserved RNG per run).
    run_policy:
        Reserved run-policy slot (``"same_seed_isolated"`` today;
        ``"as_run"`` / cross-device policies land here without a reshape).
    model_mode:
        ``"train"`` or ``"eval"`` as received.
    environment_witnesses:
        Named ``(witness, value)`` lines: deterministic-algorithms flag,
        cuDNN flags, TF32 flags, ``CUBLAS_WORKSPACE_CONFIG``, thread counts.
    first_divergence:
        The first EXECUTION-ORDER row where two runs separated, or ``None``.
    divergent_ops:
        Labels of all ops that separated across any compared pair of runs.
    nonfinite_ops:
        Labels whose payloads carried NaN/Inf in run 1 (recorded separately;
        matching non-finite masks still count as agreement under ``exact``).
    unchecked_payloads:
        Ops whose payloads could not be compared (unsaved/non-tensor); a
        divergence inside them cannot be excluded.
    rng_consuming_ops:
        Labels observed consuming host RNG, attributed to the CONSUMING op
        (the per-op state snapshot is PRE-op, so a transition observed at row
        k belongs to op k-1 -- the off-by-one is corrected here), with the
        stochastic-function-name table as the fallback basis.
    rng_attribution_basis:
        ``"pre_op_rng_snapshots"`` or ``"stochastic_func_names"``.
    host_entropy_witnesses:
        Trace-recorded host-entropy facts, quoted, never re-derived.
    seed_sensitivity_verdict:
        SECOND named verdict slot: ``None`` unless ``seed_sensitivity=True``
        was requested; then ``sensitive_to_seed`` / ``insensitive_under_test``
        / ``inconclusive`` from exactly one extra differently-seeded run.
    remedies:
        Actionable next steps.
    message:
        Human-readable summary; discloses total capture count up front.
    """

    verdict: str
    runs: int
    seed: int
    policy: str
    rtol: float | None
    atol: float | None
    isolation: str
    run_policy: str
    model_mode: str
    environment_witnesses: tuple[tuple[str, str], ...]
    first_divergence: dict[str, Any] | None
    divergent_ops: tuple[str, ...]
    nonfinite_ops: tuple[str, ...]
    unchecked_payloads: int
    rng_consuming_ops: tuple[str, ...]
    rng_attribution_basis: str
    host_entropy_witnesses: tuple[str, ...]
    seed_sensitivity_verdict: str | None
    remedies: tuple[str, ...] = field(default_factory=tuple)
    message: str = ""

    @property
    def first_bad_thing(self) -> FirstBadThing:
        """Project the first divergence into the shared vocabulary."""

        row = self.first_divergence
        return FirstBadThing(
            found=self.verdict == "nondeterminism_observed",
            tool="check_determinism",
            kind="nondeterminism" if self.verdict == "nondeterminism_observed" else "none",
            label=str(row["op"]) if row else None,
            label_status="final" if row else "none",
            module=None,
            source_line=str(row.get("source_line")) if row and row.get("source_line") else None,
            coverage="complete" if self.unchecked_payloads == 0 else "found_first_among_checked",
            uncertainty_zone=(),
            detection_basis="double_run",
            message=self.message,
        )


def _fp32_witness(value: Any) -> str:
    """Render one legacy fp32 view; ``None`` means no legacy equivalent."""

    return "<per-backend fp32_precision policy>" if value is None else str(value)


def _environment_witnesses() -> tuple[tuple[str, str], ...]:
    """Collect the named environment witness lines (read, never mutated)."""

    legacy_fp32, _ = read_legacy_fp32_controls()
    witnesses: list[tuple[str, str]] = [
        (
            "torch.are_deterministic_algorithms_enabled",
            str(torch.are_deterministic_algorithms_enabled()),
        ),
        ("torch.backends.cudnn.deterministic", str(torch.backends.cudnn.deterministic)),
        ("torch.backends.cudnn.benchmark", str(torch.backends.cudnn.benchmark)),
        # Legacy TF32 views raise under an fp32_precision policy with no legacy
        # equivalent (torch >= 2.9); read them safely and witness the controls.
        ("torch.backends.cudnn.allow_tf32", _fp32_witness(legacy_fp32["cudnn_allow_tf32"])),
        (
            "torch.backends.cuda.matmul.allow_tf32",
            _fp32_witness(legacy_fp32["cuda_matmul_allow_tf32"]),
        ),
        ("torch.backends.fp32_precision", str(snapshot_fp32_precision_controls())),
        ("CUBLAS_WORKSPACE_CONFIG", os.environ.get("CUBLAS_WORKSPACE_CONFIG", "<unset>")),
        ("torch.get_num_threads", str(torch.get_num_threads())),
        ("torch.get_num_interop_threads", str(torch.get_num_interop_threads())),
    ]
    return tuple(witnesses)


def _values_agree(
    a: torch.Tensor, b: torch.Tensor, *, rtol: float | None, atol: float | None
) -> bool:
    """Compare two payloads under the declared policy.

    Exact policy: elementwise equality with MATCHING non-finite masks counting
    as agreement (NaN positions must match; the non-finites themselves are
    recorded separately by the caller).
    """

    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if rtol is None and atol is None:
        if torch.is_floating_point(a) or torch.is_complex(a):
            nan_a = torch.isnan(a)
            if not torch.equal(nan_a, torch.isnan(b)):
                return False
            finite_equal = (a == b) | nan_a
            return bool(finite_equal.all().item())
        return bool(torch.equal(a, b))
    return bool(torch.allclose(a, b, rtol=rtol or 0.0, atol=atol or 0.0, equal_nan=True))


def _rng_consuming_ops(trace: Trace) -> tuple[tuple[str, ...], str]:
    """Attribute host-RNG consumption to the CONSUMING op.

    The per-op RNG snapshot (``func_rng_states``) is taken PRE-op, so a state
    transition observed at row k belongs to op k-1 -- the off-by-one is
    corrected here. When snapshots are unavailable the stochastic
    function-name table is the disclosed fallback basis.
    """

    ops = _compute_ops(trace)
    digests: list[Any] = []
    for op in ops:
        states = getattr(op, "func_rng_states", None)
        digests.append(repr(states) if states else None)
    if any(digest is not None for digest in digests):
        consumers: list[str] = []
        for index in range(len(ops) - 1):
            before, after = digests[index], digests[index + 1]
            if before is not None and after is not None and before != after:
                consumers.append(_op_label(ops[index]))
        if consumers:
            return tuple(dict.fromkeys(consumers)), "pre_op_rng_snapshots"
    from ._precision import _STOCHASTIC_FUNC_NAMES

    named = tuple(
        _op_label(op)
        for op in ops
        if str(getattr(op, "func_name", "")).rstrip("_") in _STOCHASTIC_FUNC_NAMES
    )
    return named, "stochastic_func_names"


def _host_entropy_witnesses(trace: Trace) -> tuple[str, ...]:
    """Quote trace-recorded host-entropy facts (never re-derived here)."""

    witnesses: list[str] = []
    for attribute in (
        "host_rng_consumed",
        "host_nondeterminism_witnesses",
        "nondeterministic_sources",
    ):
        value = getattr(trace, attribute, None)
        if value:
            witnesses.append(f"{attribute}={value!r}")
    return tuple(witnesses)


def _compare_runs(
    reference: Trace,
    candidate: Trace,
    *,
    rtol: float | None,
    atol: float | None,
) -> tuple[list[dict[str, Any]], int, list[str]]:
    """Walk two runs in execution order; return (divergences, unchecked, nonfinite)."""

    from ._nan import _nonfinite_kind

    reference_ops = {_op_label(op): op for op in _compute_ops(reference)}
    candidate_ops = {_op_label(op): op for op in _compute_ops(candidate)}
    divergences: list[dict[str, Any]] = []
    unchecked = 0
    nonfinite: list[str] = []
    for label, ref_op in reference_ops.items():
        cand_op = candidate_ops.get(label)
        if cand_op is None:
            divergences.append({"op": label, "why": "structural: op absent in a same-seed rerun"})
            continue
        ref_out, ref_reason = _safe_out(ref_op)
        cand_out, cand_reason = _safe_out(cand_op)
        if ref_reason is not None or cand_reason is not None:
            unchecked += 1
            continue
        if not isinstance(ref_out, torch.Tensor) or not isinstance(cand_out, torch.Tensor):
            unchecked += 1
            continue
        if isinstance(ref_out, torch.Tensor) and _nonfinite_kind(ref_out) != "none":
            nonfinite.append(label)
        if not _values_agree(ref_out, cand_out, rtol=rtol, atol=atol):
            divergences.append(
                {
                    "op": label,
                    "why": "value",
                    "source_line": _source_line(ref_op),
                    "func_name": str(getattr(ref_op, "func_name", "")),
                }
            )
    structural_only_b = [label for label in candidate_ops if label not in reference_ops]
    for label in structural_only_b:
        divergences.append({"op": label, "why": "structural: op only in the rerun"})
    return divergences, unchecked, nonfinite


@dataclass(frozen=True)
class _PolicyFacts:
    """The declared run policy shared by every report builder."""

    seed: int
    policy: str
    rtol: float | None
    atol: float | None
    model_mode: str
    witnesses: tuple[tuple[str, str], ...]
    seed_sensitivity: bool


def _inconclusive_deepcopy_report(
    copy_error: BaseException, facts: _PolicyFacts
) -> DeterminismReport:
    """Build the typed inconclusive report for a deepcopy-refusing model."""

    seed = facts.seed
    seed_sensitivity = facts.seed_sensitivity
    return DeterminismReport(
        verdict="inconclusive",
        runs=0,
        seed=seed,
        policy=facts.policy,
        rtol=facts.rtol,
        atol=facts.atol,
        isolation="fresh_deepcopy_per_run",
        run_policy="same_seed_isolated",
        model_mode=facts.model_mode,
        environment_witnesses=facts.witnesses,
        first_divergence=None,
        divergent_ops=(),
        nonfinite_ops=(),
        unchecked_payloads=0,
        rng_consuming_ops=(),
        rng_attribution_basis="stochastic_func_names",
        host_entropy_witnesses=(),
        seed_sensitivity_verdict="inconclusive" if seed_sensitivity else None,
        remedies=(
            "provide a model_factory-constructed fresh instance per run and "
            "compare traces with tl.debug.first_divergence",
        ),
        message=(
            "inconclusive: the model cannot be deep-copied for isolated reruns "
            f"({type(copy_error).__name__}: {copy_error})."
        ),
    )


def _compare_all_runs(
    traces: list[Any], *, rtol: float | None, atol: float | None
) -> tuple[list[dict[str, Any]], int, tuple[str, ...]]:
    """Compare every rerun against run 1; return (divergences, unchecked, nonfinite)."""

    reference = traces[0]
    divergences: list[dict[str, Any]] = []
    unchecked = 0
    nonfinite: tuple[str, ...] = ()
    for candidate in traces[1:]:
        pair_divergences, pair_unchecked, pair_nonfinite = _compare_runs(
            reference, candidate, rtol=rtol, atol=atol
        )
        divergences.extend(pair_divergences)
        unchecked = max(unchecked, pair_unchecked)
        if not nonfinite:
            nonfinite = tuple(dict.fromkeys(pair_nonfinite))
    return divergences, unchecked, nonfinite


def _build_remedies(
    verdict: str,
    unchecked: int,
    witnesses: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    """Assemble the actionable remedies for one verdict."""

    remedies: list[str] = []
    if verdict == "nondeterminism_observed":
        remedies.append(
            "compare rng_consuming_ops against the first divergence; seed-"
            "insensitive nondeterminism usually points at atomics/cudnn.benchmark"
        )
        if any(value == "True" for name, value in witnesses if "benchmark" in name):
            remedies.append(
                "cudnn.benchmark=True autotunes kernels per shape; set "
                "torch.backends.cudnn.benchmark=False for repeatability"
            )
    if unchecked:
        remedies.append(
            f"{unchecked} payload(s) were not comparable (unsaved/non-tensor); "
            "re-trace with a wider save= for complete coverage"
        )
    return tuple(remedies)


def _seed_probe_verdict(
    reference: Any, probe: Any, *, rtol: float | None, atol: float | None
) -> str:
    """Return the SECOND named verdict from the one extra differently-seeded run."""

    probe_divergences, probe_unchecked, _ = _compare_runs(reference, probe, rtol=rtol, atol=atol)
    if probe_divergences:
        return "sensitive_to_seed"
    if probe_unchecked:
        return "inconclusive"
    return "insensitive_under_test"


def _same_seed_verdict(divergences: list[dict[str, Any]], unchecked: int, reference: Any) -> str:
    """Settle the three-state same-seed verdict (never "deterministic")."""

    if divergences:
        return "nondeterminism_observed"
    if unchecked and not any(_safe_out(op)[1] is None for op in _compute_ops(reference)):
        return "inconclusive"
    return "repeatable_under_test"


def _report_message(report: DeterminismReport, total_captures: int) -> str:
    """Compose the summary line; total capture cost is disclosed up front."""

    claim = (
        "exact"
        if report.policy == "exact"
        else f"numerical (rtol={report.rtol}, atol={report.atol})"
    )
    message = (
        f"{report.verdict}: {report.runs} isolated same-seed (seed={report.seed}) runs "
        f"compared under the {claim} policy; {total_captures} captures executed in total."
    )
    if report.first_divergence is not None:
        message += f" First divergence in execution order: {report.first_divergence['op']}."
    if report.seed_sensitivity_verdict is not None:
        message += f" Seed sensitivity (1 extra run): {report.seed_sensitivity_verdict}."
    return message


def check_determinism(
    model: Any,
    input_args: Any,
    input_kwargs: dict[Any, Any] | None = None,
    *,
    runs: int = 2,
    seed: int = 0,
    rtol: float | None = None,
    atol: float | None = None,
    seed_sensitivity: bool = False,
    trace_kwargs: dict[str, Any] | None = None,
) -> DeterminismReport:
    """Answer ONE controlled question: do N isolated same-seed runs agree?

    Every run executes on a fresh deep copy (released from TorchLens
    preparation, BatchNorm buffers isolated per run) with cloned inputs under
    preserved, identically-seeded RNG. The verdict is three-state and never
    generalizes to "deterministic"; different-seed sensitivity is the
    separate, explicitly-requested ``seed_sensitivity=True`` question (exactly
    ONE extra differently-seeded run -- total captures = runs + 1, disclosed).

    Parameters
    ----------
    model:
        Source model; never mutated (deepcopy-impossible models return an
        ``inconclusive`` verdict with a ``model_factory`` remedy).
    input_args:
        Forward input value or positional-argument list/tuple.
    input_kwargs:
        Optional forward keyword arguments.
    runs:
        Number of same-seed isolated executions (>= 2).
    seed:
        Seed for every compared run.
    rtol:
        Optional relative tolerance; supplying either tolerance changes the
        claim from exact to NUMERICAL repeatability.
    atol:
        Optional absolute tolerance.
    seed_sensitivity:
        Opt into the SECOND named verdict from one extra differently-seeded
        run.
    trace_kwargs:
        Extra ``tl.trace`` keyword arguments for every run.

    Returns
    -------
    DeterminismReport
        Typed report; ``report.verdict`` is the same-seed answer and
        ``report.seed_sensitivity_verdict`` stays ``None`` unless requested.

    Raises
    ------
    InvalidArgumentError
        If ``runs < 2`` (code ``determinism_runs_invalid``).
    """

    from .._errors import InvalidArgumentError
    from ._rerun import isolated_capture

    if runs < 2:
        raise InvalidArgumentError(
            f"check_determinism needs at least 2 same-seed runs, got {runs}",
            code="determinism_runs_invalid",
            remedy="pass runs=2 or more",
        )
    policy = "exact" if (rtol is None and atol is None) else "tolerance"
    model_mode = "train" if getattr(model, "training", False) else "eval"
    witnesses = _environment_witnesses()

    try:
        copy.deepcopy(model)
    except Exception as copy_error:  # noqa: BLE001 - typed inconclusive, not a crash.
        return _inconclusive_deepcopy_report(
            copy_error,
            _PolicyFacts(
                seed=seed,
                policy=policy,
                rtol=rtol,
                atol=atol,
                model_mode=model_mode,
                witnesses=witnesses,
                seed_sensitivity=seed_sensitivity,
            ),
        )

    total_captures = runs + (1 if seed_sensitivity else 0)
    traces: list[Any] = []
    try:
        for _ in range(runs):
            traces.append(
                isolated_capture(model, input_args, input_kwargs, seed=seed, **(trace_kwargs or {}))
            )
        reference = traces[0]
        divergences, unchecked, nonfinite = _compare_all_runs(traces, rtol=rtol, atol=atol)
        rng_ops, rng_basis = _rng_consuming_ops(reference)
        host_entropy = _host_entropy_witnesses(reference)

        seed_sensitivity_verdict: str | None = None
        if seed_sensitivity:
            probe = isolated_capture(
                model, input_args, input_kwargs, seed=seed + 1, **(trace_kwargs or {})
            )
            traces.append(probe)
            seed_sensitivity_verdict = _seed_probe_verdict(reference, probe, rtol=rtol, atol=atol)

        divergent_labels = tuple(dict.fromkeys(str(row["op"]) for row in divergences))
        first_row = divergences[0] if divergences else None
        verdict = _same_seed_verdict(divergences, unchecked, reference)
        remedies = _build_remedies(verdict, unchecked, witnesses)

        report = DeterminismReport(
            verdict=verdict,
            runs=runs,
            seed=seed,
            policy=policy,
            rtol=rtol,
            atol=atol,
            isolation="fresh_deepcopy_per_run",
            run_policy="same_seed_isolated",
            model_mode=model_mode,
            environment_witnesses=witnesses,
            first_divergence=first_row,
            divergent_ops=divergent_labels,
            nonfinite_ops=nonfinite,
            unchecked_payloads=unchecked,
            rng_consuming_ops=rng_ops,
            rng_attribution_basis=rng_basis,
            host_entropy_witnesses=host_entropy,
            seed_sensitivity_verdict=seed_sensitivity_verdict,
            remedies=tuple(remedies),
            message="",
        )
        return replace(report, message=_report_message(report, total_captures))
    finally:
        for captured in traces:
            cleanup = getattr(captured, "cleanup", None)
            if cleanup is not None:
                cleanup()
