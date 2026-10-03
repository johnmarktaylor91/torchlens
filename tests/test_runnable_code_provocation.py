"""Provocation coverage for previously test-orphaned ``RunnableErrorCode`` members (r25-3).

Every test drives a REAL production raise site through the public API (``tl.trace``,
``tl.save``, ``tl.load``, ``Trace.run``) and asserts the frozen code string on
``exc.fields["code"]`` (or on the surfaced report/diagnostics for
divergence-classification and producer-preflight codes). No raise site is
monkeypatched; where physically constructing a condition is impractical the test
tampers surrounding state instead -- a saved artifact's descriptor JSON (the exact
adversarial input the checks exist to catch) or a loaded trace's resolved-callable
table (forcing the replayed call to genuinely misbehave at execution time).

ALLOWLIST (unprovokable without touching fenced internals): none -- all 12 target
codes are provoked below, plus the B8-33 run-time ``run_capability_unavailable``
pin, the r25-4 ``input_alias_topology_unresolved`` disclosure row, and the B8-27
preflight message summary.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import (
    PathDivergenceError,
    PoisonedRunError,
    RunCapabilityUnavailableError,
    RunnablePreflightError,
    RunPreconditionError,
)
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness, RunnableErrorCode


def _capture_options() -> CaptureOptions:
    """Return the standard runnable-capable capture options."""

    return CaptureOptions(intervention_ready=True, capture_container_structure=True, cache=False)


class BranchModel(nn.Module):
    """Data-dependent scalar-bool branch over one linear projection."""

    def __init__(self) -> None:
        """Build the shared projection."""

        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale the projection by a branch taken on the input sign."""

        if bool(x.sum() > 0):
            return self.lin(x) * 2.0
        return self.lin(x) * 3.0


def _branch_input() -> torch.Tensor:
    """Return the positive-branch capture input."""

    return torch.ones(2, 4)


def _save_branch_artifact(tmp_path: Path, name: str) -> Path:
    """Capture ``BranchModel`` on the positive branch and save it runnable."""

    torch.manual_seed(0)
    path = tmp_path / name
    trace = tl.trace(BranchModel().eval(), _branch_input(), capture=_capture_options())
    trace.save(path, level="runnable", include_weights=True)
    return path


def _tamper_run_descriptor(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    """Rewrite the artifact's sparse run descriptor JSON through ``mutate``."""

    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mutate(manifest["run"])
    manifest_path.write_text(json.dumps(manifest))


def _wrap_loaded_callable(
    loaded: Any, call_id: str, wrap: Callable[[Callable[..., Any]], Callable[..., Any]]
) -> None:
    """Replace one resolved callable on a loaded trace with a misbehaving wrapper.

    This tampers surrounding state only: the production contract checks and their
    raise sites run untouched against the genuinely misbehaving replayed call.
    """

    callables = loaded._runnable.callables_by_call_id
    callables[call_id] = wrap(callables[call_id])


# ---------------------------------------------------------------------------
# Producer-preflight codes (surface on RunnablePreflightError diagnostics).
# ---------------------------------------------------------------------------


def test_missing_control_classification_preflight_and_summary_message(
    tmp_path: Path,
) -> None:
    """A terminal unclassified scalar-bool refuses runnable save with the code.

    Also pins the B8-27 preflight message contract: the refusal message must
    summarize the first diagnostic (code, text, stage) instead of the historical
    content-free one-liner.
    """

    class TerminalBool(nn.Module):
        """Compute a scalar bool that escapes without a classified consumer."""

        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # ``and`` consumes ``__bool__`` outside every classified construct
            # (if/while/bool-cast/assert), leaving the terminal bool unclassified.
            _ = (x.sum() > 0) and True
            return self.lin(x)

    trace = tl.trace(TerminalBool().eval(), torch.randn(2, 4), capture=_capture_options())
    with pytest.raises(RunnablePreflightError) as caught:
        trace.save(tmp_path / "terminal-bool.tlspec", level="runnable")
    error = caught.value
    assert error.fields["code"] == RunnableErrorCode.SPARSE_PREFLIGHT_FAILED.value
    codes = {diag.code for diag in error.fields["diagnostics"]}
    assert RunnableErrorCode.MISSING_CONTROL_CLASSIFICATION in codes
    # B8-27: the message inlines the first diagnostic's code, text, and stage.
    message = str(error)
    first = error.fields["diagnostics"][0]
    assert f"[{first.code.value}]" in message
    assert first.message.split()[0] in message
    assert first.detection_stage in message
    assert "exc.fields['diagnostics']" in message


# ---------------------------------------------------------------------------
# Run-time input contract codes.
# ---------------------------------------------------------------------------


def test_input_tree_mismatch_on_non_tensor_input(tmp_path: Path) -> None:
    """A non-tensor leaf at a recorded tensor site refuses typed."""

    loaded = tl.load(_save_branch_artifact(tmp_path, "tree.tlspec"))
    with pytest.raises(RunPreconditionError) as caught:
        loaded.run(inputs=3)
    assert caught.value.fields["code"] == RunnableErrorCode.INPUT_TREE_MISMATCH.value


def test_input_dtype_mismatch_on_upcast_input(tmp_path: Path) -> None:
    """A same-shape float64 input diverges typed on the dtype contract."""

    loaded = tl.load(_save_branch_artifact(tmp_path, "dtype-in.tlspec"))
    with pytest.raises(PathDivergenceError) as caught:
        loaded.run(inputs=torch.ones(2, 4, dtype=torch.float64))
    assert caught.value.fields["code"] == RunnableErrorCode.INPUT_DTYPE_MISMATCH.value


# ---------------------------------------------------------------------------
# Run-time output/production/mutation contract codes.
# ---------------------------------------------------------------------------


def test_output_shape_mismatch_on_data_dependent_shape(tmp_path: Path) -> None:
    """A boolean-mask model replayed on a different mask count diverges typed."""

    class MaskModel(nn.Module):
        """Boolean advanced indexing: output shape depends on input values."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[x > 0]

    path = tmp_path / "mask.tlspec"
    trace = tl.trace(
        MaskModel().eval(),
        torch.tensor([1.0, -1.0, 2.0, -2.0]),
        capture=_capture_options(),
    )
    trace.save(path, level="runnable")
    loaded = tl.load(path)
    with pytest.raises(PathDivergenceError) as caught:
        loaded.run(inputs=torch.tensor([1.0, 1.0, 2.0, -2.0]))
    assert caught.value.fields["code"] == RunnableErrorCode.OUTPUT_SHAPE_MISMATCH.value


def test_output_dtype_mismatch_on_upcasting_call(tmp_path: Path) -> None:
    """A replayed call genuinely producing the wrong dtype diverges typed."""

    loaded = tl.load(_save_branch_artifact(tmp_path, "dtype-out.tlspec"))
    _wrap_loaded_callable(
        loaded,
        "call:10",
        lambda orig: lambda *args, **kwargs: orig(*args, **kwargs).to(torch.float64),
    )
    with pytest.raises(PathDivergenceError) as caught:
        loaded.run(inputs=_branch_input())
    assert caught.value.fields["code"] == RunnableErrorCode.OUTPUT_DTYPE_MISMATCH.value


def test_output_structure_mismatch_on_tampered_output_path(tmp_path: Path) -> None:
    """A tampered recorded output path fails the structure contract typed."""

    path = _save_branch_artifact(tmp_path, "struct.tlspec")

    def tamper(run: dict[str, Any]) -> None:
        for slot in run["tensor_slots"]:
            if slot["slot_id"] == "slot:linear_1_3:1":
                slot["output_path"] = [0]

    _tamper_run_descriptor(path, tamper)
    loaded = tl.load(path)
    with pytest.raises(PathDivergenceError) as caught:
        loaded.run(inputs=_branch_input())
    assert caught.value.fields["code"] == RunnableErrorCode.OUTPUT_STRUCTURE_MISMATCH.value


def test_slot_production_mismatch_on_unproduced_output_source(tmp_path: Path) -> None:
    """A final call producing the wrong container leaves the output slot unproduced."""

    loaded = tl.load(_save_branch_artifact(tmp_path, "produce.tlspec"))
    _wrap_loaded_callable(
        loaded,
        "call:19",
        lambda orig: lambda *args, **kwargs: {"x": orig(*args, **kwargs)},
    )
    with pytest.raises(RunPreconditionError) as caught:
        loaded.run(inputs=_branch_input(), on_divergence="return_diverged")
    assert caught.value.fields["code"] == RunnableErrorCode.SLOT_PRODUCTION_MISMATCH.value


def test_missing_tensor_slot_on_consumer_of_unproduced_slot(tmp_path: Path) -> None:
    """A later call referencing a never-produced slot refuses typed at run time."""

    loaded = tl.load(_save_branch_artifact(tmp_path, "missing-slot.tlspec"))
    _wrap_loaded_callable(
        loaded,
        "call:10",
        lambda orig: lambda *args, **kwargs: {"x": orig(*args, **kwargs)},
    )
    with pytest.raises(RunPreconditionError) as caught:
        loaded.run(inputs=_branch_input(), on_divergence="return_diverged")
    assert caught.value.fields["code"] == RunnableErrorCode.MISSING_TENSOR_SLOT.value


def test_mutation_version_mismatch_on_covert_input_mutation(tmp_path: Path) -> None:
    """A non-inplace recorded call that bumps an input tensor version diverges typed."""

    loaded = tl.load(_save_branch_artifact(tmp_path, "mutate.tlspec"))

    def mutating(orig: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for value in args:
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    value.mul_(1.0)  # byte-identical, version-bumping
                    break
            return orig(*args, **kwargs)

        return wrapper

    _wrap_loaded_callable(loaded, "call:10", mutating)
    with pytest.raises(PathDivergenceError) as caught:
        loaded.run(inputs=_branch_input())
    assert caught.value.fields["code"] == RunnableErrorCode.MUTATION_VERSION_MISMATCH.value


# ---------------------------------------------------------------------------
# Control-flow divergence codes.
# ---------------------------------------------------------------------------


def test_scalar_bool_divergence_on_flipped_branch_input(tmp_path: Path) -> None:
    """An input flipping a recorded scalar-bool witness diverges typed."""

    loaded = tl.load(_save_branch_artifact(tmp_path, "flip.tlspec"))
    with pytest.raises(PathDivergenceError) as caught:
        loaded.run(inputs=-_branch_input())
    assert caught.value.fields["code"] == RunnableErrorCode.SCALAR_BOOL_DIVERGENCE.value


def test_conditional_arm_divergence_on_fast_live_module_flip() -> None:
    """A fast live run whose branch selects a different module diverges typed."""

    class ModuleBranch(nn.Module):
        """Branch between two distinct atomic modules on the input sign."""

        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(4, 4)
            self.b = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if bool(x.sum() > 0):
                return self.a(x)
            return self.b(x)

    model = ModuleBranch().eval()
    live = tl.trace(model, _branch_input(), save=tl.in_module("a"))
    same = live.run(inputs=_branch_input(), fast=True)
    assert same.report.path_faithfulness is PathFaithfulness.VERIFIED
    with pytest.raises(PathDivergenceError) as caught:
        live.run(inputs=-_branch_input(), fast=True)
    assert caught.value.fields["code"] == RunnableErrorCode.CONDITIONAL_ARM_DIVERGENCE.value


# ---------------------------------------------------------------------------
# Poisoned-run refusal.
# ---------------------------------------------------------------------------


def test_poisoned_run_refused_on_faithful_consumers(tmp_path: Path) -> None:
    """A return_diverged poisoned Trace refuses export and tabular consumption typed."""

    loaded = tl.load(_save_branch_artifact(tmp_path, "poison.tlspec"))
    result = loaded.run(inputs=-_branch_input(), on_divergence="return_diverged")
    assert result.report.poisoned
    with pytest.raises(PoisonedRunError) as export_caught:
        tl.save(result.trace, tmp_path / "poisoned-export.tlspec")
    assert export_caught.value.fields["code"] == RunnableErrorCode.POISONED_RUN_REFUSED.value
    with pytest.raises(PoisonedRunError) as pandas_caught:
        result.trace.to_pandas()
    assert pandas_caught.value.fields["code"] == RunnableErrorCode.POISONED_RUN_REFUSED.value


# ---------------------------------------------------------------------------
# B8-33 run-time half: analysis-only loads refuse run() typed.
# ---------------------------------------------------------------------------


def test_run_capability_unavailable_on_analysis_only_load(tmp_path: Path) -> None:
    """An analysis-level artifact loads fine but refuses ``run()`` typed."""

    path = tmp_path / "analysis.tlspec"
    trace = tl.trace(nn.Linear(4, 4).eval(), torch.randn(2, 4), capture=_capture_options())
    tl.save(trace, path)
    loaded = tl.load(path)
    with pytest.raises(RunCapabilityUnavailableError) as caught:
        loaded.run(inputs=torch.randn(2, 4))
    assert caught.value.fields["code"] == RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value


# ---------------------------------------------------------------------------
# R25-4: the unresolved alias ceiling code is observable on the report.
# ---------------------------------------------------------------------------


def test_input_alias_topology_unresolved_code_surfaces_on_report(tmp_path: Path) -> None:
    """The unresolved-topology ceiling surfaces its documented code on the report.

    Same-storage congruent-stride views above the enumeration cap defeat the
    three-valued alias engine (relation ``unknown``): the verdict is capped at
    UNVERIFIABLE and the report must carry a PASSED disclosure row naming
    ``input_alias_topology_unresolved`` -- previously an unnamed boolean no
    caller could branch on.
    """

    class TwoInput(nn.Module):
        """Consume two tensor inputs without mutation."""

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a[:8].sum() + b[:8].sum()

    from torchlens.errors import TorchLensCaptureGapWarning

    base = torch.randn(3 * 70000 + 4)
    a = base[0::3][:69000]
    b = base[3::3][:69000]
    path = tmp_path / "alias-unknown.tlspec"
    # The unprovable same-storage topology already discloses a capture gap at
    # capture time (input_copy_semantics_unverifiable) -- expected here, since the
    # unresolved relation is exactly what this provocation constructs.
    with pytest.warns(TorchLensCaptureGapWarning):
        trace = tl.trace(TwoInput().eval(), (a, b), capture=_capture_options())
    trace.save(path, level="runnable")
    result = tl.load(path).run(inputs=(a, b))
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    rows = [
        check
        for check in result.report.contract_checks
        if check.name == RunnableErrorCode.INPUT_ALIAS_TOPOLOGY_UNRESOLVED.value
    ]
    assert len(rows) == 1
    disclosure = rows[0]
    # A ceiling is not a contradiction: the row PASSES (a failed check would
    # wrongly classify DIVERGED) while its diagnostic names the documented code.
    assert disclosure.passed
    assert disclosure.diagnostic is not None
    assert disclosure.diagnostic.code is RunnableErrorCode.INPUT_ALIAS_TOPOLOGY_UNRESOLVED
