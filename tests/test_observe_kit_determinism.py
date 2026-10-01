"""Observe-kit item 9: check_determinism -- one controlled verdict, cost disclosed.

The default answers exactly one question (do N isolated same-seed runs
agree?) with a three-state verdict that never generalizes to "deterministic";
seed sensitivity is a separately-requested second verdict from exactly one
extra disclosed run.
"""

from __future__ import annotations

import random

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.debug import check_determinism

pytestmark = pytest.mark.smoke


class _DropoutModel(nn.Module):
    """Linear + dropout: same-seed repeatable, different-seed sensitive."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the RNG-consuming dropout after the linear layer."""

        return self.dropout(self.linear(x))


class _HostEntropyModel(nn.Module):
    """Consumes OS entropy inside forward: never same-seed repeatable."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix non-seeded host entropy into the output."""

        import secrets

        return x + float(secrets.randbelow(1_000_000)) / 1_000_000.0


class _BatchNormModel(nn.Module):
    """Train-mode BatchNorm: buffer writes must not read as nondeterminism."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(4)
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run BatchNorm in whatever mode the module carries."""

        return self.linear(self.norm(x))


def test_eval_model_is_repeatable_and_caller_state_untouched() -> None:
    """Deterministic eval forward: repeatable_under_test; caller RNG intact."""

    model = _DropoutModel()
    model.eval()
    x = torch.randn(2, 4)
    torch.manual_seed(31)
    random.seed(31)
    torch_state = torch.get_rng_state()
    python_state = random.getstate()
    report = check_determinism(model, x, runs=2, seed=5)
    assert report.verdict == "repeatable_under_test"
    assert report.runs == 2
    assert report.policy == "exact"
    assert report.model_mode == "eval"
    assert report.seed_sensitivity_verdict is None
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    witness_names = [name for name, _value in report.environment_witnesses]
    assert "torch.backends.cudnn.benchmark" in witness_names
    assert "CUBLAS_WORKSPACE_CONFIG" in witness_names


def test_train_dropout_repeats_and_names_the_dropout_op_itself() -> None:
    """Same-seed dropout REPEATS; the RNG consumer list names dropout, not its successor."""

    model = _DropoutModel()
    model.train()
    report = check_determinism(model, torch.randn(2, 4), runs=2, seed=7)
    assert report.verdict == "repeatable_under_test"
    assert any(label.startswith("dropout") for label in report.rng_consuming_ops), (
        report.rng_consuming_ops
    )
    assert report.rng_attribution_basis in (
        "pre_op_rng_snapshots",
        "stochastic_func_names",
    )


def test_batchnorm_train_buffer_writes_are_not_nondeterminism() -> None:
    """Per-run fresh deepcopy isolates BatchNorm buffers across runs."""

    model = _BatchNormModel()
    model.train()
    report = check_determinism(model, torch.randn(4, 4), runs=3, seed=11)
    assert report.verdict == "repeatable_under_test"


def test_host_entropy_observed_as_divergence() -> None:
    """OS-entropy consumption inside forward yields an observed divergence."""

    report = check_determinism(_HostEntropyModel(), torch.randn(2, 4), runs=2, seed=3)
    assert report.verdict == "nondeterminism_observed"
    assert report.first_divergence is not None
    shared = report.first_bad_thing
    assert shared.tool == "check_determinism"
    assert shared.kind == "nondeterminism"
    assert shared.detection_basis == "double_run"


def test_seed_sensitivity_is_optin_with_disclosed_extra_run() -> None:
    """seed_sensitivity=True adds ONE run and a SECOND named verdict."""

    model = _DropoutModel()
    model.train()
    report = check_determinism(model, torch.randn(2, 4), runs=2, seed=7, seed_sensitivity=True)
    assert report.verdict == "repeatable_under_test"
    assert report.seed_sensitivity_verdict == "sensitive_to_seed"
    assert "3 captures executed in total" in report.message

    eval_model = _DropoutModel()
    eval_model.eval()
    eval_report = check_determinism(
        eval_model, torch.randn(2, 4), runs=2, seed=7, seed_sensitivity=True
    )
    assert eval_report.seed_sensitivity_verdict == "insensitive_under_test"


def test_same_run_twice_self_check() -> None:
    """Harness ACCEPTANCE: the harness itself never manufactures divergence.

    Two isolated captures of a deterministic model compared through the same
    machinery agree exactly -- this is the self-check as an acceptance test,
    not a hidden per-call calibration run.
    """

    model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2))
    report = check_determinism(model, torch.randn(3, 4), runs=4, seed=0)
    assert report.verdict == "repeatable_under_test"
    assert report.divergent_ops == ()


def test_already_traced_model_works() -> None:
    """The deepcopy-after-trace KeyError class never reaches this tool."""

    model = _DropoutModel()
    model.eval()
    x = torch.randn(2, 4)
    prior = tl.trace(model, x)
    try:
        report = check_determinism(model, x, runs=2, seed=1)
        assert report.verdict == "repeatable_under_test"
    finally:
        prior.cleanup()


def test_runs_below_two_refuse_typed() -> None:
    """runs=1 cannot support a repeatability claim; the refusal teaches."""

    from torchlens._errors import InvalidArgumentError

    with pytest.raises(InvalidArgumentError) as excinfo:
        check_determinism(_DropoutModel(), torch.randn(2, 4), runs=1)
    assert excinfo.value.fields["code"] == "determinism_runs_invalid"


def test_undeepcopyable_model_is_inconclusive_with_factory_remedy() -> None:
    """A deepcopy-refusing model returns inconclusive, never a crash."""

    class _Undeepcopyable(nn.Module):
        def __deepcopy__(self, memo: dict) -> nn.Module:
            raise RuntimeError("this model refuses deepcopy")

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Identity forward."""

            return x

    report = check_determinism(_Undeepcopyable(), torch.randn(2, 4), runs=2, seed=0)
    assert report.verdict == "inconclusive"
    assert any("model_factory" in remedy for remedy in report.remedies)
