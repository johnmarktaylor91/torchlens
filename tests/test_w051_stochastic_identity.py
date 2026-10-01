"""W051-STOCH / AUD-CODE 2.1: trace-backed population identity is a pure function of the captures.

The derived-seed law folds ``population_identity`` into every draw, so an
``id()``-salted address identity silently broke rerun reproducibility for
every trace-backed population. The address is now built from persisted
capture facts plus per-op geometry; ``digest_population=True`` hashes the
retained payloads it stamps ``content`` for, and refuses when there are none.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.intervention import reference, sample_from, sampling_records


class _TwoBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.b(torch.relu(self.a(x))))


def _capture(model: nn.Module, seed: int, **kwargs: object) -> tl.Trace:
    torch.manual_seed(seed)
    return tl.trace(
        model,
        torch.randn(5, 4),
        capture=tl.options.CaptureOptions(intervention_ready=True, random_seed=seed, **kwargs),
    )


def _fixture() -> tuple[tl.Trace, list[tl.Trace]]:
    torch.manual_seed(0)
    model = _TwoBlock()
    subject = _capture(model, 100)
    donors = [_capture(model, seed) for seed in (1, 2, 3)]
    return subject, donors


@pytest.mark.smoke
def test_address_identity_ignores_object_identity() -> None:
    """Two populations over EQUAL captures share one identity; the object address never enters."""

    subject, donors = _fixture()
    first = reference(donors, origin="d")
    torch.manual_seed(0)
    fresh_model = _TwoBlock()
    fresh_donors = [_capture(fresh_model, seed) for seed in (1, 2, 3)]
    second = reference(fresh_donors, origin="d")
    assert first.digest_kind == "address"
    assert first.population_identity == second.population_identity
    plan_a = sample_from(first, seed=3)
    plan_b = sample_from(second, seed=3)
    assert plan_a.donor_group_id == plan_b.donor_group_id
    fork_a = subject.fork()
    fork_a.do(tl.label("relu_1_2"), tl.patch_from(plan_a))
    fork_b = subject.fork()
    fork_b.do(tl.label("relu_1_2"), tl.patch_from(plan_b))
    assert sampling_records(fork_a)[-1]["donor_ids"] == sampling_records(fork_b)[-1]["donor_ids"]


@pytest.mark.smoke
def test_content_digest_hashes_retained_payloads() -> None:
    """Same facts + same geometry + DIFFERENT values: one address, two content identities."""

    torch.manual_seed(0)
    model = _TwoBlock()
    torch.manual_seed(1)
    run_one = tl.trace(model, torch.randn(5, 4), capture=tl.options.CaptureOptions(random_seed=7))
    torch.manual_seed(2)
    run_two = tl.trace(model, torch.randn(5, 4), capture=tl.options.CaptureOptions(random_seed=7))
    address_one = reference([run_one], origin="o")
    address_two = reference([run_two], origin="o")
    assert address_one.population_identity == address_two.population_identity
    content_one = reference([run_one], origin="o", digest_population=True)
    content_two = reference([run_two], origin="o", digest_population=True)
    assert content_one.digest_kind == content_two.digest_kind == "content"
    assert content_one.population_identity != content_two.population_identity
    again = reference([run_one], origin="o", digest_population=True)
    assert again.content_digest == content_one.content_digest


@pytest.mark.smoke
def test_content_digest_refuses_without_retained_payloads() -> None:
    """The content stamp must be earned: no retained payload anywhere refuses typed."""

    torch.manual_seed(0)
    model = _TwoBlock()
    with pytest.warns(UserWarning, match="matched zero sites"):
        unsaved = tl.trace(model, torch.randn(5, 4), save=tl.func("tanh"))
    assert not any(op.has_saved_activation for op in unsaved.ops)
    with pytest.raises(InvalidArgumentError) as info:
        reference([unsaved], origin="o", digest_population=True)
    assert info.value.fields["code"] == "population_digest_unavailable"
    # The address identity stays available for the same capture.
    assert reference([unsaved], origin="o").digest_kind == "address"


_TWO_PROCESS_SCRIPT = textwrap.dedent(
    """
    import torch
    from torch import nn
    import torchlens as tl
    from torchlens.intervention import reference, sample_from, sampling_records

    class TwoBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 4)
            self.b = nn.Linear(4, 4)
        def forward(self, x):
            return torch.relu(self.b(torch.relu(self.a(x))))

    def capture(model, seed):
        torch.manual_seed(seed)
        return tl.trace(model, torch.randn(5, 4), capture=tl.options.CaptureOptions(
            intervention_ready=True, random_seed=seed))

    torch.manual_seed(0)
    model = TwoBlock()
    subject = capture(model, 100)
    donors = [capture(model, seed) for seed in (1, 2, 3)]
    for digest in (False, True):
        ref = reference(donors, origin="d", digest_population=digest)
        plan = sample_from(ref, seed=3)
        fork = subject.fork()
        fork.do(tl.label("relu_1_2"), tl.patch_from(plan))
        record = sampling_records(fork)[-1]
        print(ref.digest_kind, ref.population_identity, plan.donor_group_id,
              record["derived_seed"], record["donor_ids"])
    """
)


@pytest.mark.heavy
def test_two_process_golden_reproduces_every_draw() -> None:
    """The flagship claim, measured: two processes with different hash salts draw identically."""

    root = os.path.dirname(os.path.dirname(os.path.abspath(tl.__file__)))
    outputs = []
    for hash_seed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONPATH=root)
        completed = subprocess.run(
            [sys.executable, "-c", _TWO_PROCESS_SCRIPT],
            capture_output=True,
            text=True,
            env=env,
            check=True,
            timeout=240,
        )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        assert len(lines) == 2, completed.stdout
        outputs.append(lines)
    assert outputs[0] == outputs[1], f"process-salted draw: {outputs}"
    assert outputs[0][0].startswith("address ") and outputs[0][1].startswith("content ")
