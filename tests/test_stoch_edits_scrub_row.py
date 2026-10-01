"""F02 row gate: the causal-scrubbing expressibility row (edits memo B1, offline analog).

The flagship gallery row scaled to the R0 fixture discipline: a REAL
config-built GPT-2 (the sprint's zero-network realism fixture -- the same
2-layer/64-dim `GPT2LMHeadModel` the R0 gate runs), >=6 population examples
over 2 prompt "templates", agreement-conditioned donors at a pass-qualified
residual site drawn from a SECOND set of captured runs through
``tl.reference`` + ``tl.sample_from`` + ``tl.patch_from``:

- no cross-template donor is ever drawn (the agreement law);
- whole-event coherence: the patched value IS one donor run's recorded value
  at the same site, bit-exact;
- the same seed reproduces donor ids AND downstream logits byte-identically;
- a different seed changes the realized draw;
- the scrub changes the output logits (a real effect, never "did not crash");
- class-size and audit retention pins (envelope + ACT row + the sampling
  disclosure riding FireRecord.determinism_note through the ONE builder).

The pretrained-checkpoint leg (openai-community/gpt2 with a calibrated
%-loss-recovered band, both transformers locks) is the Tier-B release row and
runs in the R1/network venue, not at this gate. Credits: Redwood Research's
causal scrubbing -- the agreement-conditioned resample this family serves.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

import torchlens as tl
from tests.real_model.r0.families import _token_ids, build_gpt2
from torchlens.intervention import OneDatum, reference, sample_from, sampling_records

# slow: each leg captures a real (config-built) GPT-2 subject plus six donor
# runs, ~40-45s measured -- honestly tiered above the smoke/heavy budgets.
pytestmark = [pytest.mark.real_model, pytest.mark.slow]


def _capture(model: torch.nn.Module, ids: torch.Tensor) -> tl.Trace:
    return tl.trace(
        model,
        ids,
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )


@pytest.fixture(scope="module")
def scrub_world() -> Any:
    """One GPT-2, one subject run, six donor runs over two templates."""

    model = build_gpt2("eager")
    base_ids = _token_ids()
    subject_ids = base_ids.clone()
    donors: list[tl.Trace] = []
    datums: list[dict[str, str]] = []
    generator = torch.Generator().manual_seed(99)
    for index in range(6):
        template = "A" if index < 3 else "B"
        ids = base_ids.clone()
        # Template A perturbs the tail tokens, template B the head tokens --
        # two families of prompts with a shared skeleton.
        span = slice(5, 8) if template == "A" else slice(0, 3)
        ids[0, span] = torch.randint(0, 512, (3,), generator=generator)
        donors.append(_capture(model, ids))
        datums.append({"template": template})
    subject = _capture(model, subject_ids)
    try:
        yield {"model": model, "subject": subject, "donors": donors, "datums": datums}
    finally:
        subject.cleanup()
        for donor in donors:
            donor.cleanup()


def _residual_site(log: tl.Trace) -> str:
    """Pick a stable mid-network residual-stream site (an `add` op label)."""

    adds = [label for label in log.layer_labels if label.startswith("add")]
    assert adds, "the GPT-2 capture must expose residual add sites"
    return adds[len(adds) // 2]


def test_causal_scrub_row_agreement_seed_and_audit(scrub_world: dict[str, object]) -> None:
    """The expressibility row: conditioned donors, honest draws, full audit."""

    subject: tl.Trace = scrub_world["subject"]  # type: ignore[assignment]
    donors: list[tl.Trace] = scrub_world["donors"]  # type: ignore[assignment]
    datums: list[dict[str, str]] = scrub_world["datums"]  # type: ignore[assignment]
    site = _residual_site(subject)
    baseline_site = subject[site].out
    baseline_logits = subject.output_ops[0].out if subject.output_ops else None

    population = reference(
        donors,
        origin="six donor runs, two prompt templates, config-built gpt2 (R0 fixture)",
        data=datums,
    )
    plan = sample_from(
        population,
        agree_on=lambda datum: datum["template"],
        matching=OneDatum({"template": "A"}),
        seed=1234,
    )
    fork = subject.fork()
    fork.do(tl.label(site), tl.patch_from(plan))
    record = sampling_records(fork)[-1]

    # Agreement law: the donor comes from template A's class, never B's.
    assert record["eligible_count"] == 3, "class-size pin: three template-A donors"
    (donor_index,) = record["donor_ids"]
    assert donor_index in (0, 1, 2), "no cross-template donor, ever"

    # Whole-event coherence: the patched value IS the donor run's recorded
    # value at this site, bit-exact.
    donor_value = donors[donor_index][site].out
    assert torch.equal(fork[site].out, donor_value)
    assert not torch.equal(fork[site].out, baseline_site), "a real scrub, not a no-op"

    # The scrub has a real downstream effect on the logits.
    if baseline_logits is not None and fork.output_ops:
        assert not torch.equal(fork.output_ops[0].out, baseline_logits)

    # Same seed -> same donor AND byte-identical downstream values.
    fork_again = subject.fork()
    fork_again.do(
        tl.label(site),
        tl.patch_from(
            sample_from(
                population,
                agree_on=lambda datum: datum["template"],
                matching=OneDatum({"template": "A"}),
                seed=1234,
            )
        ),
    )
    again = sampling_records(fork_again)[-1]
    assert again["donor_ids"] == record["donor_ids"]
    assert torch.equal(fork_again[site].out, fork[site].out)

    # A different seed changes the realized draw (derived seed always; with
    # three eligible donors the id may collide, the derivation must not).
    fork_other = subject.fork()
    fork_other.do(
        tl.label(site),
        tl.patch_from(
            sample_from(
                population,
                agree_on=lambda datum: datum["template"],
                matching=OneDatum({"template": "A"}),
                seed=4321,
            )
        ),
    )
    other = sampling_records(fork_other)[-1]
    assert other["derived_seed"] != record["derived_seed"]

    # Audit retention: one envelope, one canonical row, the sampling note on
    # the FireRecord (the ONE builder's lift), digest-kind disclosure.
    envelopes = [
        row
        for row in fork.state_history
        if isinstance(row, dict) and row.get("op") == "intervention_event"
    ]
    assert len(envelopes) == 1 and envelopes[0]["status"] == "fired"
    fires = list(fork.ops[site].interventions or ())
    assert fires and "sampling[patch_from]" in (fires[-1].determinism_note or "")
    assert record["digest_kind"] == "address"
    assert record["population_count"] == 6


def test_recursive_post_scrub_donor_works(scrub_world: dict[str, object]) -> None:
    """B1's recursive leg: a scrubbed fork serves as the donor for a second scrub."""

    subject: tl.Trace = scrub_world["subject"]  # type: ignore[assignment]
    donors: list[tl.Trace] = scrub_world["donors"]  # type: ignore[assignment]
    site = _residual_site(subject)
    first = subject.fork()
    first.do(
        tl.label(site),
        tl.patch_from(
            sample_from(
                reference(donors[:3], origin="template-A donors"),
                seed=7,
            )
        ),
    )
    # The scrubbed FORK is itself a legitimate single-member population; a
    # class of one fires the D15 singleton disclosure (a deterministic patch
    # wearing a sampler's name -- the reviewer must SEE it).
    second = subject.fork()
    with pytest.warns(Warning, match="agreement class of size 1"):
        second.do(
            tl.label(site),
            tl.patch_from(sample_from(reference([first], origin="post-scrub donor"), seed=8)),
        )
    assert torch.equal(second[site].out, first[site].out)
