"""Pre-admission host-capacity refusal tests.

The load-bearing case is the real Mixtral-8x7B proposal that OOM-killed a worker
on the 16 GiB host: its recipe kwargs are reproduced verbatim below, and the
check must refuse it without importing transformers or allocating a byte.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.host_capacity import (
    BYTES_PER_PARAMETER,
    CAPACITY_DEFERRAL_DISPOSITION,
    HOST_MEMORY_ENV_VAR,
    OVERCOMMIT_ALLOWANCE,
    CapacityDeferralError,
    CapacityVerdict,
    HostCapacity,
    append_capacity_deferral_row,
    assess_model_capacity,
    build_capacity_deferral_row,
    capacity_deferral_path,
    estimate_parameter_count,
    host_capacity,
    load_capacity_deferral_rows,
)

#: 16 GiB, the host this refusal was built for.
SMALL_HOST = HostCapacity(physical_memory_bytes=16 * 2**30, memory_source="test")

#: The exact ``library_recipe`` from the m5915 Mixtral 8x7B author result whose
#: worker was terminated by signal 9 after roughly two minutes of allocation.
MIXTRAL_RECIPE: dict[str, Any] = {
    "distribution": "transformers",
    "entrypoint": None,
    "kwargs": {
        "config": {
            "__construct__": {
                "kwargs": {
                    "attention_dropout": 0.0,
                    "head_dim": 128,
                    "hidden_act": "silu",
                    "hidden_size": 4096,
                    "intermediate_size": 14336,
                    "max_position_embeddings": 32768,
                    "num_attention_heads": 32,
                    "num_experts_per_tok": 2,
                    "num_hidden_layers": 32,
                    "num_key_value_heads": 8,
                    "num_local_experts": 8,
                    "output_router_logits": False,
                    "rms_norm_eps": 1e-05,
                    "rope_theta": 1000000.0,
                    "router_aux_loss_coef": 0.001,
                    "router_jitter_noise": 0.0,
                    "sliding_window": None,
                    "tie_word_embeddings": False,
                    "use_cache": False,
                    "vocab_size": 32000,
                },
                "module": "transformers",
                "symbol": "MixtralConfig",
            }
        }
    },
    "module": "transformers",
    "pretrained_disable_fields": [],
    "symbol": "MixtralForCausalLM",
    "version": "4.57.1",
}

#: A real small library recipe (m4334, ``timm.models.dla.dla60``). A named
#: library symbol declares no size at all, so no bound is derivable and the
#: model must be admitted.
NAMED_SYMBOL_RECIPE: dict[str, Any] = {
    "distribution": "timm",
    "kwargs": {"pretrained": False},
    "module": "timm.models.dla",
    "pretrained_disable_fields": ["pretrained"],
    "symbol": "dla60",
    "version": "1.0.28",
}


def test_mixtral_is_refused_before_any_allocation() -> None:
    """The real Mixtral proposal is deferred on a 16 GiB host."""

    assessment = assess_model_capacity(MIXTRAL_RECIPE, host=SMALL_HOST)
    assert assessment.verdict is CapacityVerdict.DEFER
    assert assessment.deferred
    estimate = assessment.estimate
    assert estimate is not None
    # A strict lower bound: Mixtral 8x7B really has about 46.5B parameters, and
    # the bound understates that because it counts two feed-forward projections
    # per expert where the real gated block has three.
    assert 30e9 < estimate.parameter_count_lower_bound < 46.5e9
    assert estimate.parameter_count_lower_bound > SMALL_HOST.admissible_parameter_ceiling
    # The refusal states both halves of the decision it made.
    payload = assessment.to_json()
    assert payload["estimate"]["parameter_count_lower_bound"] > 0
    assert payload["threshold"]["host_physical_memory_bytes"] == 16 * 2**30
    assert payload["threshold"]["overcommit_allowance"] == OVERCOMMIT_ALLOWANCE
    assert payload["overage_ratio"] > 1.0


def test_estimate_locates_the_nested_construct_config() -> None:
    """The bound is derived from the recipe's own nested constructor kwargs."""

    estimate = estimate_parameter_count(MIXTRAL_RECIPE)
    assert estimate is not None
    assert estimate.config_path.endswith("kwargs.config.__construct__.kwargs")
    assert estimate.terms["hidden_size"]["value"] == 4096
    assert estimate.terms["depth"]["value"] == 32
    assert estimate.terms["feedforward_per_layer"]["experts"] == 8
    assert estimate.terms["embedding"]["untied_output_head"] is True


def test_the_same_model_is_admitted_on_a_bigger_host() -> None:
    """The ceiling is a host figure, so bigger hardware admits without a code change."""

    big_host = HostCapacity(physical_memory_bytes=512 * 2**30, memory_source="test")
    assessment = assess_model_capacity(MIXTRAL_RECIPE, host=big_host)
    assert assessment.verdict is CapacityVerdict.ADMIT
    assert not assessment.deferred


def test_named_library_symbol_is_admitted_and_says_so() -> None:
    """A recipe with no declared layer stack is admitted, not silently sized."""

    assessment = assess_model_capacity(NAMED_SYMBOL_RECIPE, host=SMALL_HOST)
    assert assessment.verdict is CapacityVerdict.NOT_DERIVABLE
    assert not assessment.deferred
    assert assessment.estimate is None
    assert "no parameter bound could be derived" in assessment.explanation


@pytest.mark.parametrize(
    "recipe",
    [
        {},
        {"kwargs": {}},
        {"kwargs": {"hidden_size": 4096}},
        {"kwargs": {"num_hidden_layers": 32}},
        None,
        [1, 2, 3],
    ],
)
def test_unsizeable_recipes_never_refuse(recipe: Any) -> None:
    """An unknown size is never treated as a large size."""

    assessment = assess_model_capacity(recipe, host=SMALL_HOST)
    assert assessment.verdict is CapacityVerdict.NOT_DERIVABLE
    assert not assessment.deferred


def test_models_at_the_practical_ceiling_are_still_attempted() -> None:
    """A 7B-class dense model is still attempted on 16 GiB.

    The practical instantiation ceiling on this host is near 4B parameters, so
    this model very probably will NOT fit and will OOM. It is attempted anyway:
    the refusal threshold sits at roughly 8.6B parameters, and the whole band
    between "probably will not fit" and "is refused unseen" is deliberately left
    to the run. An OOM costs one wasted run; a false refusal costs a catalog
    entry forever, because the authoring stage runs once per model.
    """

    # Llama-7B geometry: hidden 4096, 32 layers, ffn 11008, vocab 32000 tied.
    recipe = {
        "kwargs": {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
            "vocab_size": 32000,
        }
    }
    estimate = estimate_parameter_count(recipe)
    assert estimate is not None
    # A lower bound below the model's real ~6.7B, and below the 8.59B ceiling.
    assert 5e9 < estimate.parameter_count_lower_bound < 6.7e9
    assert estimate.parameter_count_lower_bound < SMALL_HOST.admissible_parameter_ceiling
    assert assess_model_capacity(recipe, host=SMALL_HOST).verdict is CapacityVerdict.ADMIT


def test_host_capacity_reads_the_machine() -> None:
    """The host figure comes from the machine, not a hardcoded constant."""

    host = host_capacity(environ={})
    assert host.physical_memory_bytes > 0
    assert host.memory_source in {
        "sysconf:SC_PHYS_PAGES*SC_PAGE_SIZE",
        "sysctl:hw.memsize",
    }
    assert host.admissible_parameter_bytes == int(
        host.physical_memory_bytes * OVERCOMMIT_ALLOWANCE
    )
    assert (
        host.admissible_parameter_ceiling
        == host.admissible_parameter_bytes // BYTES_PER_PARAMETER
    )


def test_host_capacity_honours_the_operator_override() -> None:
    """An explicit host declaration is used verbatim and is attributed."""

    host = host_capacity(environ={HOST_MEMORY_ENV_VAR: str(1024 * 2**30)})
    assert host.physical_memory_bytes == 1024 * 2**30
    assert host.memory_source == f"env:{HOST_MEMORY_ENV_VAR}"


@pytest.mark.parametrize("value", ["", "0", "-1", "sixteen"])
def test_host_capacity_refuses_a_malformed_override(value: str) -> None:
    """A malformed host declaration fails loudly instead of silently defaulting."""

    with pytest.raises(CapacityDeferralError):
        host_capacity(environ={HOST_MEMORY_ENV_VAR: value})


def _mixtral_row() -> dict[str, Any]:
    """Return a complete deferral row for the real Mixtral assessment."""

    return build_capacity_deferral_row(
        stable_id="m5915",
        work_id="work-m5915",
        name="Mixtral 8x7B",
        campaign_id="c1-mech",
        run_id="run-1",
        machine_id="mymini",
        created_at="2026-08-04T02:52:37.527339Z",
        assessment=assess_model_capacity(MIXTRAL_RECIPE, host=SMALL_HOST),
    )


def test_deferral_row_carries_its_estimate_and_threshold() -> None:
    """The record is self-describing enough to re-judge the call later."""

    row = _mixtral_row()
    assert row["disposition"] == CAPACITY_DEFERRAL_DISPOSITION
    assert row["capacity"]["estimate"]["parameter_count_lower_bound"] > 0
    assert row["capacity"]["threshold"]["admissible_parameter_ceiling"] > 0
    assert "physical memory" in row["recheck_hint"]
    assert row["row_sha256"].startswith("sha256:")


def test_only_a_refusal_may_be_recorded_as_a_deferral() -> None:
    """An admitting assessment cannot be written to the deferral ledger."""

    with pytest.raises(CapacityDeferralError):
        build_capacity_deferral_row(
            stable_id="m4334",
            work_id="work-m4334",
            name="DLA-60",
            campaign_id="c1-mech",
            run_id="run-1",
            machine_id="mymini",
            created_at="2026-08-04T02:52:37.527339Z",
            assessment=assess_model_capacity(NAMED_SYMBOL_RECIPE, host=SMALL_HOST),
        )


def test_deferral_ledger_appends_idempotently(tmp_path: Path) -> None:
    """Re-deriving the same refusal on a resume is a no-op, not a conflict."""

    path = capacity_deferral_path(tmp_path)
    row = _mixtral_row()
    assert append_capacity_deferral_row(path, row) == row
    assert append_capacity_deferral_row(path, row) == row
    loaded = load_capacity_deferral_rows([path])
    assert len(loaded) == 1
    assert loaded[0]["stable_id"] == "m5915"
    assert loaded[0]["disposition"] == CAPACITY_DEFERRAL_DISPOSITION


def test_deferral_ledger_refuses_a_conflicting_row(tmp_path: Path) -> None:
    """One work generation cannot carry two different capacity verdicts."""

    path = capacity_deferral_path(tmp_path)
    append_capacity_deferral_row(path, _mixtral_row())
    conflicting = build_capacity_deferral_row(
        stable_id="m5915",
        work_id="work-m5915",
        name="Mixtral 8x7B",
        campaign_id="c1-mech",
        run_id="run-2",
        machine_id="mymini",
        created_at="2026-08-04T03:52:37.527339Z",
        assessment=assess_model_capacity(MIXTRAL_RECIPE, host=SMALL_HOST),
    )
    with pytest.raises(CapacityDeferralError):
        append_capacity_deferral_row(path, conflicting)


def test_deferral_ledger_refuses_a_tampered_row(tmp_path: Path) -> None:
    """A row whose estimate was edited after the fact does not load."""

    path = capacity_deferral_path(tmp_path)
    row = _mixtral_row()
    append_capacity_deferral_row(path, row)
    tampered = path.read_text(encoding="utf-8").replace(
        '"verdict":"defer"', '"verdict":"admit"'
    )
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(CapacityDeferralError):
        load_capacity_deferral_rows([path])
