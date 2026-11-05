"""interp-engine point addresses resolve to the same sites as TransformerLens names.

interp-engine names a site by a canonical address (``resid_post.5``,
``embeddings``). Each ``interp-engine`` translation row is paired here with the
TransformerLens hook name interp-engine itself maps that point to (its
``mappers`` table at the pinned version); on a real-class tiny GPT-2 (plain
MLP, learned positions) and a tiny Llama (gated MLP, rotary) both names must
resolve to the same module, facet, site key and pass index, and serve the same
captured tensor. The engine's points with no torchlens site, this trace's sites
with no engine point, and the written points file are checked against a saved
copy of the engine's point format (``tests/fixtures/interp_engine_points_v1_14_3.json``).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

import torchlens as tl
import torchlens.mechinterp as mi
from torchlens.mechinterp._aliases import _package_json
from torchlens.mechinterp._heads import _attention_modules
from torchlens.mechinterp._interp_engine import _write_interp_engine_points

sys.path.insert(0, str(Path(__file__).resolve().parent / "real_model" / "r0"))

GENERATION = "interp-engine"
LIVE = ("real", "reconstructed_read_only")
MODELS = ("gpt2", "llama")
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "interp_engine_points_v1_14_3.json"

#: interp-engine point -> the TransformerLens name interp-engine maps it to.
POINT_PAIRS: tuple[tuple[str, str], ...] = (
    ("embeddings", "hook_embed"),
    ("resid_pre.{L}", "blocks.{L}.hook_resid_pre"),
    ("attn_in.{L}", "blocks.{L}.attn.hook_in"),
    ("value.{L}", "blocks.{L}.attn.hook_v"),
    ("attn_scores.{L}", "blocks.{L}.attn.hook_attn_scores"),
    ("attn_probs.{L}", "blocks.{L}.attn.hook_pattern"),
    ("z.{L}", "blocks.{L}.attn.hook_z"),
    ("attn_out.{L}", "blocks.{L}.attn.hook_out"),
    ("resid_mid.{L}", "blocks.{L}.hook_resid_mid"),
    ("mlp_in.{L}", "blocks.{L}.mlp.hook_in"),
    ("mlp_pre.{L}", "blocks.{L}.mlp.hook_pre"),
    ("mlp_pre_linear.{L}", "blocks.{L}.mlp.hook_pre_linear"),
    ("mlp_act.{L}", "blocks.{L}.mlp.hook_post"),
    ("mlp_out.{L}", "blocks.{L}.mlp.hook_out"),
    ("resid_post.{L}", "blocks.{L}.hook_resid_post"),
    # interp-engine gives lm_head no TransformerLens name (logits are returned).
    ("lm_head", "logits"),
)
#: Points that must be live on both tiny models (the residual and sublayer boundaries).
CORE = ("embeddings", "resid_pre", "attn_in", "attn_out", "resid_mid", "mlp_in", "mlp_act")
CORE += ("mlp_out", "resid_post", "lm_head")


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _capture(builder_name: str):
    import families

    model = getattr(families, builder_name)("eager")
    torch.manual_seed(4)
    x = torch.randint(0, 500, (1, 6))
    return model, tl.trace(model, x, save="all")


@pytest.fixture(scope="module")
def traces():
    """Traces keyed by model; the models stay referenced for the module's life."""

    built = {name: _capture(f"build_{name}") for name in MODELS}
    try:
        yield {name: trace for name, (_, trace) in built.items()}
    finally:
        for _, trace in built.values():
            trace.cleanup()


def _names(pattern: str, n_layers: int) -> list[str]:
    if "{L}" not in pattern:
        return [pattern]
    return [pattern.replace("{L}", str(layer)) for layer in range(n_layers)]


def _payload(trace, resolution) -> torch.Tensor:
    payload = trace.modules[resolution.module_address].facets[resolution.facet]
    value = getattr(payload, "value", payload)
    assert isinstance(value, torch.Tensor), resolution.query
    return value


def _coordinates(resolution) -> tuple:
    return (
        resolution.status,
        resolution.module_address,
        resolution.facet,
        resolution.site_key,
        resolution.pass_index,
    )


def _parse(text: str, grammar: dict[str, str], fields: list[str]) -> dict[str, Any]:
    """interp-engine's parse_address over the saved grammar (point addresses only)."""

    head, *rest = text.split(".")
    assert re.match(grammar["name"], head) and head != grammar["module_path_prefix"], text
    parsed: dict[str, Any] = dict.fromkeys(fields)
    parsed["name"] = head
    if rest and re.match(grammar["layer"], rest[0]):
        parsed["layer"] = int(rest.pop(0))
    for token in rest:
        match = re.match(grammar["coordinate"], token)
        assert match and match.group(1) in fields[2:], (text, token)
        parsed[match.group(1)] = int(match.group(2))
    return parsed


@pytest.mark.parametrize("model", MODELS)
def test_interp_engine_point_resolves_to_the_transformerlens_site(traces, model):
    trace = traces[model]
    n_layers = len(_attention_modules(trace))
    assert n_layers == 2
    live = set()
    for point_pattern, tlens_pattern in POINT_PAIRS:
        pairs = zip(_names(point_pattern, n_layers), _names(tlens_pattern, n_layers), strict=True)
        for point, tlens_name in pairs:
            ours = mi.resolve_alias(trace, point)
            theirs = mi.resolve_alias(trace, tlens_name)
            assert ours.generation == GENERATION, point
            if ours.status == "structurally_absent":
                assert ours.remedy, (model, point)
                continue
            assert _coordinates(ours) == _coordinates(theirs), (model, point, tlens_name)
            if ours.status in LIVE:
                live.add(point.split(".")[0])
                assert torch.equal(_payload(trace, ours), _payload(trace, theirs)), point
    assert set(CORE) <= live, (model, sorted(set(CORE) - live))


def test_gpt2_rows_name_the_gpt2_modules(traces):
    trace = traces["gpt2"]
    expected = {
        "embeddings": "transformer.wte",
        "resid_post.1": "transformer.h.1",
        "attn_out.0": "transformer.h.0.attn",
        "mlp_act.1": "transformer.h.1.mlp",
        "mlp_pre.0": "transformer.h.0.mlp",
        # The logits facet sits on the model that owns the head (the trace root).
        "lm_head": "self",
    }
    for point, module_address in expected.items():
        found = mi.resolve_alias(trace, point)
        assert found.status in LIVE, (point, found.status, found.remedy)
        assert found.module_address == module_address, (point, found.module_address)


def test_gpt2_mlp_act_is_the_activation_the_model_runs(traces):
    """mlp_act is the down projection's input: GPT-2's own tanh GELU, not the exact GELU."""

    from transformers.activations import ACT2FN

    trace = traces["gpt2"]
    for layer in range(2):
        act = mi.resolve_alias(trace, f"mlp_act.{layer}")
        pre = mi.resolve_alias(trace, f"mlp_pre.{layer}")
        neurons, up = _payload(trace, act), _payload(trace, pre)
        assert torch.equal(neurons, ACT2FN["gelu_new"](up)), layer
        assert not torch.equal(neurons, torch.nn.functional.gelu(up)), layer


def test_mlp_projection_points_follow_the_mlp_kind(traces):
    """A plain MLP has mlp_pre and refuses mlp_pre_linear; a gated MLP the reverse."""

    plain = mi.resolve_alias(traces["gpt2"], "mlp_pre_linear.0")
    assert plain.status == "structurally_absent" and "gated MLPs only" in plain.remedy
    assert mi.resolve_alias(traces["gpt2"], "mlp_pre.0").facet == "up_out"
    gated = mi.resolve_alias(traces["llama"], "mlp_pre.0")
    assert gated.status == "structurally_absent" and "gate projection" in gated.remedy
    assert mi.resolve_alias(traces["llama"], "mlp_pre_linear.0").status in LIVE


def test_report_lists_points_without_a_torchlens_site(traces):
    registry = _package_json("interp_engine_points.json")
    unmapped = {
        p["name"]: p["no_torchlens_site"] for p in registry["points"] if "no_torchlens_site" in p
    }
    assert len(unmapped) == 18
    for model, trace in traces.items():
        report = mi.alias_report(trace, generation=GENERATION)
        assert {r.generation for r in report} == {GENERATION}
        # 14 per-layer rows at 2 layers, 2 whole-model rows, then one entry per absent point.
        assert len(report) == 14 * 2 + 2 + len(unmapped), model
        tail = report[-len(unmapped) :]
        assert {r.query: r.remedy for r in tail} == unmapped
        assert all(r.status == "structurally_absent" and r.facet is None for r in tail)
        mapped = report[: -len(unmapped)]
        assert all(r.facet is not None for r in mapped)
        assert {"final_norm", "attn_out_post", "q_norm_in", "resid_streams"} <= set(unmapped)


def test_points_file_reports_sites_without_a_point(traces, tmp_path):
    data = json.loads(_write_interp_engine_points(traces["gpt2"], tmp_path / "p.json").read_text())
    missing = {(row["module_address"], row["facet"]) for row in data["no_interp_engine_point"]}
    # Learned positions, the projections before attention, the final norm's input.
    assert ("transformer.wpe", "lookup") in missing
    for layer in range(2):
        assert (f"transformer.h.{layer}.attn", "q") in missing
        assert (f"transformer.h.{layer}.attn", "k") in missing
    assert any(facet == "final_norm_input" for _, facet in missing)
    assert all(row["reason"] for row in data["no_interp_engine_point"])
    served = {
        (p["torchlens"]["module_address"], p["torchlens"]["facet"])
        for p in data["points"]
        if p["address"] in data["capture_points"]
    }
    assert not served & missing
    assert {row["point"] for row in data["no_torchlens_site"]} >= {"final_norm", "mlp_out_post"}


def test_registry_matches_the_saved_point_format():
    """The shipped registry is the saved one, and every point is mapped or reasoned absent."""

    fixture = _fixture()
    registry = _package_json("interp_engine_points.json")
    assert registry["source"]["commit"] == fixture["source"]["commit"]
    assert registry["source"]["version"] == fixture["source"]["version"]
    assert registry["address"]["fields"] == fixture["address_fields"]
    assert [[p["name"], p["scope"]] for p in registry["points"]] == fixture["points"]
    assert len(fixture["points"]) == 34
    rows = [r for r in mi.translation_table() if r["generation"] == GENERATION]
    mapped = {str(r["tlens"]).split(".")[0]: r for r in rows}
    assert len(mapped) == len(rows)
    scope = dict(fixture["points"])
    for point in registry["points"]:
        name = point["name"]
        assert (name in mapped) != ("no_torchlens_site" in point), name
        if name in mapped:
            assert ("{L}" in mapped[name]["tlens"]) == (scope[name] == "layer"), name


def test_saved_grammar_reads_the_engine_examples():
    fixture = _fixture()
    for text, fields in fixture["examples"].items():
        assert _parse(text, fixture["grammar"], fixture["address_fields"]) == fields, text


def test_points_file_round_trips_through_the_engine_format(traces, tmp_path):
    fixture = _fixture()
    fields = fixture["address_fields"]
    names = {name for name, _ in fixture["points"]}
    for model, trace in traces.items():
        path = _write_interp_engine_points(trace, tmp_path / f"{model}.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["schema"] == "torchlens_interp_engine_points_v1"
        assert data["interp_engine"]["address_fields"] == fields
        assert data["interp_engine"]["commit"] == fixture["source"]["commit"]
        addresses = [record["address"] for record in data["points"]]
        assert set(data["capture_points"]) <= set(addresses)
        assert len(data["capture_points"]) >= 2 * 10
        for record in data["points"]:
            parsed = _parse(record["address"], fixture["grammar"], fields)
            assert {key: record[key] for key in fields} == parsed, record["address"]
            assert record["name"] in names
            rebuilt = ".".join(
                [record["name"]] + ([str(record["layer"])] if record["layer"] is not None else [])
            )
            assert rebuilt == record["address"]
        assert {row["point"] for row in data["no_torchlens_site"]} <= names
