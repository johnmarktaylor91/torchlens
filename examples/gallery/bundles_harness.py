"""Gallery entry: bundles of clean and ablated runs, and a two-model draft-and-critique harness.

Run: ``python examples/gallery/bundles_harness.py --out DIR [--steer-layer L]
[--critic gpt2|distilgpt2] [--draft-tokens N]``. Downloads GPT-2 small and DistilGPT2 (public,
about 900 MB together) at pinned revisions and the torchvision ResNet-18 ImageNet weights
(about 45 MB) on first use.

This entry is a method demonstration rather than the reproduction of one paper. It checks two
claims a comparison tool has to get right:

1. Bundle half. A comparison of a clean run with a zero-ablated run of the same model on the
   same input must report, at every module, exactly the difference two plain PyTorch hook runs
   give (GPT-2 small and ResNet-18). A pair of runs whose sites cannot line up one to one must
   be refused, not compared.
2. Harness half. In a two-model pipeline (GPT-2 small drafts a sentence, DistilGPT2 reads it
   and writes a verdict), the bundle must record the same tokens as running the two models one
   after the other by hand, the ``fed_by`` token spans must point at the critic's prompt tokens
   that really hold the draft, and steering the drafter must reach the critic only through the
   tokens it reads: a steer that moves the draft changes the critic's prompt, and a steer too
   small to move the draft must leave the critique reported unchanged.

Both halves are saved as one bundle, reloaded, and the spans and the comparison replayed from
the reloaded copy.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import _gallery as gallery  # the sibling module; scripts here run from this folder
import torch
from torch import nn

import torchlens as tl

DRAFTER_ID = "openai-community/gpt2"
DRAFTER_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"  # pragma: allowlist secret
CRITICS = {
    "distilgpt2": ("distilbert/distilgpt2", "2290a62682d06624634c1f46a6ad5be0f47f38aa"),
    "gpt2": (DRAFTER_ID, DRAFTER_REVISION),
}
CNN_WEIGHTS = "ResNet18_Weights.IMAGENET1K_V1"

CLAIM = (
    "Method demonstration (no single paper): a clean-versus-ablated comparison is valid only "
    "when the two runs line up and differ by the stated edit, as in the clean-versus-ablated "
    "comparisons of activation-patching work; and in a two-model draft-and-critique pipeline, "
    "an edit to the drafter's representation reaches the critic only through the tokens the "
    "critic reads."
)
DRAFT_PROMPT = "The Eiffel Tower is in the city of"
CRITIC_PREFIX = "Statement:"
CRITIC_SUFFIX = "\nIs the statement true or false? Answer:"
DRAFT_TOKENS = 12
CRITIC_TOKENS = 8
ABLATION_SITE = "transformer.h.6.mlp"
CNN_ABLATION_SITE = "layer1.0.relu"
STEER_LAYER = 6
STEER_SEED = 0
STRONG_STRENGTH = 120.0
WEAK_STRENGTH = 1e-3
OTHER_PROMPT = "The Colosseum is in"

#: Rules fixed before the first run of this entry.
DIFF_RULE = (
    "at every module call that both the TorchLens bundle and two plain hook runs record with "
    "one tensor output, (ablated - clean) from the bundle equals (ablated - clean) from the "
    "plain hooks: max abs difference 0 in float32"
)
COMPARE_RULE = (
    "Bundle.compare('clean', 'ablated', metric='relative_l2') at the lm_head output equals the "
    "relative L2 distance of the plain hook runs' logits within 1e-6 (relative)"
)
COMPARE_TOLERANCE = 1e-6
COMPOSITION_RULE = (
    "running the drafter, decoding, building the critic prompt and running the critic by hand "
    "(plain greedy loops, no TorchLens) gives exactly the tokens the bundle recorded: the draft, "
    "the critic prompt and the critique, unsteered and steered (the hand-run steer is a plain "
    "forward hook adding strength x vector)"
)
SPANS_RULE = (
    "the critique fed_by draft row found with token_spans='auto' is basis 'matched', and every "
    "span's draft token ids equal the critic-prompt token ids it points at in the hand-run "
    "pipeline"
)
REACH_RULE = (
    "a steer that changes the draft is reported as status 'steered' for the draft and "
    "'absorbed' or 'changed' for the critique, and the critique's prompt divergence is the "
    "first position where the two hand-run critic prompts differ"
)
UNCHANGED_RULE = (
    "a steer too small to change the drafted tokens (strength 1e-3) leaves the draft tokens "
    "equal and is reported with the critique status 'unchanged'"
)
REFUSE_RULE = (
    "comparing two GPT-2 runs on prompts of different token counts refuses with "
    "bundle_gate_input_values_differ or metric_shape_mismatch (the pair cannot line up site "
    "by site)"
)
REFUSE_CODES = ("bundle_gate_input_values_differ", "metric_shape_mismatch")
ORDERING_RULE = (
    "comparing the critique with the draft it read refuses with bundle_gate_ordering_topology "
    "(members joined by fed_by are not comparison operands)"
)


# region: shared
def greedy(runner: Any, ids: torch.Tensor, n_new: int) -> torch.Tensor:
    """Greedy decoding without a cache: the same loop for TorchLens and plain runs."""

    with torch.no_grad():
        for _ in range(n_new):
            logits = runner(ids, use_cache=False).logits
            ids = torch.cat([ids, logits[:, -1].argmax(dim=-1, keepdim=True)], dim=1)
    return ids


def encode(tokenizer: Any, text: str) -> torch.Tensor:
    """Token ids of ``text`` as a batch of one."""

    return torch.tensor([tokenizer.encode(text)])


def critic_prompt(tokenizer: Any, draft_text: str) -> torch.Tensor:
    """The critic's prompt: the decoded draft inside a fixed template, tokenized as text."""

    return encode(tokenizer, CRITIC_PREFIX + draft_text + CRITIC_SUFFIX)


def steer_vector(width: int) -> torch.Tensor:
    """A fixed unit-norm direction in the residual stream."""

    vector = torch.randn(width, generator=torch.Generator().manual_seed(STEER_SEED))
    return vector / vector.norm()


def single_tensor(value: Any) -> torch.Tensor | None:
    """The tensor a module call returned, when it returned exactly one tensor."""

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and len(value) == 1 and torch.is_tensor(value[0]):
        return value[0]
    return None


# endregion


# region: torchlens
def clean_and_ablated(model: nn.Module, x: Any, site: str, kwargs: dict) -> tl.Bundle:
    """Capture a clean run, fork it, zero-ablate one module and bundle the two."""

    clean = tl.trace(
        model,
        x,
        input_kwargs=kwargs,
        save="all",
        capture=tl.options.CaptureOptions(replayable=True),
    )
    ablated = clean.fork("ablated")
    ablated.do(tl.module(site), tl.zero_ablate())
    return tl.Bundle({"clean": clean, "ablated": ablated}, baseline="clean")


def bundle_diffs(bundle: tl.Bundle) -> dict[tuple[str, int], torch.Tensor]:
    """(ablated - clean) at every module call with one tensor output, read from the bundle."""

    diffs = {}
    clean, ablated = bundle["clean"], bundle["ablated"]
    for module in clean.modules:
        address = str(module.address)
        for index in range(module.num_calls):
            outs = module.calls[index].outs
            other = ablated.modules[address].calls[index].outs
            if len(outs) == 1 and torch.is_tensor(outs[0]):
                diffs[(address, index)] = other[0] - outs[0]
    return diffs


def harness(drafter: nn.Module, critic: nn.Module, tokenizer: Any, spec: Any) -> tuple:
    """Draft, decode, critique; record both calls as one bundle with a fed_by row."""

    bundle = tl.Bundle()
    texts = {}

    def call(name: str, model: nn.Module, prompt: torch.Tensor, n: int, edit: Any) -> str:
        out = greedy(edit.bind(model) if edit else model, prompt, n)
        trace = tl.trace(model, out, input_kwargs={"use_cache": False}, intervene=edit)
        binding = {"stream": "input", "batch_row": 0, "completion": [prompt.shape[1], out.shape[1]]}
        tl.io.attach_sidecar(trace, "torchlens.boundary_facts", binding)
        bundle.add(trace, names=name)
        texts[name] = out
        return tokenizer.decode(out[0, prompt.shape[1] :])

    draft_text = call("draft", drafter, encode(tokenizer, DRAFT_PROMPT), DRAFT_TOKENS, spec)
    call("critique", critic, critic_prompt(tokenizer, draft_text), CRITIC_TOKENS, None)
    bundle.relate("fed_by", "critique", "draft", token_spans="auto", label="critique")
    return bundle, texts


def steer_spec(strength: float, layer: int) -> Any:
    """Add ``strength`` times the fixed direction to one GPT-2 MLP's output at every token."""

    site = f"transformer.h.{layer}.mlp"
    return tl.when(tl.module(site), tl.steer(steer_vector(768), strength=strength, dim=-1))


# endregion


# region: plain
def plain_outputs(model: nn.Module, x: Any, kwargs: dict, ablate: str | None) -> dict:
    """Every module call's single-tensor output from one plain forward, optionally ablating."""

    records: dict[tuple[str, int], torch.Tensor] = {}
    counts: dict[str, int] = {}
    handles = []

    def make_hook(name: str) -> Any:
        def hook(module: nn.Module, args: Any, out: Any) -> Any:
            if name == ablate:
                out = torch.zeros_like(out)
            index = counts.get(name, 0)
            counts[name] = index + 1
            tensor = single_tensor(out)
            if tensor is not None:
                records[(name, index)] = tensor.detach().clone()
            return out

        return hook

    for name, module in model.named_modules():
        if name:
            handles.append(module.register_forward_hook(make_hook(name)))
    try:
        with torch.no_grad():
            model(x, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
    return records


def plain_pipeline(drafter: nn.Module, critic: nn.Module, tokenizer: Any, steer: tuple) -> dict:
    """Run drafter then critic by hand; ``steer`` is ``(layer, strength)`` or ``()``."""

    handle = None
    if steer:
        layer, strength = steer
        vector = steer_vector(768)
        mlp = drafter.get_submodule(f"transformer.h.{layer}.mlp")
        handle = mlp.register_forward_hook(lambda m, a, out: out + vector * strength)
    try:
        prompt = encode(tokenizer, DRAFT_PROMPT)
        draft = greedy(drafter, prompt, DRAFT_TOKENS)
    finally:
        if handle is not None:
            handle.remove()
    c_prompt = critic_prompt(tokenizer, tokenizer.decode(draft[0, prompt.shape[1] :]))
    critique = greedy(critic, c_prompt, CRITIC_TOKENS)
    return {"draft": draft, "critic_prompt": c_prompt, "critique": critique}


# endregion


def load_models(critic_name: str) -> tuple[nn.Module, nn.Module, Any]:
    """Load the drafter, the critic (eager attention, float32) and the shared tokenizer."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    def load(model_id: str, revision: str) -> nn.Module:
        return AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, attn_implementation="eager"
        ).eval()

    critic_id, critic_revision = CRITICS[critic_name]
    tokenizer = AutoTokenizer.from_pretrained(DRAFTER_ID, revision=DRAFTER_REVISION)
    return load(DRAFTER_ID, DRAFTER_REVISION), load(critic_id, critic_revision), tokenizer


def diff_check(bundle: tl.Bundle, model: nn.Module, x: Any, kwargs: dict, site: str) -> dict:
    """Compare the bundle's per-module differences with two plain hook runs."""

    ours = bundle_diffs(bundle)
    clean = plain_outputs(model, x, kwargs, None)
    ablated = plain_outputs(model, x, kwargs, site)
    plain = {key: ablated[key] - clean[key] for key in clean if key in ablated}
    shared = [key for key in plain if key in ours and plain[key].shape == ours[key].shape]
    worst = max((float((ours[k] - plain[k]).abs().max()) for k in shared), default=float("nan"))
    nonzero = sum(1 for k in shared if bool(plain[k].abs().max() > 0))
    return {
        "max_abs_difference": worst,
        "module_calls_compared": len(shared),
        "calls_changed_by_the_ablation": nonzero,
        "plain_calls_not_in_bundle": sorted(f"{a}#{i}" for a, i in plain if (a, i) not in ours),
        "bundle_calls_not_in_plain": sorted(f"{a}#{i}" for a, i in ours if (a, i) not in plain),
    }


def lm_head_row(bundle: tl.Bundle) -> float | None:
    """Bundle.compare's relative L2 row at the lm_head output, or ``None`` when absent."""

    clean = bundle["clean"]
    labels = set()
    for label in clean.modules["lm_head"].calls[0].output_ops:
        labels.add(str(label))
        labels.add(str(getattr(clean.ops[label], "layer_label", label)))
    rows = bundle.compare("clean", "ablated", metric="relative_l2")
    found = [float(distance) for site, distance in rows if str(site) in labels]
    return found[0] if found else None


def bundle_half(drafter: nn.Module, tokenizer: Any) -> tuple[tl.Bundle, dict, dict]:
    """Clean versus ablated on GPT-2 small and ResNet-18, each against plain hooks."""

    from torchvision.models import ResNet18_Weights, resnet18

    ids = encode(tokenizer, DRAFT_PROMPT)
    kwargs = {"use_cache": False}
    gpt2 = clean_and_ablated(drafter, ids, ABLATION_SITE, kwargs)
    gpt2_check = diff_check(gpt2, drafter, ids, kwargs, ABLATION_SITE)
    with torch.no_grad():
        clean_logits = drafter(ids, use_cache=False).logits
        plain_ablated = plain_outputs(drafter, ids, kwargs, ABLATION_SITE)
    ablated_logits = plain_ablated[("lm_head", 0)]
    plain_distance = float((ablated_logits - clean_logits).norm() / clean_logits.norm())
    row = lm_head_row(gpt2)
    compare = {"bundle_compare": row, "plain": plain_distance}
    compare["passed"] = row is not None and abs(row - plain_distance) <= COMPARE_TOLERANCE * max(
        1.0, plain_distance
    )
    cnn = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    image = torch.randn(1, 3, 224, 224, generator=torch.Generator().manual_seed(0))
    cnn_bundle = clean_and_ablated(cnn, image, CNN_ABLATION_SITE, {})
    cnn_check = diff_check(cnn_bundle, cnn, image, {}, CNN_ABLATION_SITE)
    return gpt2, {"gpt2": gpt2_check, "resnet18": cnn_check}, compare


def refusal_code(bundle: tl.Bundle, left: str, right: str) -> dict:
    """Try a comparison; return the refusal's type and code, or note that it ran."""

    try:
        rows = bundle.compare(left, right)
    except Exception as exc:  # the control reads which refusal fired
        fields = getattr(exc, "fields", None) or {}
        code = getattr(exc, "code", None) or fields.get("code")
        return {"refused": True, "error": type(exc).__name__, "code": code}
    return {"refused": False, "rows": len(rows)}


def input_controls(drafter: nn.Module, tokenizer: Any) -> dict:
    """Different-length prompts must refuse; same-length different prompts are reported."""

    first = encode(tokenizer, DRAFT_PROMPT)
    other = encode(tokenizer, OTHER_PROMPT)
    swapped = first.clone()
    swapped[0, 1] = tokenizer.encode(" Louvre")[0]
    kwargs = {"use_cache": False}
    traces = {
        name: tl.trace(drafter, ids, input_kwargs=kwargs, save="all")
        for name, ids in (("a", first), ("b", other), ("c", swapped))
    }
    different = refusal_code(tl.Bundle({"a": traces["a"], "b": traces["b"]}), "a", "b")
    same_length = refusal_code(tl.Bundle({"a": traces["a"], "c": traces["c"]}), "a", "c")
    lengths = {"a": int(first.shape[1]), "b": int(other.shape[1]), "c": int(swapped.shape[1])}
    return {"different_lengths": different, "same_length": same_length, "lengths": lengths}


def ids_of(trace: Any) -> list[int]:
    """The token ids a member's capture retained."""

    return [int(v) for v in trace.input_ops[0].out.reshape(-1)]


def composition(bundle: tl.Bundle, texts: dict, plain: dict, prefix: str = "") -> dict:
    """Compare the harness's recorded tokens with the hand-run pipeline's."""

    return {
        "draft": torch.equal(texts["draft"], plain["draft"]),
        "critic_prompt": torch.equal(
            texts["critique"][:, : plain["critic_prompt"].shape[1]], plain["critic_prompt"]
        ),
        "critique": torch.equal(texts["critique"], plain["critique"]),
        "retained_ids": ids_of(bundle[prefix + "draft"]) == plain["draft"][0].tolist()
        and ids_of(bundle[prefix + "critique"]) == plain["critique"][0].tolist(),
    }


def span_check(bundle: tl.Bundle, plain: dict) -> dict:
    """Check the fed_by row's spans against the hand-run token ids."""

    row = next(r for r in bundle.relations if r.kind == "fed_by")
    spans = [list(map(int, span)) for span in row.params["token_spans"]]
    draft, reader = plain["draft"][0].tolist(), plain["critique"][0].tolist()
    agree = all(draft[s0:s1] == reader[d0:d1] for s0, s1, d0, d1 in spans)
    return {
        "basis": row.params["basis"],
        "coverage": row.params.get("coverage"),
        "spans": spans,
        "ids_agree": agree,
        "passed": row.params["basis"] == "matched" and agree and bool(spans),
    }


def first_difference(left: torch.Tensor, right: torch.Tensor) -> int | None:
    """First position where two id rows differ (the shorter length when one extends the other)."""

    a, b = left[0].tolist(), right[0].tolist()
    for index, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def harness_half(drafter: nn.Module, critic: nn.Module, tokenizer: Any, layer: int) -> dict:
    """The unsteered harness, a steered alternative, a too-weak alternative, and hand runs."""

    base, base_texts = harness(drafter, critic, tokenizer, None)
    strong, strong_texts = harness(drafter, critic, tokenizer, steer_spec(STRONG_STRENGTH, layer))
    plain_base = plain_pipeline(drafter, critic, tokenizer, ())
    plain_strong = plain_pipeline(drafter, critic, tokenizer, (layer, STRONG_STRENGTH))
    plain_weak = plain_pipeline(drafter, critic, tokenizer, (layer, WEAK_STRENGTH))
    checks = {
        "unsteered": composition(base, base_texts, plain_base),
        "steered": composition(strong, strong_texts, plain_strong),
        "spans": span_check(base, plain_base),
    }
    report = base.add_alternative(strong, steered=("draft",))
    calls = {row.call: row.status for row in report.calls}
    divergence = base.token_divergence("critique", "cf_critique")
    weak_base, _ = harness(drafter, critic, tokenizer, None)
    weak, _ = harness(drafter, critic, tokenizer, steer_spec(WEAK_STRENGTH, layer))
    weak_report = weak_base.add_alternative(weak, steered=("draft",))
    checks["reach"] = {
        "statuses": calls,
        "prompt_divergence": divergence.prompt_divergence,
        "plain_prompt_divergence": first_difference(
            plain_base["critic_prompt"], plain_strong["critic_prompt"]
        ),
        "table": report.table(),
    }
    checks["weak"] = {
        "draft_unchanged": torch.equal(plain_weak["draft"], plain_base["draft"]),
        "statuses": {row.call: row.status for row in weak_report.calls},
    }
    checks["texts"] = {
        "draft": tokenizer.decode(plain_base["draft"][0]),
        "critique": tokenizer.decode(plain_base["critique"][0]),
        "steered_draft": tokenizer.decode(plain_strong["draft"][0]),
        "steered_critique": tokenizer.decode(plain_strong["critique"][0]),
    }
    checks["bundle"] = base
    return checks


def make_replay(expected_row: float | None, spans: list, divergence: int | None) -> Any:
    """Build the replay: spans, the divergence and the compare row from the reloaded bundle."""

    def replay(loaded: Any) -> dict[str, Any]:
        row = next(r for r in loaded.relations if r.kind == "fed_by")
        got_spans = [list(map(int, span)) for span in row.params["token_spans"]]
        got_row = lm_head_row(loaded)
        got_divergence = loaded.token_divergence("critique", "cf_critique").prompt_divergence
        row_ok = (
            got_row is not None
            and expected_row is not None
            and abs(got_row - expected_row) <= COMPARE_TOLERANCE * max(1.0, expected_row)
        )
        return {
            "matched": row_ok and got_spans == spans and got_divergence == divergence,
            "basis": row.params["basis"],
            "spans": got_spans,
            "lm_head_relative_l2": got_row,
            "expected_lm_head_relative_l2": expected_row,
            "prompt_divergence": got_divergence,
        }

    return replay


def add_checks(entry: gallery.Entry, diffs: dict, compare: dict, half: dict, inputs: dict) -> None:
    """Turn the measurements into the entry's oracles and controls."""

    for arch in ("gpt2", "resnet18"):
        found = diffs[arch]
        ok = found["module_calls_compared"] > 0 and found["max_abs_difference"] == 0.0
        entry.oracles[f"bundle_diff_{arch}"] = gallery.Check(DIFF_RULE, found, ok)
    entry.oracles["bundle_compare"] = gallery.Check(COMPARE_RULE, compare, compare["passed"])
    runs = {"unsteered": half["unsteered"], "steered": half["steered"]}
    entry.oracles["composition"] = gallery.Check(
        COMPOSITION_RULE, runs, all(all(run.values()) for run in runs.values())
    )
    spans = half["spans"]
    entry.oracles["fed_by_spans"] = gallery.Check(SPANS_RULE, spans, spans["passed"])
    reach = half["reach"]
    statuses = reach["statuses"]
    reach_ok = (
        statuses.get("draft") == "steered"
        and statuses.get("critique") in ("absorbed", "changed")
        and reach["prompt_divergence"] is not None
        and reach["prompt_divergence"] == reach["plain_prompt_divergence"]
    )
    entry.oracles["steer_reaches_critic"] = gallery.Check(REACH_RULE, reach, reach_ok)
    weak = half["weak"]
    weak_ok = weak["draft_unchanged"] and weak["statuses"].get("critique") == "unchanged"
    entry.controls["weak_steer_unchanged"] = gallery.Check(UNCHANGED_RULE, weak, weak_ok)
    different = inputs["different_lengths"]
    entry.controls["different_inputs_refuse"] = gallery.Check(
        REFUSE_RULE,
        {**different, "lengths": inputs["lengths"]},
        different["refused"] and different.get("code") in REFUSE_CODES,
    )
    ordering = refusal_code(half["bundle"], "draft", "critique")
    entry.controls["ordering_refuse"] = gallery.Check(
        ORDERING_RULE,
        ordering,
        ordering["refused"] and ordering.get("code") == "bundle_gate_ordering_topology",
    )


def run(out_dir: Path | None, critic_name: str, layer: int, draft_tokens: int) -> dict:
    """Measure both halves, run the controls, save the bundle, reload, replay and report."""

    global DRAFT_TOKENS
    started = time.time()
    DRAFT_TOKENS = draft_tokens
    drafter, critic, tokenizer = load_models(critic_name)
    gpt2, diffs, compare = bundle_half(drafter, tokenizer)
    half = harness_half(drafter, critic, tokenizer, layer)
    inputs = input_controls(drafter, tokenizer)
    critic_id, critic_revision = CRITICS[critic_name]
    entry = gallery.Entry(
        name="bundles_harness",
        claim=CLAIM,
        pinned={
            "drafter": f"{DRAFTER_ID}@{DRAFTER_REVISION} float32 eager attention",
            "critic": f"{critic_id}@{critic_revision} float32 eager attention",
            "cnn": f"torchvision resnet18 {CNN_WEIGHTS}; input randn seed 0",
            "prompts": {"draft": DRAFT_PROMPT, "critic_template": [CRITIC_PREFIX, CRITIC_SUFFIX]},
            "decoding": f"greedy, {draft_tokens} draft tokens, {CRITIC_TOKENS} critique tokens",
            "steer": f"unit randn(768) seed {STEER_SEED} at transformer.h.{layer}.mlp, "
            f"strength {STRONG_STRENGTH} (weak {WEAK_STRENGTH})",
            "ablations": [ABLATION_SITE, CNN_ABLATION_SITE],
        },
        verbs=("capture", "select", "read", "do", "compare", "save"),
        architectures=("GPT-2 small", "ResNet-18", f"{critic_name} (critic)"),
    )
    add_checks(entry, diffs, compare, half, inputs)
    entry.measured = {
        "texts": half["texts"],
        "same_length_different_prompts": inputs["same_length"],
    }
    bundle = half["bundle"]
    bundle.add(gpt2["clean"], names="clean")
    bundle.add(gpt2["ablated"], names="ablated")
    entry.measured["members"] = list(bundle.names)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "bundles_harness.tlspec"
        bundle.save(path, level="portable", overwrite=True)
        replay = make_replay(
            compare["bundle_compare"], half["spans"]["spans"], half["reach"]["prompt_divergence"]
        )
        entry.artifact = gallery.artifact_record(path, replay)
    source = Path(__file__)
    entry.lines = {
        "torchlens": gallery.count_region_lines(source, "torchlens"),
        "plain": gallery.count_region_lines(source, "plain"),
        "shared_setup": gallery.count_region_lines(source, "shared"),
        "reference": "not measured (no reference implementation)",
    }
    return gallery.report(entry, out_dir, "cpu", started)


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument("--critic", choices=sorted(CRITICS), default="distilgpt2")
    parser.add_argument("--steer-layer", type=int, default=STEER_LAYER, help="GPT-2 block 0-11")
    parser.add_argument("--draft-tokens", type=int, default=DRAFT_TOKENS)
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    run(args.out, args.critic, args.steer_layer, args.draft_tokens)


if __name__ == "__main__":
    main()
