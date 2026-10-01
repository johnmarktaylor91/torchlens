"""The TLens-agreement demo (mikit D19's first public demo; RG21 interim).

Run: ``python -m torchlens.mechinterp._demo`` (needs the gpt2 checkpoint in
the local HF cache and, for the parity section, transformer-lens installed).

Assertion-carrying by design (the single-source-teaching posture): every
printed claim is asserted in the same breath, so the demo cannot drift from
the truth it advertises. Promotion to gallery row RG21 is a rename.
"""

from __future__ import annotations

import os
from typing import Any


def tlens_agreement_demo() -> dict[str, Any]:
    """Decomposition + DLA agreeing with TransformerLens on real gpt2."""

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import torchlens as tl
    import torchlens.mechinterp as mi

    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt = "The Eiffel Tower is located in the city of"
    ids = torch.tensor([tokenizer(prompt)["input_ids"]])
    log = tl.trace(
        model,
        ids,
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )

    dec = mi.residual_decomposition(log)
    assert dec.identity_receipt["result"] == "bitwise_equal"  # noqa: S101 -- the demo IS its own gate
    print(f"decomposition: {len(dec)} writers, {dec.grading}, BITWISE identity")

    paris = tokenizer(" Paris")["input_ids"][0]
    london = tokenizer(" London")["input_ids"][0]
    dla = mi.direct_logit_contributions(log, answer=paris, vs=london)
    assert dla.identity_receipt["result"] == "verified"  # noqa: S101 -- the demo IS its own gate
    print("DLA (' Paris' - ' London') top contributors:")
    for label, score in dla.top(5):
        print(f"  {label}: {score:+.3f}")

    inspection = mi.inspect_prompt(log, answer=paris, tokenizer=tokenizer)
    print(inspection.bos_disclosure)
    print(f"' Paris' rank: {inspection.answers[0].rank}")

    scores = mi.head_scores(log, kind="induction")
    print("induction heads on the FUSED default-loaded model (no eager reload):")
    for name, score in scores.top(3):
        print(f"  {name}: {score:.3f}")

    results: dict[str, Any] = {
        "decomposition_rows": len(dec),
        "dla_top": dla.top(5),
        "induction_top": scores.top(3),
    }

    try:
        from transformer_lens import HookedTransformer
    except ImportError:
        print("(transformer-lens not installed; parity section skipped)")
        return results

    tlens = HookedTransformer.from_pretrained_no_processing("gpt2")
    tlens.eval()
    tokens = tlens.to_tokens(prompt)
    _, cache = tlens.run_with_cache(tokens)
    log2 = tl.trace(tlens, tokens, capture=tl.options.CaptureOptions(layers_to_save="all"))
    dec2 = mi.residual_decomposition(log2)
    stack, _labels = cache.decompose_resid(layer=-1, return_labels=True)
    exact = sum(
        1 for index in range(len(dec2)) if torch.equal(dec2.rows[index].value, stack[index])
    )
    assert exact == len(dec2) == 26  # noqa: S101 -- the demo IS its own gate
    print(f"same-object parity: {exact}/26 decomposed rows torch.equal to TLens's own cache")
    results["same_object_exact_rows"] = exact
    return results


if __name__ == "__main__":
    tlens_agreement_demo()
