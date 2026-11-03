"""Fit a Jacobian lens for GPT-2 small with public TorchLens primitives and read it out.

The Jacobian lens transports a block's residual state into the final layer's
basis with ``J_l = E[d h_final / d h_l]`` before the model's own final norm and
unembedding; the logit lens is the special case ``J_l = I``. The recipe is the
one in ``docs/recipes/jacobian_lens.md``: a hook callable bound with
``tl.when(...).bind(model)`` keeps each block's live output, batched
``torch.autograd.grad`` gives the Jacobian rows, and ``logit_lens(lens=...)``
reads the fitted lens out beside the plain logit lens. ``--check`` compares the
fit against ``torch.func.jacrev`` of the whole model with the block output
replaced, which never touches TorchLens.

The eight-sentence corpus (about a hundred source positions) is a demonstration fit:
the late blocks read out sensibly, the early blocks' top tokens are noise. The
reference method averages over thousands of positions; pass more text to
``CORPUS`` for a usable lens.

Run (downloads the checkpoint on first use)::

    python examples/jacobian_lens_demo.py --check
"""

from __future__ import annotations

import argparse
import functools
import operator
import time

import torch

import torchlens as tl
from torchlens._semantic import logit_lens

CORPUS = (
    "The quick brown fox jumps over the lazy dog while the farmer watches from the porch.",
    "In 1969 the first crew landed on the Moon and returned safely to Earth a few days later.",
    "She opened the old book, and a pressed flower fell out from between the yellowed pages.",
    "The committee will meet on Thursday to review the budget for the coming fiscal year.",
    "Water boils at one hundred degrees Celsius at sea level, but lower on a mountain.",
    "After the storm passed, the children ran outside to splash in the puddles on the street.",
    "The museum's new wing houses paintings from the Dutch Golden Age and early photography.",
    "He tuned the guitar carefully before the concert, listening to each string in turn.",
)
PROBE = "The capital of France is Paris and the capital of Italy is"


def block_output(out):
    """Unwrap a GPT-2 block's output, a tensor or a tuple whose first item is the tensor."""
    return out[0] if isinstance(out, tuple) else out


def residual_states(model, blocks, ids):
    """Run one forward and return each block's live (graph-attached) output, in block order."""
    states = {}

    def keep(address):
        def hook(activation, *, ctx):
            states[address] = activation
            return activation

        return hook

    spec = functools.reduce(operator.and_, [tl.when(tl.module(a), keep(a)) for a in blocks])
    spec.bind(model)(ids)
    return [states[address] for address in blocks]


def fit_jacobian_lens(model, blocks, prompts, *, skip_first, chunk):
    """Return the corpus-average Jacobian of the last block's residual w.r.t. each earlier block's."""
    d_model = model.config.n_embd
    total = {address: torch.zeros(d_model, d_model) for address in blocks[:-1]}
    for ids in prompts:
        states = residual_states(model, blocks, ids)
        valid = torch.arange(skip_first, ids.shape[1] - 1)
        target = states[-1][0, valid].sum(0)
        rows = {address: [] for address in blocks[:-1]}
        for cotangents in torch.eye(d_model).split(chunk):
            grads = torch.autograd.grad(
                target,
                states[:-1],
                grad_outputs=cotangents,
                is_grads_batched=True,
                retain_graph=True,
            )
            for address, grad in zip(blocks[:-1], grads):
                rows[address].append(grad[:, 0, valid].sum(1) / len(valid))
        for address in blocks[:-1]:
            total[address] += torch.cat(rows[address])
    return {address: matrix / len(prompts) for address, matrix in total.items()}


def model_jacobian(model, index, ids, *, skip_first, chunk_size):
    """Independent check: jacrev of the whole model with block ``index``'s output replaced."""
    layers, seen = model.transformer.h, {}
    handles = [
        layer.register_forward_hook(lambda m, i, out, k=k: seen.__setitem__(k, block_output(out)))
        for k, layer in enumerate(layers)
    ]
    model(ids)
    for handle in handles:
        handle.remove()
    valid = torch.arange(skip_first, ids.shape[1] - 1)

    def suffix(h):
        patch = layers[index].register_forward_hook(
            lambda m, i, out: (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        )
        grab = layers[-1].register_forward_hook(
            lambda m, i, out: seen.__setitem__("last", block_output(out))
        )
        model(ids)
        patch.remove()
        grab.remove()
        return seen["last"][0, valid].sum(0)

    jac = torch.func.jacrev(suffix, chunk_size=chunk_size)(seen[index].detach())
    return jac[:, 0, valid].sum(1) / len(valid)


def readout(model, tokenizer, blocks, lenses, ids, *, top):
    """Print the Jacobian lens and the logit lens top tokens at the last position per block."""
    final_norm, unembed = model.transformer.ln_f, model.lm_head
    lens = {address: (lambda h, J=J: unembed(final_norm(h @ J.T))) for address, J in lenses.items()}
    trace = tl.trace(model, ids, capture=tl.options.CaptureOptions(layers_to_save="all"))
    jacobian = logit_lens(trace, lens=lens, layers=list(lens), verify=False)
    plain = logit_lens(trace, layers=list(lens), verify=False)
    print(f"prompt: {tokenizer.decode(ids[0])!r}")
    print(f"{'block':<16} {'jacobian lens':<40} logit lens")
    for j_entry, p_entry in zip(jacobian.entries, plain.entries):
        j_tokens = [tokenizer.decode(t) for t in j_entry.logits[0, -1].topk(top).indices]
        p_tokens = [tokenizer.decode(t) for t in p_entry.logits[0, -1].topk(top).indices]
        print(f"{j_entry.address:<16} {str(j_tokens):<40} {p_tokens}")
    model_tokens = [tokenizer.decode(t) for t in plain.final_logits[0, -1].topk(top).indices]
    print(f"{'model':<16} {model_tokens}")


def main() -> int:
    """Fit, read out and optionally check a Jacobian lens for GPT-2 small."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--skip-first", type=int, default=4, help="source positions skipped")
    parser.add_argument("--chunk", type=int, default=96, help="cotangents per backward pass")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--check", action="store_true", help="compare against torch.func.jacrev")
    parser.add_argument("--check-layers", default="0,5,10")
    parser.add_argument("--check-chunk", type=int, default=4, help="jacrev chunk_size (memory)")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    from transformers import AutoTokenizer, GPT2LMHeadModel

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model, use_cache=False).eval()
    blocks = [f"transformer.h.{index}" for index in range(model.config.n_layer)]
    prompts = [tokenizer(text, return_tensors="pt").input_ids for text in CORPUS]

    started = time.perf_counter()
    lenses = fit_jacobian_lens(model, blocks, prompts, skip_first=args.skip_first, chunk=args.chunk)
    positions = sum(max(ids.shape[1] - 1 - args.skip_first, 0) for ids in prompts)
    print(
        f"fit {len(lenses)} lenses from {len(prompts)} prompts ({positions} source positions) "
        f"in {time.perf_counter() - started:.1f}s; a demonstration fit, not a usable lens"
    )
    probe = tokenizer(PROBE, return_tensors="pt").input_ids
    readout(model, tokenizer, blocks, lenses, probe, top=args.top)

    if not args.check:
        return 0
    one_prompt = fit_jacobian_lens(
        model, blocks, [probe], skip_first=args.skip_first, chunk=args.chunk
    )
    worst = 0.0
    for index in (int(x) for x in args.check_layers.split(",")):
        started = time.perf_counter()
        reference = model_jacobian(
            model, index, probe, skip_first=args.skip_first, chunk_size=args.check_chunk
        )
        error = float((one_prompt[blocks[index]] - reference).norm() / reference.norm())
        worst = max(worst, error)
        print(
            f"check {blocks[index]}: rel error {error:.2e} ({time.perf_counter() - started:.1f}s)"
        )
    print(f"check {'passed' if worst < 1e-4 else 'FAILED'}: worst rel error {worst:.2e}")
    return 0 if worst < 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
