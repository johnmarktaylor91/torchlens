"""Fit the Jacobian lens to a causal language model, finding its blocks and readout with TorchLens.

Usage: python jlens_five.py MODEL_ID   (tested on five architectures; a CPU is enough)
"""

import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import torchlens as tl

CORPUS = [
    "The cat sat on the mat while rain fell outside.",
    "A little boat crossed the lake before the sun went down.",
]
PROMPTS = {
    "The Eiffel Tower is in the city of": "Paris",
    "The opposite of hot is": "cold",
    "Paris is the capital of": "France",
    "The largest planet in the solar system is": "Jupiter",
}
SKIP_FIRST = 1  # positions skipped at the start of each sentence
CHUNK = 16  # cotangents per backward pass; the spot check compares the first CHUNK rows

model_id = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, use_cache=False)
model.eval()


def ids(text):
    return tokenizer(text, return_tensors="pt").input_ids


with torch.no_grad():  # reference logits from a plain forward first: RWKV rescales weights on it
    ids0 = ids(CORPUS[0])
    logits = model(ids0).logits

# --- The TorchLens part: find the blocks and the readout, and get the block outputs. ---


def readout_steps(trace, start, logits_shape):
    # The trace path from the last block's output to the logits, breadth-first. A step is a
    # leaf module the path enters (applied whole) or any other op (replayed).
    parent, queue = {start.label: None}, [start]
    while not (queue[0].is_output and queue[0].out.shape == logits_shape):
        op = queue.pop(0)
        for child in op.get_children():
            if child.label not in parent:
                parent[child.label], queue = op, [*queue, child]
    path, op = [], queue[0]
    while op.label != start.label:
        path, op = [(op, parent[op.label])] + path, parent[op.label]
    steps = []
    for op, prev in path:
        module = op.modules[-1].rsplit(":", 1)[0] if op.modules else None
        if module is not None and list(model.get_submodule(module).children()):
            module = None  # an op directly inside a container module (RWKV's rescale): replay it
        if module is None and op.func is not None:
            steps.append((op, prev))
        elif module is not None and module not in steps:
            steps.append(module)
    return steps


# The block stack: the largest group of same-class
# siblings with one same-shape 3-D output each.
capture = tl.options.CaptureOptions(
    save_mode="view",
    save_args=True,
)
trace = tl.trace(model, ids0, capture=capture)
groups = {}
for call in trace.module_calls:
    outs = call.outs
    if len(outs) != 1 or outs[0].dim() != 3:
        continue
    parent = call.address.rpartition(".")[0]
    key = parent, call.class_name, outs[0].shape
    group = groups.setdefault(key, {})
    group[call.address] = call
biggest = max(groups.values(), key=len)
calls = list(biggest.values())
blocks = [call.address for call in calls]
last = trace.ops[calls[-1].output_ops[0]]
steps = readout_steps(trace, last, logits.shape)
last_output = last.out
del trace
VIEW = tl.options.CaptureOptions(save_mode="view")


def block_outputs(text):
    # Block outputs, live on the autograd graph.
    x = ids(text)
    trace = tl.trace(model, x, capture=VIEW)
    return [trace[b].out for b in blocks]


# --- Plain PyTorch from here on. ---


def readout(x):
    for step in steps:
        if isinstance(step, str):
            x = model.get_submodule(step)(x)
            continue
        op, prev = step
        slots, mine = op.call.parent_arg_positions, (prev.label, prev.layer_label)
        args = [x if slots["args"].get(i) in mine else a for i, a in enumerate(op.call.saved_args)]
        kwargs = {
            k: x if slots["kwargs"].get(k) in mine else v
            for k, v in (op.call.saved_kwargs or {}).items()
        }
        if any(torch.is_tensor(a) and a is not x for a in [*args, *kwargs.values()]):
            raise ValueError(f"readout op {op.label} reads an activation other than the stream")
        x = op.func(*args, **kwargs)
    return x


def jacobian_rows(text):
    # d(last block's output, summed over positions) / d(each earlier block's output),
    # averaged over source positions: one sentence's share of the lens.
    states = block_outputs(text)
    target = states[-1][0, SKIP_FIRST:-1].sum(0)
    rows = [[] for _ in states[:-1]]
    for cotangents in torch.eye(target.numel()).split(CHUNK):
        grads = torch.autograd.grad(
            target, states[:-1], cotangents, retain_graph=True, is_grads_batched=True
        )
        for row, grad in zip(rows, grads):
            row.append(grad[:, 0, SKIP_FIRST:-1].mean(1).detach())
    return [torch.cat(row) for row in rows]


def plain_rows(text, layer):
    # The first rows again from plain forward hooks, no TorchLens: by scalar autograd (exact)
    # and by the same batched call as the fit (to measure PyTorch's own batched round-off).
    seen = {}
    hooks = [
        model.get_submodule(b).register_forward_hook(
            lambda _m, _a, out, b=b: seen.update({b: out[0] if isinstance(out, tuple) else out})
        )
        for b in (blocks[layer], blocks[-1])
    ]
    model(ids(text))
    for hook in hooks:
        hook.remove()
    target = seen[blocks[-1]][0, SKIP_FIRST:-1].sum(0)
    source, cotangents = seen[blocks[layer]], torch.eye(target.numel())[:CHUNK]
    scalar = [torch.autograd.grad(target[i], source, retain_graph=True)[0] for i in range(CHUNK)]
    batched = torch.autograd.grad(target, source, cotangents, is_grads_batched=True)[0]
    return torch.stack(scalar)[:, 0, SKIP_FIRST:-1].mean(1), batched[:, 0, SKIP_FIRST:-1].mean(1)


def relative_error(rows, exact):
    return float((rows - exact).norm() / exact.norm())


with torch.no_grad():
    readout_error = float((readout(last_output) - logits).abs().max())
per_sentence = [jacobian_rows(text) for text in CORPUS]
J = [sum(rows) / len(CORPUS) for rows in zip(*per_sentence)]

# Spot check at three blocks: the fitted rows against exact scalar autograd. The limit is 1e-5,
# or plain PyTorch's own batched-vs-scalar difference on the same rows if that is larger.
jacobian_error, torch_error = 0.0, 0.0
for layer in (0, len(J) // 2, len(J) - 1):
    exact, batched = plain_rows(CORPUS[0], layer)
    jacobian_error = max(jacobian_error, relative_error(per_sentence[0][layer][:CHUNK], exact))
    torch_error = max(torch_error, relative_error(batched, exact))
jacobian_limit = max(1e-5, torch_error)

with torch.no_grad():
    reads = {}
    for prompt, answer in PROMPTS.items():
        # A hit is the answer's first token (Granite splits " Paris" into "Par" + "is").
        piece = tokenizer.decode(tokenizer(" " + answer, add_special_tokens=False).input_ids[0])
        states = [h[:, -1:] for h in block_outputs(prompt)]
        lenses = {"jacobian": [h @ m.T for h, m in zip(states, J)] + states[-1:], "logit": states}
        top = {
            name: [tokenizer.decode(int(readout(h)[0, -1].argmax())) for h in hs]
            for name, hs in lenses.items()
        }
        first = {
            name: next((i for i, t in enumerate(tokens) if t.strip() == piece.strip()), None)
            for name, tokens in top.items()
        }
        says = top["logit"][-1]
        reads[prompt] = {"answer": answer, "piece": piece, "says": says, "first": first, "top": top}

nonzero = all(bool(m.abs().sum() > 0) for m in J)
ok = len(blocks) >= 4 and readout_error <= 1e-4 and jacobian_error <= jacobian_limit and nonzero
names = [s if isinstance(s, str) else s[0].func_name for s in steps]
print(f"{model_id}: {len(blocks)} blocks, {blocks[0]} to {blocks[-1]}; readout {names}")
print(f"readout error {readout_error:.1e} (limit 1e-4); Jacobian rows vs scalar autograd, relative")
print(f"error {jacobian_error:.1e} (limit {jacobian_limit:.1e}, PyTorch batched {torch_error:.1e})")
print(f"every Jacobian nonzero: {nonzero}")
for prompt, r in reads.items():
    j, g = r["first"]["jacobian"], r["first"]["logit"]
    print(
        f"  {prompt!r}: {r['answer']} first at layer {j} (Jacobian), {g} (logit); says {r['says']!r}"
    )
print("VERDICT", "PASS" if ok else "FAIL")
result = {"model": model_id, "blocks": blocks, "readout": names, "readout_error": readout_error}
result |= {"jacobian_error": jacobian_error, "torch_batched_error": torch_error}
result |= {"jacobian_limit": jacobian_limit, "nonzero": nonzero, "reads": reads, "pass": ok}
print("RESULT " + json.dumps(result))
