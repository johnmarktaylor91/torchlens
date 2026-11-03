"""Fit a captured-state Jacobian lens and compare independent estimators on CPU.

Run ``python examples/gallery/jacobian_lens.py --out DIR --reference-repo URL``.
The reference checkout is supplied explicitly and checked against REFERENCE_COMMIT.
No weights are downloaded by the executed documentation's tiny stand-in.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import _gallery as gallery
import torch

import torchlens as tl

CLAIM = (
    "Verbalizable Representations Form a Global Workspace in Language Models "
    "(Transformer Circuits, 6 July 2026, arXiv 2607.15495 v1): a corpus-average "
    "Jacobian transports a block's residual state into the final residual basis "
    "for readout through the model's final normalization and unembedding."
)
MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"  # pragma: allowlist secret
MAMBA_ID = "state-spaces/mamba-130m-hf"
MAMBA_REVISION = "1e76775f628fbf1350fbe4dbb3d971ba64af25a1"  # pragma: allowlist secret
REFERENCE_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"  # pragma: allowlist secret
SEED = 0
J_ERROR_MAX = 1e-3
OVERLAP_MIN = 0.9
TOP_K = 10
CHUNK = 8
SKIP_FIRST = 1
# Original miniature text corpus, dedicated to CC0; no external dataset is needed.
CORPUS = (
    "The cat sat on the mat while rain fell outside.",
    "A little boat crossed the lake before the sun went down.",
)
PROBES = ("The capital of Italy is", "The small animal has a", ":-) The middle of this face is the")
CORPUS_SHA256 = hashlib.sha256(("\n".join(CORPUS) + "\n").encode()).hexdigest()


def unwrap(output: Any) -> torch.Tensor:
    """Return a block's residual tensor from tensor or tuple outputs.

    Returns
    -------
    torch.Tensor
        Live block residual.
    """
    return output[0] if isinstance(output, tuple) else output


def model_parts(model: Any, arch: str) -> tuple:
    """Resolve block addresses, blocks and final normalization.

    Returns
    -------
    tuple
        Addresses, block modules and final normalization.
    """
    if arch == "gpt2":
        return (
            [f"transformer.h.{i}" for i in range(len(model.transformer.h))],
            model.transformer.h,
            model.transformer.ln_f,
        )
    return (
        [f"backbone.layers.{i}" for i in range(len(model.backbone.layers))],
        model.backbone.layers,
        model.backbone.norm_f,
    )


# region: torchlens
def fit_captured(model: Any, addresses: list[str], prompts: list[torch.Tensor]) -> tuple:
    """Fit all pre-final-block lenses from live captured block outputs.

    Returns
    -------
    tuple
        Mean Jacobians and the final fitting trace for persistence.
    """
    sums: dict[int, torch.Tensor] = {}
    trace = None
    for ids in prompts:
        trace = tl.trace(model, ids, capture=tl.options.CaptureOptions(save_mode="view"))
        states = [trace[address].out for address in addresses]
        target = states[-1][0, SKIP_FIRST:-1].sum(0)
        dim = target.numel()
        chunks: dict[int, list[torch.Tensor]] = {i: [] for i in range(len(states) - 1)}
        for cotangent in torch.eye(dim).split(CHUNK):
            gradients = torch.autograd.grad(
                target,
                states[:-1],
                grad_outputs=cotangent,
                is_grads_batched=True,
                retain_graph=True,
            )
            for index, gradient in enumerate(gradients):
                chunks[index].append(gradient[:, 0, SKIP_FIRST:-1].mean(1).detach())
        for index, rows in chunks.items():
            matrix = torch.cat(rows)
            sums[index] = sums.get(index, torch.zeros_like(matrix)) + matrix
    return {index: matrix / len(prompts) for index, matrix in sums.items()}, trace


def capture_readouts(
    model: Any,
    tokenizer: Any,
    addresses: list[str],
    norm: Any,
    matrices: dict[int, torch.Tensor],
    prompts: tuple[str, ...],
) -> tuple:
    """Apply the fitted lens to held-out live residuals.

    Returns
    -------
    tuple
        Layer-by-prompt top-token indices and detached residuals.
    """
    tokens: dict[int, list[list[int]]] = {index: [] for index in matrices}
    residuals: dict[int, list[torch.Tensor]] = {index: [] for index in matrices}
    for text in prompts:
        ids = tokenizer(text, return_tensors="pt").input_ids
        trace = tl.trace(model, ids, capture=tl.options.CaptureOptions(save_mode="view"))
        for index, matrix in matrices.items():
            value = trace[addresses[index]].out[0, -1].detach()
            residuals[index].append(value)
            tokens[index].append(decode(model, norm, value, matrix))
    return tokens, residuals


# endregion


# region: plain
def fit_autograd(model: Any, blocks: Any, prompts: list[torch.Tensor]) -> dict[int, torch.Tensor]:
    """Fit exact Jacobian rows with ordinary hooks and scalar reverse-mode autograd.

    Returns
    -------
    dict
        Corpus-average exact Jacobians, independently of capture and batched VJPs.
    """
    # Keep the independent prompt/dimension/layer reduction explicit for auditing;
    # this essential numerical nesting exceeds the cognitive-complexity default.
    sums: dict[int, torch.Tensor] = {}
    for ids in prompts:
        states: dict[int, torch.Tensor] = {}
        handles = [
            block.register_forward_hook(
                lambda m, args, output, index=index, states=states: states.__setitem__(
                    index, unwrap(output)
                )
            )
            for index, block in enumerate(blocks)
        ]
        try:
            model(ids, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        target = states[len(blocks) - 1][0, SKIP_FIRST:-1].sum(0)
        source = [states[index] for index in range(len(blocks) - 1)]
        rows: dict[int, list[torch.Tensor]] = {index: [] for index in range(len(source))}
        for value in target:
            gradients = torch.autograd.grad(value, source, retain_graph=True)
            for index, gradient in enumerate(gradients):
                rows[index].append(gradient[0, SKIP_FIRST:-1].mean(0).detach())
        for index in rows:
            matrix = torch.stack(rows[index])
            sums[index] = sums.get(index, torch.zeros_like(matrix)) + matrix
    return {index: matrix / len(prompts) for index, matrix in sums.items()}


def decode(model: Any, norm: Any, residual: torch.Tensor, matrix: torch.Tensor) -> list[int]:
    """Read the transported residual through the model's final norm and unembedding.

    Returns
    -------
    list[int]
        Top ten token ids.
    """
    with torch.no_grad():
        return model.lm_head(norm(residual @ matrix.T)).topk(TOP_K).indices.tolist()


def overlap(left: list[int], right: list[int]) -> float:
    """Return top-ten set overlap.

    Returns
    -------
    float
        Intersection size divided by ten.
    """
    return len(set(left) & set(right)) / TOP_K


# endregion


def reference_module(out: Path, url: str | None) -> Any:
    """Clone the explicitly supplied reference and verify its commit.

    Returns
    -------
    module
        The pinned reference's fitting module.
    """
    root = out / "reference"
    path = root / "jlens"
    if not url or not url.startswith("https://github.com/"):
        raise ValueError("--reference-repo must supply the paper's HTTPS GitHub repository")
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--quiet", url, str(path)], check=True, timeout=120)
    subprocess.run(
        ["git", "-C", str(path), "checkout", "--quiet", REFERENCE_COMMIT], check=True, timeout=30
    )
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, timeout=10
    ).strip()
    if actual != REFERENCE_COMMIT:
        raise ValueError("reference revision mismatch")
    sys.path.insert(0, str(path))
    return importlib.import_module("jlens.fitting")


def fit_reference(
    model: Any, tokenizer: Any, fitting: Any, n_layers: int
) -> dict[int, torch.Tensor]:
    """Run the pinned reference estimator on the identical model, text and positions.

    Returns
    -------
    dict
        Prompt-average reference Jacobians.
    """
    from jlens.hf import from_hf

    adapter = from_hf(model, tokenizer, force_bos=False)
    results = [
        fitting.jacobian_for_prompt(
            adapter, text, list(range(n_layers - 1)), dim_batch=CHUNK, skip_first=SKIP_FIRST
        )[0]
        for text in CORPUS
    ]
    for param in model.parameters():
        param.requires_grad_(True)
    return {index: sum(result[index] for result in results) / len(results) for index in results[0]}


def save_lens(
    trace: Any,
    matrices: dict[int, torch.Tensor],
    residuals: dict[int, list[torch.Tensor]],
    model: Any,
    norm: Any,
    out: Path,
    arch: str,
    layer: int,
) -> dict:
    """Save fitted matrices plus fitting trace and replay a held-out readout.

    Returns
    -------
    dict
        Portable trace record and checked matrix-sidecar hash.
    """
    weights = out / f"{arch}_jacobians.pt"
    torch.save(matrices, weights)
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    trace.annotations["gallery_jacobian"] = {
        "file": weights.name,
        "sha256": digest,
        "layer": layer,
        "held_out_residual": residuals[layer][0].tolist(),
    }
    target = out / f"{arch}_jacobian_lens.tlspec"
    trace.save(target, level="portable", overwrite=True)
    expected = decode(model, norm, residuals[layer][0], matrices[layer])

    def replay(loaded: Any) -> dict[str, Any]:
        """Verify the companion matrix bytes and reproduce the held-out top ten."""
        record = loaded.annotations["gallery_jacobian"]
        sidecar = out / record["file"]
        same = hashlib.sha256(sidecar.read_bytes()).hexdigest() == record["sha256"]
        restored = torch.load(sidecar, map_location="cpu", weights_only=True)
        actual = decode(
            model, norm, torch.tensor(record["held_out_residual"]), restored[record["layer"]]
        )
        return {
            "matched": same and actual == expected,
            "top10": actual,
            "matrix_sha256": digest,
            "matrix_bytes": weights.stat().st_size,
            "uses_pinned_model_weights": True,
        }

    return gallery.artifact_record(target, replay)


def measure_arch(arch: str, args: argparse.Namespace, entry: gallery.Entry, fitting: Any) -> None:
    """Measure one architecture's oracles, controls and artifact.

    Parameters
    ----------
    arch:
        GPT-2 or Mamba.
    args:
        Parsed CLI arguments.
    entry:
        Measurement record receiving checks.
    fitting:
        Pinned reference estimator, needed for GPT-2.
    """
    # One architecture's checks share the same fitted matrices and residuals;
    # retaining that measurement scope exceeds the cognitive-complexity default.
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, MambaForCausalLM

    identifier, revision = (
        (MODEL_ID, MODEL_REVISION) if arch == "gpt2" else (MAMBA_ID, MAMBA_REVISION)
    )
    tokenizer = AutoTokenizer.from_pretrained(identifier, revision=revision)
    if args.quick:
        config = GPT2Config(
            n_layer=3, n_head=2, n_embd=16, vocab_size=50257, n_positions=128, use_cache=False
        )
        config._attn_implementation = "eager"
        model = GPT2LMHeadModel(config).eval()
    elif arch == "gpt2":
        model = GPT2LMHeadModel.from_pretrained(
            identifier, revision=revision, attn_implementation="eager", use_cache=False
        ).eval()
    else:
        model = MambaForCausalLM.from_pretrained(
            identifier, revision=revision, use_cache=False
        ).eval()
    addresses, blocks, norm = model_parts(model, arch)
    prompts = [tokenizer(text, return_tensors="pt").input_ids for text in CORPUS]
    reference = (
        fit_reference(model, tokenizer, fitting, len(blocks))
        if arch == "gpt2"
        else fit_autograd(model, blocks, prompts)
    )
    print(f"PROGRESS {arch}: independent estimator complete", flush=True)
    matrices, trace = fit_captured(model, addresses, prompts)
    errors = {
        index: float((matrices[index] - reference[index]).norm() / reference[index].norm())
        for index in matrices
    }
    probes = (args.prompt,) if args.prompt else PROBES
    tokens, residuals = capture_readouts(model, tokenizer, addresses, norm, matrices, probes)
    overlaps = {
        index: [
            overlap(actual, decode(model, norm, value, reference[index]))
            for actual, value in zip(tokens[index], residuals[index], strict=True)
        ]
        for index in matrices
    }
    entry.oracles[f"{arch}_jacobian"] = gallery.Check(
        "relative Frobenius error < 1e-3 at every pre-final layer",
        errors,
        all(value < J_ERROR_MAX for value in errors.values()),
    )
    entry.oracles[f"{arch}_readout"] = gallery.Check(
        "top-10 overlap >= 0.9 at every layer on every held-out prompt",
        overlaps,
        all(value >= OVERLAP_MIN for values in overlaps.values() for value in values),
    )
    layer = min(args.layer, len(blocks) - 2)
    other = (layer + max(1, (len(blocks) - 1) // 2)) % (len(blocks) - 1)
    wrong_error = float((matrices[layer].T - reference[layer]).norm() / reference[layer].norm())
    shuffled = [
        overlap(actual, decode(model, norm, value, matrices[other]))
        for actual, value in zip(tokens[layer], residuals[layer], strict=True)
    ]
    entry.controls[f"{arch}_transpose"] = gallery.Check(
        "transposed Jacobian fails relative-error bar (>= 1e-3)",
        {"layer": layer, "error": wrong_error},
        wrong_error >= J_ERROR_MAX,
    )
    entry.controls[f"{arch}_wrong_layer"] = gallery.Check(
        "applying the half-stack-distant layer's J drops mean top-10 overlap below 0.9",
        {"layer": layer, "other": other, "overlaps": shuffled},
        sum(shuffled) / len(shuffled) < OVERLAP_MIN,
    )
    artifact = save_lens(trace, matrices, residuals, model, norm, args.out, arch, layer)
    entry.measured[arch] = {
        "artifact": artifact,
        "top10": tokens,
        "oracle": "pinned reference implementation" if arch == "gpt2" else "full scalar autograd",
    }
    entry.oracles[f"{arch}_replay"] = gallery.Check(
        "portable trace and companion matrices reproduce held-out readout",
        artifact["replay"],
        artifact["replay"]["matched"],
    )
    entry.artifact = artifact


def main() -> int:
    """Fit and report both architecture measurements.

    Returns
    -------
    int
        Zero; the report carries the verdict.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--reference-repo")
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--prompt")
    parser.add_argument("--arch", choices=("both", "gpt2", "mamba"), default="both")
    args = parser.parse_args()
    if args.layer < 0:
        parser.error("layer must be nonnegative")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "corpus.txt").write_text("\n".join(CORPUS) + "\n")
    started = time.time()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    architectures = (
        ("gpt2",) if args.quick else (("gpt2", "mamba") if args.arch == "both" else (args.arch,))
    )
    entry = gallery.Entry(
        "jacobian_lens",
        CLAIM,
        {
            "gpt2_revision": MODEL_REVISION,
            "mamba_revision": MAMBA_REVISION,
            "reference_commit": REFERENCE_COMMIT,
            "corpus_sha256": CORPUS_SHA256,
            "seed": SEED,
            "skip_first": SKIP_FIRST,
            "quick": args.quick,
        },
    )
    fitting = reference_module(args.out, args.reference_repo) if "gpt2" in architectures else None
    for arch in architectures:
        measure_arch(arch, args, entry, fitting)
    entry.oracles["coverage"] = gallery.Check(
        "both pretrained GPT-2 small and Mamba-130m measured",
        list(architectures),
        architectures == ("gpt2", "mamba") and not args.quick,
    )
    entry.architectures = architectures
    entry.verbs = ("capture", "read", "save")
    entry.lines = {
        "torchlens": gallery.count_region_lines(Path(__file__), "torchlens"),
        "plain": gallery.count_region_lines(Path(__file__), "plain"),
        "reference": "not measured",
    }
    gallery.report(entry, args.out, "cpu", started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
