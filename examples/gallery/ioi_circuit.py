"""Recover the IOI head circuit with EAP-IG and test two negative controls.

Run ``python examples/gallery/ioi_circuit.py --out DIR [--quick]`` on CPU.
The full measurement uses 1,000 pinned MIB training rows, five integration
steps and seed 0. Quick runs exercise the pipeline and cannot pass the full oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import _gallery as gallery
import torch

import torchlens as tl
import torchlens.mechinterp as mi

CLAIM = (
    "Wang et al., Interpretability in the Wild: a Circuit for Indirect Object Identification "
    "in GPT-2 small (ICLR 2023, arXiv 2211.00593): a circuit of 26 heads implements IOI. "
    "This entry ranks heads by summed absolute outgoing EAP-IG edge scores."
)
MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"  # pragma: allowlist secret
DATA_ID = "mib-bench/ioi"
DATA_REVISION = "e5f3468f3af4c0883be35cd3bced8c711c95d286"  # pragma: allowlist secret
DATA_SHA256 = (
    "2ee41861d725eee2b71da3846872d5e5ea29d9d8c9aa3a8954f8430584b2f243"  # pragma: allowlist secret
)
SEED = 0
ROWS = 1000
STEPS = 5
CONTROL_ROWS = 32
BATCH_ROWS = 8
AUC_MIN = 0.80
SCRAMBLED_BAND = 0.10
REVERSED_MAX = -0.90
HEAD_GROUPS = {
    "duplicate token": ((0, 1), (0, 10), (3, 0)),
    "previous token": ((2, 2), (4, 11)),
    "induction": ((5, 5), (5, 8), (5, 9), (6, 9)),
    "s2 inhibition": ((7, 3), (7, 9), (8, 6), (8, 10)),
    "name mover": ((9, 6), (9, 9), (10, 0)),
    "backup name mover": ((9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (10, 10), (11, 2), (11, 9)),
    "negative": ((10, 7), (11, 10)),
}
# Wang membership and connections transcribed from the published ACDC IOI oracle,
# acdc/ioi/utils.py at bc99ace817974b5584b7ee203d596a8e2bbcd399.
CONNECTIONS = (
    ("input", "backup name mover", "kv"),
    ("input", "duplicate token", "qkv"),
    ("input", "name mover", "kv"),
    ("input", "negative", "kv"),
    ("input", "previous token", "qkv"),
    ("input", "s2 inhibition", "q"),
    ("backup name mover", "logits", ""),
    ("duplicate token", "s2 inhibition", "kv"),
    ("induction", "s2 inhibition", "kv"),
    ("name mover", "logits", ""),
    ("negative", "logits", ""),
    ("previous token", "induction", "kv"),
    ("s2 inhibition", "backup name mover", "q"),
    ("s2 inhibition", "name mover", "q"),
    ("s2 inhibition", "negative", "q"),
)


def top_heads(scores: dict[str, float], k: int) -> list[str]:
    """Rank heads by absolute outgoing edge mass.

    Returns
    -------
    list[str]
        Head names, descending score with lexical tie breaking.
    """
    heads = head_scores(scores)
    return sorted(heads, key=lambda name: (-heads[name], name))[:k]


def head_scores(scores: dict[str, float]) -> dict[str, float]:
    """Reduce edge scores to all 144 GPT-2 small heads.

    Returns
    -------
    dict
        Head name to sum of absolute outgoing scores.
    """
    heads = {f"a{layer}.h{head}": 0.0 for layer in range(12) for head in range(12)}
    for edge, score in scores.items():
        sender = edge.split("->")[0]
        if sender in heads:
            heads[sender] += abs(score)
    return heads


# region: plain
def roc_auc(scores: dict[str, float], positives: set[str]) -> float:
    """Compute pairwise ROC AUC, including half credit for ties.

    Returns
    -------
    float
        Probability a positive ranks above a negative.
    """
    yes = [score for name, score in scores.items() if name in positives]
    no = [score for name, score in scores.items() if name not in positives]
    return sum((a > b) + 0.5 * (a == b) for a in yes for b in no) / (len(yes) * len(no))


def wang_edges() -> set[str]:
    """Expand the paper's head-class connections into 182 graph edges.

    Returns
    -------
    set[str]
        Head-level oracle edges in component-graph vocabulary.
    """
    groups = {
        group: [f"a{layer}.h{head}" for layer, head in pairs]
        for group, pairs in HEAD_GROUPS.items()
    } | {"input": ["input"], "logits": ["logits"]}
    return {
        f"{sender}->{target}<{role}>" if role else f"{sender}->{target}"
        for source, destination, roles in CONNECTIONS
        for sender in groups[source]
        for target in groups[destination]
        for role in (roles or [""])
        if sender == "input"
        or target == "logits"
        or int(sender[1:].split(".")[0]) < int(target[1:].split(".")[0])
    }


# endregion


def load_rows(tokenizer: Any, count: int) -> list[dict[str, Any]]:
    """Download, hash-check and select the first MIB-compatible training rows.

    Returns
    -------
    list[dict]
        Rows whose three answer-name token lengths agree.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATA_ID, "train.json", repo_type="dataset", revision=DATA_REVISION)
    blob = Path(path).read_bytes()
    if hashlib.sha256(blob).hexdigest() != DATA_SHA256:
        raise ValueError("MIB train.json hash differs from the pinned input")
    rows = []
    for row in json.loads(blob):
        lengths = {
            len(tokenizer(" " + row["metadata"][key], add_special_tokens=False).input_ids)
            for key in ("indirect_object", "subject", "random_c")
        }
        if len(lengths) == 1:
            rows.append(row)
        if len(rows) == count:
            return rows
    raise ValueError(f"only {len(rows)} compatible rows, wanted {count}")


def token_batch(tokenizer: Any, rows: list[dict[str, Any]], scramble: bool) -> tuple:
    """Tokenize clean/corrupt prompts with BOS and right padding.

    Returns
    -------
    tuple
        Two token tensors and the metric's position and answer indices.
    """
    # Keep paired tokenization, seeded scrambling and answer alignment together;
    # this essential data-alignment logic exceeds the cognitive-complexity default.
    rng = random.Random(SEED)
    clean, corrupt, answers, subjects, last = [], [], [], [], []
    for row in rows:
        texts = (row["prompt"], row["s2_io_flip_counterfactual"]["prompt"])
        encoded = [[tokenizer.bos_token_id] + tokenizer.encode(text) for text in texts]
        if scramble:
            names = {
                tokenizer.encode(" " + row["metadata"][key])[0]
                for key in ("indirect_object", "subject", "random_c")
            }
            # Replace every name occurrence independently with an ordinary random token.
            # No repeated subject/indirect-object name structure remains.
            encoded = [
                [rng.randrange(1000, 20000) if x in names else x for x in ids] for ids in encoded
            ]
        clean.append(encoded[0])
        corrupt.append(encoded[1])
        last.append(len(encoded[0]) - 1)
        answers.append(tokenizer.encode(" " + row["metadata"]["indirect_object"])[0])
        subjects.append(tokenizer.encode(" " + row["metadata"]["subject"])[0])
    width = max(len(ids) for ids in clean + corrupt)
    tensors = [
        torch.tensor([ids + [tokenizer.bos_token_id] * (width - len(ids)) for ids in side])
        for side in (clean, corrupt)
    ]
    return (*tensors, torch.tensor(last), torch.tensor(answers), torch.tensor(subjects))


# region: torchlens
def score_rows(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    steps: int,
    *,
    reverse: bool = False,
    scramble: bool = False,
) -> dict[str, float]:
    """Measure mean EAP-IG edge scores over bounded batches.

    Returns
    -------
    dict
        Edge name to mean score, with no missing or unreachable edges hidden.
    """
    sums: dict[str, float] = {}
    for start in range(0, len(rows), BATCH_ROWS):
        clean, corrupt, last, answer, subject = token_batch(
            tokenizer, rows[start : start + BATCH_ROWS], scramble
        )
        batch = torch.arange(len(clean))

        def metric(
            logits: torch.Tensor,
            *,
            batch: torch.Tensor = batch,
            last: torch.Tensor = last,
            answer: torch.Tensor = answer,
            subject: torch.Tensor = subject,
        ) -> torch.Tensor:
            """Return MIB's negative IO-minus-subject logit difference."""
            return -(logits[batch, last, answer] - logits[batch, last, subject])

        table = mi.edge_scores(
            model,
            corrupt if reverse else clean,
            corrupted_input=clean if reverse else corrupt,
            metric=metric,
            steps=steps,
            example_reduction="sum",
        )
        for row in table.rows():
            if row.status != "ok" or row.score is None:
                raise RuntimeError(f"edge {row.address} has status {row.status}")
            name = row.address[0]
            sums[name] = sums.get(name, 0.0) + float(row.score)
        print(
            f"PROGRESS rows={min(start + BATCH_ROWS, len(rows))}/{len(rows)} reverse={reverse} scramble={scramble}",
            flush=True,
        )
    return {name: value / len(rows) for name, value in sums.items()}


def save_measurement(
    model: Any, ids: torch.Tensor, scores: dict[str, float], out: Path, k: int
) -> dict:
    """Save a representative capture and the complete score table, then replay the ranking.

    Returns
    -------
    dict
        Artifact record including the reloaded top-k heads.
    """
    trace = tl.trace(model, ids)
    trace.annotations["gallery_ioi"] = {"scores": scores, "k": k, "representative_only": True}
    target = out / "ioi_circuit.tlspec"
    trace.save(target, level="portable", overwrite=True)
    (out / "edge_scores.json").write_text(json.dumps(scores, sort_keys=True) + "\n")
    expected = top_heads(scores, k)

    def replay(loaded: Any) -> dict[str, Any]:
        """Recompute the ranking from the table persisted beside the captured values."""
        actual = top_heads(loaded.annotations["gallery_ioi"]["scores"], k)
        return {"matched": actual == expected, "top_heads": actual, "k": k}

    return gallery.artifact_record(target, replay)


# endregion


def draw_measurement(scores: dict[str, float], out: Path) -> None:
    """Draw the recorded head ROC and strongest head-to-head connections.

    Parameters
    ----------
    scores:
        Complete aggregate edge table from one run; no uncertainty intervals.
    out:
        Destination for PNG and vector SVG figures.
    """
    import matplotlib.pyplot as plt

    # Question: does outgoing attribution recover Wang membership? Primary: ROC.
    # Context: chance diagonal; all 144 heads, 26 positives, one fixed-seed run.
    # Diagnostic: 20 largest absolute head-to-head edges; q/k/v scores stay separate.
    # Expected direction: above chance. Audience: gallery; 760-pixel display width.
    heads = head_scores(scores)
    members = {f"a{layer}.h{head}" for group in HEAD_GROUPS.values() for layer, head in group}
    ordered = sorted(heads, key=lambda name: (-heads[name], name))
    true_positive = false_positive = 0
    xs, ys = [0.0], [0.0]
    for index, name in enumerate(ordered):
        true_positive += name in members
        false_positive += name not in members
        if index == len(ordered) - 1 or heads[name] != heads[ordered[index + 1]]:
            xs.append(false_positive / (len(heads) - len(members)))
            ys.append(true_positive / len(members))
    fig, ax = plt.subplots(figsize=(7.6, 5.2), layout="constrained")
    ax.plot(xs, ys, color="#0072b2", linewidth=2)
    ax.plot([0, 1], [0, 1], color="#888888", linestyle=":")
    ax.text(0.65, 0.5, "Chance", color="#555555")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Wang head membership: ROC AUC {roc_auc(heads, members):.4f}")
    for suffix in ("png", "svg"):
        fig.savefig(out / f"ioi_roc.{suffix}", dpi=200)
    plt.close(fig)
    draw_connections(scores, out)


def draw_connections(scores: dict[str, float], out: Path) -> None:
    """Draw the strongest directed head connections, with Wang classes colored.

    Parameters
    ----------
    scores:
        Signed edge scores; width encodes absolute magnitude, not causal necessity.
    out:
        Figure destination.
    """
    import matplotlib.pyplot as plt

    colors = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9", "#000000")
    fig, ax = plt.subplots(figsize=(7.6, 6.4), layout="constrained")
    edges = [name for name in scores if re.fullmatch(r"a\d+\.h\d+->a\d+\.h\d+<[qkv]>", name)]
    selected = sorted(edges, key=lambda name: (-abs(scores[name]), name))[:20]
    scale = max((abs(scores[name]) for name in selected), default=1.0)
    for name in selected:
        source, target = name.split("->")
        x0, y0 = map(int, re.findall(r"\d+", source))
        x1, y1 = map(int, re.findall(r"\d+", target))
        ax.annotate(
            "",
            (x1, y1),
            (x0, y0),
            arrowprops={
                "arrowstyle": "->",
                "color": "#888888",
                "alpha": 0.65,
                "lw": 0.5 + 2 * abs(scores[name]) / scale,
            },
            zorder=1,
        )
    ax.scatter(
        [i for i in range(12) for _ in range(12)],
        list(range(12)) * 12,
        s=18,
        color="#dddddd",
        zorder=2,
    )
    for (label, group), color in zip(HEAD_GROUPS.items(), colors, strict=True):
        ax.scatter(
            [layer for layer, _ in group],
            [head for _, head in group],
            s=48,
            color=color,
            label=label,
            zorder=3,
        )
    ax.set_xticks(range(12))
    ax.set_yticks(range(12))
    ax.set_xlabel("Layer (zero-based)")
    ax.set_ylabel("Head index (zero-based)")
    ax.set_xlim(-0.4, 11.4)
    ax.set_ylim(-0.4, 11.4)
    ax.set_title("Twenty strongest head-to-head attribution edges")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        fontsize=9,
        frameon=False,
        title="Wang head classes; grey nodes are other heads",
    )
    for suffix in ("png", "svg"):
        fig.savefig(out / f"ioi_circuit.{suffix}", dpi=200)
    plt.close(fig)


def main() -> int:
    """Run the fixed oracle, controls, persistence and report.

    Returns
    -------
    int
        Zero; the summary carries the scientific verdict.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--rows", type=int, default=ROWS)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--k", type=int, default=26)
    args = parser.parse_args()
    if min(args.rows, args.steps, args.k) < 1:
        parser.error("rows, steps and k must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    from transformers import AutoTokenizer, GPT2LMHeadModel

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = GPT2LMHeadModel.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, attn_implementation="eager"
    ).eval()
    count, steps = (2, 1) if args.quick else (args.rows, args.steps)
    rows = load_rows(tokenizer, count)
    scores = score_rows(model, tokenizer, rows, steps)
    (args.out / "edge_scores.json").write_text(json.dumps(scores, sort_keys=True) + "\n")
    members = {f"a{layer}.h{head}" for pairs in HEAD_GROUPS.values() for layer, head in pairs}
    auc = roc_auc(head_scores(scores), members)
    entry = gallery.Entry(
        "ioi_circuit",
        CLAIM,
        {
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "data": DATA_ID,
            "data_revision": DATA_REVISION,
            "data_sha256": DATA_SHA256,
            "seed": SEED,
            "rows": count,
            "steps": steps,
            "control_rows": min(CONTROL_ROWS, count),
        },
    )
    entry.oracles["wang_heads"] = gallery.Check(
        "head ROC AUC >= 0.80 on 1000 rows, m=5",
        {"auc": auc, "rows": count, "steps": steps},
        auc >= AUC_MIN and count == ROWS and steps == STEPS,
    )
    subset = rows[:CONTROL_ROWS]
    normal = score_rows(model, tokenizer, subset, steps)
    reversed_scores = score_rows(model, tokenizer, subset, steps, reverse=True)
    keys = sorted(normal)
    correlation = float(
        torch.corrcoef(
            torch.tensor(
                [[normal[k] for k in keys], [reversed_scores[k] for k in keys]], dtype=torch.float64
            )
        )[0, 1]
    )
    scrambled = score_rows(model, tokenizer, subset, steps, scramble=True)
    control_auc = roc_auc(head_scores(scrambled), members)
    entry.controls["scrambled_names"] = gallery.Check(
        "abs(head ROC AUC - 0.5) <= 0.10 on first 32 rows with independent random replacements at every name occurrence",
        {"auc": control_auc, "rows": len(subset)},
        abs(control_auc - 0.5) <= SCRAMBLED_BAND,
    )
    entry.controls["reversed_roles"] = gallery.Check(
        "Pearson(edge scores normal, clean/corrupt reversed) <= -0.90 on first 32 rows",
        {"pearson": correlation, "rows": len(subset)},
        correlation <= REVERSED_MAX,
    )
    restricted = {
        name: abs(value)
        for name, value in scores.items()
        if re.fullmatch(r"(?:input|a\d+\.h\d+)->(?:logits|a\d+\.h\d+<[qkv]>)", name)
    }
    edge_top = sorted(restricted, key=lambda name: (-restricted[name], name))[:182]
    oracle_edges = wang_edges()
    if len(oracle_edges) != 182:
        raise AssertionError(f"Wang oracle has {len(oracle_edges)} edges, expected 182")
    entry.measured = {
        "head_recall_at_26": len(set(top_heads(scores, 26)) & members) / 26,
        "edge_recall_at_182": len(set(edge_top) & oracle_edges) / 182,
        "top_heads": top_heads(scores, args.k),
        "exact_edge_patching": "not run: outside the CPU time box",
        "artifact_scope": "complete aggregate table plus one representative clean trace; intermediate integration traces are not retained",
    }
    ids = token_batch(tokenizer, rows[:1], False)[0]
    entry.artifact = save_measurement(model, ids, scores, args.out, args.k)
    entry.verbs = ("capture", "read", "save")
    entry.architectures = ("GPT-2 small",)
    entry.lines = {
        "torchlens": gallery.count_region_lines(Path(__file__), "torchlens"),
        "plain": gallery.count_region_lines(Path(__file__), "plain"),
        "reference": "not measured",
    }
    draw_measurement(scores, args.out)
    gallery.report(entry, args.out, "cpu", started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
