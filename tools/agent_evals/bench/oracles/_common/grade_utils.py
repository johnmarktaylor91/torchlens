"""Deterministic grading helpers shared by every task grader.

No judge model, no randomness: every function is a plain numerical check.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class Bar:
    name: str
    passed: bool
    value: float | None
    threshold: float | None
    note: str = ""


@dataclass
class Report:
    task: str
    bars: list[Bar] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.bars) and all(b.passed for b in self.bars)

    def add(self, name, passed, value=None, threshold=None, note=""):
        v = None if value is None else float(value)
        self.bars.append(Bar(name, bool(passed), v, threshold, note))

    def fail_all(self, names, note):
        for n in names:
            self.bars.append(Bar(n, False, None, None, note))

    def to_dict(self):
        return {"task": self.task, "passed": self.passed,
                "bars": [asdict(b) for b in self.bars]}

    def emit(self):
        for b in self.bars:
            flag = "PASS" if b.passed else "FAIL"
            print(f"  {flag} {b.name}: value={b.value} threshold={b.threshold} {b.note}",
                  file=sys.stderr)
        print(json.dumps(self.to_dict()))


def load_npz(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def load_json(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        return json.load(f)


def rel_fro(a, b):
    """||a - b||_F / ||b||_F, b is the oracle."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    den = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / den) if den > 0 else float(np.linalg.norm(a - b))


def max_abs(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape or a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def _ranks(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.shape != b.shape or a.size < 2:
        return float("nan")
    return pearson(_ranks(a), _ranks(b))


def topk_overlap(a, b, k):
    """Mean |top-k(a) & top-k(b)| / k over leading axes; last axis is scores or ids."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return 0.0
    a2 = a.reshape(-1, a.shape[-1])
    b2 = b.reshape(-1, b.shape[-1])
    tot = 0.0
    for x, y in zip(a2, b2):
        tot += len(set(x[:k].tolist()) & set(y[:k].tolist())) / k
    return tot / max(1, len(a2))


def roc_auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    if scores.shape != labels.shape:
        return float("nan")
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = _ranks(scores)
    return float((r[labels].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def iou(a, b):
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    if a.shape != b.shape:
        return 0.0
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def upper_tri(m):
    m = np.asarray(m)
    iu = np.triu_indices(m.shape[0], k=1)
    return m[iu]


def rdm_correlation(a, b):
    return pearson(upper_tri(a), upper_tri(b))


def grader_main(task_id, grade_fn):
    """Standard CLI for graders: grade_fn(task_dir, outputs, heldout_outputs, cache) -> Report."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--task-dir", required=True)
    p.add_argument("--outputs", required=True)
    p.add_argument("--heldout-outputs", default=None)
    p.add_argument("--cache", default=None)
    a = p.parse_args()
    cache = a.cache or os.path.join(a.task_dir, ".work", "oracle_cache")
    os.makedirs(cache, exist_ok=True)
    rep = grade_fn(a.task_dir, a.outputs, a.heldout_outputs, cache)
    rep.emit()
    return 0 if rep.passed else 1
