"""Memory of the facet patching helpers: weight arguments, trace lifetime, repeated runs.

The helpers capture with ``save_args=True`` and keep a clean, a corrupted and a
patched trace alive at once. Each trace used to hold its own copy of every
weight argument; on GPT-2 small (tied 50257 x 768 embedding, consumed twice per
forward) ``tl.mechinterp.patch_heads_grid`` exceeded 5.9 GiB. Parameter
arguments of the helpers' internal traces are now saved by reference, and the
forward that writes a parameter falls back to the copy path.

These gate object lifetime (no helper trace, nor the model, outlives the call)
and repeated-run RSS in a fresh process, not one code path: a weak-key cache can
keep its own key alive through the cached value, which only a lifetime check
sees. The model is a hand-written GPT-2-shaped LM (the built-in
``GPT2Attention`` facet recipe, tied ``wte``/``lm_head``); the GPT-2 small
figure itself needs the real model.
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import textwrap
import weakref
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.mechinterp  # noqa: F401  (registers tl.mechinterp)
from torchlens._data_classes.trace import Trace
from torchlens._semantic import patching
from torchlens._semantic._state_guard import _CounterfactualStateGuard, _teardown

SEQ = 5


class GPT2Attention(nn.Module):
    """GPT-2-shaped causal self-attention: fused ``c_attn``, per-head split, ``c_proj``."""

    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        self.num_heads = n_head
        self.head_dim = n_embd // n_head
        self.embed_dim = n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        q, k, v = self.c_attn(x).split(self.embed_dim, dim=-1)
        q = q.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / self.head_dim**0.5
        causal = torch.ones(seq, seq, dtype=torch.bool).tril()
        pattern = scores.masked_fill(~causal, float("-inf")).softmax(-1)
        z = (pattern @ v).transpose(1, 2).reshape(batch, seq, self.embed_dim)
        return self.c_proj(z)


class _Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        self.attn = GPT2Attention(n_embd, n_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(x)


class TinyLM(nn.Module):
    """Two GPT-2-shaped blocks between a tied token embedding and ``lm_head``."""

    def __init__(self, vocab: int = 256, n_embd: int = 16, n_head: int = 4) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.wte = nn.Embedding(vocab, n_embd)
        self.h = nn.ModuleList([_Block(n_embd, n_head), _Block(n_embd, n_head)])
        self.lm_head = nn.Linear(n_embd, vocab, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.wte(ids)
        for block in self.h:
            x = block(x)
        return self.lm_head(x)


class _WritesItsWeight(TinyLM):
    """A forward that writes a parameter in place after reading it (version moves)."""

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        out = super().forward(ids)
        with torch.no_grad():
            self.h[0].attn.c_proj.weight.add_(0.0)
        return out


def _prompts(vocab: int) -> tuple[torch.Tensor, torch.Tensor]:
    clean = torch.randint(0, vocab, (1, SEQ), generator=torch.Generator().manual_seed(1))
    corrupt = torch.randint(0, vocab, (1, SEQ), generator=torch.Generator().manual_seed(2))
    return clean, corrupt


def _metric(trace: Any) -> torch.Tensor:
    logits = trace[trace.output_layers[0]].out
    return logits[0, -1, 3] - logits[0, -1, 7]


def _saved_weight_args(log: Any, model: nn.Module) -> list[tuple[torch.Tensor, nn.Parameter]]:
    """Return (saved argument, parameter) pairs for every saved arg shaped like a weight."""

    by_shape: dict[tuple[int, ...], list[nn.Parameter]] = {}
    for parameter in model.parameters():
        by_shape.setdefault(tuple(parameter.shape), []).append(parameter)
    pairs: list[tuple[torch.Tensor, nn.Parameter]] = []
    for op in log.ops:
        for value in [*(op.call.saved_args or ()), *(op.call.saved_kwargs or {}).values()]:
            if not isinstance(value, torch.Tensor):
                continue
            for parameter in by_shape.get(tuple(value.shape), ()):
                if torch.equal(value, parameter.detach()):
                    pairs.append((value, parameter))
                    break
    return pairs


def _shares_storage(value: torch.Tensor, parameter: torch.Tensor) -> bool:
    return value.untyped_storage().data_ptr() == parameter.untyped_storage().data_ptr()


def test_helper_traces_save_weight_arguments_by_reference() -> None:
    """Every weight argument of a helper trace is an alias of the parameter, not a copy."""

    model = TinyLM().eval()
    clean, corrupt = _prompts(256)
    guard = _CounterfactualStateGuard(model)
    guard.open()
    clean_log = corrupted_log = None
    try:
        clean_log, corrupted_log = patching._baseline_traces(
            model, clean, corrupt, trace_kwargs=None, guard=guard
        )
        for log in (clean_log, corrupted_log):
            pairs = _saved_weight_args(log, model)
            # The tied embedding alone is read twice (lookup and lm_head).
            assert len(pairs) >= 2 + 2 * 4
            assert all(_shares_storage(value, parameter) for value, parameter in pairs)
            assert not any(isinstance(value, nn.Parameter) for value, _ in pairs)
    finally:
        _teardown(guard, clean_log, corrupted_log)
    # A user's own save_args=True trace still copies.
    own = tl.trace(model, clean, capture=tl.options.CaptureOptions(save_args=True))
    try:
        pairs = _saved_weight_args(own, model)
        assert pairs and not any(_shares_storage(value, parameter) for value, parameter in pairs)
    finally:
        own.cleanup()


def test_a_forward_that_writes_a_parameter_recaptures_on_the_copy_path() -> None:
    """A moved version counter drops the by-reference trace and copies every argument."""

    model = _WritesItsWeight().eval()
    clean, corrupt = _prompts(256)
    guard = _CounterfactualStateGuard(model)
    guard.open()
    clean_log = corrupted_log = None
    try:
        clean_log, corrupted_log = patching._baseline_traces(
            model, clean, corrupt, trace_kwargs=None, guard=guard
        )
        for log in (clean_log, corrupted_log):
            pairs = _saved_weight_args(log, model)
            assert pairs and not any(
                _shares_storage(value, parameter) for value, parameter in pairs
            )
    finally:
        _teardown(guard, clean_log, corrupted_log)
    # The grid gives the same answer through the copy path as the plain model does.
    plain = TinyLM().eval()
    expected = tl.mechinterp.patch_heads_grid(plain, clean, corrupt, _metric).values
    got = tl.mechinterp.patch_heads_grid(model, clean, corrupt, _metric).values
    torch.testing.assert_close(got, expected)


def test_patch_heads_grid_matches_the_copy_path() -> None:
    """Keeping weights by reference changes no grid value."""

    model = TinyLM().eval()
    clean, corrupt = _prompts(256)
    by_reference = tl.mechinterp.patch_heads_grid(model, clean, corrupt, _metric).values
    original = patching._capture_by_reference
    patching._capture_by_reference = lambda _model, _guard, capture: capture()
    try:
        copied = tl.mechinterp.patch_heads_grid(model, clean, corrupt, _metric).values
    finally:
        patching._capture_by_reference = original
    torch.testing.assert_close(by_reference, copied, rtol=0, atol=0)
    assert by_reference.abs().sum() > 0


def test_no_helper_trace_or_the_model_outlives_the_call() -> None:
    """After the grid and per-head patching return, their traces and the model are freed."""

    gc.collect()
    # Hold the traces that already exist (an earlier test in this worker may
    # still own one) so the check below counts only traces this call created;
    # comparing bare counts fails when such a trace is freed mid-test.
    traces_before = [obj for obj in gc.get_objects() if isinstance(obj, Trace)]
    model = TinyLM().eval()
    clean, corrupt = _prompts(256)
    tl.mechinterp.patch_heads_grid(model, clean, corrupt, _metric)
    tl.facets.patching.activation_patch_attention_heads(
        model, clean, corrupt, _metric, facet_name="z"
    )
    model_ref = weakref.ref(model)
    del model
    gc.collect()
    assert model_ref() is None, "a helper kept the model alive after returning"
    survivors = [
        obj
        for obj in gc.get_objects()
        if isinstance(obj, Trace) and not any(obj is held for held in traces_before)
    ]
    assert not survivors, f"{len(survivors)} helper trace(s) outlived the call"


_RSS_SCRIPT = textwrap.dedent(
    """
    import gc, json, resource, sys
    sys.path.insert(0, sys.argv[1])
    import torch
    import torchlens as tl
    import torchlens.mechinterp
    from test_patching_helper_memory import TinyLM, _metric, _prompts

    scale = 1 if sys.platform == "darwin" else 1024  # ru_maxrss: bytes on macOS, KiB on Linux

    def peak_mb():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 2**20

    def rss_mb():
        try:
            import psutil
        except ImportError:
            with open("/proc/self/statm") as handle:
                return int(handle.read().split()[1]) * resource.getpagesize() / 2**20
        return psutil.Process().memory_info().rss / 2**20

    vocab = 50257
    model = TinyLM(vocab=vocab, n_embd=128, n_head=4).eval()
    params_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20
    clean, corrupt = _prompts(vocab)
    tl.trace(model, clean).cleanup()
    gc.collect()
    base_peak = peak_mb()
    after = []
    for _ in range(5):
        tl.mechinterp.patch_heads_grid(model, clean, corrupt, _metric)
        gc.collect()
        after.append(rss_mb())
    print(json.dumps({
        "params_mb": params_mb, "peak_growth_mb": peak_mb() - base_peak, "rss_mb": after,
    }))
    """
)


@pytest.mark.heavy
@pytest.mark.skipif(
    sys.platform not in ("darwin", "linux"), reason="needs ru_maxrss and a readable RSS"
)
def test_repeated_patch_heads_grid_peak_and_rss_stay_flat() -> None:
    """In a fresh process: peak growth stays near one weight copy, repeated runs add nothing.

    Measured on macOS (25 MiB of tied weights): 34 MiB peak growth by reference,
    207 MiB on the copy path; RSS after runs 1..5 flat within 4 MiB on both.
    """

    result = subprocess.run(
        [sys.executable, "-c", _RSS_SCRIPT, str(Path(__file__).parent)],
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
        env={"OMP_NUM_THREADS": "1", **_inherited_env()},
    )
    report = json.loads(result.stdout.strip().splitlines()[-1])
    params_mb = report["params_mb"]
    # The state guard's snapshot is one copy; the copy path held ~8 (2 per live trace).
    assert report["peak_growth_mb"] < 3 * params_mb, report
    rss = report["rss_mb"]
    assert rss[-1] - rss[0] < 24, report


def _inherited_env() -> dict[str, str]:
    import os

    return {key: value for key, value in os.environ.items() if key != "OMP_NUM_THREADS"}
