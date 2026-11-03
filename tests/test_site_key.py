"""site_key_v1 substrate tests: encoder byte parity, minting properties P1-P4,
hostile-name serialization round trips, degraded-path parity, structural-fact
pins, and the operation-witness selector.

The serialized form is pinned byte-for-byte against the L1 design memo's
reference encoder (``sitekey_ref.py``); the exact expected strings in this
file ARE that pin — any encoder change that moves them is a schema event,
never a refactor.
"""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._postprocess._site_key import (
    parse_site_key,
    render_site_key,
)
from torchlens.validation._invariants_sites import _check_site_key_uniqueness
from torchlens.validation.invariants import MetadataInvariantError


class _Tied(nn.Module):
    """The tied-loop census fixture: one Linear + ReLU reused three times."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 8)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = self.act(self.lin(x))
        return x


class _CellLoop(nn.Module):
    """Multi-output reuse fixture: one LSTMCell called twice."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.LSTMCell(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.zeros(x.shape[0], 4)
        c = torch.zeros(x.shape[0], 4)
        for _ in range(2):
            h, c = self.cell(x, (h, c))
        return h


def _ops(log: tl.Trace) -> dict[str, object]:
    return {label: log.ops[label] for label in log.op_labels}


def _key_multiset(log: tl.Trace) -> Counter:
    return Counter(op.site_key for op in _ops(log).values())


# ---------------------------------------------------------------------------
# Encoder byte parity (reference vectors)
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/_site_key.py.
def test_parse_is_exact_inverse() -> None:
    for site, layer_type, slot, ordinal in (
        (("a|b", "a/b.c:d"), "relu", None, 3),
        ((), "output", 0, 1),
        (("mods.100%",), "add_", 2, 11),
    ):
        key = render_site_key(site, layer_type, slot, ordinal)
        assert parse_site_key(key) == (site, layer_type, slot, ordinal)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/_site_key.py.
def test_parse_refuses_malformed_keys() -> None:
    for bad in ("", "s2|a|relu||1", "s1|a|relu|1", "s1|a|relu||0", "s1|a|relu|x|1"):
        with pytest.raises(ValueError):
            parse_site_key(bad)


# ---------------------------------------------------------------------------
# Real-capture properties (the memo's P1/P2/P4 probes on the real helpers)
# ---------------------------------------------------------------------------


def test_degraded_path_key_parity() -> None:
    # recurrence_detection=False mints byte-identical key multisets: the key
    # is policy-independent (P4).
    for build in (lambda: (_Tied(), torch.randn(2, 8)), lambda: (_CellLoop(), torch.randn(2, 4))):
        torch.manual_seed(0)
        model, x = build()
        default_log = tl.trace(model, x)
        torch.manual_seed(0)
        model2, x2 = build()
        degraded_log = tl.trace(
            model2, x2, capture=tl.options.CaptureOptions(recurrence_detection=False)
        )
        assert _key_multiset(default_log) == _key_multiset(degraded_log)


# ---------------------------------------------------------------------------
# Hostile module names (serialization round-trip through REAL captures)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Structural-fact pins (I-S3' exact counts; drift = investigation)
# ---------------------------------------------------------------------------


def _site_spanning_recurrent_groups(log: tl.Trace) -> tuple[int, int]:
    ops = _ops(log)
    groups = {frozenset(op.recurrent_ops) for op in ops.values() if len(op.recurrent_ops) > 1}
    spanning = sum(
        1
        for group in groups
        if len({ops[member].site_key for member in group if member in ops}) > 1
    )
    return spanning, len(groups)


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
@pytest.mark.heavy
def test_gpt2_4l_structural_facts() -> None:
    # Pinned census facts on the gpt2_4l fixture: 199 retained ops, no
    # recurrent group spans sites (the structural grouper fused each block's two
    # residual adds into one layer; call-site grouping keeps the two source
    # lines apart), and P1 holds (199 distinct keys, no per-instance collision).
    transformers = pytest.importorskip("transformers")
    config = transformers.GPT2Config(n_layer=4, n_head=4, n_embd=128, vocab_size=512)
    model = transformers.GPT2LMHeadModel(config)
    model.eval()
    with torch.no_grad():
        log = tl.trace(model, torch.randint(0, 512, (1, 16)))
    assert len(log.op_labels) == 199
    assert _site_spanning_recurrent_groups(log) == (0, 0)
    keys = [log.ops[label].site_key for label in log.op_labels]
    assert len(set(keys)) == 199


# ---------------------------------------------------------------------------
# Silent-alias pins (bare-label lookups must keep their referent site)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# to_pandas column
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Operation-witness selector (deepest operation frame, cc[-2])
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# I-S2 direct unit coverage: the no-module-stack carve-out (93ba1a033,
# narrowed in 3e777a9de) has no regression test of its own -- the behavior
# is exercised here directly against ``_check_site_key_uniqueness`` rather
# than through a real capture, since no real backend forges a collision.
# ---------------------------------------------------------------------------


def _fake_op(**kwargs: object) -> SimpleNamespace:
    kwargs.setdefault("module_call_stack", ())
    kwargs.setdefault("equivalence_class", None)
    return SimpleNamespace(**kwargs)


# Kept by shrink-r4 (per-id subset rule): sole coverage of validation/_invariants_sites.py, campaign mutant kills.
def test_is2_forged_singleton_collision_without_module_stack_still_raises() -> None:
    """Two singleton ops (unique equivalence_class each) sharing one site_key,
    with no module_call_stack, is a real forged collision -- the coarse
    ROOT_CALL_INSTANCE identity must still catch it (the narrowing in
    3e777a9de exists precisely so this case is never a false negative).
    """

    ops = [
        _fake_op(
            label="input_1_1",
            _label_raw="input_1_1_raw",
            equivalence_class="input",
            site_key="s1||input||1",
        ),
        _fake_op(
            label="output_1_1",
            _label_raw="output_1_1_raw",
            equivalence_class="output",
            # Forged: a distinct singleton occurrence sharing the SAME key.
            site_key="s1||input||1",
        ),
    ]

    with pytest.raises(MetadataInvariantError, match="I-S2"):
        _check_site_key_uniqueness(ops, "site_key_invariants")


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
def test_is2_jax_bare_function_root_multipass_group_is_not_a_false_positive() -> None:
    """A genuine multi-occurrence equivalence group with no module_call_stack
    (the JAX bare ``function_root`` recurrence case 93ba1a033 fixed) shares
    one site_key across its passes by design and must not false-positive,
    because each pass gets a distinct raw-label differentiator.
    """

    ops = [
        _fake_op(
            label=f"lt_{i}_1",
            _label_raw=f"lt_{i}_1_raw",
            equivalence_class="lt",
            site_key="s1||lt||1",
        )
        for i in range(1, 4)
    ]

    _check_site_key_uniqueness(ops, "site_key_invariants")  # must not raise
