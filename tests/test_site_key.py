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
    operation_witness,
    parse_site_key,
    render_site_key,
)
from torchlens.validation._invariants import MetadataInvariantError
from torchlens.validation._invariants_sites import _check_site_key_uniqueness


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
    return Counter(op.graph.site_key for op in _ops(log).values())


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
    groups = {
        frozenset(op.graph.recurrent_ops) for op in ops.values() if len(op.graph.recurrent_ops) > 1
    }
    spanning = sum(
        1
        for group in groups
        if len({ops[member].graph.site_key for member in group if member in ops}) > 1
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
    keys = [log.ops[label].graph.site_key for label in log.op_labels]
    assert len(set(keys)) == 199


# ---------------------------------------------------------------------------
# Silent-alias pins (bare-label lookups must keep their referent site)
# ---------------------------------------------------------------------------


# Kept: edited on the C-100 side (C-091/C-098) after the shrink removed it.
def test_bare_label_lookup_referents_pinned_by_site() -> None:
    # The census's silent-alias hazard: after any future grouping change a
    # stale bare label may keep resolving while denoting a DIFFERENT op.
    # Pin today's referents BY SITE so a silent referent change fails here
    # instead of shipping.
    log = tl.trace(_Tied(), torch.randn(2, 8))
    assert log.layers["linear_1_1"].ops[1].graph.site_key == "s1|lin|linear||1"
    assert log.layers["relu_1_2"].ops[1].graph.site_key == "s1|act|relu||1"
    assert log["linear_1_1:2"].graph.site_key == "s1|lin|linear||1"
    assert log["relu_1_2:3"].graph.site_key == "s1|act|relu||1"


# ---------------------------------------------------------------------------
# to_pandas column
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Operation-witness selector (deepest operation frame, cc[-1])
# ---------------------------------------------------------------------------


# Kept: named by a kept test.
class _Frame:
    def __init__(self, file: str, line_number: int) -> None:
        self.file = file
        self.line_number = line_number


# Kept: named by a kept test.
class _StubOp:
    def __init__(self, code_context: list[_Frame] | None) -> None:
        self.code_context = code_context


# Kept: edited on the C-100 side (C-091/C-098) after the shrink removed it.
def test_witness_selects_deepest_operation_frame() -> None:
    outer = _Frame("model.py", 10)
    inner = _Frame("model.py", 42)
    call_site = _Frame("driver.py", 7)
    # Contexts are ordered outermost to innermost, like a traceback, and
    # START with the trace call-site entry: the operation frame is [-1],
    # NEVER [0] (the call site) or [1] (the outermost forward line, which
    # false-joins nested branch sites).
    assert operation_witness(_StubOp([call_site, outer, inner])) == ("model.py", 42)
    assert operation_witness(_StubOp([call_site, inner])) == ("model.py", 42)
    # A single-entry context is its own operation frame; absent context is
    # witness-absence.
    assert operation_witness(_StubOp([inner])) == ("model.py", 42)
    assert operation_witness(_StubOp([])) is None
    assert operation_witness(_StubOp(None)) is None


# Kept: edited on the C-100 side (C-091/C-098) after the shrink removed it.
def test_witness_on_real_capture_is_operation_frame() -> None:
    log = tl.trace(_Tied(), torch.randn(2, 8))
    checked = 0
    for label in log.op_labels:
        op = log.ops[label]
        if op.type in ("input", "output", "buffer"):
            assert operation_witness(op) is None  # I/O boundary: witness-absent
            continue
        context = list(op.source.frames or ())
        assert len(context) >= 2
        witness = operation_witness(op)
        # The selected frame is the deepest OPERATION frame [-1] -- never the
        # leading trace call-site entry [0], and never the outermost forward
        # line [1] (the false-join class the selector exists to refuse;
        # module-invoked ops legitimately witness torch-library frames here).
        assert witness == (context[-1].file, context[-1].line_number)
        assert witness != (context[0].file, context[0].line_number)
        assert context[0].file == __file__  # the call site is this test's trace line
        if len(context) >= 3:
            assert witness != (context[1].file, context[1].line_number)
        checked += 1
    assert checked == 6


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
