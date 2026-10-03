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
from torchlens.postprocess._site_key import (
    ROOT_CALL_INSTANCE,
    SITE_KEY_PREFIX,
    SiteKeyMinter,
    call_instance_id,
    escape_site_component,
    operation_witness,
    parse_site_key,
    render_site_key,
    site_axis,
    unescape_site_component,
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


def test_reference_encoder_vectors() -> None:
    # Exact vectors from the design memo / reference encoder.
    assert render_site_key(("a|b",), "relu", None, 1) == "s1|a%7Cb|relu||1"
    assert render_site_key((), "input", None, 1) == "s1||input||1"
    assert render_site_key(("enc", "enc.mlp"), "linear", 0, 2) == "s1|enc/enc.mlp|linear|0|2"
    assert escape_site_component("a/b%c|d") == "a%2Fb%25c%7Cd"
    assert escape_site_component("\x01") == "%01"
    # Unicode letters stay readable (unescaped).
    assert escape_site_component("unicodé") == "unicodé"
    assert unescape_site_component("a%2Fb%25c%7Cd") == "a/b%c|d"


@pytest.mark.smoke
def test_parse_is_exact_inverse() -> None:
    for site, layer_type, slot, ordinal in (
        (("a|b", "a/b.c:d"), "relu", None, 3),
        ((), "output", 0, 1),
        (("mods.100%",), "add_", 2, 11),
    ):
        key = render_site_key(site, layer_type, slot, ordinal)
        assert parse_site_key(key) == (site, layer_type, slot, ordinal)


def test_parse_refuses_malformed_keys() -> None:
    for bad in ("", "s2|a|relu||1", "s1|a|relu|1", "s1|a|relu||0", "s1|a|relu|x|1"):
        with pytest.raises(ValueError):
            parse_site_key(bad)


def test_site_axis_representation_parity() -> None:
    # The canonical normalizer is defined on BOTH live representations:
    # build-time (address, pass) pairs and serialized "address:pass" strings.
    pairs = (("enc", 1), ("enc.block:0", 2))
    strings = ("enc:1", "enc.block:0:2")
    assert site_axis(pairs) == site_axis(strings) == ("enc", "enc.block:0")
    assert call_instance_id(pairs) == call_instance_id(strings) == "enc.block:0:2"
    assert call_instance_id(()) == ROOT_CALL_INSTANCE


def test_minter_ordinals_restart_per_call_instance() -> None:
    minter = SiteKeyMinter()
    first = minter.mint((("m", 1),), "relu", None)
    second = minter.mint((("m", 1),), "relu", None)
    fresh_instance = minter.mint((("m", 2),), "relu", None)
    assert first == "s1|m|relu||1"
    assert second == "s1|m|relu||2"
    assert fresh_instance == "s1|m|relu||1"


# ---------------------------------------------------------------------------
# Real-capture properties (the memo's P1/P2/P4 probes on the real helpers)
# ---------------------------------------------------------------------------


def test_tied_loop_exact_key_pins() -> None:
    # Ground-truth pin: reused-module ops share ONE key across call
    # instances (P2) and the ordinal counter restarts per call instance.
    log = tl.trace(_Tied(), torch.randn(2, 8))
    keys = {label: log.ops[label].site_key for label in log.op_labels}
    assert keys["input_1:1"] == "s1||input||1"
    assert keys["output_1:1"] == "s1||output||1"
    for pass_index in (1, 2, 3):
        assert keys[f"linear_1_1:{pass_index}"] == "s1|lin|linear||1"
        assert keys[f"relu_1_2:{pass_index}"] == "s1|act|relu||1"
    assert len(set(keys.values())) == 4


def test_multi_output_slots_split_sites() -> None:
    # Co-outputs of one call occupy distinct output slots => distinct keys;
    # both slots stay call-instance-stable across the two cell calls.
    log = tl.trace(_CellLoop(), torch.randn(2, 4))
    keys = {label: log.ops[label].site_key for label in log.op_labels}
    slot_keys = {key for key in keys.values() if key is not None and "lstmcell" in key}
    assert slot_keys == {"s1|cell|lstmcell|0|1", "s1|cell|lstmcell|1|1"}


def test_every_retained_op_has_prefixed_key() -> None:
    # I-S1 capture-time totality on a fresh capture.
    for model, x in ((_Tied(), torch.randn(2, 8)), (_CellLoop(), torch.randn(2, 4))):
        log = tl.trace(model, x)
        for label in log.op_labels:
            key = log.ops[label].site_key
            assert isinstance(key, str) and key.startswith(SITE_KEY_PREFIX + "|"), (
                label,
                key,
            )


def test_uniqueness_per_key_and_call_instance() -> None:
    log = tl.trace(_Tied(), torch.randn(2, 8))
    seen: Counter = Counter()
    for label in log.op_labels:
        op = log.ops[label]
        stack = tuple(op.module_call_stack or ())
        seen[(op.site_key, stack[-1] if stack else ROOT_CALL_INSTANCE)] += 1
    assert all(count == 1 for count in seen.values())


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


def test_cross_capture_key_stability() -> None:
    # The bridging property: two captures of the same program agree on keys
    # even though process-local identity (barcodes, ids) differs.
    torch.manual_seed(0)
    first = tl.trace(_Tied(), torch.randn(2, 8))
    torch.manual_seed(1)
    second = tl.trace(_Tied(), torch.randn(2, 8))
    assert _key_multiset(first) == _key_multiset(second)


# ---------------------------------------------------------------------------
# Hostile module names (serialization round-trip through REAL captures)
# ---------------------------------------------------------------------------


class _HostileInner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.r = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.r(x)


def test_hostile_module_names_roundtrip_through_capture() -> None:
    hostile_names = ("a|b", "a/b", "a:b", "unicodé", "x_raw", "100%")

    class Hostile(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mods = nn.ModuleDict()
            for name in hostile_names:
                self.mods[name] = _HostileInner()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for module in self.mods.values():
                x = module(x)
            return x

    log = tl.trace(Hostile(), torch.randn(2, 3))
    relu_keys = [
        log.ops[label].site_key for label in log.op_labels if "relu" in (log.ops[label].type or "")
    ]
    assert len(relu_keys) == len(hostile_names)
    expected_sites = [(f"mods.{name}", f"mods.{name}.r") for name in hostile_names]
    parsed_sites = [parse_site_key(key)[0] for key in relu_keys]
    assert parsed_sites == expected_sites
    # Exact escaped rendering pinned for the two separator-bearing names.
    assert relu_keys[0] == "s1|mods.a%7Cb/mods.a%7Cb.r|relu||1"
    assert relu_keys[1] == "s1|mods.a%2Fb/mods.a%2Fb.r|relu||1"


def test_build_vs_serialized_axis_parity_on_real_capture() -> None:
    # I-S4's core: recomputing the site axis from the SERIALIZED
    # module_call_stack reproduces the build-time (op.modules) axis.
    log = tl.trace(_Tied(), torch.randn(2, 8))
    for label in log.op_labels:
        op = log.ops[label]
        assert site_axis(op.modules) == site_axis(op.module_call_stack)
        assert call_instance_id(op.modules) == call_instance_id(op.module_call_stack)


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


def test_tied_loop_structural_facts() -> None:
    # Pinned census fact: tied_loop mints 2 recurrent groups, 0 site-spanning
    # (across-call-instance recurrence shares keys).
    log = tl.trace(_Tied(), torch.randn(2, 8))
    assert _site_spanning_recurrent_groups(log) == (0, 2)


@pytest.mark.heavy
def test_gpt2_4l_structural_facts() -> None:
    # Pinned census facts on the gpt2_4l fixture: 199 retained ops, all four
    # recurrent groups are within-call residual-add pairs SPANNING sites, and
    # P1 holds (199 distinct keys, no per-instance collision).
    transformers = pytest.importorskip("transformers")
    config = transformers.GPT2Config(n_layer=4, n_head=4, n_embd=128, vocab_size=512)
    model = transformers.GPT2LMHeadModel(config)
    model.eval()
    with torch.no_grad():
        log = tl.trace(model, torch.randint(0, 512, (1, 16)))
    assert len(log.op_labels) == 199
    assert _site_spanning_recurrent_groups(log) == (4, 4)
    keys = [log.ops[label].site_key for label in log.op_labels]
    assert len(set(keys)) == 199


# ---------------------------------------------------------------------------
# Silent-alias pins (bare-label lookups must keep their referent site)
# ---------------------------------------------------------------------------


def test_bare_label_lookup_referents_pinned_by_site() -> None:
    # The census's silent-alias hazard: after any future grouping change a
    # stale bare label may keep resolving while denoting a DIFFERENT op.
    # Pin today's referents BY SITE so a silent referent change fails here
    # instead of shipping.
    log = tl.trace(_Tied(), torch.randn(2, 8))
    assert log["linear_1_1"].ops[1].site_key == "s1|lin|linear||1"
    assert log["relu_1_2"].ops[1].site_key == "s1|act|relu||1"
    assert log["linear_1_1:2"].site_key == "s1|lin|linear||1"
    assert log["relu_1_2:3"].site_key == "s1|act|relu||1"


# ---------------------------------------------------------------------------
# to_pandas column
# ---------------------------------------------------------------------------


def test_site_key_reaches_to_pandas() -> None:
    pytest.importorskip("pandas")
    log = tl.trace(_Tied(), torch.randn(2, 8))
    frame = log.to_pandas()
    assert "site_key" in frame.columns
    assert set(frame["site_key"]) == {
        "s1||input||1",
        "s1|lin|linear||1",
        "s1|act|relu||1",
        "s1||output||1",
    }


# ---------------------------------------------------------------------------
# Operation-witness selector (deepest operation frame, cc[-2])
# ---------------------------------------------------------------------------


class _Frame:
    def __init__(self, file: str, line_number: int) -> None:
        self.file = file
        self.line_number = line_number


class _StubOp:
    def __init__(self, code_context: list[_Frame] | None) -> None:
        self.code_context = code_context


def test_witness_selects_deepest_operation_frame() -> None:
    outer = _Frame("model.py", 10)
    inner = _Frame("model.py", 42)
    call_site = _Frame("driver.py", 7)
    # Live contexts are ordered shallow-to-deep and END with the trace
    # call-site entry: the operation frame is [-2], NEVER [0] (the outermost
    # forward line, which false-joins nested branch sites).
    assert operation_witness(_StubOp([outer, inner, call_site])) == ("model.py", 42)
    assert operation_witness(_StubOp([inner, call_site])) == ("model.py", 42)
    # Degenerate single-entry context (not producible by tl.trace) falls
    # back to its only frame; absent context is witness-absence.
    assert operation_witness(_StubOp([inner])) == ("model.py", 42)
    assert operation_witness(_StubOp([])) is None
    assert operation_witness(_StubOp(None)) is None


def test_witness_on_real_capture_is_operation_frame() -> None:
    log = tl.trace(_Tied(), torch.randn(2, 8))
    checked = 0
    for label in log.op_labels:
        op = log.ops[label]
        if op.type in ("input", "output", "buffer"):
            assert operation_witness(op) is None  # I/O boundary: witness-absent
            continue
        context = list(op.code_context or ())
        assert len(context) >= 2
        witness = operation_witness(op)
        # The selected frame is the deepest OPERATION frame [-2] -- never the
        # appended trace call-site entry [-1], and never the outermost
        # forward line [0] (the false-join class the selector exists to
        # refuse; module-invoked ops legitimately witness torch-library
        # frames here).
        assert witness == (context[-2].file, context[-2].line_number)
        assert witness != (context[-1].file, context[-1].line_number)
        if len(context) >= 3:
            assert witness != (context[0].file, context[0].line_number)
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


@pytest.mark.smoke
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


@pytest.mark.smoke
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


@pytest.mark.smoke
def test_is2_jax_multipass_group_still_catches_a_genuine_duplicate() -> None:
    """Within a genuine multi-occurrence group, two passes that collide on
    BOTH site_key and raw label (not merely the shared equivalence_class)
    must still raise -- the raw-label differentiator narrows the carve-out,
    it does not disable I-S2 inside the group.
    """

    ops = [
        _fake_op(
            label="lt_1_1",
            _label_raw="lt_1_1_raw",
            equivalence_class="lt",
            site_key="s1||lt||1",
        ),
        _fake_op(
            label="lt_2_1",
            _label_raw="lt_1_1_raw",  # forged duplicate raw label
            equivalence_class="lt",
            site_key="s1||lt||1",
        ),
    ]

    with pytest.raises(MetadataInvariantError, match="I-S2"):
        _check_site_key_uniqueness(ops, "site_key_invariants")
