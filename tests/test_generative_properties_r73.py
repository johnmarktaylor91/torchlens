"""Seeded generative property sweeps for the three R73 ungenerated surfaces.

R73 (4 hunt passes): the repo's strongest contracts were covered by
enumerated examples only —

1. **selector/predicate algebra**: 85 combinator uses, a fixed 278-cell
   matrix, and ZERO generated ``&``/``|``/``~`` composition trees checked
   against a set-algebra oracle;
2. **the input-path key codec** (r69/r71 D): a fixed 11-key list and exactly
   4 hand-picked sentinels, on the surface whose whole job is escaping
   adversarial keys (mid-string ``\\x00``, near-marker collisions, NaN
   payloads, tuple nesting);
3. **merged canonical-JSON determinism**: asserted only as identity on
   already-canonical fixed payloads — no key-order-permutation adversary.

Each sweep is SEEDED (deterministic by default; override the seed with
``TORCHLENS_FUZZ_SEED`` for a fresh-seed fuzz leg — the R73 "exploration, not
just pinning" half without adding a property-testing dependency).
"""

from __future__ import annotations

import json
import os
import random
import struct
import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._input_walk import (
    decode_mapping_key,
    encode_mapping_key,
    reserved_input_path_components,
)
from torchlens._merged._artifact import canonical_json_bytes

#: Deterministic default; a nightly fuzz leg may inject fresh seeds.
_SEED = int(os.environ.get("TORCHLENS_FUZZ_SEED", "20260815"))


# ---------------------------------------------------------------------------
# 1. Selector algebra: random composition trees vs a set-algebra oracle
# ---------------------------------------------------------------------------


class _TinySelectorNet(nn.Module):
    """Small nested CNN giving the atoms distinct, overlapping match sets."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.features = nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU(), nn.Conv2d(2, 2, 3))
        self.head = nn.Linear(2 * 4 * 4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run conv/relu/add/flatten/linear so every atom matches something."""

        x = self.features(x)
        x = torch.relu(x)
        x = x + 1
        x = x.flatten(1)
        return self.head(x)


#: Atom name -> selector factory. Factories, not instances: every evaluated
#: composition builds FRESH selector objects, so the sweep also proves
#: composition does not corrupt shared atom state.
_ATOM_FACTORIES = {
    "relu": lambda: tl.func("relu"),
    "conv": lambda: tl.func("conv2d"),
    "features": lambda: tl.in_module("features"),
    "add": lambda: tl.func("add"),
    "linear": lambda: tl.func("linear"),
}


@pytest.fixture(scope="module")
def _selector_trace():
    """One cached trace evaluated by every composition in the sweep."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log = tl.trace(_TinySelectorNet(), torch.rand(1, 1, 8, 8))
    try:
        yield log
    finally:
        log.cleanup()


def _labels(log, selector) -> frozenset[str]:
    """Evaluate one selector post-hoc and return its matched label set."""

    return frozenset(str(label) for label in log.find_sites(selector, max_fanout=10**6).labels())


def _random_tree(rng: random.Random, depth: int):
    """Generate one random composition tree over the atom vocabulary."""

    if depth == 0 or rng.random() < 0.3:
        return ("atom", rng.choice(sorted(_ATOM_FACTORIES)))
    op = rng.choice(("and", "or", "not"))
    if op == "not":
        return ("not", _random_tree(rng, depth - 1))
    return (op, _random_tree(rng, depth - 1), _random_tree(rng, depth - 1))


def _build_selector(tree):
    """Materialize a composition tree into fresh selector objects."""

    kind = tree[0]
    if kind == "atom":
        return _ATOM_FACTORIES[tree[1]]()
    if kind == "not":
        return ~_build_selector(tree[1])
    left, right = _build_selector(tree[1]), _build_selector(tree[2])
    return (left & right) if kind == "and" else (left | right)


def _oracle(tree, atom_sets: dict[str, frozenset[str]], universe: frozenset[str]) -> frozenset[str]:
    """Pure set-algebra evaluation of a composition tree (the oracle)."""

    kind = tree[0]
    if kind == "atom":
        return atom_sets[tree[1]]
    if kind == "not":
        return universe - _oracle(tree[1], atom_sets, universe)
    left = _oracle(tree[1], atom_sets, universe)
    right = _oracle(tree[2], atom_sets, universe)
    return (left & right) if kind == "and" else (left | right)


# Kept: sole branch coverage of _ir/_selector_universe.py, _selection/resolver.py, _selection/selector_base.py and more; a sole mutation kill.
def test_selector_composition_matches_set_algebra_oracle(_selector_trace) -> None:
    """120 random &/|/~ trees evaluate exactly as set algebra over atom sets."""

    log = _selector_trace
    atom_sets = {name: _labels(log, factory()) for name, factory in _ATOM_FACTORIES.items()}
    relu = _ATOM_FACTORIES["relu"]
    universe = _labels(log, relu() | ~relu())
    # Guard against a degenerate oracle: atoms must be non-empty, distinct,
    # and strictly inside the universe, or the sweep proves nothing.
    assert all(atom_sets.values()), f"degenerate atom sets: {atom_sets}"
    assert len(set(atom_sets.values())) == len(atom_sets)
    assert all(labels < universe for labels in atom_sets.values())

    rng = random.Random(_SEED)
    mismatches = []
    for index in range(120):
        tree = _random_tree(rng, depth=3)
        expected = _oracle(tree, atom_sets, universe)
        actual = _labels(log, _build_selector(tree))
        if actual != expected:
            mismatches.append((index, tree, sorted(expected), sorted(actual)))
    assert not mismatches, (
        f"selector algebra diverged from the set-algebra oracle (seed {_SEED}); "
        f"first mismatches: {mismatches[:3]}"
    )


# ---------------------------------------------------------------------------
# 2. Input-path key codec: adversarial generated keys round-trip injectively
# ---------------------------------------------------------------------------


def _key_bits(value) -> object:
    """Comparison form that is exact for floats (NaN payloads, -0.0 sign)."""

    if type(value) is float:
        return ("f", struct.pack(">d", value))
    if type(value) is tuple:
        return tuple(_key_bits(item) for item in value)
    return (type(value).__name__, value)


def _random_scalar_key(rng: random.Random):
    """Draw one admitted scalar key, biased toward adversarial shapes."""

    roll = rng.random()
    if roll < 0.45:
        alphabet = "\x00|\\:abzé中 \ttl_k"
        base = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 8)))
        if rng.random() < 0.4:
            # Near-marker collisions: reserved sentinels and their neighbors.
            base = (
                rng.choice(
                    [*sorted(reserved_input_path_components()), "\x00tlk:", "\x00tlk:s:", "\x00"]
                )
                + base
            )
        return base
    if roll < 0.6:
        return rng.randint(-(2**40), 2**40)
    if roll < 0.7:
        return rng.choice([True, False, None])
    # Floats: normals, -0.0, infs, and NaNs with random payload bits.
    sub = rng.random()
    if sub < 0.5:
        return rng.uniform(-1e300, 1e300)
    if sub < 0.7:
        return rng.choice([0.0, -0.0, float("inf"), float("-inf")])
    payload = rng.getrandbits(51) | (0x7FF8 << 48)
    return struct.unpack(">d", payload.to_bytes(8, "big"))[0]


def _random_key(rng: random.Random, depth: int = 2):
    """Draw one admitted key: a scalar or a (nested) tuple of admitted keys."""

    if depth > 0 and rng.random() < 0.25:
        return tuple(_random_key(rng, depth - 1) for _ in range(rng.randint(0, 4)))
    return _random_scalar_key(rng)


# Kept: sole branch coverage of _input_walk.py; the catch of its regression fault.
def test_key_codec_round_trips_generated_adversarial_keys() -> None:
    """decode(encode(k)) == k bit-exactly for 400 generated adversarial keys."""

    rng = random.Random(_SEED)
    reserved = reserved_input_path_components()
    for index in range(400):
        key = _random_key(rng)
        token = encode_mapping_key(key)
        assert token not in reserved, (index, key, token)
        decoded = decode_mapping_key(token)
        assert _key_bits(decoded) == _key_bits(key), (
            f"codec round-trip broke at case {index} (seed {_SEED}): "
            f"{key!r} -> {token!r} -> {decoded!r}"
        )
        assert encode_mapping_key(key) == token, "codec is not deterministic"


# ---------------------------------------------------------------------------
# 3. Merged canonical JSON: key-insertion-order permutation invariance
# ---------------------------------------------------------------------------


def _random_json_value(rng: random.Random, depth: int):
    """Draw one JSON-able value with unicode keys and mixed scalar leaves."""

    if depth == 0 or rng.random() < 0.4:
        return rng.choice(
            [
                rng.randint(-1000, 1000),
                rng.uniform(-10, 10),
                "".join(rng.choice("abé中 z0") for _ in range(rng.randint(0, 6))),
                True,
                False,
                None,
            ]
        )
    if rng.random() < 0.5:
        return [_random_json_value(rng, depth - 1) for _ in range(rng.randint(0, 4))]
    return {
        f"k{index}_{rng.choice('abé中')}": _random_json_value(rng, depth - 1)
        for index in range(rng.randint(0, 5))
    }


def _shuffled_copy(value, rng: random.Random):
    """Deep copy with every dict rebuilt in a random key-insertion order."""

    if isinstance(value, dict):
        items = list(value.items())
        rng.shuffle(items)
        return {key: _shuffled_copy(inner, rng) for key, inner in items}
    if isinstance(value, list):
        return [_shuffled_copy(item, rng) for item in value]
    return value


# Kept: sole branch coverage of _merged/_artifact.py.
def test_canonical_json_bytes_is_key_order_invariant() -> None:
    """Equal payloads with permuted dict insertion orders serialize identically."""

    rng = random.Random(_SEED + 2)
    for index in range(100):
        payload = _random_json_value(rng, depth=3)
        permuted = _shuffled_copy(payload, rng)
        assert json.loads(json.dumps(payload)) == json.loads(json.dumps(permuted))
        assert canonical_json_bytes(payload) == canonical_json_bytes(permuted), (
            f"canonical_json_bytes is insertion-order-sensitive at case {index} "
            f"(seed {_SEED + 2}): {payload!r}"
        )
    # And it must remain VALUE-sensitive (the invariance is not constancy).
    assert canonical_json_bytes({"a": 1}) != canonical_json_bytes({"a": 2})


# Kept: sole branch coverage of _merged/_artifact.py, _merged/_errors.py; smoke tier id (frozen).
@pytest.mark.smoke
def test_canonical_json_bytes_refuses_noncanonical_inputs_typed() -> None:
    """R73: the canonical encoder is an integrity surface -- refuse, never coerce.

    Fail-before: ``{1: "a"}`` and ``{"1": "a"}`` (likewise ``True``/``"true"``)
    silently produced IDENTICAL bytes, so two structurally distinct payloads
    shared one attested SHA-256 digest; NaN/Inf, lone surrogates, bytes
    leaves, and mixed-type keys escaped as raw ValueError/UnicodeEncodeError/
    TypeError past the module's typed refusal surface.
    """

    from torchlens._merged import MergedErrorCode
    from torchlens._merged._errors import MergedArtifactError

    bad_payloads = [
        {1: "a"},  # int key: coerced to "1" -> digest collision with {"1": "a"}
        {True: "a"},  # bool key: coerced to "true"
        {1.0: "a"},  # float key
        {1: "a", "b": 2},  # mixed keys: raw TypeError from sort_keys
        {"a": float("nan")},
        {"a": float("inf")},
        {"nested": [{"deep": float("-inf")}]},
        {"a": "\udc80"},  # lone surrogate: escaped dumps, died at .encode
        {"a": b"raw-bytes"},
    ]
    for payload in bad_payloads:
        with pytest.raises(MergedArtifactError) as excinfo:
            canonical_json_bytes(payload)
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value, (
            f"payload {payload!r} refused with the wrong code"
        )

    # The collision pair itself: the string-keyed form still encodes, and it is
    # now the ONLY spelling that produces these bytes.
    assert canonical_json_bytes({"1": "a"}) == b'{"1":"a"}\n'
