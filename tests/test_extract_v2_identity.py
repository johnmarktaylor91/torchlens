"""T-CALLABLE-IDENTITY: the D8 classifier battery (lane F18, item 15).

The merged acceptance shape: legitimate transforms classify COMPLETE with
zero false refusals, collision pairs SEPARATE (defaults, kw-only defaults,
bound methods differing only in instance state, closure tensors with
identical repr), must-refuse cases refuse AND name the offending reference
(the self-referential container demotes, never ``RecursionError``), and
mutable-state hazards are caught. Digests are stable across processes and
across TorchLens wrap states.
"""

from __future__ import annotations

import functools

import pytest
import torch
from torch import nn

from torchlens._extraction import classify_callable

_SCALE = 2.5


def _module_level_probe(t, factor=2.0):
    """Scale by a default (module-level: co_flags match a fresh interpreter)."""

    return t * factor


def _digest(fn) -> str:
    """Classify one callable and return its digest."""

    return classify_callable(fn)["digest"]


# --- legitimate transforms classify COMPLETE (false-refusal guard) -----------------


@pytest.mark.smoke
def test_legitimate_transforms_classify_complete() -> None:
    """The modal transform shapes all measure with nothing opaque."""

    tensor_stat = torch.tensor([1.0, 2.0])

    legit = [
        lambda t: t.mean(dim=1),
        lambda t: torch.nn.functional.relu(t),
        lambda t: t.float(),  # builtin float name must NOT refuse (D8)
        lambda t: t * _SCALE,  # module-level constant global
        lambda t: t - tensor_stat,  # closure tensor BY VALUE
        functools.partial(torch.clamp, min=0.0),
        torch.abs,
        lambda t, factor=2.0: t * factor,
        lambda t, *, bias=1.0: t + bias,
    ]
    for fn in legit:
        record = classify_callable(fn)
        assert record["classification"] == "complete", (
            record["opaque_references"],
            getattr(fn, "__qualname__", fn),
        )
        assert record["digest"].startswith("blake2b:")


@pytest.mark.smoke
def test_nn_module_and_bound_method_values_measure() -> None:
    """nn.Module callables fold forward + the D6 state digest."""

    torch.manual_seed(0)
    module = nn.Linear(3, 3)
    record = classify_callable(module)
    assert record["classification"] == "complete"

    class _Scaler:
        """Instance-stateful scaler."""

        def __init__(self, factor: float) -> None:
            """Store the factor."""

            self.factor = factor

        def scale(self, tensor: torch.Tensor) -> torch.Tensor:
            """Scale by the instance factor."""

            return tensor * self.factor

    bound = classify_callable(_Scaler(2.0).scale)
    assert bound["classification"] == "complete"


# --- collision pairs SEPARATE ---------------------------------------------------------


def test_default_argument_collisions_separate() -> None:
    """Two functions differing only in a default value get distinct digests."""

    def f_two(t, factor=2.0):
        """Scale by 2 by default."""

        return t * factor

    def f_three(t, factor=3.0):
        """Scale by 3 by default."""

        return t * factor

    assert _digest(f_two) != _digest(f_three)


def test_kwonly_default_collisions_separate() -> None:
    """Keyword-only defaults fold by value too."""

    def f_a(t, *, bias=1.0):
        """Add 1."""

        return t + bias

    def f_b(t, *, bias=2.0):
        """Add 2."""

        return t + bias

    assert _digest(f_a) != _digest(f_b)


def test_closure_tensor_collisions_separate() -> None:
    """Closure tensors with identical repr (truncation) still separate."""

    big_a = torch.zeros(10_000)
    big_b = torch.zeros(10_000)
    big_b[7777] = 1e-30  # invisible to repr truncation, visible to bytes

    def use_a(t):
        """Subtract big_a."""

        return t[..., :1] + big_a.sum()

    def use_b(t):
        """Subtract big_b."""

        return t[..., :1] + big_b.sum()

    # NOTE: distinct FUNCTIONS with distinct cells; the digest must separate
    # on the cell BYTES even though the code is identical up to names.
    assert _digest(use_a) != _digest(use_b)


def test_bound_method_instance_state_separates() -> None:
    """Bound methods differing only in __self__ state separate (D8)."""

    class _Scaler:
        """Instance-stateful scaler."""

        def __init__(self, factor: float) -> None:
            """Store the factor."""

            self.factor = factor

        def scale(self, tensor: torch.Tensor) -> torch.Tensor:
            """Scale by the instance factor."""

            return tensor * self.factor

    assert _digest(_Scaler(2.0).scale) != _digest(_Scaler(3.0).scale)


def test_partial_argument_collisions_separate() -> None:
    """functools.partial args/keywords fold by value."""

    assert _digest(functools.partial(torch.clamp, min=0.0)) != _digest(
        functools.partial(torch.clamp, min=1.0)
    )


# --- must-refuse cases refuse AND name the reference ------------------------------------


def test_set_closure_demotes_named() -> None:
    """Unordered containers are blind territory: partial, named."""

    blind = {"a", "b"}

    def uses_set(t):
        """Depend on a set's size."""

        return t * float(len(blind))

    record = classify_callable(uses_set)
    assert record["classification"] == "partial"
    assert any("set_unordered" in ref for ref in record["opaque_references"])


@pytest.mark.smoke
def test_self_referential_container_demotes_never_recurses() -> None:
    """A cyclic closure container demotes with a named cycle, never crashes."""

    cyclic: list = [1.0]
    cyclic.append(cyclic)

    def uses_cycle(t):
        """Depend on a self-referential list."""

        return t * cyclic[0]

    record = classify_callable(uses_cycle)
    assert record["classification"] == "partial"
    assert any("cycle" in ref for ref in record["opaque_references"])


def test_overbound_container_demotes_named() -> None:
    """A container past the size bound demotes instead of hashing forever."""

    huge = list(range(100_000))

    def uses_huge(t):
        """Depend on a huge list."""

        return t * float(huge[0])

    record = classify_callable(uses_huge)
    assert record["classification"] == "partial"
    assert any("container_over" in ref for ref in record["opaque_references"])


def test_non_allowlisted_module_demotes_named() -> None:
    """A module outside the launch allowlist demotes with its name."""

    import wave as wave_module  # stdlib but NOT in the pure allowlist

    def uses_wave(t):
        """Reference a non-allowlisted module."""

        return t * float(len(wave_module.__name__))

    record = classify_callable(uses_wave)
    assert record["classification"] == "partial"
    assert any("module(wave)" in ref for ref in record["opaque_references"])


@pytest.mark.smoke
def test_opaque_instance_demotes_named() -> None:
    """An arbitrary object global demotes with its type named."""

    class _Opaque:
        """No usable value semantics."""

        __slots__ = ()

    blob = _Opaque()

    def uses_blob(t):
        """Depend on an opaque object."""

        return t if blob else t

    record = classify_callable(uses_blob)
    assert record["classification"] == "partial"
    assert any("_Opaque" in ref for ref in record["opaque_references"])


# --- mutable-state hazards + stability ----------------------------------------------------


def test_mutable_closure_state_changes_digest() -> None:
    """Mutating observed closure state changes the digest (hazard caught)."""

    state = {"scale": 2.0}

    def uses_state(t):
        """Scale by mutable dict state."""

        return t * state["scale"]

    before = _digest(uses_state)
    state["scale"] = 3.0
    assert _digest(uses_state) != before


def test_wrap_state_does_not_change_torch_function_identity() -> None:
    """torch.abs folds as its pinned identity whether or not TL wrapped it.

    TorchLens wraps torch functions lazily; the wrapper is a FunctionType
    whose globals hold mutable capture registries, so identity must key on
    the pinned ``torch.abs@version``, never the wrapper's internals.
    """

    reference = classify_callable(torch.abs)
    import torchlens as tl

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU()).eval()
    tl.trace(model, torch.randn(2, 2))  # ensures wrappers installed
    wrapped = classify_callable(torch.abs)
    assert wrapped["digest"] == reference["digest"]
    assert wrapped["classification"] == "complete"


def test_digest_stable_across_processes() -> None:
    """The digest is a portable fact: identical in a fresh interpreter."""

    import subprocess
    import sys
    from pathlib import Path

    # exec() the SAME source both here and in the fresh interpreter: pytest's
    # assertion-rewriting import loader compiles this module's own functions
    # differently, which is a compilation-context difference, not a process
    # dependence.
    probe_source = "def probe(t, factor=2.0):\n    'Scale by a default.'\n    return t * factor\n"
    namespace: dict = {}
    # dont_inherit: exec() otherwise inherits the calling module's __future__
    # compiler flags (this file imports annotations), which land in co_flags.
    exec(compile(probe_source, "<probe>", "exec", dont_inherit=True), namespace)
    here = _digest(namespace["probe"])
    script = (
        "import sys; sys.path.insert(0, sys.argv[1])\n"
        "from torchlens._extraction import classify_callable\n"
        f"source = {probe_source!r}\n"
        "namespace = {}\n"
        "exec(compile(source, '<probe>', 'exec', dont_inherit=True), namespace)\n"
        "print(classify_callable(namespace['probe'])['digest'])\n"
    )
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", script, str(repo_root)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    fresh = result.stdout.strip()
    assert fresh == here, "the digest is process-independent"
