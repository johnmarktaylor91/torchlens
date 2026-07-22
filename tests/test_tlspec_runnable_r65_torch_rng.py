"""r65 CLUSTER Z: torch RNG API surface completeness -- no false VERIFIED, no collateral.

Finding (r64, Sol correctness Finding 2): the host-nondeterminism registry covered
clocks / OS entropy / python-``random`` / numpy but NOT torch's OWN Python-level RNG
APIs -- an in-forward ``torch.seed()`` / ``torch.initial_seed()`` (off-seed) /
``torch.manual_seed()`` produced a false VERIFIED. The deciding asymmetry: sparse
replay RE-EXECUTES the recorded tensor RNG ops under the run seed but NEVER re-executes
host code, so a Python-level mutation of the global torch engine desyncs every
downstream DAG RNG op from both the capture and the fresh-run oracle.

Frozen dispositions (r65 PLAN_AGREED, both labs converged):

* ``seed`` family = ENTROPY -> permanent ceiling (``capture_seed=None``).
* ``manual_seed`` / ``set_rng_state`` family / ``fork_rng`` (transitively) = in-forward
  MUTATION -> permanent ceiling.
* ``initial_seed`` family = REPLAYABLE_READ -> consumed-flag only: run at the capture
  seed stays VERIFIED, any other/absent seed ceilings (the python-``random`` analog).
* ``get_rng_state`` family = NO ROW -- the state-tensor return is already covered by
  the r39 tensor->host escape belt (branch-on-state-bytes -> INCOMPLETE_SCALAR_ESCAPE,
  never VERIFIED; store-only stays VERIFIED); a row would over-ceiling
  ``torch.utils.checkpoint(preserve_rng_state=True)``, which round-trips
  VERIFIED+ATTESTED today.

The meta-tests are the anti-edge: a torch upgrade that grows the RNG surface (a new
module endpoint or a new ``torch._C.Generator`` method) is a RED test until it is given
an explicit disposition -- the r39 "one more name" doctrine, now covering torch itself.

r67 C1 (r66 free-F1 / corr1-1 / hon1-F5 / hon1-F6) extends the doctrine with RETURN
CLOSURE over ALL Generator receivers: every public ``torch.Generator`` method carries a
:class:`GeneratorMethodRow` whose per-receiver-class dispositions are machine-checked to
be closed under the method's return family (a host scalar can never leave the classifier
unwitnessed on ANY receiver; Generator returns are structural only because they re-enter
the same classifier), the default identity set is a dynamically re-resolved routing
cache, and the module-discovery meta-test owns NO module list of its own (it swept
``torch.mtia`` and ``torch.xpu.random`` into the surface -- the shared production/test
blind spot r66 hon1-F6 flagged).
"""

from __future__ import annotations

import shutil
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io.runnable import build_sparse_run_descriptor
from torchlens.options import CaptureOptions
from torchlens.runnable import (
    NumericAttestationStatus,
    PathFaithfulness,
    WitnessCompleteness,
)
from torchlens.utils.rng import (
    _DEFAULT_GENERATOR_HOLDER_MODULES,
    _TORCH_RNG_DEVICE_SPEC,
    _TORCH_RNG_MODULE_SPECS,
    GENERATOR_METHOD_DISPOSITIONS,
    GENERATOR_METHOD_TABLE,
    GENERATOR_RETURN_FAMILIES,
    HOST_NONDETERMINISM_REGISTRY,
    TORCH_RNG_DISPOSITIONS,
    TORCH_RNG_SURFACE,
    _torch_rng_holder_module,
    host_nondeterminism_monitor,
    log_current_rng_states,
    set_random_seed,
    set_rng_from_saved_states,
)

_MONITORED_DISPOSITIONS = frozenset({"entropy", "mutation", "replayable_read"})

# Generator-table dispositions that land in the permanent-ceiling ``channels`` set.
_CEILING_DISPOSITIONS = frozenset({"entropy", "mutation", "instance_read"})

# Generator methods that may raise a capability error on unsupported engines (CPU
# philox offset / graph-safe state). The c_call classifier marks AT ENTRY, so the
# capability raise never under-marks; the live return-family check skips them when
# the scratch engine cannot execute them.
_CAPABILITY_GATED_METHODS = frozenset(
    {"get_offset", "set_offset", "graphsafe_get_state", "graphsafe_set_state"}
)

# Pre-window held references (module import time), the r41 held-ref spelling.
_HELD_MANUAL_SEED = torch.manual_seed
_HELD_INITIAL_SEED = torch.initial_seed


def _rng_surface_names(module: types.ModuleType) -> dict[str, Any]:
    """Enumerate the RNG-relevant public names of one torch surface module."""

    names: dict[str, Any] = {}
    for name in dir(module):
        if name.startswith("_"):
            continue
        try:
            value = getattr(module, name, None)
        except Exception:
            continue
        if isinstance(value, types.ModuleType):
            continue
        lowered = name.lower()
        if (
            "seed" in lowered
            or "rng" in lowered
            or name in ("Generator", "default_generator", "default_generators")
        ):
            names[name] = value
    return names


def _discover_rng_bearing_modules() -> dict[str, dict[str, Any]]:
    """Independent, no-test-owned-list discovery of loaded RNG-bearing torch modules.

    Sweeps ``sys.modules`` for every already-loaded PUBLIC-path torch module (no
    imports, so no surface is created by the test itself) and returns each module's
    RNG-relevant public endpoints. This is the r67 anti-blind-spot: r66 hon1-F6 found
    that production AND its meta-test shared one hand-maintained module list that
    omitted ``torch.mtia`` -- this sweep owns no list, so a torch upgrade or import
    graph change that loads one more RNG-bearing module goes RED here.
    """

    found: dict[str, dict[str, Any]] = {}
    for module_name, module in sorted(list(sys.modules.items())):
        if not isinstance(module, types.ModuleType):
            continue
        if module_name != "torch" and not module_name.startswith("torch."):
            continue
        if any(segment.startswith("_") for segment in module_name.split(".")):
            continue
        endpoints = _rng_surface_names(module)
        if endpoints:
            found[module_name] = endpoints
    return found


@pytest.fixture()
def _torch_rng_state_guard():
    """Snapshot/restore global torch RNG state around a mutating monitor probe."""

    state = torch.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available() and torch.cuda.is_initialized()
        else None
    )
    try:
        yield
    finally:
        torch.set_rng_state(state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


# ======================================================================================
# T-Z1 -- surface enumeration + registry derivation meta-tests (the anti-edge)
# ======================================================================================


@pytest.mark.smoke
def test_torch_rng_surface_enumeration_complete() -> None:
    """Every discovered torch RNG endpoint has an explicit frozen disposition.

    r67: the sweep is the INDEPENDENT no-list module discovery -- a torch upgrade (or
    an import-graph change) that loads one more RNG-bearing public torch module or
    endpoint is a RED test until classified in ``TORCH_RNG_SURFACE``, never a silent
    false VERIFIED. Coverage is granted by exactly three rules: (a) an explicit table
    row for the spelling; (b) object IDENTITY with an already-classified row's object
    (re-export aliases -- receiver-identity classification and the held-code layer are
    spelling-independent, and only identity-bearing objects qualify: callables,
    classes, Generator instances -- never interned data like the empty
    ``default_generators`` tuple); (c) a non-torch-owned class/function/typing form
    that merely shares an RNG-ish name (``from typing import Generator``). Anything
    else fails.
    """

    table_targets = {row.target for row in TORCH_RNG_SURFACE}
    covered_object_ids: set[int] = set()
    for row in TORCH_RNG_SURFACE:
        module_path, _, attr_name = row.target.rpartition(".")
        holder = _torch_rng_holder_module(module_path)
        if holder is not None and hasattr(holder, attr_name):
            covered_object_ids.add(id(getattr(holder, attr_name)))

    discovered = _discover_rng_bearing_modules()
    # Sanity: the sweep itself must be alive (an empty sweep would vacuously pass).
    assert {"torch", "torch.random", "torch.cuda"} <= set(discovered)

    for module_path, endpoints in discovered.items():
        for name, value in endpoints.items():
            target = f"{module_path}.{name}"
            if target in table_targets:
                continue  # (a) explicit row for this spelling
            if (callable(value) or isinstance(value, (type, torch.Generator))) and id(
                value
            ) in covered_object_ids:
                continue  # (b) identity alias of a classified object
            if (
                isinstance(value, (type, types.FunctionType, types.BuiltinFunctionType))
                or type(value).__module__ == "typing"
            ):
                owner = getattr(value, "__module__", None)
                if isinstance(owner, str) and owner.split(".")[0] != "torch":
                    continue  # (c) non-torch-owned name collision
            raise AssertionError(
                f"torch RNG endpoint {target} has NO frozen disposition; classify it "
                "in TORCH_RNG_SURFACE / _TORCH_RNG_MODULE_SPECS (rng.py) before "
                "shipping"
            )
    for row in TORCH_RNG_SURFACE:
        assert row.disposition in TORCH_RNG_DISPOSITIONS, row


@pytest.mark.smoke
def test_mtia_and_xpu_random_surfaces_classified() -> None:
    """r66 hon1-F6 named pin: the mtia / xpu.random module specs actually resolve.

    Feature-detected: a torch build without the module skips. On torch 2.8 the mtia
    surface is ``get_rng_state``/``set_rng_state``; any upgrade that grows it lights
    up through the same ``hasattr`` detection plus the discovery sweep above.
    """

    targets = {row.target for row in TORCH_RNG_SURFACE}
    spec_paths = {path for path, _spec in _TORCH_RNG_MODULE_SPECS}
    assert {"torch.mtia", "torch.xpu.random"} <= spec_paths
    for module_path in ("torch.mtia", "torch.xpu", "torch.xpu.random"):
        module = _torch_rng_holder_module(module_path)
        if module is None:
            continue
        for name in ("seed", "manual_seed", "set_rng_state", "get_rng_state"):
            if hasattr(module, name):
                assert f"{module_path}.{name}" in targets, f"{module_path}.{name}"


@pytest.mark.smoke
def test_mtia_mutation_spelling_marks_fail_closed() -> None:
    """The mtia mutation patch marks at entry even when the deviceless call raises."""

    mtia = _torch_rng_holder_module("torch.mtia")
    if mtia is None or not hasattr(mtia, "set_rng_state"):
        pytest.skip("torch.mtia RNG surface not present on this torch")
    state = torch.get_rng_state()
    with host_nondeterminism_monitor(None) as result:
        try:
            mtia.set_rng_state(state)
        except Exception:
            pass
    assert "torch.mtia.set_rng_state" in result.channels


@pytest.mark.smoke
def test_default_generator_resolver_covers_every_device_spec_module() -> None:
    """The dynamic-membership resolver spans exactly the device-spec base modules.

    A future device namespace added to ``_TORCH_RNG_MODULE_SPECS`` with the device
    spec but missing from the resolver's holder list is a RED test (its default
    generators would silently classify as private receivers).
    """

    device_bases = {
        path.removesuffix(".random")
        for path, spec in _TORCH_RNG_MODULE_SPECS
        if spec is _TORCH_RNG_DEVICE_SPEC
    }
    assert device_bases == set(_DEFAULT_GENERATOR_HOLDER_MODULES)


@pytest.mark.smoke
def test_torch_rng_monitored_rows_match_registry() -> None:
    """The registry's torch module-patch rows derive 1:1 from the disposition table."""

    surface_monitored = {
        row.target: row.disposition
        for row in TORCH_RNG_SURFACE
        if row.disposition in _MONITORED_DISPOSITIONS
    }
    registry_torch = {
        row.target: row.classification
        for row in HOST_NONDETERMINISM_REGISTRY
        if row.target.startswith("torch") and row.strategy == "module_patch"
    }
    assert registry_torch == surface_monitored
    receiver_rows = [
        row
        for row in HOST_NONDETERMINISM_REGISTRY
        if row.target == "torch.Generator" and row.strategy == "receiver_profile"
    ]
    assert len(receiver_rows) == 1, "all-receiver Generator receiver_profile row missing"


@pytest.mark.smoke
def test_get_rng_state_family_has_no_registry_row() -> None:
    """The Z-a NO-ROW ruling is structural: get_rng_state must never grow a row.

    The state-tensor return rides the r39 escape belt; a row would over-ceiling
    ``torch.utils.checkpoint(preserve_rng_state=True)`` (VERIFIED+ATTESTED today).
    """

    for row in HOST_NONDETERMINISM_REGISTRY:
        assert "get_rng_state" not in row.target, (
            f"{row.target}: get_rng_state family must stay structurally covered "
            "(escape belt), never a monitor row -- see the r65 Z-a ruling"
        )
    for row in TORCH_RNG_SURFACE:
        if "get_rng_state" in row.target:
            assert row.disposition == "structurally_covered", row


@pytest.mark.smoke
def test_torch_rng_monitor_installs_and_restores_every_patch() -> None:
    """Every monitored row's patch installs in-window and restores identity-exact."""

    resolvable = []
    for row in TORCH_RNG_SURFACE:
        if row.disposition not in _MONITORED_DISPOSITIONS:
            continue
        module_path, _, attr_name = row.target.rpartition(".")
        holder = _torch_rng_holder_module(module_path)
        if holder is not None and hasattr(holder, attr_name):
            resolvable.append((row.target, holder, attr_name, getattr(holder, attr_name)))
    assert resolvable, "no monitored torch RNG rows resolved -- table build broken"
    with host_nondeterminism_monitor(None):
        for target, holder, attr_name, pre in resolvable:
            assert getattr(holder, attr_name) is not pre, (
                f"monitored row {target} left UNPATCHED inside the monitor window"
            )
    for target, holder, attr_name, pre in resolvable:
        assert getattr(holder, attr_name) is pre, (
            f"monitor exit did not restore {target} to the identical pre-window object"
        )


@pytest.mark.smoke
def test_generator_method_table_return_closure() -> None:
    """r67 C1 immunizer: one row per public Generator name, closed under its return.

    r66 corr1-1's lesson verbatim: "the enumeration test proves every method has a
    row, but not that a row's disposition is semantically closed under its return
    value." This test checks BOTH: a torch upgrade that adds a public Generator name
    is RED until it has exactly one row, and every row's per-receiver-class
    dispositions must be closed under its declared return family --

    * ``host_scalar``: a marking disposition in BOTH columns (a bare Python int can
      never leave the classifier unwitnessed on ANY receiver); ``replayable_read`` is
      legal ONLY in the proven-default column, the non-default column must ceiling;
    * ``state_tensor``: structural columns must carry the named r39 escape-belt proof;
    * ``generator`` / ``self_generator``: structural columns must carry the named
      classifier-closure proof (the returned Generator re-enters this classifier);
    * ``device_attr``: inert both columns, and genuinely non-callable.
    """

    generator_type = type(torch.default_generator)
    assert generator_type is torch.Generator
    public_names = {name for name in dir(generator_type) if not name.startswith("_")}
    table_methods = [row.method for row in GENERATOR_METHOD_TABLE]
    assert len(table_methods) == len(set(table_methods)), "duplicate Generator rows"
    assert set(table_methods) == public_names, (
        "GENERATOR_METHOD_TABLE out of sync with the public torch.Generator surface: "
        f"missing rows {sorted(public_names - set(table_methods))}, orphan rows "
        f"{sorted(set(table_methods) - public_names)}"
    )
    for row in GENERATOR_METHOD_TABLE:
        assert row.return_family in GENERATOR_RETURN_FAMILIES, row
        for disposition in (row.default_disposition, row.nondefault_disposition):
            assert disposition is None or disposition in GENERATOR_METHOD_DISPOSITIONS, row
        if row.return_family == "host_scalar":
            assert row.default_disposition is not None, (
                f"{row.method}: host-scalar return unwitnessed on default receivers"
            )
            assert row.nondefault_disposition in _CEILING_DISPOSITIONS, (
                f"{row.method}: host-scalar return must CEILING on non-default "
                "receivers (instance history is untracked; replayable is "
                "default-column-only)"
            )
        elif row.return_family == "state_tensor":
            if row.default_disposition is None or row.nondefault_disposition is None:
                assert "r39" in row.note, (
                    f"{row.method}: structural state-tensor column without a named "
                    "escape-belt proof"
                )
        elif row.return_family in ("generator", "self_generator"):
            if row.default_disposition is None or row.nondefault_disposition is None:
                assert "classifier" in row.note, (
                    f"{row.method}: structural Generator-return column without a "
                    "named classifier-closure proof"
                )
        else:  # device_attr
            assert row.default_disposition is None and row.nondefault_disposition is None
            assert not callable(getattr(generator_type, row.method, None)), row


@pytest.mark.smoke
def test_generator_method_table_live_return_families() -> None:
    """Each row's declared return family matches the LIVE return on a scratch engine.

    Capability-gated methods (CPU philox offset / graph-safe state) may raise on the
    scratch engine; anything else raising, or returning outside its declared family,
    is a RED test -- the closure argument is only sound if the families are real.
    """

    def _args_for(scratch: torch.Generator, method: str) -> tuple[Any, ...]:
        if method == "manual_seed":
            return (1234,)
        if method == "set_state":
            return (scratch.get_state(),)
        if method == "set_offset":
            return (0,)
        if method == "graphsafe_set_state":
            return (scratch.clone_state(),)
        return ()

    for row in GENERATOR_METHOD_TABLE:
        if row.return_family == "device_attr":
            assert isinstance(torch.Generator().device, torch.device)
            continue
        scratch = torch.Generator()
        try:
            result = getattr(scratch, row.method)(*_args_for(scratch, row.method))
        except (RuntimeError, NotImplementedError):
            assert row.method in _CAPABILITY_GATED_METHODS, (
                f"{row.method}: unexpected capability raise on a scratch generator"
            )
            continue
        if row.return_family == "host_scalar":
            assert isinstance(result, int) and not isinstance(result, bool), row
        elif row.return_family == "state_tensor":
            assert isinstance(result, torch.Tensor), row
        elif row.return_family == "self_generator":
            assert result is scratch, row
        else:  # generator
            assert isinstance(result, torch.Generator) and result is not scratch, row


@pytest.mark.smoke
def test_unknown_generator_method_flags_uncertainty() -> None:
    """A future unclassified Generator method is monitor UNCERTAINTY, never a miss."""

    class _FakeBoundMethod:
        __name__ = "future_scalar_method"

        def __init__(self, receiver: torch.Generator) -> None:
            self.__self__ = receiver

    monitor = host_nondeterminism_monitor(None)
    with monitor as result:
        monitor._classify_c_call(None, _FakeBoundMethod(torch.Generator()))
    assert result.uncertain
    assert any(
        "generator_method:future_scalar_method" in detail for detail in result.uncertain_detail
    )


# ======================================================================================
# Monitor-level behavior (no capture; smoke-fast)
# ======================================================================================


@pytest.mark.smoke
def test_monitor_dispositions_route_to_the_right_result_set(_torch_rng_state_guard) -> None:
    """Mutation/entropy land in ceiling ``channels``; initial_seed in ``replayable_reads``."""

    with host_nondeterminism_monitor(None) as result:
        torch.manual_seed(123)
    assert "torch.manual_seed" in result.channels
    assert not result.replayable_reads

    with host_nondeterminism_monitor(None) as result2:
        torch.initial_seed()
    assert not result2.channels, sorted(result2.channels)
    assert "torch.initial_seed" in result2.replayable_reads

    with host_nondeterminism_monitor(None) as result3:
        torch.seed()
    assert "torch.seed" in result3.channels


@pytest.mark.smoke
def test_monitor_held_ref_spellings_are_witnessed(_torch_rng_state_guard) -> None:
    """Pre-window ``from torch import manual_seed`` aliases cannot bypass the patch."""

    with host_nondeterminism_monitor(None) as result:
        _HELD_MANUAL_SEED(55)
    assert "torch.manual_seed" in result.channels

    with host_nondeterminism_monitor(None) as result2:
        _HELD_INITIAL_SEED()
    assert not result2.channels
    assert "torch.initial_seed" in result2.replayable_reads


@pytest.mark.smoke
def test_monitor_default_generator_receiver_classified(_torch_rng_state_guard) -> None:
    """Direct default-generator method calls classify like the delegating module APIs."""

    with host_nondeterminism_monitor(None) as result:
        torch.default_generator.manual_seed(9)
    assert "torch.default_generator.manual_seed" in result.channels

    with host_nondeterminism_monitor(None) as result2:
        torch.default_generator.initial_seed()
    assert not result2.channels
    assert "torch.default_generator.initial_seed" in result2.replayable_reads

    # Preservation pin (r67 table, non-default column): a USER-constructed local
    # generator's MUTATIONS touch instance state only -- inert, no over-trigger. Its
    # SCALAR READS are ceiled by the same table (see the r67 receiver-matrix tests).
    local_generator = torch.Generator()
    with host_nondeterminism_monitor(None) as result3:
        local_generator.manual_seed(7)
        local_generator.set_state(local_generator.get_state())
    assert not result3.channels
    assert not result3.replayable_reads
    assert not result3.uncertain


# ======================================================================================
# r67 C1 -- all-receiver Generator matrix (r66 free-F1 / corr1-1 / hon1-F5)
# ======================================================================================


@pytest.mark.smoke
def test_private_generator_seed_is_ceiled_every_receiver_spelling() -> None:
    """r66 free-F1: ``Generator().seed()`` draws OS entropy -- witnessed on EVERY
    receiver class (temporary, pre-window held, subclass)."""

    with host_nondeterminism_monitor(None) as result:
        torch.Generator().seed()
    assert "torch.Generator.seed" in result.channels

    held = torch.Generator()
    with host_nondeterminism_monitor(None) as result2:
        held.seed()
    assert "torch.Generator.seed" in result2.channels

    class _SubGenerator(torch.Generator):
        pass

    with host_nondeterminism_monitor(None) as result3:
        _SubGenerator().seed()
    assert "torch.Generator.seed" in result3.channels, (
        "inherited C method on a Generator SUBCLASS receiver escaped the classifier"
    )


@pytest.mark.smoke
def test_private_generator_initial_seed_is_ceiled_not_replayable() -> None:
    """Non-default ``initial_seed()`` is instance history: ceiling, never replayable."""

    generator = torch.Generator()
    generator.manual_seed(5)
    with host_nondeterminism_monitor(None) as result:
        generator.initial_seed()
    assert "torch.Generator.initial_seed" in result.channels
    assert not result.replayable_reads


@pytest.mark.smoke
def test_clone_state_return_closure_ceils_the_second_read() -> None:
    """r66 corr1-1: ``default.clone_state().initial_seed()`` cannot escape.

    ``clone_state`` itself is structural (marks nothing); the RETURNED clone re-enters
    the all-receiver classifier, so its scalar read lands on the non-default ceiling
    -- the accepted, documented over-ceiling for a clone of a seeded default.
    """

    with host_nondeterminism_monitor(None) as result:
        clone = torch.default_generator.clone_state()
    assert not result.channels
    assert not result.replayable_reads
    assert not result.uncertain

    with host_nondeterminism_monitor(None) as result2:
        clone.initial_seed()
    assert "torch.Generator.initial_seed" in result2.channels
    assert not result2.replayable_reads


@pytest.mark.smoke
def test_get_offset_marks_fail_closed_at_entry() -> None:
    """r66 hon1-F5: ``get_offset`` ceilings on every receiver; the classifier marks at
    c_call entry, so even the CPU capability raise never under-marks."""

    generator = torch.Generator()
    with host_nondeterminism_monitor(None) as result:
        try:
            generator.get_offset()
        except RuntimeError:
            pass
    assert "torch.Generator.get_offset" in result.channels


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA default generator")
def test_cuda_default_get_offset_is_ceiled(_torch_rng_state_guard) -> None:
    """r66 hon1-F5 device path: a successful default-receiver offset read ceilings."""

    torch.cuda.init()
    device_generator = torch.cuda.default_generators[0]
    with host_nondeterminism_monitor(None) as result:
        offset = device_generator.get_offset()
    assert isinstance(offset, int)
    assert "torch.default_generator.get_offset" in result.channels
    assert not result.replayable_reads


@pytest.mark.smoke
def test_device_default_populated_mid_window_selects_default_column(monkeypatch) -> None:
    """Dynamic membership: a device default that appears MID-WINDOW (lazy device
    init) still classifies with the default-receiver column on a routing-cache miss."""

    xpu = _torch_rng_holder_module("torch.xpu")
    if xpu is None:
        pytest.skip("torch.xpu module not loaded on this torch")
    generator = torch.Generator()
    with host_nondeterminism_monitor(None) as result:
        monkeypatch.setattr(xpu, "default_generators", (generator,), raising=False)
        generator.initial_seed()
    assert "torch.default_generator.initial_seed" in result.replayable_reads
    assert not result.channels


@pytest.mark.smoke
def test_monitor_get_only_reads_mark_nothing(_torch_rng_state_guard) -> None:
    """The get_rng_state family and Generator state reads are zero-collateral."""

    with host_nondeterminism_monitor(None) as result:
        torch.get_rng_state()
        torch.random.get_rng_state()
        torch.default_generator.get_state()
    assert not result.channels
    assert not result.replayable_reads
    assert not result.uncertain


@pytest.mark.smoke
def test_monitor_fork_rng_ceilings_transitively(_torch_rng_state_guard) -> None:
    """``fork_rng`` needs no direct row: its restore transits the set_rng_state patch."""

    with host_nondeterminism_monitor(None) as result:
        with torch.random.fork_rng(devices=[]):
            pass
    assert "torch.set_rng_state" in result.channels


@pytest.mark.smoke
def test_monitor_suppresses_torchlens_owned_seeding(_torch_rng_state_guard) -> None:
    """TL-owned seed/snapshot/restore never reads as model host nondeterminism."""

    with host_nondeterminism_monitor(None) as result:
        set_random_seed(3)
        states = log_current_rng_states()
        set_rng_from_saved_states(states)
    assert not result.channels, sorted(result.channels)
    assert not result.replayable_reads
    assert not result.uncertain


# ======================================================================================
# T-Z2 -- capture/save/load/run behavioral pins
# ======================================================================================


class _TorchSeedBranch(nn.Module):
    """Branch on fresh OS entropy via ``torch.seed()`` (permanent ceiling)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pick a tensor op from unseedable entropy."""
        if torch.seed() % 2 == 0:
            return x + 10
        return x * 10


class _InitialSeedBranch(nn.Module):
    """Branch on ``torch.initial_seed()`` -- fully determined by the capture seed."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pick a tensor op from the seeded default engine's identity scalar."""
        if torch.initial_seed() % 2 == 0:
            return x + 10
        return x * 10


class _ManualSeedInForward(nn.Module):
    """In-forward host mutation of the global torch engine (permanent ceiling)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reseed the global engine mid-path, then draw from it."""
        torch.manual_seed(999)
        return x + torch.randn_like(x)


class _HeldManualSeedInForward(nn.Module):
    """The held-reference spelling of the same mutation (r41 bypass attempt)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reseed through a pre-window alias of ``torch.manual_seed``."""
        _HELD_MANUAL_SEED(999)
        return x + torch.randn_like(x)


class _ForkRngInForward(nn.Module):
    """``fork_rng`` scope in-forward: ops inside drew from a host-chosen stream."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Draw inside a forked-and-restored RNG scope."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            y = x + torch.randn_like(x)
        return y


class _GetStateStoreOnly(nn.Module):
    """Store-only ``get_rng_state`` read: zero-collateral by the Z-a ruling."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read the engine state tensor without folding it into control flow."""
        _ = torch.get_rng_state()
        return x * 2


class _GetStateBranch(nn.Module):
    """Branch on engine-state BYTES: covered by the r39 scalar-escape belt, not a row."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fold state bytes into control flow (INCOMPLETE_SCALAR_ESCAPE)."""
        state = torch.get_rng_state()
        if int(state[0]) >= 0:
            return x + 10
        return x * 10


class _DropoutRandn(nn.Module):
    """Replayable tensor RNG ops -- captured DAG calls, never registry rows."""

    def __init__(self) -> None:
        super().__init__()
        self.drop = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Draw from the seeded engines through captured ops only."""
        return self.drop(x) + torch.randn_like(x) * 0.1


class _CheckpointModel(nn.Module):
    """``torch.utils.checkpoint`` stores engine state in forward (get-only)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the linear layer under reentrant activation checkpointing."""
        return torch.utils.checkpoint.checkpoint(self.lin, x, use_reentrant=True)


class _Deterministic(nn.Module):
    """A torch-RNG-free model that must stay VERIFIED with zero channels."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a deterministic affine transform."""
        return x * 3 + 1


class _PrivateGenSeedBranch(nn.Module):
    """r66 free-F1 verbatim: branch on a TEMPORARY private generator's OS entropy."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pick a tensor op from a user-constructed generator's entropy scalar."""
        if torch.Generator().seed() % 2 == 0:
            return x + 10
        return x * 10


class _HeldGenSeedBranch(nn.Module):
    """r66 free-F1 held spelling: the model HOLDS the private generator."""

    def __init__(self) -> None:
        super().__init__()
        self.gen = torch.Generator()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pick a tensor op from the held generator's entropy scalar."""
        if self.gen.seed() % 2 == 0:
            return x + 10
        return x * 10


class _CloneStateSeedBranch(nn.Module):
    """r66 corr1-1 verbatim: leak the default seed through an unheld local clone."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on the returned clone's initial_seed scalar."""
        clone = torch.default_generator.clone_state()
        if clone.initial_seed() % 2 == 0:
            return x + 10
        return x * 10


def _capture(model: nn.Module, x: torch.Tensor, *, seed: int) -> tl.Trace:
    """Capture a runnable-ready trace under a fixed seed."""
    return tl.trace(
        model,
        x,
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
            random_seed=seed,
        ),
    )


def _roundtrip_run(
    model: nn.Module,
    x: torch.Tensor,
    *,
    capture_seed: int,
    run_seed: int | None,
    tmp: Path,
    include_weights: bool = False,
) -> tl.RunResult:
    """Capture, save runnable, reload, and run under ``run_seed``."""
    trace = _capture(model, x, seed=capture_seed)
    path = tmp / "torch_rng.tlspec"
    shutil.rmtree(path, ignore_errors=True)
    trace.save(path, level="runnable", include_activations=True, include_weights=include_weights)
    return tl.load(path).run(inputs=x, seed=run_seed)


def test_torch_seed_profile_is_permanently_unreplayable() -> None:
    """ENTROPY: the descriptor records consumption with NO identifiable seed."""
    trace = _capture(_TorchSeedBranch(), torch.tensor([2.0]), seed=1)
    profile = build_sparse_run_descriptor(trace).rng_profile
    assert profile.host_rng_consumed is True
    assert profile.capture_seed is None
    assert any("torch.seed" in name for name in trace._runnable_host_rng_channels)


def test_torch_seed_every_run_unverifiable(tmp_path: Path) -> None:
    """ENTROPY: even the capture-seed run ceilings (nothing reproduces OS entropy)."""
    result = _roundtrip_run(
        _TorchSeedBranch(), torch.tensor([2.0]), capture_seed=1, run_seed=1, tmp=tmp_path
    )
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_initial_seed_profile_keeps_capture_seed() -> None:
    """REPLAYABLE_READ: consumed-flag only -- the capture seed survives."""
    trace = _capture(_InitialSeedBranch(), torch.tensor([2.0]), seed=1)
    profile = build_sparse_run_descriptor(trace).rng_profile
    assert profile.host_rng_consumed is True
    assert profile.capture_seed == 1
    assert not trace._runnable_host_rng_channels
    assert "torch.initial_seed" in trace._runnable_host_rng_replayable_reads


def test_initial_seed_on_seed_is_verified_and_attested(tmp_path: Path) -> None:
    """REPLAYABLE_READ: reproducing the capture seed is an honest VERIFIED replay."""
    result = _roundtrip_run(
        _InitialSeedBranch(), torch.tensor([2.0]), capture_seed=1, run_seed=1, tmp=tmp_path
    )
    assert result.output.tolist() == [20.0]
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED


@pytest.mark.parametrize("run_seed", [2, None])
def test_initial_seed_off_seed_is_unverifiable(run_seed: int | None, tmp_path: Path) -> None:
    """The r64 false-VERIFIED verbatim: off-seed/seedless initial_seed runs ceiling."""
    result = _roundtrip_run(
        _InitialSeedBranch(),
        torch.tensor([2.0]),
        capture_seed=1,
        run_seed=run_seed,
        tmp=tmp_path,
    )
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


@pytest.mark.parametrize(
    "model_cls", [_ManualSeedInForward, _HeldManualSeedInForward, _ForkRngInForward]
)
def test_in_forward_mutation_ceilings_every_run(model_cls: type, tmp_path: Path) -> None:
    """MUTATION: replay never re-executes host code, so even on-seed runs ceiling.

    The held-reference spelling and the ``fork_rng`` restore are witnessed identically
    to the module-attr spelling.
    """
    trace = _capture(model_cls(), torch.tensor([2.0]), seed=1)
    profile = build_sparse_run_descriptor(trace).rng_profile
    assert profile.host_rng_consumed is True
    assert profile.capture_seed is None
    assert trace._runnable_host_rng_channels
    result = _roundtrip_run(
        model_cls(), torch.tensor([2.0]), capture_seed=1, run_seed=1, tmp=tmp_path
    )
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_get_rng_state_store_only_stays_verified(tmp_path: Path) -> None:
    """Z-a zero-collateral pin: a store-only state read is not host consumption."""
    trace = _capture(_GetStateStoreOnly(), torch.tensor([2.0]), seed=1)
    assert build_sparse_run_descriptor(trace).rng_profile.host_rng_consumed is False
    result = _roundtrip_run(
        _GetStateStoreOnly(), torch.tensor([2.0]), capture_seed=1, run_seed=1, tmp=tmp_path
    )
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED


def test_get_rng_state_branch_on_bytes_never_verified(tmp_path: Path) -> None:
    """Z-a structural coverage pin: state-byte control flow rides the escape belt."""
    trace = _capture(_GetStateBranch(), torch.tensor([2.0]), seed=1)
    descriptor = build_sparse_run_descriptor(trace)
    assert descriptor.witness_completeness is WitnessCompleteness.INCOMPLETE_SCALAR_ESCAPE
    for run_seed in (1, 9):
        result = _roundtrip_run(
            _GetStateBranch(),
            torch.tensor([2.0]),
            capture_seed=1,
            run_seed=run_seed,
            tmp=tmp_path,
        )
        assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE


def test_tensor_rng_ops_fixed_seed_stay_verified(tmp_path: Path) -> None:
    """No-over-trigger: dropout / randn_like are captured DAG ops, never ceilinged."""
    trace = _capture(_DropoutRandn(), torch.tensor([1.0, 2.0, 3.0, 4.0]), seed=7)
    assert build_sparse_run_descriptor(trace).rng_profile.host_rng_consumed is False
    result = _roundtrip_run(
        _DropoutRandn(),
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        capture_seed=7,
        run_seed=7,
        tmp=tmp_path,
    )
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


def test_checkpoint_model_stays_verified_attested(tmp_path: Path) -> None:
    """The Z-a collateral guard verbatim (probe za4): checkpoint round-trips clean."""
    x = torch.randn(2, 4)
    trace = _capture(_CheckpointModel(), x, seed=5)
    assert build_sparse_run_descriptor(trace).rng_profile.host_rng_consumed is False
    assert not trace._runnable_host_rng_channels
    result = _roundtrip_run(
        _CheckpointModel(), x, capture_seed=5, run_seed=5, tmp=tmp_path, include_weights=True
    )
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED


# ======================================================================================
# r67 C1 -- capture/save/load/run regressions for the r66 escapes
# ======================================================================================


@pytest.mark.parametrize("model_cls", [_PrivateGenSeedBranch, _HeldGenSeedBranch])
def test_r67_private_generator_seed_ceilings_every_run(model_cls: type, tmp_path: Path) -> None:
    """r66 free-F1 (CRITICAL): both private-generator seed spellings ceiling.

    Before r67 these captured with ZERO channels and replayed
    VERIFIED+ATTESTED while a fresh oracle flipped the branch ~50% of the time.
    """

    trace = _capture(model_cls(), torch.tensor([2.0]), seed=1)
    profile = build_sparse_run_descriptor(trace).rng_profile
    assert profile.host_rng_consumed is True
    assert profile.capture_seed is None
    assert any("torch.Generator.seed" in name for name in trace._runnable_host_rng_channels)
    result = _roundtrip_run(
        model_cls(), torch.tensor([2.0]), capture_seed=1, run_seed=1, tmp=tmp_path
    )
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


@pytest.mark.parametrize("run_seed", [1, 2])
def test_r67_clone_state_initial_seed_ceilings_every_run(run_seed: int, tmp_path: Path) -> None:
    """r66 corr1-1 (HIGH): the clone's second read ceilings every run.

    ``run_seed=2`` is the original false-VERIFIED construction; ``run_seed=1`` pins
    the documented conservative over-ceiling (clone lineage is untracked, so even the
    capture-seed run is UNVERIFIABLE by design -- fail-closed first).
    """

    trace = _capture(_CloneStateSeedBranch(), torch.tensor([2.0]), seed=1)
    assert "torch.Generator.initial_seed" in trace._runnable_host_rng_channels
    profile = build_sparse_run_descriptor(trace).rng_profile
    assert profile.host_rng_consumed is True
    assert profile.capture_seed is None
    result = _roundtrip_run(
        _CloneStateSeedBranch(),
        torch.tensor([2.0]),
        capture_seed=1,
        run_seed=run_seed,
        tmp=tmp_path,
    )
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA default generator")
def test_r67_cuda_default_get_offset_ceilings_every_run(
    _torch_rng_state_guard, tmp_path: Path
) -> None:
    """r66 hon1-F5 (device-gated): a default-receiver offset branch ceilings."""

    class _OffsetBranch(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Branch on the philox consumption offset of the CUDA default engine."""
            if torch.cuda.default_generators[0].get_offset() == 0:
                return x + 10
            return x * 10

    torch.cuda.init()
    trace = _capture(_OffsetBranch(), torch.tensor([2.0]), seed=1)
    assert "torch.default_generator.get_offset" in trace._runnable_host_rng_channels
    result = _roundtrip_run(
        _OffsetBranch(), torch.tensor([2.0]), capture_seed=1, run_seed=1, tmp=tmp_path
    )
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


# ======================================================================================
# T-Z3 -- negative control + restoration identity across a real capture
# ======================================================================================


def test_deterministic_model_records_zero_torch_rng() -> None:
    """A torch-RNG-free forward records nothing on either result set."""
    trace = _capture(_Deterministic(), torch.tensor([2.0]), seed=1)
    assert build_sparse_run_descriptor(trace).rng_profile.host_rng_consumed is False
    assert trace._runnable_host_rng_channels == ()
    assert trace._runnable_host_rng_replayable_reads == ()
    assert trace._runnable_rng_monitor_uncertain is False


def test_capture_restores_torch_rng_attrs_identity_exact() -> None:
    """A completed capture leaves every monitored torch RNG attr the identical object."""
    resolvable = []
    for row in TORCH_RNG_SURFACE:
        if row.disposition not in _MONITORED_DISPOSITIONS:
            continue
        module_path, _, attr_name = row.target.rpartition(".")
        holder = _torch_rng_holder_module(module_path)
        if holder is not None and hasattr(holder, attr_name):
            resolvable.append((row.target, holder, attr_name, getattr(holder, attr_name)))
    _capture(_Deterministic(), torch.tensor([2.0]), seed=1)
    for target, holder, attr_name, pre in resolvable:
        assert getattr(holder, attr_name) is pre, f"capture leaked a monitor patch for {target}"
