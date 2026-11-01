"""The frozen host-nondeterminism channel registry and the torch RNG surface tables.

Every monitored channel (clock, entropy, construction, RNG primitive, RNG instance, global
engine state) is one :class:`HostNondeterminismRow`; the monitor installs its observers
FROM :data:`HOST_NONDETERMINISM_REGISTRY` and the coverage tests read the same table.
"""

from __future__ import annotations

import datetime as _datetime_module
import sys as _sys_module
from dataclasses import dataclass
from types import (
    ModuleType,
)
from typing import Any

import torch

from .._torch_compat import (
    HAS_GENERATOR_CLONE_STATE,
    HAS_GENERATOR_GRAPHSAFE_GET_STATE,
    HAS_GENERATOR_GRAPHSAFE_SET_STATE,
    HAS_GENERATOR_PHILOX_STATE,
    torch_module_exists_without_importing,
)
from . import engine_state

# ======================================================================================
# Host-nondeterminism channel monitor (r37 hon1_2; r39 CLASS A enumeration completeness).
#
# The global-engine snapshots above cover only the two REPLAYABLE engines (module
# ``random``, legacy ``np.random``). Every OTHER host entropy / clock / RNG-instance
# channel is monitored here over a data-driven FROZEN registry
# (:data:`HOST_NONDETERMINISM_REGISTRY`). A positive touch permanently ceilings the
# capture (``host_rng_consumed=True`` with NO identifiable seed -> every replay
# UNVERIFIABLE + NOT_APPLICABLE). Monitor uncertainty -- install/chain/inventory/restore
# failure, an observed-but-unclassified host event, or a live pre-existing non-owner
# Python thread whose ephemeral draws are unwitnessable on <=3.11 -- is itself recorded
# and downgrades capture completeness to INCOMPLETE; it NEVER reads as "no consumption".
#
# A negative witness (``host_rng_consumed=False``) is honest ONLY when every required
# observer over the declared channel/thread surface installed, classified, stayed
# installed, and restored exactly. This is the r39 enumeration-completeness invariant:
# "one more RNG/clock name" is a failing registry meta-test, not a false VERIFIED.
#
# Residual tail (outside any Python-visible call surface, disclaimed in contract s11):
# direct ``/dev/urandom`` file reads; ctypes / user C-extension entropy or clock reads;
# legacy ``RandomState()`` C-level construction entropy (its DRAWS stay digest-witnessed).
# ======================================================================================


@dataclass(frozen=True)
class HostNondeterminismRow:
    """One declared host-nondeterminism channel in the frozen monitoring registry.

    ``family`` groups the channel (``clock`` / ``entropy`` / ``construction`` /
    ``rng_primitive`` / ``rng_instance`` / ``rng_global_state``); ``target`` is its identity;
    ``strategy`` is HOW it is observed (see the coverage meta-test's allowlist);
    ``thread_scope`` is WHERE a positive can be observed (``any`` -- thread-independent
    module/class patch or process inventory; ``owner`` -- capture owner thread only;
    ``hooked`` -- owner plus every thread started in-window and profile-hooked);
    ``classification`` records the semantic role used by the arg/receiver classifiers.
    """

    family: str
    target: str
    strategy: str
    thread_scope: str
    classification: str


# ---- Clock vocabulary (r39 R2/R2b) ----------------------------------------------------
#
# The current-clock readers below all read the CURRENT wall/monotonic clock; a positive
# marks on ANY covered thread. Module-attr readers fire thread-independently; the
# immutable ``datetime`` classmethods (which CANNOT be class-patched -- extension types)
# are classified by c_call identity on the owner + hooked threads.

_CLOCK_COUNTER_NAMES = (
    "time",
    "time_ns",
    "monotonic",
    "monotonic_ns",
    "perf_counter",
    "perf_counter_ns",
    "process_time",
    "process_time_ns",
    "thread_time",
    "thread_time_ns",
    "clock_gettime",
    "clock_gettime_ns",
)
"""``time.*`` readers that always read the current clock (no explicit-time argument)."""

# Implicit-now converters: ``localtime()`` / ``strftime(fmt)`` etc. read the CURRENT
# clock when their optional explicit-time argument is absent/None, but are pure
# TRANSFORMS when given an explicit time. ``time_arg_index`` is the position of that
# optional argument (``strftime`` takes ``(format, [t])`` -> index 1; the rest index 0).
_CLOCK_IMPLICIT_NOW: tuple[tuple[str, int], ...] = (
    ("localtime", 0),
    ("gmtime", 0),
    ("asctime", 0),
    ("ctime", 0),
    ("strftime", 1),
)

# Immutable ``datetime`` current-clock readers, classified by c_call identity
# ``(receiver_type, method_name)`` because the receiver types are unpatchable C types.
_DATETIME_CLOCK_READERS: tuple[tuple[Any, str], ...] = (
    (_datetime_module.datetime, "now"),
    (_datetime_module.datetime, "utcnow"),
    (_datetime_module.datetime, "today"),
    (_datetime_module.date, "today"),
)


# ---- Torch global-RNG API surface (r65 CLUSTER Z) --------------------------------------
#
# torch's OWN Python-level RNG APIs are host-nondeterminism channels too. The asymmetry
# that decides every disposition below: sparse replay RE-EXECUTES the recorded tensor RNG
# ops (dropout / randn_like / ...) under the run seed, but it NEVER re-executes HOST code
# -- so an in-forward Python-level mutation of the global torch engine desyncs every
# downstream DAG RNG op from BOTH the capture and the fresh-run oracle (a python/numpy
# in-forward reseed, by contrast, is re-run by the conceptual oracle and stays honest
# through the existing snapshot compare). Every public name of the torch / torch.random /
# torch.cuda(.random) / torch.mps / torch.mtia / torch.xpu(.random) RNG surfaces carries
# an EXPLICIT disposition in the frozen closed vocabulary below; the independent no-list
# module-discovery meta-test (r67) makes a torch upgrade that grows this surface -- or
# loads one more RNG-bearing module -- a FAILING test, never a silent false-VERIFIED
# (the r39 "one more name" doctrine, now covering torch itself).

TORCH_RNG_DISPOSITIONS: frozenset[str] = frozenset(
    {"entropy", "mutation", "replayable_read", "structurally_covered", "op_captured"}
)
"""Closed disposition vocabulary for :data:`TORCH_RNG_SURFACE` rows.

``entropy`` -- reads fresh OS entropy (and mutates engine state): permanent ceiling.
``mutation`` -- in-forward host mutation of global engine state: permanent ceiling
(replay re-executes DAG RNG ops but never host code, see the asymmetry note above).
``replayable_read`` -- leaks a host scalar fully determined by the seeded default
engine (``initial_seed``): consumed-flag only, so a run at the capture seed stays
``verified`` and any other/absent seed ceilings (the python-``random`` analog).
``structurally_covered`` -- no monitor row; honesty is carried by an existing net
(each row's note names the covering mechanism and its pinned proof).
``op_captured`` -- tensor RNG draw ops (``randn`` / ``dropout`` / ``normal_`` ...):
captured DAG calls replayed seeded; never enumerated here (the capture spine owns them).
"""


@dataclass(frozen=True)
class TorchRngSurfaceRow:
    """One torch RNG API endpoint with its frozen honesty disposition (r65 CLUSTER Z).

    ``target`` is the dotted module-attribute spelling (each re-export spelling gets its
    own row so patch installation and the held-reference layer cover every alias);
    ``disposition`` is a member of :data:`TORCH_RNG_DISPOSITIONS`; ``note`` carries the
    covering-mechanism proof pointer for ``structurally_covered`` rows.
    """

    target: str
    disposition: str
    note: str


# Per-module endpoint specs, feature-detected at build. ``get_rng_state`` family rows are
# deliberately structurally_covered (NO monitor row): the returned state TENSOR is already
# covered by the r39 tensor->host escape belt (branch-on-state-bytes ->
# INCOMPLETE_SCALAR_ESCAPE, never VERIFIED; a store-only read stays VERIFIED), and a row
# would over-ceiling ``torch.utils.checkpoint(preserve_rng_state=True)``, which
# round-trips VERIFIED+ATTESTED today (r65 probe za4).
_TORCH_RNG_CORE_SPEC: tuple[tuple[str, str, str], ...] = (
    ("seed", "entropy", "draws OS entropy and reseeds the global engine"),
    ("manual_seed", "mutation", "in-forward host mutation of the global engine"),
    ("initial_seed", "replayable_read", "scalar read fully determined by the capture seed"),
    ("set_rng_state", "mutation", "in-forward host mutation of the global engine"),
    ("get_rng_state", "structurally_covered", "state-tensor return; r39 escape belt"),
)
_TORCH_RNG_DEVICE_SPEC: tuple[tuple[str, str, str], ...] = _TORCH_RNG_CORE_SPEC + (
    ("seed_all", "entropy", "draws OS entropy and reseeds every device engine"),
    ("manual_seed_all", "mutation", "in-forward host mutation of every device engine"),
    ("set_rng_state_all", "mutation", "in-forward host mutation of every device engine"),
    ("get_rng_state_all", "structurally_covered", "state-tensor return; r39 escape belt"),
)
_TORCH_ACCELERATOR_RNG_SPEC: tuple[tuple[str, str, str], ...] = (
    ("initial_seed", "replayable_read", "scalar read fully determined by the capture seed"),
    ("get_rng_state", "structurally_covered", "state-tensor return; r39 escape belt"),
    ("get_rng_state_all", "structurally_covered", "state-tensor return; r39 escape belt"),
)
"""torch 2.14's generic accelerator-agnostic RNG surface (``torch.accelerator.random``).

A DELIBERATELY SEPARATE object from :data:`_TORCH_RNG_DEVICE_SPEC`, even though every
row's content is a subset of it: ``torch.accelerator`` is a cross-backend ROUTER (it
defers to whichever concrete accelerator -- cuda/xpu/mtia -- is current) and holds no
``default_generators`` list of its own, so it must NOT join
:data:`_DEFAULT_GENERATOR_HOLDER_MODULES`'s identity-matched module set the way
``torch.cuda``/``torch.xpu``/``torch.mtia`` do (``test_default_generator_resolver_covers_
every_device_spec_module`` asserts that set is EXACTLY the device-spec holder modules).
It also has no ``seed``/``manual_seed``/``set_rng_state``/``*_all`` setters at all, so
reusing the full device spec here would also be a content mismatch, not just an identity
one.
"""
_TORCH_RNG_MODULE_SPECS: tuple[tuple[str, tuple[tuple[str, str, str], ...]], ...] = (
    ("torch", _TORCH_RNG_CORE_SPEC),
    ("torch.random", _TORCH_RNG_CORE_SPEC),
    ("torch.cuda", _TORCH_RNG_DEVICE_SPEC),
    ("torch.cuda.random", _TORCH_RNG_DEVICE_SPEC),
    ("torch.mps", _TORCH_RNG_CORE_SPEC),
    # r67 C1 (hon1-F6): ``torch.mtia`` carries the full feature-detected device RNG
    # spec -- on torch 2.8 only ``get_rng_state``/``set_rng_state`` resolve, and any
    # torch upgrade that grows the mtia surface lights up through the same
    # ``hasattr`` feature detection instead of a hand-list edit.
    ("torch.mtia", _TORCH_RNG_DEVICE_SPEC),
    ("torch.xpu", _TORCH_RNG_DEVICE_SPEC),
    # r67 C1: ``torch.xpu.random`` re-exports the xpu RNG surface exactly like
    # ``torch.cuda.random`` does for cuda; found by the independent no-list module
    # discovery immunizer (the same shared-blind-spot class as mtia).
    ("torch.xpu.random", _TORCH_RNG_DEVICE_SPEC),
    # torch 2.14 adds ``torch.accelerator.random`` (generic accelerator-agnostic
    # RNG surface, eagerly imported by ``torch.accelerator``, itself eagerly
    # imported by ``torch/__init__.py``). Its own dedicated spec (not the
    # shared device spec -- see that spec's docstring).
    ("torch.accelerator.random", _TORCH_ACCELERATOR_RNG_SPEC),
)
# Non-function endpoints the enumeration meta-test still demands dispositions for.
_TORCH_RNG_STRUCTURAL_EXTRAS: tuple[tuple[str, str], ...] = (
    (
        "torch.Generator",
        "construction is deterministic (probed constant initial seed); EVERY receiver's "
        "method c_calls dispatch through GENERATOR_METHOD_TABLE (r67 C1 all-receiver "
        "closure), and instance draws feed ops only through the opaque generator= "
        "kwarg, refused at save",
    ),
    (
        "torch.random.Generator",
        "same object as torch.Generator (construction deterministic; all-receiver "
        "method classification)",
    ),
    (
        "torch.default_generator",
        "receiver_profile registry row: method c_calls on ANY torch.Generator receiver "
        "dispatch through GENERATOR_METHOD_TABLE; process/device defaults select the "
        "default-receiver column via dynamic identity membership",
    ),
    (
        "torch.random.default_generator",
        "same object as torch.default_generator (receiver-classified)",
    ),
    (
        "torch.random.fork_rng",
        "pure composition of get/set_rng_state + manual_seed: its in-forward RESTORE "
        "transits the module-patched set_rng_state mutation rows (behaviorally pinned)",
    ),
    (
        "torch.cuda.default_generators",
        "device default generators; identity-joined into the default-receiver column "
        "of GENERATOR_METHOD_TABLE dynamically (re-resolved on a routing-cache miss, "
        "so a default populated mid-forward is still classified default)",
    ),
    (
        "torch.xpu.default_generators",
        "device default generators; identity-joined dynamically exactly like the cuda "
        "entry (r67 C1 closes the r65 xpu identity-join gap)",
    ),
    (
        "torch.mtia.default_generators",
        "device default generators; identity-joined dynamically exactly like the cuda "
        "entry (feature-detected: absent on torch builds without an mtia generator "
        "surface)",
    ),
    (
        "torch.utils.data.graph_settings.apply_random_seed",
        "draws ONLY from the caller-provided generator through a captured tensor RNG "
        "op (torch.empty(()).random_(generator=rng)) whose generator= kwarg is refused "
        "at save; the .item() scalar transits the r39 tensor->host escape belt, and "
        "datapipe set_seed() mutations are instance state, not global-engine state",
    ),
    (
        "torch.utils.data.graph_settings.apply_shuffle_seed",
        "deprecated alias delegating to apply_random_seed (same structural coverage)",
    ),
)
# Structural extras whose MODULE is never eagerly imported by torch or torchlens
# (unlike every _TORCH_RNG_STRUCTURAL_EXTRAS target above, all already sitting in
# sys.modules by the time torch/torchlens finish their own imports): requiring
# sys.modules presence here would make coverage depend on which OTHER test
# happened to import the module first in the session. find_spec proves existence
# without paying the module's own import cost (W21 cold-start guarantee).
_TORCH_RNG_UNIMPORTED_MODULE_EXTRAS: tuple[tuple[str, str], ...] = (
    (
        "torch.distributed.tensor.parallel.api.TensorParallelRNGTracker",
        "tensor-parallel distributed-training RNG tracker (torch 2.2 re-export of "
        "torch.distributed._tensor.random.TensorParallelRNGTracker). NOT eagerly "
        "imported by bare `import torch`/`import torchlens` -- it only lands in "
        "sys.modules when some OTHER already-collected test imports "
        "torch.distributed.tensor.parallel, so it belongs in this find_spec-checked "
        "group, not the sys.modules-checked one above (the exact failure mode this "
        "group exists to avoid). Construction only READS the device engine's state "
        "(get_rng_state); its seeding/region methods compose "
        "set_seed -> fork_rng -> device get/set_rng_state, transiting the same "
        "module-patched mutation rows torch.random.fork_rng already covers above. "
        "Active tensor_parallel usage is independently refused at capture entry by "
        "DistributedCaptureUnsupportedError (the tensor_parallel finding) before this "
        "tracker's state could ever reach a captured forward",
    ),
    (
        "torch.distributed.tensor.parallel.api.is_rng_supported_mesh",
        "pure capability predicate (torch 2.2): reads the device handle and returns "
        "whether it exposes set_rng_state, at most warning on an unsupported device "
        "mesh -- no entropy draw, no engine mutation, nothing that could desync a "
        "replay",
    ),
    (
        "torch.distributed.pipeline.sync.checkpoint.restore_rng_states",
        "torch.distributed.pipeline was removed from torch (gone by the 2.3 era; only "
        "the torch>=2.1 floor still carries it). Its checkpoint recomputation pairs "
        "save_rng_states/restore_rng_states exactly like torchgpipe's upstream "
        "implementation, whose restore calls torch.set_rng_state/torch.cuda.set_rng_state "
        "on the saved snapshot -- its in-forward RESTORE transits the module-patched "
        "set_rng_state/cuda.set_rng_state mutation rows, same composition as "
        "torch.random.fork_rng above (behaviorally pinned)",
    ),
    (
        "torch.distributed.pipeline.sync.checkpoint.save_rng_states",
        "the SAVE half of the same torchgpipe-derived pair above: calls "
        "torch.get_rng_state/torch.cuda.get_rng_state and returns the state "
        "tensor(s) -- structurally covered the same way those module-patched "
        "get_rng_state rows already are (r39 tensor->host escape belt)",
    ),
)


def _torch_rng_holder_module(module_path: str) -> ModuleType | None:
    """Resolve a torch RNG surface module WITHOUT importing anything new.

    All candidate modules are imported by ``torch/__init__`` itself when present, so
    ``sys.modules`` resolution is exact feature detection: an absent module (an old
    torch without ``torch.xpu``) simply contributes no rows.
    """

    module = _sys_module.modules.get(module_path)
    return module if isinstance(module, ModuleType) else None


def _build_torch_rng_surface() -> tuple[TorchRngSurfaceRow, ...]:
    """Assemble the frozen torch RNG API disposition table (feature-detected)."""

    rows: list[TorchRngSurfaceRow] = []
    for module_path, spec in _TORCH_RNG_MODULE_SPECS:
        module = _torch_rng_holder_module(module_path)
        if module is None:
            continue
        for name, disposition, note in spec:
            if hasattr(module, name):
                rows.append(TorchRngSurfaceRow(f"{module_path}.{name}", disposition, note))
    for target, note in _TORCH_RNG_STRUCTURAL_EXTRAS:
        module_path, _, name = target.rpartition(".")
        module = _torch_rng_holder_module(module_path)
        if module is not None and hasattr(module, name):
            rows.append(TorchRngSurfaceRow(target, "structurally_covered", note))
    for target, note in _TORCH_RNG_UNIMPORTED_MODULE_EXTRAS:
        # Unlike _TORCH_RNG_STRUCTURAL_EXTRAS's targets (already imported by
        # the time torch/torchlens finish their own imports), these modules
        # are never eagerly imported by anything; sys.modules-gating them
        # would make coverage depend on which OTHER test imported them first
        # (W21). The spec walk proves existence without importing the module
        # or a missing ancestor (find_spec would import the ancestor, and an
        # ancestor guard would bring the order dependence back); the
        # attribute itself is trusted present.
        module_path, _, _name = target.rpartition(".")
        if torch_module_exists_without_importing(module_path):
            rows.append(TorchRngSurfaceRow(target, "structurally_covered", note))
    return tuple(rows)


TORCH_RNG_SURFACE: tuple[TorchRngSurfaceRow, ...] = _build_torch_rng_surface()
"""Frozen disposition table for the torch Python-level RNG API surface (r65).

Every ``entropy`` / ``mutation`` / ``replayable_read`` row here becomes a
``module_patch`` row of :data:`HOST_NONDETERMINISM_REGISTRY` (same builder), so the
monitor install, the held-reference layer, and the coverage meta-tests all derive from
ONE table. ``structurally_covered`` rows carry their covering-mechanism proof in
``note`` and deliberately install nothing.
"""

# ---- All-receiver Generator method table (r67 C1) --------------------------------------
#
# r65 classified Generator method c_calls by RECEIVER IDENTITY (process defaults only),
# which let scalar-returning methods on user-constructed / held / RETURNED generators
# escape unwitnessed (r66 free-F1 ``torch.Generator().seed()``, corr1-1
# ``default.clone_state().initial_seed()``, hon1-F5 ``get_offset()``): the returned
# Python int leaks OS entropy or engine/instance history into control flow with no
# channel recorded -> false VERIFIED+ATTESTED. r67 replaces the default-only dict with
# ONE authoritative method table dispatched for EVERY ``isinstance(receiver,
# torch.Generator)`` c_call (subclasses included -- an inherited C method's c_call
# receiver is the subclass instance). Each row is CLOSED UNDER ITS RETURN VALUE:
#
# * ``host_scalar`` returns carry a marking disposition in BOTH receiver columns
#   (nothing downstream can see a bare Python int);
# * ``state_tensor`` returns are structural ONLY because the r39 tensor->host
#   escape belt covers the returned tensor's value use;
# * ``generator`` / ``self_generator`` returns are structural ONLY because the
#   returned Generator re-enters this same all-receiver classifier (closure) --
#   its later scalar reads cannot escape;
# * an UNKNOWN public method on ANY Generator receiver flags monitor uncertainty
#   (INCOMPLETE), never a silent miss.
#
# The default identity set survives ONLY as a routing cache (which column applies),
# re-resolved dynamically on a miss -- never an install-time honesty boundary.

GENERATOR_METHOD_DISPOSITIONS: frozenset[str] = frozenset(
    {"entropy", "mutation", "replayable_read", "instance_read"}
)
"""Closed disposition vocabulary for :data:`GENERATOR_METHOD_TABLE` columns.

``entropy`` / ``mutation`` / ``replayable_read`` exactly as in
:data:`TORCH_RNG_DISPOSITIONS`. ``instance_read`` -- a host scalar read of
engine/instance HISTORY (``get_offset``; ``initial_seed`` on a non-default receiver):
not reproduced by any run seed -> permanent ceiling. ``None`` in a column = inert or
structurally covered there (the row's ``note`` names the covering mechanism).
"""

GENERATOR_RETURN_FAMILIES: frozenset[str] = frozenset(
    {
        "host_scalar",
        "state_tensor",
        "generator",
        "self_generator",
        "device_attr",
        "state_tensor_tuple",
    }
)
"""Closed return-family vocabulary for :data:`GENERATOR_METHOD_TABLE` rows.

``host_scalar`` -- Python int; ``state_tensor`` -- ``torch.Tensor`` engine state;
``generator`` -- a NEW ``torch.Generator``; ``self_generator`` -- returns the
receiver (fluent setter); ``device_attr`` -- non-callable getset attribute;
``state_tensor_tuple`` -- a fixed tuple of freshly-minted ``torch.Tensor``
engine-state values (``philox_state``'s ``(seed, offset, intragraph_offset)``).
"""


@dataclass(frozen=True)
class GeneratorMethodRow:
    """One public ``torch.Generator`` method with its per-receiver-class dispositions.

    ``default_disposition`` applies when the receiver is a proven process/device
    default generator; ``nondefault_disposition`` applies to every other receiver
    (user-constructed, model-held, cloned, subclass). ``note`` carries the covering-
    mechanism proof for structural (``None``) columns; the return-closure meta-test
    machine-checks that each row's dispositions are closed under ``return_family``.
    """

    method: str
    return_family: str
    default_disposition: str | None
    nondefault_disposition: str | None
    note: str


_GENERATOR_METHOD_ROWS: tuple[GeneratorMethodRow, ...] = (
    GeneratorMethodRow(
        "seed",
        "host_scalar",
        "entropy",
        "entropy",
        "draws OS entropy in C++ (bypassing the patched os.urandom funnel) and returns "
        "it as a Python int: the scalar return IS the entropy channel on EVERY "
        "receiver (r66 free-F1)",
    ),
    GeneratorMethodRow(
        "manual_seed",
        "self_generator",
        "mutation",
        None,
        "default: in-forward host mutation of a global engine. Non-default: instance "
        "state only -- inert (the pinned private-manual_seed no-over-trigger); the "
        "returned receiver re-enters this classifier",
    ),
    GeneratorMethodRow(
        "set_state",
        "self_generator",
        "mutation",
        None,
        "default: in-forward host mutation of a global engine. Non-default: instance "
        "state only -- inert; the returned receiver re-enters this classifier",
    ),
    GeneratorMethodRow(
        "set_offset",
        "self_generator",
        "mutation",
        None,
        "default: in-forward host mutation of a global engine's philox offset. "
        "Non-default: instance state only -- inert; the returned receiver re-enters "
        "this classifier; capability-gated (raises on CPU)",
    ),
    GeneratorMethodRow(
        "graphsafe_set_state",
        "self_generator",
        "mutation",
        None,
        "default: in-forward host mutation of a global engine. Non-default: instance "
        "state only -- inert; the returned receiver re-enters this classifier; "
        "capability-gated (raises on CPU)",
    ),
    GeneratorMethodRow(
        "initial_seed",
        "host_scalar",
        "replayable_read",
        "instance_read",
        "default: scalar fully determined by the capture seed (consumed-flag only). "
        "Non-default: instance/clone history is untracked -> ceiling (the accepted, "
        "documented over-ceiling for a clone of a seeded default; r66 corr1-1)",
    ),
    GeneratorMethodRow(
        "get_offset",
        "host_scalar",
        "instance_read",
        "instance_read",
        "philox consumption offset: a host int of engine HISTORY (advanced by every "
        "prior draw, reset by manual_seed) -- no belt sees it, no run seed reproduces "
        "it -> ceiling on every receiver; marks at c_call entry, so the CPU "
        "capability-raise still marks fail-closed (r66 hon1-F5)",
    ),
    GeneratorMethodRow(
        "philox_state",
        "state_tensor_tuple",
        "mutation",
        "mutation",
        "torch 2.14: reserves `increment` Philox outputs, ADVANCING the engine's "
        "internal offset on every receiver (c10::GeneratorImpl::philox_state; only "
        "Philox-based engines implement it, so CPU's default mt19937 generator "
        "raises NotImplementedError -- capability-gated like get_offset/set_offset). "
        "Marks mutation on BOTH columns (never structural): unlike get_state, this "
        "call changes engine state as a side effect, so a non-default receiver's "
        "mutation is still a real, honestly-marked consumption, not inert instance "
        "state (r67 C1 honesty-first: unknown consumption must ceiling)",
    ),
    GeneratorMethodRow(
        "get_state",
        "state_tensor",
        None,
        None,
        "structural: the returned state TENSOR rides the r39 tensor->host escape belt "
        "(branch-on-state-bytes -> INCOMPLETE_SCALAR_ESCAPE); a row would over-ceiling "
        "torch.utils.checkpoint(preserve_rng_state=True) (pinned za4)",
    ),
    GeneratorMethodRow(
        "graphsafe_get_state",
        "generator",
        None,
        None,
        "structural closure: returns a Generator that re-enters this all-receiver "
        "classifier -- its later scalar reads are classified by their own rows; "
        "capability-gated (raises on CPU)",
    ),
    GeneratorMethodRow(
        "clone_state",
        "generator",
        None,
        None,
        "structural closure: returns a NEW Generator that re-enters this all-receiver "
        "classifier, so default.clone_state().initial_seed() lands on the non-default "
        "instance_read ceiling (r66 corr1-1)",
    ),
    GeneratorMethodRow(
        "device",
        "device_attr",
        None,
        None,
        "non-callable getset attribute (never emits a c_call); inert metadata",
    ),
)

_OPTIONAL_GENERATOR_METHOD_CAPABILITIES: dict[str, bool] = {
    "clone_state": HAS_GENERATOR_CLONE_STATE,
    "graphsafe_get_state": HAS_GENERATOR_GRAPHSAFE_GET_STATE,
    "graphsafe_set_state": HAS_GENERATOR_GRAPHSAFE_SET_STATE,
    "philox_state": HAS_GENERATOR_PHILOX_STATE,
}

GENERATOR_METHOD_TABLE: tuple[GeneratorMethodRow, ...] = tuple(
    row
    for row in _GENERATOR_METHOD_ROWS
    if _OPTIONAL_GENERATOR_METHOD_CAPABILITIES.get(row.method, True)
)
"""The live-surface all-receiver ``torch.Generator`` method table (r67 C1)."""

_GENERATOR_METHOD_ROWS_BY_NAME: dict[str, GeneratorMethodRow] = {
    row.method: row for row in GENERATOR_METHOD_TABLE
}

# Device modules whose ``default_generators`` tuples join the default-receiver column.
# Resolved through ``sys.modules`` only (never an import) and via a plain module-attr
# read (never device initialization); an uninitialized device contributes an empty
# tuple and a torch build without the module contributes nothing.
_DEFAULT_GENERATOR_HOLDER_MODULES: tuple[str, ...] = ("torch.cuda", "torch.xpu", "torch.mtia")


def _resolve_default_generator_ids() -> frozenset[int]:
    """Identity set of the CURRENTLY populated process/device default generators.

    Called at monitor install (routing-cache seed) and again on any Generator-receiver
    cache miss (r67 C1 dynamic membership): a device default generator populated
    MID-FORWARD (lazy device init) still selects the default-receiver column, without
    imports and without forcing device initialization.
    """

    ids: set[int] = set()
    process_default = getattr(torch, "default_generator", None)
    if process_default is not None:
        ids.add(id(process_default))
    for module_path in _DEFAULT_GENERATOR_HOLDER_MODULES:
        holder = _torch_rng_holder_module(module_path)
        if holder is None:
            continue
        for device_generator in tuple(getattr(holder, "default_generators", ()) or ()):
            ids.add(id(device_generator))
    return frozenset(ids)


def _build_host_nondeterminism_registry() -> tuple[HostNondeterminismRow, ...]:
    """Assemble the frozen channel registry the runtime classifiers are built from."""

    rows: list[HostNondeterminismRow] = []
    for name in _CLOCK_COUNTER_NAMES:
        rows.append(
            HostNondeterminismRow("clock", f"time.{name}", "module_patch", "any", "current_reader")
        )
    for name, _index in _CLOCK_IMPLICIT_NOW:
        rows.append(
            HostNondeterminismRow("clock", f"time.{name}", "module_patch", "any", "current_reader")
        )
    rows.append(HostNondeterminismRow("clock", "os.times", "module_patch", "any", "current_reader"))
    rows.append(
        HostNondeterminismRow(
            "clock", "resource.getrusage", "module_patch", "any", "current_reader"
        )
    )
    for receiver, method in _DATETIME_CLOCK_READERS:
        target = f"datetime.{receiver.__name__}.{method}"
        rows.append(
            HostNondeterminismRow("clock", target, "c_call_identity", "hooked", "current_reader")
        )
    rows.append(HostNondeterminismRow("entropy", "os.urandom", "module_patch", "any", "funnel"))
    rows.append(HostNondeterminismRow("entropy", "os.getrandom", "module_patch", "any", "funnel"))
    rows.append(
        HostNondeterminismRow("entropy", "random._urandom", "module_patch", "any", "funnel")
    )
    rows.append(
        HostNondeterminismRow(
            "construction", "numpy.random.default_rng", "module_patch", "any", "construction"
        )
    )
    rows.append(
        HostNondeterminismRow(
            "construction",
            "numpy.random.bit_generator.randbits",
            "construction_entropy",
            "any",
            "construction",
        )
    )
    for cls_name in ("random.Random", "random.SystemRandom", "_random.Random"):
        for method in ("random", "getrandbits", "randbytes"):
            rows.append(
                HostNondeterminismRow(
                    "rng_primitive", f"{cls_name}.{method}", "class_patch", "any", "primitive"
                )
            )
    rows.append(
        HostNondeterminismRow(
            "rng_instance",
            "numpy.random.Generator/RandomState/BitGenerator",
            "receiver_profile",
            "hooked",
            "instance_draw",
        )
    )
    rows.append(
        HostNondeterminismRow(
            "rng_instance", "_random.Random", "receiver_profile", "hooked", "instance_draw"
        )
    )
    rows.append(
        HostNondeterminismRow(
            "rng_instance",
            "numpy.random.Generator/RandomState/BitGenerator",
            "state_inventory",
            "any",
            "instance_draw",
        )
    )
    # r65 CLUSTER Z: torch's own Python-level RNG APIs, derived from the ONE frozen
    # disposition table so the monitor install, the held-reference layer, and the
    # coverage meta-tests can never drift apart. ``replayable_read`` rows mark the
    # NON-ceiling ``replayable_reads`` result set (consumed-flag only; run at the
    # capture seed stays verified); entropy/mutation rows ceiling permanently.
    _torch_family_by_disposition = {
        "entropy": "entropy",
        "mutation": "rng_mutation",
        "replayable_read": "rng_replayable_read",
    }
    for surface_row in TORCH_RNG_SURFACE:
        family = _torch_family_by_disposition.get(surface_row.disposition)
        if family is not None:
            rows.append(
                HostNondeterminismRow(
                    family, surface_row.target, "module_patch", "any", surface_row.disposition
                )
            )
    # r67 C1: ONE all-receiver row -- method c_calls on ANY ``torch.Generator``
    # (process/device defaults, user-constructed, held, cloned, subclass) dispatch
    # through GENERATOR_METHOD_TABLE; the default identity set is only the
    # column-routing cache, re-resolved dynamically on a miss.
    rows.append(
        HostNondeterminismRow(
            "rng_instance",
            "torch.Generator",
            "receiver_profile",
            "hooked",
            "generator_method",
        )
    )
    # W051 (2.17): the Python / legacy-NumPy GLOBAL-engine state surface (consumed-only).
    for family, target, strategy, scope, classification in engine_state.registry_rows():
        rows.append(HostNondeterminismRow(family, target, strategy, scope, classification))
    return tuple(rows)


HOST_NONDETERMINISM_REGISTRY: tuple[HostNondeterminismRow, ...] = (
    _build_host_nondeterminism_registry()
)
"""Frozen data-driven inventory of every monitored host-nondeterminism channel.

The runtime classifiers are built FROM this registry; the coverage meta-test asserts
every row has a valid strategy + thread policy and that no numpy draw-method NAME is
enumerated anywhere (receiver typing + state digests cover new draw-method names). A
future stdlib clock/RNG endpoint is a failing meta-test until it is given a registry row.

r41 (hon1_1): every ``module_patch`` row (and the construction-entropy alias) carries a
SECOND observation layer -- the ORIGINAL builtin's identity is registered at monitor
install, before the module attribute is replaced, so a pre-window held reference
(``from time import time`` / ``from os import urandom`` at the top of a model or helper
module) is classified from the ``c_call`` profile event on the owner and every in-window
hooked thread. The held-ref layer adds NO new rows and NO new strategy vocabulary; the
r41 held-ref immunizer derives its obligations from ``strategy == "module_patch"``, so a
future module-patch row without identity registration (or without a probe recipe) is a
RED test, not a silent gap.
"""
