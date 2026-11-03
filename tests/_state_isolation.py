"""Per-test state isolation: warn-once sentinels, registries, caches, RNG.

Registered by ``tests/conftest.py`` through ``pytest_plugins``; every fixture
here is autouse so no test module opts in or out.
"""

from __future__ import annotations

import os
import random
import sys
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType

import numpy as np
import pytest
import torch

from torchlens import _state

_MISSING = object()
_WARN_ONCE_SENTINELS: tuple[tuple[str, str, object], ...] = (
    # NOTE: this sentinel is a weakref.WeakSet in the package (type-keyed cache
    # eviction, F3a); resetting it to a plain set() would strong-pin model
    # classes — the reset must preserve the weak container type.
    ("torchlens._capture_state_helpers", "_VALIDATION_DEEPCOPY_WARNING_TYPES", weakref.WeakSet()),
    ("torchlens._capture_state_helpers", "_COMPILED_MODEL_UNWRAP_WARNED", False),
    ("torchlens._capture_state_helpers", "_COMPILED_FORCED_EAGER_WARNED", False),
    ("torchlens._io", "_LEGACY_THREAD_WARNING_EMITTED", {"flag": False}),
    ("torchlens._io.bundle", "_NONPERSISTENT_DISCLOSURE_WARNED", False),
    ("torchlens._io.bundle", "_UNATTESTABLE_ACTIVATION_DISCLOSURE_WARNED", False),
    ("torchlens._state", "_functorch_warning_emitted", False),
    ("torchlens._state", "_dynamo_warning_emitted", False),
    ("torchlens.backends.tf._tf_compat", "_warned_missing_capabilities", set()),
    ("torchlens.backends.torch._opaque_call_advisory", "_WARNED_CODES", set()),
    ("torchlens.backends.torch.legacy_ctors", "_WARNED_FOREIGN_NEW", set()),
    ("torchlens.backends.torch.ops", "_UNSUPPORTED_OUTPUT_CONTAINER_WARNED", set()),
    ("torchlens._data_classes._op_save", "_WARNED_REFERENCE_SAVE_MODE", False),
    ("torchlens.distributed._lifecycle", "_AUTO_ARM_DEGRADATION", None),
    ("torchlens.recording._storage_resolver", "_WARNED_REFERENCE_SAVE_MODE", False),
    ("torchlens._postprocess.ast_branches", "_source_drift_warned", set()),
    # SAE splice warnings (latents size, zero variance): once per session per code.
    ("torchlens.bridge.sae", "_WARNED_CODES", set()),
    # Per-unit statistics state-size warning: once per session per code.
    ("torchlens.stats._per_unit", "_WARNED_CODES", set()),
    # keep_grad differentiable-replay disclosure: once per session.
    ("torchlens.intervention._gradient_contract", "_WARNED_CODES", set()),
    # F17 quickstart nongold-value gate: once-per-trace raw-read warning keyed
    # by id(trace) (plain int keys, so a plain set() reset pins nothing).
    ("torchlens._quickstart._gate", "_WARNED_TRACE_IDS", set()),
    # F04 one-backward reads: the once-per-process frozen-default disclosure
    # (read_frozen_default_linearization) resets per test.
    ("torchlens.attribution.onebackward._frozen", "_DISCLOSURE_WARNED", set()),
    ("torchlens._utils._torch_compat", "_warned_missing_capabilities", set()),
    # Capture-scoped CUDA RNG retry latch (grind p5, B2P3-16): re-armed per
    # capture in the package, and reset per test here so a test that trips it
    # via a direct log_current_rng_states() call cannot degrade later tests.
    ("torchlens._utils.rng.streams", "_cuda_rng_unusable", False),
    ("torchlens._capture.prepare.door", "_BATCHNORM_TRAIN_STATS_WARNED", False),
    ("torchlens._utils.introspection", "_col_offset_cache_warned", False),
    ("torchlens.validation._stock_layer_grads", "_PASS_INDEX_PARSE_WARNED", False),
    ("torchlens._visualization._render_common", "_SIBLING_ORDER_WARNING_EMITTED", False),
    # The functional (global-mutated) copy moved to _render_ordering with the
    # renderer-thinning extraction; _render_dot keeps only an inert
    # star-import binding.
    ("torchlens._visualization._render_ordering", "_SIBLING_ORDER_WARNING_EMITTED", False),
    ("torchlens._visualization.auto_collapse", "_COUNT_MISMATCH_WARNING_EMITTED", False),
    # WeakSet of Traces that already got the collapse-ceiling decline warning;
    # a plain set() reset would strong-pin those Traces for the session.
    ("torchlens._visualization.collapse_optimizer", "_CEILING_WARNED_TRACES", weakref.WeakSet()),
    ("torchlens._visualization._collapse_disclosures", "_BUDGET_WARNED_TRACES", weakref.WeakSet()),
)


# ---------------------------------------------------------------------------
# Coverage: auto-generate text report when pytest --cov is used
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="session")
def _isolated_torchlens_cache(tmp_path_factory: pytest.TempPathFactory):
    """Point ``TORCHLENS_CACHE_DIR`` at a fresh per-session directory.

    ``tl.trace(..., cache=True)`` without an explicit ``cache_dir=`` falls back to
    ``~/.cache/torchlens`` (read lazily from ``TORCHLENS_CACHE_DIR`` on every call).
    A shared on-disk cache makes cache-hit assertions order- and history-dependent:
    a stale entry left by an earlier session or another worktree turns a
    first-capture cache-miss assertion into a phantom failure. Every test session
    gets its own empty cache root instead; the prior environment is restored on
    teardown so the suite never leaks state into the invoking shell.
    """

    prior = os.environ.get("TORCHLENS_CACHE_DIR")
    os.environ["TORCHLENS_CACHE_DIR"] = str(tmp_path_factory.mktemp("torchlens_cache"))
    yield
    if prior is None:
        os.environ.pop("TORCHLENS_CACHE_DIR", None)
    else:
        os.environ["TORCHLENS_CACHE_DIR"] = prior


def _copy_sentinel_value(value: object) -> object:
    """Return an independent snapshot of a supported warn-once value.

    Parameters
    ----------
    value:
        Boolean, set, or dictionary sentinel value.

    Returns
    -------
    object
        Independent mutable copy, or the original immutable value.
    """

    if isinstance(value, (set, dict, weakref.WeakSet)):
        return value.copy()
    return value


def _set_sentinel_default(module: ModuleType, name: str, default: object) -> None:
    """Reset one imported warn-once sentinel to its cold-process value.

    Parameters
    ----------
    module:
        Imported module that owns the sentinel.
    name:
        Module attribute name.
    default:
        Cold-process sentinel value.
    """

    _assign_sentinel(module, name, _copy_sentinel_value(default))


def _assign_sentinel(module: ModuleType, name: str, value: object) -> None:
    """Write a sentinel value, mutating container sentinels IN PLACE.

    Rebinding a set/dict/WeakSet sentinel booby-traps every ``from module
    import NAME`` alias in a test module: the alias keeps the OLD object, so
    its reads and ``.clear()`` calls silently target dead state after the
    first fixture cycle (r7 R76/R77 b2 remainder — the
    ``_WARNED_DEPRECATIONS`` from-import in test_conditional_lifecycle was
    exactly this). Same-type containers are cleared and refilled instead;
    immutable sentinels (bools, None) still rebind.
    """

    current = getattr(module, name, _MISSING)
    if type(current) is type(value) and isinstance(current, (set, dict, weakref.WeakSet)):
        current.clear()
        current.update(value)
    else:
        setattr(module, name, value)


@pytest.fixture(autouse=True)
def _reset_warn_once_sentinels() -> Iterator[None]:
    """Give every test fresh warn-once state and restore its incoming state."""

    snapshots: dict[tuple[str, str], object] = {}
    for module_name, name, default in _WARN_ONCE_SENTINELS:
        module = sys.modules.get(module_name)
        if module is None:
            snapshots[(module_name, name)] = _MISSING
            continue
        snapshots[(module_name, name)] = _copy_sentinel_value(getattr(module, name, _MISSING))
        _set_sentinel_default(module, name, default)

    try:
        yield
    finally:
        for module_name, name, default in _WARN_ONCE_SENTINELS:
            module = sys.modules.get(module_name)
            if module is None:
                continue
            prior = snapshots[(module_name, name)]
            if prior is _MISSING:
                _set_sentinel_default(module, name, default)
            else:
                _assign_sentinel(module, name, prior)


#: Public content registries that tests mutate through PUBLIC registration
#: APIs with no unregister spelling (r7 R76, sol b2 MED): a registered
#: container class or custom op rule was a permanent process-global, so
#: full-suite and targeted runs saw different registry state depending on
#: which tests had run first. Snapshot/restore per test, same lazy
#: sys.modules discipline as the warn-once sentinels. The third column
#: names the registry's own lock attribute (or None): the container
#: registry has a writer/iterator race lock that the restore must honor.
#:
#: r7 R76/R77 adjudication (fixwave-7): two same-day parallel lanes each
#: landed a twin of this fixture — one lazy but lockless, one locked but
#: force-importing ``capture.flops`` suite-wide. This is the MERGED single
#: fixture: lazy module lookup (flops stays unimported until a test needs
#: it) AND lock-disciplined container access.
_CONTENT_REGISTRIES: tuple[tuple[str, str, str | None], ...] = (
    ("torchlens._ir.container", "_CONTAINER_REGISTRY", "_CONTAINER_REGISTRY_LOCK"),
    ("torchlens._capture.flops", "_CUSTOM_OP_RULES", None),
)


@contextmanager
def _registry_lock(module: object, lock_name: str | None) -> Iterator[None]:
    """Hold the registry's own lock when it declares one."""

    if lock_name is None:
        yield
    else:
        with getattr(module, lock_name):
            yield


@pytest.fixture(autouse=True)
def _restore_content_registries() -> Iterator[None]:
    """Restore registered-container and custom-op-rule state after every test."""

    snapshots: dict[tuple[str, str], object] = {}
    for module_name, name, lock_name in _CONTENT_REGISTRIES:
        module = sys.modules.get(module_name)
        if module is None:
            snapshots[(module_name, name)] = _MISSING
            continue
        with _registry_lock(module, lock_name):
            snapshots[(module_name, name)] = dict(getattr(module, name))
    try:
        yield
    finally:
        for module_name, name, lock_name in _CONTENT_REGISTRIES:
            module = sys.modules.get(module_name)
            if module is None:
                continue
            prior = snapshots[(module_name, name)]
            if prior is _MISSING:
                # Module imported DURING the test: whatever it registered at
                # import time is legitimate baseline; drop only test-added
                # rows is impossible to distinguish, so leave as-is.
                continue
            with _registry_lock(module, lock_name):
                registry = getattr(module, name)
                registry.clear()
                registry.update(prior)


_CAPABILITY_DEPENDENT_CACHES: tuple[tuple[str, str], ...] = (
    # Second-layer lru_caches whose cached value DERIVES from a lazy HAS_*
    # capability probe (grind p5 §3.9: the b7fe953e class one layer down).
    # Restoring the _torch_compat latches alone leaves a value computed under
    # a stubbed runtime frozen in these caches for the whole process, so the
    # probe restore must clear them too. The census in
    # tests/test_marker_lint.py::test_capability_dependent_caches_are_cleared
    # keeps this list complete.
    ("torchlens._runnable.state_context", "_fake_tensor_mode_class"),
    ("torchlens._runnable.state_context", "_count_bounded_fake_tensor_mode_class"),
)


def _clear_capability_dependent_caches() -> None:
    """Clear every registered probe-derived lru_cache (loaded modules only)."""

    for module_name, attr in _CAPABILITY_DEPENDENT_CACHES:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        getattr(module, attr).cache_clear()


@pytest.fixture(autouse=True)
def _restore_distributed_arming() -> Iterator[None]:
    """Disarm distributed capture a test armed, so it never reaches later tests.

    Arming (``tl.distributed.arm()`` or the lazy arm at capture entry under an
    initialized process group) is process-lifetime by design and survives
    ``destroy_process_group``. A test that armed it used to leave plane-P's
    dispatch mode around every later capture on the same xdist worker, which
    turned unrelated weights-free and failure-origin tests red depending on
    test order. Disarming restores the unarmed incoming state.
    """

    lifecycle = sys.modules.get("torchlens.distributed._lifecycle")
    was_armed = lifecycle is not None and lifecycle.is_armed()
    try:
        yield
    finally:
        lifecycle = sys.modules.get("torchlens.distributed._lifecycle")
        if not was_armed and lifecycle is not None and lifecycle.is_armed():
            lifecycle.disarm()


@pytest.fixture(autouse=True)
def _restore_wrapper_diagnostic_modes() -> Iterator[None]:
    """Restore the escape-detector and completeness-witness modes after every test.

    ``wrap_torch(escape_detector=..., completeness_witness=...)`` sets
    process-wide flags that outlive the test. A census test that armed the
    witness and never restored it made every later capture on the same xdist
    worker settle through the witness: plain captures reported
    ``completeness_witness_mode == "shadow"`` and meta captures settled
    ``capture_verified=True``, tripping the weights-free settlement invariant
    in unrelated tests depending on test order. Restoring the pre-test modes
    keeps a session that deliberately armed diagnostics armed.
    """

    saved_escape = _state._escape_detector_mode
    saved_witness = _state._completeness_witness_mode
    try:
        yield
    finally:
        wrappers = sys.modules.get("torchlens.backends.torch.wrappers")
        if wrappers is not None:
            if _state._escape_detector_mode != saved_escape:
                wrappers._configure_escape_detector(saved_escape)
            if _state._completeness_witness_mode != saved_witness:
                wrappers._configure_completeness_witness(saved_witness)


@pytest.fixture(autouse=True)
def _restore_lazy_capability_probes() -> Iterator[None]:
    """Restore lazy ``HAS_*`` capability latches to their pre-test state.

    The lazy ``_torch_compat`` probes latch on first use, so a test that stubs
    ``sys.modules`` (or otherwise shims the runtime) while a probe fires
    poisons the flag for the whole process -- the recorded ``b7fe953e``
    incident class, previously patched per-test rather than systemically.
    Snapshotting before and restoring after every test bounds any mis-latch to
    the test that caused it; an un-latched probe simply re-probes on its next
    use, which is cheap and hits the real runtime. Second-layer caches built
    FROM a probed capability (``_CAPABILITY_DEPENDENT_CACHES``) are cleared in
    the same breath -- restoring the flag while a derived cache keeps the
    poisoned value would just move the incident one layer down.
    """

    from torchlens._utils import _torch_compat

    snapshot = _torch_compat.capability_probe_snapshot()
    try:
        yield
    finally:
        _torch_compat.restore_capability_probes(snapshot)
        _clear_capability_dependent_caches()


@pytest.fixture(autouse=True)
def _reset_rng_state() -> Iterator[None]:
    """Seed each test deterministically and restore all incoming RNG settings."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_initialized = torch.cuda.is_initialized()
    cuda_states = torch.cuda.get_rng_state_all() if cuda_initialized else None
    deterministic = torch.are_deterministic_algorithms_enabled()
    deterministic_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    # MLX seeds its global key from the clock, so an MLX test's random init
    # differed every run. Seed it only when a test module already imported
    # it: importing the optional backend here would load it for every test.
    mlx_random = getattr(sys.modules.get("mlx.core"), "random", None)
    mlx_state = list(mlx_random.state) if mlx_random is not None else None

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if cuda_initialized:
        torch.cuda.manual_seed_all(0)
    if mlx_random is not None:
        mlx_random.seed(0)
    torch.use_deterministic_algorithms(True)

    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.use_deterministic_algorithms(deterministic, warn_only=deterministic_warn_only)
        if mlx_random is not None and mlx_state is not None:
            mlx_random.state[:] = mlx_state
