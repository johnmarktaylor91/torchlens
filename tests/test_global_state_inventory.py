"""Trust-lane inventory and exception restoration for mutable module globals."""

from __future__ import annotations

import ast
import functools
import sys
import threading
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from _global_state_rows import _LIFECYCLE_CLASSES, _WEAK_SUBJECT_TABLES, _WEAKLY_HELD
from _source_corpus import package_ast
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens.backends.torch import _tl as torch_tl, completeness_witness, rescue
from torchlens.capture import projections, trace as capture_trace

MUTATING_METHODS = frozenset(
    {
        "append",
        "appendleft",
        "add",
        "clear",
        "discard",
        "extend",
        "insert",
        "move_to_end",
        "pop",
        "popitem",
        "popleft",
        "remove",
        "setdefault",
        "sort",
        "update",
        "__setitem__",
    }
)
"""Method names whose call mutates a container in place."""

MUTABLE_FACTORIES = frozenset(
    {
        "ChainMap",
        "Counter",
        "OrderedDict",
        "WeakKeyDictionary",
        "WeakSet",
        "WeakValueDictionary",
        "defaultdict",
        "deque",
        "dict",
        "list",
        "set",
    }
)
"""Callables whose result is a mutable container."""


def _package_python_paths(repo: Path) -> list[Path]:
    """Return every Python source file in the package.

    The inventory used to scan only ``_state.py`` plus ``capture/``,
    ``validation/`` and ``backends/torch/``, which governed a MINORITY of the
    surface it claimed: the process-global per-capture tap observers in the
    mlx/paddle backends, the mutable facet toggle, the RF rule registry, the
    warn-once sentinels in ``_io``/``visualization``/``distributed`` and the
    whole ``_torch_compat`` capability block were all out of scope, so a new
    unclassified global could land in them without failing anything.

    Parameters
    ----------
    repo:
        Repository root.

    Returns
    -------
    list[Path]
        Sorted package Python source paths.
    """

    return sorted((repo / "torchlens").rglob("*.py"))


def _module_level_mutable_bindings(tree: ast.Module) -> dict[str, str]:
    """Return module-level names bound to a mutable container, with their source.

    Parameters
    ----------
    tree:
        Parsed module.

    Returns
    -------
    dict[str, str]
        Name -> unparsed initializer for each module-level mutable binding.
    """

    bindings: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        mutable = isinstance(
            value, (ast.Dict, ast.Set, ast.List, ast.DictComp, ast.SetComp, ast.ListComp)
        )
        if isinstance(value, ast.Call):
            factory = value.func
            factory_name = (
                factory.id
                if isinstance(factory, ast.Name)
                else factory.attr
                if isinstance(factory, ast.Attribute)
                else None
            )
            mutable = mutable or factory_name in MUTABLE_FACTORIES
        if not mutable:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                bindings[target.id] = ast.unparse(value)
    return bindings


def _names_mutated_in_place(trees: dict[str, ast.Module]) -> set[str]:
    """Return every name mutated in place anywhere in the package.

    Both spellings count, because module state is routinely mutated through an
    imported module alias: a bare ``_CACHE[key] = value`` in the owning module
    AND an ``_state._dir_cache[key] = value`` from another one. The attribute
    spelling is matched by ATTRIBUTE NAME, which can over-include a same-named
    attribute on an unrelated object; over-inclusion only ever asks for one more
    classification row, whereas under-inclusion is the blind spot this closes.

    Parameters
    ----------
    trees:
        Relative path -> parsed module for the whole package.

    Returns
    -------
    set[str]
        Names observed under an in-place mutation.
    """

    mutated: set[str] = set()
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.Delete)):
                targets = (
                    node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
                )
                for target in targets:
                    if not isinstance(target, ast.Subscript):
                        continue
                    base = target.value
                    if isinstance(base, ast.Name):
                        mutated.add(base.id)
                    elif isinstance(base, ast.Attribute):
                        mutated.add(base.attr)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in MUTATING_METHODS
            ):
                base = node.func.value
                if isinstance(base, ast.Name):
                    mutated.add(base.id)
                elif isinstance(base, ast.Attribute):
                    mutated.add(base.attr)
    return mutated


def _module_alias_targets(
    relative: str, tree: ast.Module, module_files: frozenset[str]
) -> dict[str, str]:
    """Map local names bound to imported TORCHLENS modules onto their files.

    Resolves both absolute (``from torchlens import _state``) and relative
    (``from ... import _state``, ``from ..utils import rng as rng_mod``)
    module imports, so attribute rebinds through the alias can be attributed
    to the OWNING module.

    Parameters
    ----------
    relative:
        Module path relative to the repository root.
    tree:
        Parsed module.
    module_files:
        Every package Python file, as repo-relative POSIX paths.

    Returns
    -------
    dict[str, str]
        Local alias name -> owning module's repo-relative path.
    """

    package_parts = relative[: -len(".py")].split("/")[:-1]
    if relative.endswith("/__init__.py"):
        package_parts = relative[: -len("/__init__.py")].split("/")
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module is None or node.module.split(".")[0] != "torchlens":
                    continue
                base = node.module.split(".")
            else:
                keep = len(package_parts) - (node.level - 1)
                if keep < 0:
                    continue
                base = package_parts[:keep]
                if node.module:
                    base = [*base, *node.module.split(".")]
            for alias in node.names:
                module_candidate = "/".join([*base, alias.name]) + ".py"
                package_candidate = "/".join([*base, alias.name, "__init__.py"])
                if module_candidate in module_files:
                    aliases[alias.asname or alias.name] = module_candidate
                elif package_candidate in module_files:
                    aliases[alias.asname or alias.name] = package_candidate
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] != "torchlens":
                    continue
                module_candidate = alias.name.replace(".", "/") + ".py"
                package_candidate = alias.name.replace(".", "/") + "/__init__.py"
                local = alias.asname or alias.name.split(".")[0]
                if alias.asname is None and "." in alias.name:
                    continue  # ``import torchlens.x`` binds only ``torchlens``
                if module_candidate in module_files:
                    aliases[local] = module_candidate
                elif package_candidate in module_files:
                    aliases[local] = package_candidate
    return aliases


def _cross_module_attribute_rebinds(
    trees: dict[str, ast.Module], module_files: frozenset[str]
) -> set[tuple[str, str]]:
    """Return module globals rebound THROUGH an imported-module attribute.

    ``_declared_globals`` walks ``ast.Global`` only, so a control slot that is
    declared in one module and assigned exclusively from OTHERS
    (``_state._nonowner_belt_armed = True`` in ``_completeness_patches.py``)
    could never be classified: four live per-capture guards evaded the
    purported whole-package inventory this way (hunt-b2-sol R54). Attribute
    rebinds through a resolved torchlens module alias are attributed to the
    OWNING module.

    Parameters
    ----------
    trees:
        Relative path -> parsed module for the whole package.
    module_files:
        Every package Python file, as repo-relative POSIX paths.

    Returns
    -------
    set[tuple[str, str]]
        ``(owning module relative path, attribute name)`` rebind sites.
    """

    rebinds: set[tuple[str, str]] = set()
    for relative, tree in trees.items():
        aliases = _module_alias_targets(relative, tree, module_files)
        if not aliases:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AugAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                    owner = aliases.get(target.value.id)
                    if owner is not None:
                        rebinds.add((owner, target.attr))
    return rebinds


def _mutable_module_state(repo: Path) -> dict[tuple[str, str], str]:
    """Return every mutable module global in the package, with its initializer.

    Three detectors, because each alone has a structural blind spot:

    * ``global`` declarations catch REBINDING (``_flag = True``) but can never
      see a container mutated in place -- ``_CACHE[key] = value`` needs no
      ``global`` statement at all, so the whole cache/registry class was
      invisible to the previous gate.
    * Module-level mutable bindings catch the container class, qualified by
      package-wide evidence that something actually mutates them, so frozen
      lookup tables are not dragged in.
    * Cross-module attribute rebinds (``_state.flag = value`` from another
      module) need no ``global`` statement in ANY module, so scalar control
      slots assigned only through the module object were invisible to both
      detectors above (hunt-b2-sol R54).

    Parameters
    ----------
    repo:
        Repository root.

    Returns
    -------
    dict[tuple[str, str], str]
        ``(relative path, name)`` -> unparsed initializer (empty for names known
        only from a ``global`` declaration).
    """

    return _mutable_module_state_cached(repo)


@functools.lru_cache(maxsize=1)
def _mutable_module_state_cached(repo: Path) -> dict[tuple[str, str], str]:
    """One whole-package scan per session, shared by every census consumer.

    The scan costs ~9s (parse-dominated); the two heavy census tests and the
    smoke-tier hot-subset sentinel below all read this one result.
    """

    trees = {
        path.relative_to(repo).as_posix(): package_ast(path) for path in _package_python_paths(repo)
    }
    mutated_names = _names_mutated_in_place(trees)
    state: dict[tuple[str, str], str] = {}
    for relative, tree in trees.items():
        bindings = _module_level_mutable_bindings(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                for name in node.names:
                    state.setdefault((relative, name), bindings.get(name, ""))
        for name, initializer in bindings.items():
            if name in mutated_names:
                state[(relative, name)] = initializer
    module_files = frozenset(trees)
    for owner, attribute in _cross_module_attribute_rebinds(trees, module_files):
        state.setdefault((owner, attribute), "")
    return state


def _capture_scope_snapshot() -> dict[str, Any]:
    """Return high-risk per-capture global state for restoration checks.

    Returns
    -------
    dict[str, Any]
        Values that must be identical before and after a failed capture.
    """

    return {
        "logging_enabled": _state._logging_enabled,
        "active_trace": _state._active_trace,
        "active_owner_thread_id": _state._active_owner_thread_id,
        "nonowner_belt_armed": _state._nonowner_belt_armed,
        "active_fast_run_collector": _state._active_fast_run_collector,
        "active_hook_plan": _state._active_hook_plan,
        "active_intervention_spec": _state._active_intervention_spec,
        "capture_replay_templates": _state._capture_replay_templates,
        "relationship_model_id": _state._relationship_model_id,
        "relationship_model_class": _state._relationship_model_class,
        "relationship_weight_fingerprint": _state._relationship_weight_fingerprint,
        "relationship_input_id": _state._relationship_input_id,
        "relationship_input_shape_hash": _state._relationship_input_shape_hash,
        "runnable_ledger_armed": _state._runnable_ledger_armed,
        "aten_recording_armed": _state._aten_recording_armed,
        "capture_reserved_by": _state._capture_reserved_by,
        "active_label_session": torch_tl._ACTIVE_LABEL_SESSION,
        "active_witness_state": completeness_witness._ACTIVE_WITNESS_STATE,
        "rescue_active": rescue._rescue_is_active(),
        "active_recording_state": projections._active_recording_state,
        "active_capture_backend": capture_trace._ACTIVE_CAPTURE_BACKEND,
    }


class _InjectedBaseFailure(BaseException):
    """BaseException subclass used to exercise the interruption cleanup arm."""


class _RaiseMidCapture(nn.Module):
    """Run one logged op before raising a selected exception class."""

    def __init__(self, error_type: type[BaseException]) -> None:
        """Store the exception type raised during ``forward``.

        Parameters
        ----------
        error_type:
            Exception class to raise after one tensor operation.
        """

        super().__init__()
        self.error_type = error_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one operation and then raise the injected exception.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            This path never returns.

        Raises
        ------
        BaseException
            Always raises ``self.error_type`` after the logged operation.
        """

        _ = torch.relu(x)
        raise self.error_type("injected mid-capture failure")


class _BlockingCapture(nn.Module):
    """Hold a public capture open while a concurrent capture is attempted."""

    def __init__(self, entered: threading.Event, release: threading.Event) -> None:
        """Store synchronization events for the capture overlap.

        Parameters
        ----------
        entered:
            Event set after the first logged operation runs.
        release:
            Event that permits the model to finish.
        """

        super().__init__()
        self.entered = entered
        self.release = release

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Log one operation and wait until the contender is refused.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Logged activation after the release event is set.
        """

        out = torch.relu(x)
        self.entered.set()
        if not self.release.wait(timeout=5.0):
            raise TimeoutError("concurrent capture test did not release the owner")
        return out


class _PauseFromForeignThread(nn.Module):
    """Log ops before and after a NON-OWNER thread enters ``pause_logging()``.

    The foreign thread models the reachable-without-concurrent-capture case: a
    thread merely ANALYZING an older Trace (``tl.save``, validation, an ``.out``
    transform) enters the same process-global pause the owner's forward relies on.

    The foreign pause is HELD OPEN across the owner's second op group, which is
    the realistic shape: an analysis thread's ``pause_logging()`` body spans a
    whole save / validation pass, not a single statement.
    """

    def __init__(self, ops_per_side: int) -> None:
        """Store how many logged ops run on each side of the foreign pause.

        Parameters
        ----------
        ops_per_side:
            Number of logged operations before, during, and after the pause.
        """

        super().__init__()
        self.ops_per_side = ops_per_side
        self.foreign_paused = threading.Event()
        self.foreign_release = threading.Event()
        self.foreign_error: list[BaseException] = []

    def _foreign_pause(self) -> None:
        """Hold ``pause_logging()`` open from a non-owner thread."""

        try:
            with _state.pause_logging():
                self.foreign_paused.set()
                if not self.foreign_release.wait(timeout=10.0):
                    raise TimeoutError("owner never released the foreign pause")
        except BaseException as error:  # pragma: no cover - reported by the test
            self.foreign_error.append(error)
        finally:
            self.foreign_paused.set()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run ops before, during, and after a held foreign pause.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Activation after all three op groups ran.
        """

        for _ in range(self.ops_per_side):
            x = torch.relu(x)
        foreign = threading.Thread(target=self._foreign_pause)
        foreign.start()
        assert self.foreign_paused.wait(timeout=10.0), "foreign pause never entered"
        # These ops run while a NON-OWNER thread holds the pause open.
        for _ in range(self.ops_per_side):
            x = torch.relu(x)
        self.foreign_release.set()
        foreign.join(timeout=10.0)
        assert not foreign.is_alive(), "foreign pause thread did not exit"
        # And these run after the foreign thread restored what it saved.
        for _ in range(self.ops_per_side):
            x = torch.relu(x)
        return x


# heavy, not smoke (r3settle2 budget lint): the whole-process state scan
# measures ~9-10s after the fixwave-2 inventory growth.
@pytest.mark.heavy
def test_global_state_inventory_is_classified_and_shrink_only() -> None:
    """Every mutable module global in the PACKAGE has exactly one lifecycle class.

    Scope and detection are both wider than they were. The inventory covered
    ``_state.py`` plus three subpackages, and classified only ``global``
    declarations -- so the majority of the surface it claimed to govern was
    unreachable by it, and the entire mutated-in-place class (``_CACHE[key] =
    value`` needs no ``global`` statement) was structurally invisible even
    in-lane. See ``_package_python_paths`` and ``_mutable_module_state``.
    """

    repo = Path(__file__).resolve().parents[1]
    classified = set().union(*_LIFECYCLE_CLASSES)

    assert sum(len(category) for category in _LIFECYCLE_CLASSES) == len(classified), (
        "global lifecycle classes overlap"
    )
    observed = _mutable_module_state(repo)
    missing = sorted(set(observed) - classified)
    stale = sorted(classified - set(observed))
    assert not missing, (
        "unclassified mutable module state (assign it a lifecycle class in "
        f"tests/_global_state_rows.py): {missing}"
    )
    assert not stale, f"inventory rows no longer present in the package: {stale}"


#: Historically hot state-bearing surface for the SMOKE-tier census sentinel:
#: the capture/wrapper territory where the b2p2-F2 incident landed 24
#: unclassified globals across 3 commits with no commit-gate red, plus the
#: root-level state modules. Prefix-matched against package-relative paths.
_HOT_STATE_SUBSET_PREFIXES = (
    "torchlens/_state.py",
    "torchlens/_capture_state_helpers.py",
    "torchlens/_save_budget.py",
    "torchlens/backends/torch/",
    "torchlens/capture/",
    "torchlens/fastlog/",
    "torchlens/utils/rng.py",
    "torchlens/utils/_torch_compat.py",
)


@pytest.mark.smoke
def test_hot_state_subset_census_runs_in_the_commit_gate() -> None:
    """Commit-gate sentinel over the hot capture/wrapper state surface.

    The r3settle2 re-tier moved the whole-package census to ``heavy`` (its
    ~9s scan cannot fit the 5s smoke partition), which REOPENED the exact
    b2p2-F2 hole: unclassified globals accumulate for days before the mid
    backstop runs. This bounded census re-arms the COMMIT gate over the
    territory where that incident actually happened; the heavy tests remain
    the exhaustive whole-package authority.
    """

    repo = Path(__file__).resolve().parents[1]
    classified = set().union(*_LIFECYCLE_CLASSES)
    subset_paths = [
        path
        for path in _package_python_paths(repo)
        if path.relative_to(repo).as_posix().startswith(_HOT_STATE_SUBSET_PREFIXES)
    ]
    trees = {path.relative_to(repo).as_posix(): package_ast(path) for path in subset_paths}
    mutated_names = _names_mutated_in_place(trees)
    observed: set[tuple[str, str]] = set()
    for relative, tree in trees.items():
        bindings = _module_level_mutable_bindings(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                for name in node.names:
                    observed.add((relative, name))
        for name in bindings:
            if name in mutated_names:
                observed.add((relative, name))
    for owner, attribute in _cross_module_attribute_rebinds(trees, frozenset(trees)):
        observed.add((owner, attribute))
    missing = sorted(observed - classified)
    assert not missing, (
        "unclassified mutable module state landed in the hot capture/wrapper "
        "surface (assign each a lifecycle class in this file; the heavy "
        f"whole-package census is the exhaustive authority): {missing}"
    )


# heavy, not smoke (r3settle2 budget lint): same whole-process scan cost
# as the shrink-only inventory test above (~10s); the smoke-tier sentinel
# above covers the hot subset in the commit gate.
@pytest.mark.heavy
def test_weakly_held_state_is_exactly_the_declared_ledger() -> None:
    """The weak/strong split of every inventory member is frozen and exact.

    This is what keeps the inventory from being a rubber stamp. A lifecycle row
    says what a global is FOR; this says how it HOLDS its subjects, which is the
    difference between a side table that dies with its trace and one that pins
    every captured graph in the process. Both directions are checked, so neither
    weakening nor strengthening a container can pass unreviewed.
    """

    repo = Path(__file__).resolve().parents[1]
    observed = _mutable_module_state(repo)
    weakly_held = {
        entry for entry, initializer in observed.items() if "weakref.Weak" in initializer
    }

    became_strong = sorted(_WEAKLY_HELD - weakly_held)
    became_weak = sorted(weakly_held - _WEAKLY_HELD)
    assert not became_strong, (
        "declared-weak module state is no longer bound to a weakref container "
        f"(it now pins its subjects): {became_strong}"
    )
    assert not became_weak, (
        "module state became weak without updating the ledger (welcome, but the "
        f"row has to move): {became_weak}"
    )
    assert set().union(*_LIFECYCLE_CLASSES) >= _WEAKLY_HELD, (
        "weak-ledger rows missing from every lifecycle class"
    )
    assert _WEAK_SUBJECT_TABLES <= _WEAKLY_HELD, (
        "a member of the weak-subject-table CLASS is not in the weak ledger"
    )


@pytest.mark.parametrize("error_type", [RuntimeError, _InjectedBaseFailure])
def test_mid_capture_failure_restores_process_state(
    error_type: type[BaseException],
) -> None:
    """Ordinary and interruption failures leave no stale capture owner.

    Parameters
    ----------
    error_type:
        Failure class injected after one recorded operation.
    """

    tl.trace(nn.ReLU(), torch.ones(2))
    before = _capture_scope_snapshot()

    with pytest.raises(error_type, match="injected mid-capture failure"):
        tl.trace(_RaiseMidCapture(error_type), torch.ones(2))

    assert _capture_scope_snapshot() == before
    recovered = tl.trace(nn.ReLU(), torch.ones(2))
    assert any(op.func_name == "relu" for op in recovered.compute_ops)


def test_concurrent_public_capture_refuses_without_corruption() -> None:
    """Overlapping public captures fail loudly and leave the owner intact."""

    tl.trace(nn.ReLU(), torch.ones(2))
    before = _capture_scope_snapshot()
    entered = threading.Event()
    release = threading.Event()
    owner_errors: list[BaseException] = []

    def run_owner() -> None:
        """Run the capture that owns process-global logging state."""

        try:
            tl.trace(_BlockingCapture(entered, release), torch.ones(2))
        except BaseException as error:
            owner_errors.append(error)

    owner = threading.Thread(target=run_owner)
    owner.start()
    assert entered.wait(timeout=5.0), "owner capture never reached its forward"
    try:
        with pytest.raises(_state.ReentrantTraceError, match="not re-entrant"):
            tl.trace(nn.ReLU(), torch.ones(2))
    finally:
        release.set()
        owner.join(timeout=5.0)

    assert not owner.is_alive(), "owner capture did not finish after release"
    assert owner_errors == []
    assert _capture_scope_snapshot() == before


def test_refused_concurrent_capture_leaves_winner_verified() -> None:
    """A refused loser must not degrade the admitted winner's data quality.

    The admission lock made "exactly one admitted" atomic, but a loser used to
    run its capture-global side effects FIRST: model preparation swept and
    replaced the winner's live label session before the loser reached the
    typed refusal, leaving the winner with orphaned label stamps and
    ``capture_verified=False`` (runtime-probed, hunt-b2 R54). The reservation
    now refuses the loser BEFORE any pre-admission mutation, so the winner
    completes verified and the loser's model is never even prepared.
    """

    tl.trace(nn.ReLU(), torch.ones(2))
    before = _capture_scope_snapshot()
    entered = threading.Event()
    release = threading.Event()
    owner_errors: list[BaseException] = []
    owner_traces: list[Any] = []

    def run_owner() -> None:
        """Run the capture whose data quality the loser must not degrade."""

        try:
            owner_traces.append(tl.trace(_BlockingCapture(entered, release), torch.ones(2)))
        except BaseException as error:  # pragma: no cover - surfaced by asserts
            owner_errors.append(error)

    owner = threading.Thread(target=run_owner)
    owner.start()
    assert entered.wait(timeout=5.0), "owner capture never reached its forward"
    loser_model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
    try:
        with pytest.raises(_state.ReentrantTraceError, match="not re-entrant"):
            tl.trace(loser_model, torch.ones(1, 2))
    finally:
        release.set()
        owner.join(timeout=10.0)

    assert not owner.is_alive(), "owner capture did not finish after release"
    assert owner_errors == []
    assert len(owner_traces) == 1
    winner = owner_traces[0]
    assert winner.capture_verified is not False, (
        "the refused loser's pre-admission side effects degraded the winner: "
        f"capture_verified={winner.capture_verified!r}, "
        f"reason={getattr(winner, 'capture_verification_reason', None)!r}"
    )
    assert any(op.func_name == "relu" for op in winner.compute_ops)
    # The loser must have been refused BEFORE model preparation ran.
    assert loser_model not in _state._prepared_models, (
        "the refused loser's model was prepared: its label-session swap ran "
        "before the admission refusal"
    )
    assert _capture_scope_snapshot() == before


def test_refused_concurrent_record_never_installs_recording_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused ``tl.record`` must not touch the fastlog recording global.

    The loser used to install its ``RecordingState`` (overwriting the admitted
    recorder's) and only then reach the inner admission refusal, projecting the
    winner's events into the loser's state for that window. The recorder-side
    reservation refuses before the install.
    """

    from torchlens.fastlog import _recorder as recorder_module

    installs: list[Any] = []
    real_install = recorder_module.active_recording_state

    def counting_install(state: Any) -> Any:
        """Record every recording-state install before delegating."""

        installs.append(state)
        return real_install(state)

    monkeypatch.setattr(recorder_module, "active_recording_state", counting_install)

    entered = threading.Event()
    release = threading.Event()
    owner_errors: list[BaseException] = []

    def run_owner() -> None:
        """Hold a live capture open while the record() loser is refused."""

        try:
            tl.trace(_BlockingCapture(entered, release), torch.ones(2))
        except BaseException as error:  # pragma: no cover - surfaced by asserts
            owner_errors.append(error)

    owner = threading.Thread(target=run_owner)
    owner.start()
    assert entered.wait(timeout=5.0), "owner capture never reached its forward"
    try:
        with pytest.raises(_state.ReentrantTraceError, match="not re-entrant"):
            tl.record(nn.ReLU(), torch.ones(2), save=tl.func("relu"))
    finally:
        release.set()
        owner.join(timeout=10.0)

    assert owner_errors == []
    assert installs == [], (
        "the refused record() installed its RecordingState before the admission refusal fired"
    )


def test_publish_active_trace_refuses_concurrent_and_clears_owner() -> None:
    """Non-forward publication windows are admission-locked and owner-stamped.

    tf capture and paddle derived-grad replays publish ``_active_trace``
    without the logging toggle; a raw save/restore swap bypassed admission
    (silent corruption of a concurrent torch capture) and could republish a
    finished trace on restore. ``publish_active_trace`` must refuse typed
    against a live capture, set the owner thread id for the window, and clear
    both on exit.
    """

    sentinel = cast("Any", object())
    with _state.publish_active_trace(sentinel):
        assert _state._active_trace is sentinel
        assert _state._active_owner_thread_id == threading.get_ident()
        # A second publication (any thread) refuses while the window is open.
        with pytest.raises(_state.ReentrantTraceError, match="not re-entrant"):
            with _state.publish_active_trace(cast("Any", object())):
                pass  # pragma: no cover - refused above
    assert _state._active_trace is None
    assert _state._active_owner_thread_id is None

    entered = threading.Event()
    release = threading.Event()
    owner_errors: list[BaseException] = []

    def run_owner() -> None:
        """Hold a live torch capture open for the publication refusal."""

        try:
            tl.trace(_BlockingCapture(entered, release), torch.ones(2))
        except BaseException as error:  # pragma: no cover - surfaced by asserts
            owner_errors.append(error)

    owner = threading.Thread(target=run_owner)
    owner.start()
    assert entered.wait(timeout=5.0), "owner capture never reached its forward"
    try:
        with pytest.raises(_state.ReentrantTraceError, match="not re-entrant"):
            with _state.publish_active_trace(sentinel):
                pass  # pragma: no cover - refused above
    finally:
        release.set()
        owner.join(timeout=10.0)
    assert owner_errors == []


def test_publish_backward_capture_nests_same_thread_and_restores() -> None:
    """The backward publication handle keeps raw-swap nesting semantics.

    The multi-trace backward bracket and an inner ``backward()`` inside a
    traced forward legitimately nest backward windows on ONE thread, so the
    admission-locked replacement for the raw ``_active_trace`` swap must
    save/restore LIFO on the same thread, restore exactly once (idempotent
    against stacked unwind arms), and clear the owner id at the end.
    """

    outer = cast("Any", object())
    inner = cast("Any", object())
    outer_publication = _state.publish_backward_capture(
        outer, hook_plan=None, intervention_spec=None
    )
    try:
        assert _state._active_trace is outer
        assert _state._active_owner_thread_id == threading.get_ident()
        inner_publication = _state.publish_backward_capture(
            inner, hook_plan=None, intervention_spec=None
        )
        assert _state._active_trace is inner
        inner_publication.restore()
        assert _state._active_trace is outer
        inner_publication.restore()  # idempotent: must NOT re-clobber to inner
        assert _state._active_trace is outer
    finally:
        outer_publication.restore()
    assert _state._active_trace is None
    assert _state._active_owner_thread_id is None


def test_log_backward_refuses_foreign_live_window_instead_of_wedging() -> None:
    """``log_backward`` concurrent with a foreign capture window refuses typed.

    The last unconverted raw ``_active_trace`` swap: interleaved with another
    thread's live window, the unlocked save could snapshot that window's trace
    as "previous" and the ``finally`` republished it after the window closed —
    ``_active_trace`` stayed permanently non-``None`` and EVERY later capture
    was refused (a process-global wedge). The admission-locked publication
    refuses the foreign window typed instead; after the window closes the same
    backward and later captures run untouched.
    """

    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    x = torch.randn(2, 4, requires_grad=True)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads=True))
    loss = trace[trace.output_layers[0]].out.sum()

    sentinel = cast("Any", object())
    entered = threading.Event()
    release = threading.Event()
    window_errors: list[BaseException] = []

    def hold_window() -> None:
        """Hold a foreign non-forward publication window open."""

        try:
            with _state.publish_active_trace(sentinel):
                entered.set()
                release.wait(timeout=20.0)
        except BaseException as error:  # pragma: no cover - surfaced by asserts
            window_errors.append(error)

    holder = threading.Thread(target=hold_window)
    holder.start()
    assert entered.wait(timeout=5.0), "foreign window never opened"
    try:
        with pytest.raises(_state.ReentrantTraceError, match="not re-entrant"):
            trace.log_backward(loss)
    finally:
        release.set()
        holder.join(timeout=10.0)
    assert window_errors == []

    # No wedge: the globals are clean, the refused backward runs now, and a
    # later capture is admitted.
    assert _state._active_trace is None
    assert _state._active_owner_thread_id is None
    trace.log_backward(loss)
    assert int(trace.num_backward_passes) == 1
    recovered = tl.trace(nn.ReLU(), torch.ones(2))
    assert any(op.func_name == "relu" for op in recovered.compute_ops)


def test_capture_reservation_is_released_on_failure_and_nested_same_thread() -> None:
    """The reservation never leaks (wedging admission) and gates re-entry.

    A capture failing anywhere between the reservation claim and teardown must
    release the slot, or every later capture refuses forever. Same-thread
    nesting passes through ONLY with the yielded continuation token (the
    recorder hands it to the inner orchestration); a bare same-thread re-entry
    is a nested public capture from user code inside the reserved window and
    refuses typed (R55), as does a foreign thread's claim.
    """

    with pytest.raises(RuntimeError, match="injected mid-capture failure"):
        tl.trace(_RaiseMidCapture(RuntimeError), torch.ones(2))
    assert _state._capture_reserved_by is None
    recovered = tl.trace(nn.ReLU(), torch.ones(2))
    assert any(op.func_name == "relu" for op in recovered.compute_ops)

    with _state.capture_reservation() as token:
        assert _state._capture_reserved_by == threading.get_ident()
        # Sanctioned re-entry: presenting the live token passes through.
        with _state.capture_reservation(resume=token):
            assert _state._capture_reserved_by == threading.get_ident()
        # The inner exit must not release the outer claim.
        assert _state._capture_reserved_by == threading.get_ident()
        # Same-thread entry WITHOUT the live claim is the R55 nested-capture
        # hole and must refuse typed, not pass through.
        with pytest.raises(_state.ReentrantTraceError):
            with _state.capture_reservation():
                pass  # pragma: no cover - refused above
        with pytest.raises(_state.ReentrantTraceError):
            with _state.capture_reservation(resume=object()):
                pass  # pragma: no cover - refused above
        # The refused entries must not have released or reclaimed the slot.
        assert _state._capture_reserved_by == threading.get_ident()

        # R55: a bare same-thread re-entry (no token) is a nested PUBLIC
        # capture started inside the reserved window -- both captures used to
        # run to completion; it must refuse typed instead.
        with pytest.raises(_state.ReentrantTraceError):
            with _state.capture_reservation():
                pass  # pragma: no cover - refused above
        # A stale/forged token refuses identically.
        with pytest.raises(_state.ReentrantTraceError):
            with _state.capture_reservation(resume=object()):
                pass  # pragma: no cover - refused above
        # The refusals must not release or corrupt the live claim.
        assert _state._capture_reserved_by == threading.get_ident()

        foreign_error: list[BaseException] = []

        def contend() -> None:
            """Attempt a foreign-thread reservation against the live claim."""

            try:
                with _state.capture_reservation():
                    pass  # pragma: no cover - refused above
            except BaseException as error:
                foreign_error.append(error)

        contender = threading.Thread(target=contend)
        contender.start()
        contender.join(timeout=5.0)
        assert len(foreign_error) == 1
        assert isinstance(foreign_error[0], _state.ReentrantTraceError)
    assert _state._capture_reserved_by is None


def test_nested_public_capture_inside_reserved_window_refuses_typed() -> None:
    """A nested ``tl.trace`` from user code inside the reserved window refuses.

    R55 (r7 b8-sol): the same-thread reservation passthrough keyed on thread
    ident alone, so user code running inside the outer capture's reserved
    pre-admission window (here: a tensor-subclass ``__torch_function__`` fired
    by input setup) could start a nested PUBLIC capture that ran to completion
    -- BOTH captures settled (probe ``OUTER_OK``/``INNER_COMPLETED`` at the
    hunt-6 pin) instead of the documented ``ReentrantTraceError``. The
    passthrough now requires the recorder's continuation token; the nested
    entry holds none and refuses typed, and the refusal releases nothing it
    does not own.
    """

    fired: dict[str, object] = {"result": None}

    class _NestedTraceTensor(torch.Tensor):
        @classmethod
        def __torch_function__(
            cls, func: object, types: object, args: tuple = (), kwargs: dict | None = None
        ) -> object:
            kwargs = kwargs or {}
            if fired["result"] is None and _state._capture_reserved_by is not None:
                fired["result"] = "fired"
                try:
                    tl.trace(nn.ReLU(), torch.ones(2))
                    fired["result"] = "inner_completed"
                except _state.ReentrantTraceError:
                    fired["result"] = "inner_refused"
                    raise
            return super().__torch_function__(func, types, args, kwargs)

    inputs = torch.randn(2, 3).as_subclass(_NestedTraceTensor)
    with pytest.raises(_state.ReentrantTraceError):
        tl.trace(nn.Linear(3, 2), inputs)
    assert fired["result"] == "inner_refused"
    assert _state._capture_reserved_by is None
    # The refusal left admission clean: a fresh capture is admitted.
    recovered = tl.trace(nn.ReLU(), torch.ones(2))
    assert any(op.func_name == "relu" for op in recovered.compute_ops)


def test_auto_name_counter_is_atomic_under_racing_threads() -> None:
    """Racing pre-admission ``_auto_name`` calls never mint duplicate names.

    R54 (r7 b8-sol): ``_auto_name`` runs during capture setup BEFORE
    admission, outside ``active_logging()``'s guard, and its unlocked
    read-modify-write let two threads read the same counter value and stamp
    two captures with the SAME auto name. The get+increment is now atomic
    under ``_state._naming_lock``.
    """

    class _RaceNamed:
        pass

    _state.reset_naming_counter("_racenamed")
    switch_interval = sys.getswitchinterval()
    names: list[str] = []
    names_lock = threading.Lock()
    barrier = threading.Barrier(4)

    def mint(count: int) -> None:
        barrier.wait()
        local = [_state._auto_name(_RaceNamed()) for _ in range(count)]
        with names_lock:
            names.extend(local)

    sys.setswitchinterval(1e-6)
    try:
        workers = [threading.Thread(target=mint, args=(2000,)) for _ in range(4)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=30.0)
    finally:
        sys.setswitchinterval(switch_interval)
        _state.reset_naming_counter("_racenamed")
    assert len(names) == 8000
    assert len(set(names)) == 8000, "duplicate auto names minted under the race"


def test_register_container_racing_lookup_never_breaks_iteration() -> None:
    """``register_container`` racing a lookup never raises mid-iteration.

    R54 (r7 b2-sol): ``get_registered_container`` iterated the LIVE registry
    dict while public ``register_container`` mutated it from another thread --
    a registration landing mid-capture raised ``RuntimeError: dictionary
    changed size during iteration`` inside the forward walk. The reader now
    snapshots under the registry lock.
    """

    from torchlens.ir import container as container_mod

    registered_types: list[type] = []
    errors: list[BaseException] = []
    stop = threading.Event()
    switch_interval = sys.getswitchinterval()

    class _LookupProbe:
        pass

    def writer() -> None:
        try:
            for index in range(400):
                if stop.is_set():
                    break
                fresh = type(f"_RaceContainer{index}", (), {})
                registered_types.append(fresh)
                tl.register_container(
                    fresh,
                    lambda value: ([], None),
                    lambda aux, children: object(),
                )
        except BaseException as error:  # pragma: no cover - the defect signal
            errors.append(error)

    def reader() -> None:
        try:
            while not stop.is_set():
                container_mod.get_registered_container(_LookupProbe)
        except BaseException as error:  # pragma: no cover - the defect signal
            errors.append(error)

    sys.setswitchinterval(1e-6)
    try:
        reader_thread = threading.Thread(target=reader)
        writer_thread = threading.Thread(target=writer)
        reader_thread.start()
        writer_thread.start()
        writer_thread.join(timeout=30.0)
        stop.set()
        reader_thread.join(timeout=30.0)
    finally:
        sys.setswitchinterval(switch_interval)
        stop.set()
        with container_mod._CONTAINER_REGISTRY_LOCK:
            for registered in registered_types:
                container_mod._CONTAINER_REGISTRY.pop(registered, None)
    assert errors == [], f"registry race surfaced: {errors!r}"


def test_foreign_thread_pause_does_not_blind_the_owner_capture() -> None:
    """A non-owner ``pause_logging()`` never drops the owner's later ops.

    ``_PauseLogging.__enter__`` used to clear the process-global toggle
    unconditionally, so ANY thread pausing (even one only analyzing an old
    Trace) silently truncated a live capture from that instant on. The owner
    check now lives in the context manager itself, covering every call site.
    """

    ops_per_side = 3
    model = _PauseFromForeignThread(ops_per_side)
    trace = tl.trace(model, torch.ones(2))

    assert model.foreign_error == [], f"foreign pause raised {model.foreign_error!r}"
    relu_ops = [op for op in trace.compute_ops if op.func_name == "relu"]
    assert len(relu_ops) == 3 * ops_per_side, (
        "ops logged while a non-owner thread held pause_logging() are missing: "
        f"the foreign pause blinded the capture (saw {len(relu_ops)} of "
        f"{3 * ops_per_side} relu ops)"
    )
    assert _state._logging_enabled is False
    assert _state._active_owner_thread_id is None


def test_capture_admission_runs_under_the_admission_lock() -> None:
    """Admission blocks while another thread holds the admission lock.

    ``active_logging``'s refusal check and its publication of ``_active_trace`` /
    ``_active_owner_thread_id`` / ``_logging_enabled`` are separate bytecodes.
    Unlocked, two threads entering together can both pass the check, and the
    loser then overwrites the winner's owner id — after which every op the winner
    logs is dropped by the wrapper's owner-thread fast path and its Trace is
    silently short, with no error anywhere. The check and the publication
    therefore have to happen under one lock; this asserts that directly, because
    the racing window itself is only a few bytecodes wide and a probabilistic
    probe cannot gate it reliably.
    """

    before = _capture_scope_snapshot()
    entered = threading.Event()
    blocked_for_lock = threading.Event()
    admitted = threading.Event()
    failures: list[BaseException] = []

    def admit() -> None:
        """Enter and immediately leave one capture session."""

        try:
            with _state.active_logging(cast("Any", object())):
                admitted.set()
        except BaseException as error:  # pragma: no cover - reported by the test
            failures.append(error)
            admitted.set()

    with _state._capture_admission_lock:
        entered.set()
        worker = threading.Thread(target=admit)
        worker.start()
        # The worker cannot reach the check, let alone publish, while the lock
        # is held here. If admission ran outside the lock it would sail through.
        blocked_for_lock.wait(timeout=0.5)
        assert not admitted.is_set(), (
            "active_logging admitted a capture while the admission lock was "
            "held: the check-then-publish sequence is not serialized"
        )
        assert _state._active_trace is None
        assert _state._active_owner_thread_id is None

    assert admitted.wait(timeout=10.0), "admission never completed after release"
    worker.join(timeout=10.0)
    assert not worker.is_alive(), "admission worker hung"
    assert failures == [], f"admission failed after the lock was released: {failures!r}"
    assert _capture_scope_snapshot() == before


def test_contended_admission_never_publishes_partial_owner_state() -> None:
    """Under contention, an admitted session owns the globals for its whole body.

    Complements the lock test above: whatever the interleaving, a thread that is
    admitted must see its OWN owner id and trace for the entire session, and
    every other thread must get the documented refusal rather than a corrupted
    half-published state. Concurrent capture stays unsupported by design; this
    pins the admission mechanism's behavior under contention.
    """

    before = _capture_scope_snapshot()
    contenders = 4
    rounds = 25
    admitted = 0
    refused = 0
    stolen: list[tuple[int, int | None]] = []
    unexpected: list[BaseException] = []
    lock = threading.Lock()
    # Bytecode-level interleaving is what the admission lock excludes; make the
    # scheduler switch as often as possible so contention is real here.
    prior_switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for _ in range(rounds):
            barrier = threading.Barrier(contenders)

            def contend(barrier: threading.Barrier = barrier) -> None:
                """Enter ``active_logging`` at the same instant as the others."""

                nonlocal admitted, refused
                token = object()
                barrier.wait(timeout=10.0)
                try:
                    with _state.active_logging(cast("Any", token)):
                        mine = threading.get_ident()
                        for _ in range(200):
                            owner = _state._active_owner_thread_id
                            if owner != mine or _state._active_trace is not token:
                                with lock:
                                    stolen.append((mine, owner))
                                break
                except _state.ReentrantTraceError:
                    with lock:
                        refused += 1
                except BaseException as error:  # pragma: no cover - test signal
                    with lock:
                        unexpected.append(error)
                else:
                    with lock:
                        admitted += 1

            threads = [threading.Thread(target=contend) for _ in range(contenders)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=20.0)
            assert not any(thread.is_alive() for thread in threads), "contender hung"
    finally:
        sys.setswitchinterval(prior_switch_interval)

    assert unexpected == [], f"unexpected admission failure: {unexpected!r}"
    assert stolen == [], (
        "an admitted capture's owner globals were overwritten by a racing "
        f"contender (mine, observed_owner) pairs: {stolen!r}"
    )
    assert admitted + refused == contenders * rounds
    assert admitted >= 1, "no capture was admitted at all"
    assert _capture_scope_snapshot() == before


def test_unwrap_torch_refuses_during_an_active_capture() -> None:
    """Mid-capture ``unwrap_torch()`` is a typed refusal, not a silent truncation.

    Reachable single-threaded: from a forward hook, an ``activation_transform``,
    or any user callback running inside the traced forward. Removing the
    wrappers there left the rest of the forward unlogged and returned a
    truncated Trace with no error at all.
    """

    from torchlens.backends.torch.wrappers import unwrap_torch

    seen: list[BaseException] = []

    class _UnwrapMidForward(nn.Module):
        """Attempt an unwrap between two logged operations."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run one op, try to unwrap, then run another op.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                Activation after both operations.
            """

            x = torch.relu(x)
            try:
                unwrap_torch()
            except BaseException as error:
                seen.append(error)
            return torch.relu(x)

    trace = tl.trace(_UnwrapMidForward(), torch.ones(2))

    assert len(seen) == 1, "unwrap_torch() mid-capture did not refuse"
    error = seen[0]
    assert isinstance(error, tl.errors.CaptureContextError)
    assert error.fields["code"] == "unwrap_during_active_capture"
    relu_ops = [op for op in trace.compute_ops if op.func_name == "relu"]
    assert len(relu_ops) == 2, "the refused unwrap still truncated the capture"

    # The wrappers survived the refusal: the next capture needs no re-wrap.
    recovered = tl.trace(nn.ReLU(), torch.ones(2))
    assert any(op.func_name == "relu" for op in recovered.compute_ops)


def test_release_model_refuses_during_an_active_capture() -> None:
    """Mid-capture ``tl.release_model()`` is a typed refusal, not silent damage.

    The missing sibling of the ``unwrap_torch`` guard: releasing the model
    mid-forward (reachable single-threaded from a forward hook or
    ``activation_transform``) stripped the ``tl_*`` / ``._tl`` metadata the
    live capture's module attribution reads, and the capture then finished
    ``capture_verified`` with silently emptied module attribution.
    """

    seen: list[BaseException] = []

    class _ReleaseMidForward(nn.Module):
        """Attempt a self-release between two logged operations."""

        def __init__(self) -> None:
            """Build a submodule so module attribution has something to lose."""

            super().__init__()
            self.inner = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run one module, try to release, then run another op.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                Activation after both operations.
            """

            x = self.inner(x)
            try:
                tl.release_model(self)
            except BaseException as error:
                seen.append(error)
            return torch.relu(x)

    model = _ReleaseMidForward()
    trace = tl.trace(model, torch.ones(2))

    assert len(seen) == 1, "release_model() mid-capture did not refuse"
    error = seen[0]
    assert isinstance(error, tl.errors.CaptureContextError)
    assert error.fields["code"] == "release_during_active_capture"
    relu_ops = [op for op in trace.compute_ops if op.func_name == "relu"]
    assert len(relu_ops) == 2, "the refused release still truncated the capture"
    # Module attribution survived: the submodule call is still attributed.
    assert any(op.modules for op in trace.compute_ops), (
        "the refused release still emptied module attribution"
    )

    # Releasing AFTER the capture stays the supported no-questions path.
    tl.release_model(model)
    recovered = tl.trace(model, torch.ones(2))
    assert any(op.modules for op in recovered.compute_ops)


def test_cleanup_of_active_trace_refuses_mid_capture() -> None:
    """Husking the live capture's own trace mid-forward refuses typed.

    Sibling of the ``unwrap_torch`` / ``release_model`` guards: without the
    refusal the capture died later on a raw ``AttributeError`` (missing
    ``_wrapper_runtime_ws``) deep inside the commit path. Cleaning up a
    DIFFERENT, finished trace during a capture stays supported.
    """

    from torchlens import _state

    finished = tl.trace(nn.ReLU(), torch.ones(2))
    seen: list[BaseException] = []

    class _CleanupMidForward(nn.Module):
        """Attempt to husk the active trace between two logged ops."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Refused self-cleanup; allowed foreign-trace cleanup.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                Activation after both operations.
            """

            x = torch.relu(x)
            try:
                _state._active_trace.cleanup()
            except BaseException as error:
                seen.append(error)
            finished.cleanup()  # foreign finished trace: must stay allowed
            return torch.relu(x)

    trace = tl.trace(_CleanupMidForward(), torch.ones(2))

    assert len(seen) == 1, "cleanup() of the active trace mid-capture did not refuse"
    error = seen[0]
    assert isinstance(error, tl.errors.CaptureContextError)
    assert error.fields["code"] == "cleanup_during_active_capture"
    relu_ops = [op for op in trace.compute_ops if op.func_name == "relu"]
    assert len(relu_ops) == 2, "the refused cleanup still truncated the capture"
    trace.cleanup()  # post-capture cleanup stays the supported path


def test_interrupted_partial_diagnostics_still_restore_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interruption during partial-trace recovery still tears the session down.

    The failed-forward epilogue builds best-effort partial diagnostics and then
    restores the model. Both arms of that construction catch ``Exception``, so a
    ``KeyboardInterrupt`` raised inside it escaped straight past
    ``cleanup_model_session`` — leaving the user's model with TorchLens-forced
    ``requires_grad``, ``tl_*`` attributes, and an installed buffer tracker.
    """

    from torchlens import partial as partial_module

    class _FailingModel(nn.Module):
        """Register a frozen parameter and then fail the forward."""

        def __init__(self) -> None:
            """Build a model with one explicitly frozen parameter."""

            super().__init__()
            self.weight = nn.Parameter(torch.ones(2), requires_grad=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run one logged op and then raise.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                This path never returns.

            Raises
            ------
            ValueError
                Always, after one logged operation.
            """

            _ = torch.relu(x * self.weight)
            raise ValueError("forward failure with interrupted diagnostics")

    def _interrupt_partial_construction(*_args: Any, **_kwargs: Any) -> Any:
        """Interrupt partial-trace construction the way Ctrl-C would."""

        raise KeyboardInterrupt("interrupted during partial construction")

    monkeypatch.setattr(partial_module.PartialTrace, "from_trace", _interrupt_partial_construction)

    model = _FailingModel()
    before = _capture_scope_snapshot()

    with pytest.raises(KeyboardInterrupt):
        tl.trace(model, torch.ones(2))

    assert model.weight.requires_grad is False, (
        "the interrupted epilogue left TorchLens-forced requires_grad on a frozen parameter"
    )
    assert not [name for name in vars(model) if name.startswith("tl_")], (
        "the interrupted epilogue left tl_* session metadata on the model"
    )
    assert _capture_scope_snapshot() == before
    # The process is still usable for the next capture.
    recovered = tl.trace(nn.ReLU(), torch.ones(2))
    assert any(op.func_name == "relu" for op in recovered.compute_ops)
