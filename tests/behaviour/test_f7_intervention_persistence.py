"""F7: an intervened trace's recipe persists through the saved format (io.1).

``trace.save`` writes the staged intervention spec as a typed recipe beside the
records. A recipe of built-in edits loads back equal to the live one and the
loaded trace reruns it, reproducing the live intervened activations. A rule
holding a user callable is saved as a placeholder (its position and qualified
name, never its code), the manifest discloses that placeholder, and every rerun
door of the loaded trace refuses ``run_intervention_spec_not_persisted`` naming
the position. Both saves pass the save preflight reload and both loads run the
loaded-side validators, asserted through a call ledger that observes them
(nothing is switched off).
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens.errors import TorchLensError

NOT_PERSISTED = "run_intervention_spec_not_persisted"
#: Validators the save preflight reload (late.21) runs before an artifact is published.
SAVE_VALIDATORS = frozenset({"preflight_reload", "reload_check", "validate_artifact"})
#: Validators the reader runs on every load: envelope, file digests, record tables.
LOAD_VALIDATORS = frozenset(
    {"validate_artifact", "check_version_window", "verify_file_bytes", "validate_records"}
)


class _Toy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Linear(4, 6)
        self.b = torch.nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.a(x)))


def clamp_top(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
    """A user callable edit: the saved format keeps its position and name, never its code."""

    return out.clamp(max=0.1)


def _builtin_spec() -> Any:
    return tl.when(tl.func("relu"), tl.scale(0.5))


def _callable_spec() -> Any:
    return tl.when(tl.func("linear"), tl.scale(0.5)) & tl.when(tl.func("relu"), clamp_top)


@dataclass(frozen=True)
class Saved:
    """One intervened capture, its artifact and the validators its save ran."""

    live: Any
    root: Path
    save_calls: frozenset[str]


@contextmanager
def _validator_ledger() -> Iterator[set[str]]:
    """Record the name of every TorchLens function called inside the block.

    A profile hook only observes calls; every validator still runs and can
    still refuse. The previous profile hook is restored on exit.
    """

    calls: set[str] = set()

    def _profile(frame: Any, event: str, arg: Any) -> None:
        if event == "call" and str(frame.f_globals.get("__name__", "")).startswith("torchlens"):
            calls.add(frame.f_code.co_name)

    previous = sys.getprofile()
    sys.setprofile(_profile)
    try:
        yield calls
    finally:
        sys.setprofile(previous)


def _error_code(exc: BaseException) -> str:
    fields = getattr(exc, "fields", None) or {}
    return str(fields.get("code", "") or getattr(exc, "code", ""))


@pytest.fixture(scope="module")
def model() -> _Toy:
    torch.manual_seed(0)
    return _Toy().eval()


@pytest.fixture(scope="module")
def inputs() -> torch.Tensor:
    return torch.randn(2, 4, generator=torch.Generator().manual_seed(1))


def _save(
    model: _Toy, inputs: torch.Tensor, spec: Any, root: Path, *, level: str = "portable"
) -> Saved:
    live = tl.trace(model, inputs, intervene=spec)
    with _validator_ledger() as calls:
        live.save(root, level=level)
    return Saved(live, root, frozenset(calls))


@pytest.fixture(scope="module")
def builtin_saved(model: _Toy, inputs: torch.Tensor, tmp_path_factory: Any) -> Iterator[Saved]:
    saved = _save(model, inputs, _builtin_spec(), tmp_path_factory.mktemp("f7") / "builtin")
    try:
        yield saved
    finally:
        saved.live.cleanup()


@pytest.fixture(scope="module")
def callable_saved(model: _Toy, inputs: torch.Tensor, tmp_path_factory: Any) -> Iterator[Saved]:
    saved = _save(
        model, inputs, _callable_spec(), tmp_path_factory.mktemp("f7") / "callable", level="audit"
    )
    try:
        yield saved
    finally:
        saved.live.cleanup()


def _saved_outs(trace: Any) -> dict[str, torch.Tensor]:
    return {op.label: op.out for op in trace.ops if isinstance(op.out, torch.Tensor)}


def test_builtin_recipe_rerun_reproduces_live_activations(
    builtin_saved: Saved, model: _Toy, inputs: torch.Tensor
) -> None:
    """(a) A loaded built-in edit recipe reruns to the live intervened activations, element for element."""

    live_outs = _saved_outs(builtin_saved.live)
    edited = [op.label for op in builtin_saved.live.ops if op.is_edited]
    assert edited, "the built-in edit fired at no op"
    loaded = tl.load(builtin_saved.root)
    loaded.run(model, inputs)
    rerun_outs = _saved_outs(loaded)
    assert set(rerun_outs) == set(live_outs)
    for label, want in live_outs.items():
        assert torch.equal(rerun_outs[label], want), f"{label} differs after the loaded rerun"
    plain = tl.trace(model, inputs)
    for label in edited:
        assert not torch.equal(plain[label].out, rerun_outs[label]), (
            f"{label}: the loaded rerun matches the un-intervened capture, so the recipe did not run"
        )


def _target_row(target: Any) -> tuple[Any, ...]:
    return (
        target.selector_kind,
        target.selector_value,
        target.strict,
        target.slice_spec,
        target.metadata,
    )


def _edit_row(edit: Any) -> tuple[Any, ...] | None:
    if edit is None:
        return None
    return (
        edit.name,
        tuple(edit.args),
        tuple(sorted(dict(edit.kwargs).items())),
        edit.portability,
        edit.signal,
        edit.is_batch_independent,
        edit.is_compatible_with_append,
        tuple(sorted(dict(edit.metadata).items())),
    )


def _recipe_rows(recipe: Any) -> dict[str, Any]:
    """Return every column of a frozen recipe's rules in a comparable form."""

    return {
        "targets": [_target_row(target) for target in recipe.targets],
        "helper": _edit_row(recipe.helper),
        "target_value_specs": list(recipe.target_value_specs),
        "hook_specs": [
            (
                _target_row(spec.site_target),
                _edit_row(spec.helper),
                repr(spec.hook),
                spec.handle,
                spec.metadata,
            )
            for spec in recipe.hook_specs
        ],
    }


def test_loaded_recipe_equals_live_recipe(builtin_saved: Saved) -> None:
    """(b) The loaded recipe equals the live one column for column, and so do the per-op fire records."""

    live = builtin_saved.live
    loaded = tl.load(builtin_saved.root)
    assert _recipe_rows(loaded.recipe) == _recipe_rows(live.recipe)
    edited = [op.label for op in live.ops if op.is_edited]
    assert [op.label for op in loaded.ops if op.is_edited] == edited
    for label in edited:
        assert loaded[label].edits == live[label].edits, f"{label}: fire records differ"


def test_callable_rule_saves_a_disclosed_placeholder(callable_saved: Saved) -> None:
    """(c) A rule holding a user callable saves as a placeholder the manifest discloses."""

    root = callable_saved.root
    qualname = f"{clamp_top.__module__}.{clamp_top.__qualname__}"
    intervention = json.loads((root / "manifest.json").read_text())["intervention"]
    assert intervention["executable"] is False
    assert intervention["placeholders"] == [{"position": 1, "qualname": qualname}]
    recipe = json.loads((root / intervention["file"]).read_text())
    assert recipe["executable"] is False
    placeholders = [rule for rule in recipe["rules"] if rule["edit"]["portability"] != "builtin"]
    assert [rule["index"] for rule in placeholders] == [1]
    stub = placeholders[0]["edit"]["callable"]
    assert stub["$tl"] == "callable_placeholder"
    assert (stub["position"], stub["qualname"]) == (1, qualname)
    assert placeholders[0]["edit"]["portability"] == "callable_placeholder"
    for path in root.rglob("*"):
        if path.is_file():
            assert path.suffix not in {".pkl", ".pickle"}, f"pickled file {path.name}"
            assert b"clamp(max" not in path.read_bytes(), f"{path.name} holds the callable's code"


_DOORS: dict[str, Callable[[Any, _Toy, torch.Tensor], Any]] = {
    "run": lambda trace, model, inputs: trace.run(model, inputs),
    "staged_rerun": lambda trace, model, inputs: trace.do(
        tl.module("b"),
        tl.scale(2.0),
        model=model,
        inputs=inputs,
        intervention=tl.options.InterventionOptions(engine="rerun", confirm_mutation=True),
    ),
    "fork_run": lambda trace, model, inputs: trace.fork().run(model, inputs),
    "bundle_run": lambda trace, model, inputs: tl.Bundle([trace]).run(model, inputs),
}


@pytest.mark.parametrize("door", sorted(_DOORS))
def test_every_rerun_door_refuses_the_callable_placeholder(
    callable_saved: Saved, model: _Toy, inputs: torch.Tensor, door: str
) -> None:
    """(d) Every rerun door of the loaded trace refuses typed, naming the callable's position."""

    loaded = tl.load(callable_saved.root)
    with pytest.raises(TorchLensError) as refused:
        _DOORS[door](loaded, model, inputs)
    assert _error_code(refused.value) == NOT_PERSISTED
    message = str(refused.value)
    assert "rule 1" in message
    assert clamp_top.__qualname__ in message


def _retamper(root: Path, relpath: str, edit: Callable[[dict[str, Any]], None]) -> None:
    """Edit one JSON file of an artifact and restate its digest, so only content checks can refuse."""

    path = root / relpath
    data = json.loads(path.read_text())
    edit(data)
    raw = json.dumps(data).encode()
    path.write_bytes(raw)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["files"][relpath]
    for key in entry:
        if key in {"sha256", "digest"}:
            entry[key] = hashlib.sha256(raw).hexdigest()
        elif key in {"bytes", "size", "nbytes"}:
            entry[key] = len(raw)
    manifest_path.write_text(json.dumps(manifest))


def test_save_preflight_and_loaded_validators_run_on_both_traces(
    builtin_saved: Saved, callable_saved: Saved, tmp_path: Path
) -> None:
    """(e) Both saves pass the preflight reload, both loads run the validators, and they still refuse."""

    for saved in (builtin_saved, callable_saved):
        assert saved.save_calls >= SAVE_VALIDATORS, sorted(SAVE_VALIDATORS - saved.save_calls)
        with _validator_ledger() as calls:
            tl.load(saved.root)
        assert calls >= LOAD_VALIDATORS, sorted(LOAD_VALIDATORS - calls)
    forged = tmp_path / "forged"
    builtin_saved.live.save(forged, level="portable")

    def _unknown_edit(recipe: dict[str, Any]) -> None:
        recipe["rules"][0]["edit"]["name"] = "not_a_registered_edit"

    _retamper(forged, "intervention.json", _unknown_edit)
    with pytest.raises(TorchLensError):
        tl.load(forged)


@pytest.mark.parametrize(
    "fault", ["unknown_recipe_key", "unknown_capture_key", "bad_capture_index", "capture_not_array"]
)
def test_forged_capture_rule_schema_refuses(
    builtin_saved: Saved, tmp_path: Path, fault: str
) -> None:
    """Original predicates have a closed schema and cannot bypass the label-restage guard."""

    root = tmp_path / "forged_capture"
    builtin_saved.live.save(root, level="portable")

    def forge(recipe: dict[str, Any]) -> None:
        assert recipe["capture_rules"]
        if fault == "unknown_recipe_key":
            recipe["capture_rules_unknown"] = []
        elif fault == "unknown_capture_key":
            recipe["capture_rules"][0]["unknown"] = True
        elif fault == "bad_capture_index":
            recipe["capture_rules"][0]["index"] = True
        elif fault == "bad_capture_args":
            recipe["capture_rules"][0]["edit"]["args"] = 1
        elif fault == "bad_capture_slice":
            recipe["capture_rules"][0]["selector"]["slice_spec"] = {"unknown": 1}
        else:
            recipe["capture_rules"] = {}

    _retamper(root, "intervention.json", forge)
    with pytest.raises(TorchLensError) as caught:
        tl.load(root)
    assert _error_code(caught.value) == "manifest_schema_invalid"


def test_callable_edit_refuses_a_portable_trace_save(
    model: _Toy, inputs: torch.Tensor, tmp_path: Path
) -> None:
    """The portable save door refuses user code; audit saves retain the disclosed placeholder."""

    live = tl.trace(model, inputs, intervene=_callable_spec())
    with pytest.raises(TorchLensError) as caught:
        live.save(tmp_path / "portable_callable", level="portable")
    assert _error_code(caught.value) == "intervention_spec_not_portable"
