"""``Recipe.save(path)`` writes a ``kind: "recipe"`` artifact that ``tl.load`` returns as a Recipe.

The artifact holds the authored rules alone: no trace, model or fire records.
A rule whose WHERE term is a predicate has no portable form and refuses below
the audit level; an audit record is not loadable and says so typed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from example_models import TinyLinearRelu

import torchlens as tl
from torchlens.intervention.errors import OpaqueCallableError, ReplayPreconditionError


def _recipe() -> tl.Recipe:
    """Two structural rules, one carrying a tensor argument."""

    direction = torch.arange(4, dtype=torch.float32)
    return tl.when(tl.func("relu"), tl.scale(0.5)) & tl.when(
        tl.in_module("lin"), tl.steer(direction, 2.0, dim=1)
    )


def _relu_out(trace: tl.Trace) -> torch.Tensor:
    """Return the relu op output of a capture."""

    return next(trace[label].out for label in trace.layer_labels if "relu" in label)


def test_recipe_round_trips_through_tlspec(tmp_path: Path) -> None:
    recipe = _recipe()
    path = tmp_path / "recipe.tlspec"
    recipe.save(path)

    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "recipe"
    spec_json = json.loads((path / "spec.json").read_text(encoding="utf-8"))
    assert spec_json["executable"] is True
    assert len(spec_json["recipe"]["rules"]) == 2
    assert len(manifest["tensor_entries"]) == 1, "the steer direction travels as a sidecar"

    loaded = tl.load(path)
    assert isinstance(loaded, tl.Recipe)
    assert loaded.digest == recipe.digest, "rebuilt selectors and edits keep every rule identity"
    assert [rule.where_repr for rule in loaded.rules] == [rule.where_repr for rule in recipe.rules]

    torch.manual_seed(0)
    model = TinyLinearRelu()
    x = torch.randn(2, 4)
    expected = _relu_out(tl.trace(model, x, intervene=recipe))
    actual = _relu_out(tl.trace(model, x, intervene=loaded))
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, _relu_out(tl.trace(model, x)))


def test_recipe_save_refuses_an_existing_path_unless_overwrite(tmp_path: Path) -> None:
    recipe = _recipe()
    path = tmp_path / "recipe.tlspec"
    recipe.save(path)
    with pytest.raises(FileExistsError):
        recipe.save(path)
    recipe.save(path, overwrite=True)
    assert isinstance(tl.load(path), tl.Recipe)


def test_predicate_rule_is_not_portable_and_audit_record_is_not_loadable(tmp_path: Path) -> None:
    recipe = tl.when(lambda ctx: True, tl.scale(0.0))
    with pytest.raises(OpaqueCallableError):
        recipe.save(tmp_path / "portable.tlspec")
    assert not (tmp_path / "portable.tlspec").exists(), "a refused save leaves nothing behind"

    audit_path = tmp_path / "audit.tlspec"
    recipe.save(audit_path, level="audit")
    spec_json = json.loads((audit_path / "spec.json").read_text(encoding="utf-8"))
    assert spec_json["executable"] is False
    with pytest.raises(ReplayPreconditionError) as excinfo:
        tl.load(audit_path)
    assert excinfo.value.fields["code"] == "recipe_audit_only"


@pytest.mark.parametrize(
    ("file_name", "edit"),
    [
        pytest.param("spec.json", lambda data: data.pop("recipe"), id="no_rules_list"),
        pytest.param(
            "spec.json", lambda data: data["recipe"]["rules"].append("not a rule"), id="bad_rule"
        ),
        pytest.param(
            "manifest.json", lambda data: data.update(tensor_entries={}), id="entries_not_list"
        ),
    ],
)
def test_hand_edited_recipe_record_refuses_typed(
    tmp_path: Path, file_name: str, edit: Callable[[dict[str, Any]], Any]
) -> None:
    """A malformed recipe record refuses with ``recipe_record_malformed`` and names the file."""

    path = tmp_path / "recipe.tlspec"
    _recipe().save(path)
    target = path / file_name
    data = json.loads(target.read_text(encoding="utf-8"))
    edit(data)
    target.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayPreconditionError) as excinfo:
        tl.load(path)
    assert excinfo.value.fields["code"] == "recipe_record_malformed"
    assert excinfo.value.fields["remedy"]
    assert file_name in str(excinfo.value)
