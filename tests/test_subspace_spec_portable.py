"""A learned rotation travels as a portable spec: saved, reloaded, applied bit-identically.

``do(tl.subspace(site, R[:k], origin=...), verb)`` with a builtin verb stages a builtin
``subspace_edit`` that stores the verb, the orthonormal rows, the bound axis, the rows'
digest and the basis provenance. A portable save reloads it as the same projection
(never the plain verb over the whole site), and a malformed, tampered or non-orthonormal
stored basis refuses at load with a code whose remedy runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens._selection.subspace import _basis_digest
from torchlens.intervention import helpers as helpers_module
from torchlens.intervention.helpers import rebuild_builtin_helper

_SITE = "linear_1_1"
_ORIGIN = "rotation trained elsewhere"


class _Net(nn.Module):
    """``lin2(lin1(x))`` in float64: the rotation acts on ``lin1``'s output."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin1 = nn.Linear(6, 6).double()
        self.lin2 = nn.Linear(6, 3).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.lin1(x))


_MODEL = _Net().eval()
_X = torch.randn(3, 6, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
_X_SOURCE = torch.randn(3, 6, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
_ROTATION = torch.linalg.qr(
    torch.randn(6, 6, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
)[0].T.contiguous()


def _ready(x: torch.Tensor = _X) -> Any:
    return tl.trace(_MODEL, x, capture=tl.options.CaptureOptions(replayable=True))


def _edited(rows: torch.Tensor, edit: Any, **subspace_kwargs: Any) -> Any:
    fork = _ready().fork()
    fork.do(tl.subspace(_SITE, rows, origin=_ORIGIN, **subspace_kwargs), edit)
    return fork


def _save_load(fork: Any, path: Path) -> Any:
    tl.io.save_intervention(fork, path, level="portable")
    return tl.io.load_intervention_spec(path)


def _apply(spec: Any) -> Any:
    """Apply a loaded spec on a fresh capture of the same model and input."""

    fork = _ready().fork()
    for hook_spec in spec.hook_specs:
        fork.attach_hooks(hook_spec.site_target, hook_spec.helper, confirm_mutation=True)
    return fork.push()


def _out(trace: Any) -> torch.Tensor:
    return trace[trace.output_layers[0]].out


def _stored(rows: torch.Tensor) -> dict[str, Any]:
    return {
        "basis": rows,
        "axis": -1,
        "basis_sha256": _basis_digest(rows.contiguous()),
        "origin": _ORIGIN,
    }


def test_previous_opaque_projection_payload_still_loads_as_a_placeholder(tmp_path: Path) -> None:
    """A spec written before this release (an opaque projected verb) never rebuilds as the verb."""

    source = _ready(_X_SOURCE)
    session = _edited(_ROTATION[:2], tl.patch_from(source))
    tl.io.save_intervention(session, tmp_path / "opaque.tlspec", level="audit")
    loaded = tl.io.load_intervention_spec(tmp_path / "opaque.tlspec")
    helper = loaded.hook_specs[0].helper
    assert helper.portability == "opaque_audit" and helper.factory is None


_MALFORMED = {
    "axis": ("axis", "-1"),
    "origin": ("origin", ""),
    "method": ("method", 3),
    "flag": ("orthonormalised", "yes"),
    "float32-basis": ("basis", None),
    "sparse-basis": ("basis", None),
}


@pytest.mark.parametrize("case", sorted(_MALFORMED))
def test_malformed_field_refuses_and_the_remedy_runs(tmp_path: Path, case: str) -> None:
    rows = _ROTATION[:2].contiguous()
    stored = _stored(rows)
    field, value = _MALFORMED[case]
    replaced = {"float32-basis": rows.float(), "sparse-basis": rows.to_sparse()}
    stored[field] = replaced.get(case, value)
    with pytest.raises(InvalidArgumentError) as excinfo:
        rebuild_builtin_helper("subspace_edit", (tl.scale(2.0),), stored)
    assert excinfo.value.fields["code"] == "subspace_spec_invalid"
    assert excinfo.value.fields["argument"] == field
    _remedy_runs(excinfo.value, tmp_path)


def _remedy_runs(error: InvalidArgumentError, tmp_path: Path) -> None:
    """Run the refusal's spelling: re-save from the session, then load and apply."""

    assert "tl.io.save_intervention(fork, path, level='portable')" in error.fields["remedy"]
    fork = _ready().fork()
    fork.do(tl.subspace(_SITE, _ROTATION[:2], origin=_ORIGIN), tl.scale(2.0))
    path = tmp_path / "resaved.tlspec"
    tl.io.save_intervention(fork, path, level="portable")
    assert torch.equal(_apply(tl.io.load_intervention_spec(path))[_SITE].out, fork[_SITE].out)


def test_changed_rows_refuse_as_tampered_and_non_orthonormal_rows_refuse() -> None:
    rows = _ROTATION[:2].contiguous()
    stored = _stored(rows)
    stored["basis"] = rows * 1.0001
    with pytest.raises(InvalidArgumentError) as excinfo:
        rebuild_builtin_helper("subspace_edit", (tl.scale(2.0),), stored)
    assert excinfo.value.fields["code"] == "subspace_spec_basis_tampered"
    skewed = rows.clone()
    skewed[1] = rows[1] + 0.5 * rows[0]
    with pytest.raises(InvalidArgumentError) as excinfo:
        rebuild_builtin_helper("subspace_edit", (tl.scale(2.0),), _stored(skewed))
    assert excinfo.value.fields["code"] == "subspace_spec_basis_not_orthonormal"


def test_reader_without_the_registry_entry_refuses_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An older reader (no ``subspace_edit`` builtin) refuses ``intervention_helper_unknown``."""

    path = tmp_path / "new.tlspec"
    tl.io.save_intervention(_edited(_ROTATION[:1], tl.scale(2.0)), path, level="portable")
    registry = helpers_module._builtin_constructors

    def _older() -> dict[str, Any]:
        constructors = dict(registry())
        constructors.pop("subspace_edit")
        return constructors

    monkeypatch.setattr(helpers_module, "_builtin_constructors", _older)
    with pytest.raises(Exception) as excinfo:
        tl.io.load_intervention_spec(path)
    assert _coded(excinfo.value).fields["code"] == "intervention_helper_unknown"


def _coded(error: BaseException) -> Any:
    """Return the first error in the cause chain that carries a code."""

    current: BaseException | None = error
    while current is not None:
        if getattr(current, "fields", {}).get("code"):
            return current
        current = current.__cause__ or current.__context__
    raise AssertionError(f"no coded error in the chain of {error!r}")


@pytest.mark.parametrize("rank", [1, 4])
@pytest.mark.parametrize("verb", ["scale", "zero_ablate"])
def test_saved_rotation_reapplies_bit_identically(tmp_path: Path, rank: int, verb: str) -> None:
    edit = tl.scale(2.0) if verb == "scale" else tl.zero_ablate()
    session = _edited(_ROTATION[:rank], edit)
    loaded = _save_load(session, tmp_path / "rotation.tlspec")
    helper = loaded.hook_specs[0].helper
    assert helper.name == "subspace_edit" and helper.portability == "builtin"
    assert helper.args[0].name == verb
    reapplied = _apply(loaded)
    assert torch.equal(reapplied[_SITE].out, session[_SITE].out)
    assert torch.equal(_out(reapplied), _out(session))
