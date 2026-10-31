"""A learned rotation travels as a portable spec: saved, reloaded, applied bit-identically.

``do(tl.subspace(site, R[:k], origin=...), verb)`` with a builtin verb stages a builtin
``subspace_edit`` that stores the verb, the orthonormal rows, the bound axis, the rows'
digest and the basis provenance. A portable save reloads it as the same projection
(never the plain verb over the whole site), and a malformed, tampered or non-orthonormal
stored basis refuses at load with a code whose remedy runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens._selection_subspace import _basis_digest
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


@pytest.mark.parametrize("rank", [1, 4])
@pytest.mark.parametrize("verb", ["scale", "zero_ablate"])
def test_saved_rotation_reapplies_bit_identically(tmp_path: Path, rank: int, verb: str) -> None:
    edit = tl.scale(2.0) if verb == "scale" else tl.zero_ablate()
    session = _edited(_ROTATION[:rank], edit)
    loaded = _save_load(session, tmp_path / "rotation.tlspec")
    helper = loaded.hook_specs[0].helper
    assert helper.helper_name == "subspace_edit" and helper.portability == "builtin"
    assert helper.args[0].helper_name == verb
    reapplied = _apply(loaded)
    assert torch.equal(reapplied[_SITE].out, session[_SITE].out)
    assert torch.equal(_out(reapplied), _out(session))


@pytest.mark.parametrize("rank", [1, 4])
def test_reloaded_interchange_equals_the_hand_das_interchange(tmp_path: Path, rank: int) -> None:
    """``replace_with(source value)`` under the rotation is the portable DAS interchange."""

    rows = _ROTATION[:rank]
    source_value = _ready(_X_SOURCE)[_SITE].out.detach()
    session = _edited(rows, tl.replace_with(source_value))
    reapplied = _apply(_save_load(session, tmp_path / "interchange.tlspec"))
    with torch.no_grad():
        base = _MODEL.lin1(_X)
        hand = base + (source_value @ rows.T - base @ rows.T) @ rows
    assert torch.allclose(reapplied[_SITE].out, hand, rtol=0.0, atol=1e-12)
    assert torch.equal(reapplied[_SITE].out, session[_SITE].out)
    # The same interchange through patch_from in the session (not portable: opaque_audit).
    source = _ready(_X_SOURCE)
    patched = _edited(rows, tl.patch_from(source))
    assert torch.allclose(patched[_SITE].out, reapplied[_SITE].out, rtol=0.0, atol=1e-12)


def test_reloaded_edit_keeps_the_complement_and_the_basis_provenance(tmp_path: Path) -> None:
    rows = _ROTATION[:2]
    session = _edited(rows, tl.scale(3.0), method="das")
    loaded = _save_load(session, tmp_path / "provenance.tlspec")
    kwargs = dict(loaded.hook_specs[0].helper.kwargs)
    assert kwargs["origin"] == _ORIGIN and kwargs["method"] == "das"
    assert kwargs["axis"] == -1
    assert (
        kwargs["origin_sha256"]
        == tl.subspace(_SITE, rows, origin=_ORIGIN)
        .resolve(_ready())[0]
        .provenance.source.split("basis_sha256=")[1][:64]
    )
    assert kwargs["basis_sha256"] == _basis_digest(rows.contiguous())
    reapplied = _apply(loaded)
    base = _ready()[_SITE].out
    complement = torch.eye(6, dtype=torch.float64) - rows.T @ rows
    delta = (reapplied[_SITE].out - base).detach()
    assert torch.allclose(delta @ complement, torch.zeros_like(delta), atol=1e-12)
    assert not torch.allclose(reapplied[_SITE].out, base * 3.0)
    fired = [record.helper for record in reapplied[_SITE].interventions]
    assert any(dict(helper.kwargs).get("origin") == _ORIGIN for helper in fired)


def test_float32_rotation_round_trips_its_source_dtype(tmp_path: Path) -> None:
    """A non-default value (a float32 learned rotation, orthonormalised) round-trips."""

    learned = (_ROTATION[:3] * torch.tensor([[2.0], [1.0], [0.5]], dtype=torch.float64)).float()
    session = _edited(learned, tl.scale(0.5))
    loaded = _save_load(session, tmp_path / "float32.tlspec")
    kwargs = dict(loaded.hook_specs[0].helper.kwargs)
    assert kwargs["source_dtype"] == "float32"
    assert kwargs["orthonormalised"] is True
    assert kwargs["basis"].dtype == torch.float64
    assert torch.equal(_apply(loaded)[_SITE].out, session[_SITE].out)


def _stored(rows: torch.Tensor) -> dict[str, Any]:
    return {
        "basis": rows,
        "axis": -1,
        "basis_sha256": _basis_digest(rows.contiguous()),
        "origin": _ORIGIN,
    }


def test_older_payload_without_the_optional_fields_default_fills() -> None:
    """Only the required fields: method, origin digest, dtype and flag default-fill."""

    rows = _ROTATION[:2].contiguous()
    rebuilt = rebuild_builtin_helper("subspace_edit", (tl.scale(2.0),), _stored(rows))
    kwargs = dict(rebuilt.kwargs)
    assert kwargs["method"] is None
    assert kwargs["origin_sha256"] == kwargs["basis_sha256"]
    assert kwargs["source_dtype"] == "float64"
    assert kwargs["orthonormalised"] is False


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


def test_tampered_basis_refuses_at_load(tmp_path: Path) -> None:
    session = _edited(_ROTATION[:2], tl.scale(2.0))
    path = tmp_path / "tampered.tlspec"
    tl.io.save_intervention(session, path, level="portable")
    spec_json = path / "spec.json"
    text = spec_json.read_text(encoding="utf-8")
    digest = _basis_digest(_ROTATION[:2].contiguous())
    assert digest in text
    spec_json.write_text(text.replace(digest, "0" * 64), encoding="utf-8")
    with pytest.raises(Exception) as excinfo:
        tl.io.load_intervention_spec(path)
    error = _coded(excinfo.value)
    assert error.fields["code"] == "subspace_spec_basis_tampered"
    _remedy_runs(error, tmp_path)


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


def test_saved_spec_json_carries_the_fields(tmp_path: Path) -> None:
    path = tmp_path / "fields.tlspec"
    tl.io.save_intervention(_edited(_ROTATION[:4], tl.scale(2.0)), path, level="portable")
    text = json.dumps(json.loads((path / "spec.json").read_text(encoding="utf-8")))
    for field in ("subspace_edit", "basis_sha256", "origin_sha256", "source_dtype", _ORIGIN):
        assert field in text
