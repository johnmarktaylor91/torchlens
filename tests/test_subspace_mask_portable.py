"""A subspace edit restricted to some positions saves portably and reloads as the same edit.

``do(tl.subspace(tl.units(site, mask), basis, origin=...), verb)`` with a builtin verb stores
its ``within=`` element mask and the mask's sha256 in the ``subspace_edit`` payload, so a
one-position edit (LoReFT acts on one position) saves at ``level="portable"`` and reapplies
bit-identically. A malformed, tampered or partial stored mask, or a stored keyword the loader
does not know, refuses at load with a code whose remedy runs. Every other masked edit keeps
its portability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.intervention._subspace_projection import _mask_digest
from torchlens.intervention.helpers import rebuild_builtin_helper
from torchlens.selection_subspace import _basis_digest

_SITE = "linear_1_1"
_ORIGIN = "LoReFT trained elsewhere"
_D = 6


class _SeqNet(nn.Module):
    """``lin2(lin1(x))`` over ``[batch, seq, 6]`` in float64."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin1 = nn.Linear(_D, _D).double()
        self.lin2 = nn.Linear(_D, 3).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.lin1(x))


_MODEL = _SeqNet().eval()
_X = torch.randn(2, 5, _D, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
_R = torch.linalg.qr(
    torch.randn(_D, _D, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
)[0]
_W = torch.randn(4, _D, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
_B = torch.randn(4, generator=torch.Generator().manual_seed(5), dtype=torch.float64)


def _ready() -> Any:
    return tl.trace(_MODEL, _X, capture=tl.options.CaptureOptions(intervention_ready=True))


def _mask(position: int = 3) -> torch.Tensor:
    mask = torch.zeros(2, 5, _D, dtype=torch.bool)
    mask[:, position, :] = True
    return mask


def _session(edit: Any, rank: int = 4) -> Any:
    fork = _ready().fork()
    target = tl.subspace(tl.units(_SITE, _mask()), _R[:, :rank].T, origin=_ORIGIN, method="loreft")
    fork.do(target, edit)
    return fork


def _save_load(fork: Any, path: Path) -> Any:
    tl.io.save_intervention(fork, path, level="portable")
    return tl.io.load_intervention_spec(path)


def _apply(spec: Any) -> Any:
    fork = _ready().fork()
    for hook_spec in spec.hook_specs:
        fork.attach_hooks(hook_spec.site_target, hook_spec.helper, confirm_mutation=True)
    return fork.push()


_VERBS = {
    "affine_source": lambda: tl.intervention.set_affine_coordinates(_W, bias=_B),
    "scale": lambda: tl.scale(2.0),
}


@pytest.mark.parametrize("verb", sorted(_VERBS))
def test_one_position_subspace_edit_saves_portable_and_reapplies_bit_identically(
    tmp_path: Path, verb: str
) -> None:
    session = _session(_VERBS[verb]())
    staged = session.push()
    loaded = _save_load(session, tmp_path / "one_position.tlspec")
    helper = loaded.hook_specs[0].helper
    assert helper.helper_name == "subspace_edit" and helper.portability == "builtin"
    kwargs = dict(helper.kwargs)
    assert torch.equal(kwargs["mask"], _mask())
    assert kwargs["mask_sha256"] == _mask_digest(_mask())
    reapplied = _apply(loaded)
    assert torch.equal(reapplied[_SITE].out, staged[_SITE].out)
    assert torch.equal(
        reapplied[reapplied.output_layers[0]].out, staged[staged.output_layers[0]].out
    )
    untouched = ~_mask()
    assert torch.equal(reapplied[_SITE].out[untouched], _ready()[_SITE].out[untouched])


def test_reloaded_masked_edit_under_another_mask_stores_the_intersection(tmp_path: Path) -> None:
    """The session scatters through both masks; the saved edit stores their intersection."""

    session = _session(tl.intervention.set_affine_coordinates(_W, bias=_B))
    staged = session.push()
    loaded = _save_load(session, tmp_path / "first.tlspec").hook_specs[0].helper
    wider = _mask(3) | _mask(4)
    fork = _ready().fork()
    fork.do(tl.units(_SITE, wider), loaded)
    twice = fork.push()
    assert torch.equal(twice[_SITE].out, staged[_SITE].out)
    again = _save_load(fork, tmp_path / "second.tlspec")
    kwargs = dict(again.hook_specs[0].helper.kwargs)
    assert torch.equal(kwargs["mask"], _mask(3))
    assert [key for key, _ in again.hook_specs[0].helper.kwargs].count("mask") == 1
    assert torch.equal(_apply(again)[_SITE].out, staged[_SITE].out)


def test_other_masked_edits_keep_their_portability(tmp_path: Path) -> None:
    """A masked plain verb (no subspace) stays opaque_audit, exactly as before."""

    fork = _ready().fork()
    fork.do(tl.units(_SITE, _mask()), tl.scale(2.0))
    tl.io.save_intervention(fork, tmp_path / "plain.tlspec", level="audit")
    loaded = tl.io.load_intervention_spec(tmp_path / "plain.tlspec")
    saved = loaded.hook_specs[0].helper
    assert saved.portability == "opaque_audit" and saved.factory is None


def _stored(mask: Any, digest: Any) -> dict[str, Any]:
    rows = _R[:, :4].T.contiguous()
    return {
        "basis": rows,
        "axis": -1,
        "basis_sha256": _basis_digest(rows),
        "origin": _ORIGIN,
        "mask": mask,
        "mask_sha256": digest,
    }


def test_payload_without_a_mask_default_fills_to_the_whole_site() -> None:
    stored = _stored(None, None)
    rebuilt = rebuild_builtin_helper("subspace_edit", (tl.scale(2.0),), stored)
    assert "mask" not in dict(rebuilt.kwargs)
    without = {key: value for key, value in stored.items() if not key.startswith("mask")}
    older = rebuild_builtin_helper("subspace_edit", (tl.scale(2.0),), without)
    assert [key for key, _ in older.kwargs] == [key for key, _ in rebuilt.kwargs]


def _partial() -> torch.Tensor:
    mask = _mask()
    mask[0, 3, 0] = False
    return mask


_MALFORMED = {
    "not-bool": ("mask", lambda: _mask().to(torch.uint8), lambda m: _mask_digest(_mask())),
    "wrong-width": ("mask", lambda: torch.ones(2, 5, 4, dtype=torch.bool), _mask_digest),
    "partial-support": ("mask", _partial, _mask_digest),
    "digest-missing": ("mask_sha256", _mask, lambda m: None),
    "digest-without-mask": ("mask_sha256", lambda: None, lambda m: "0" * 64),
}


def _remedy_runs(error: Any, tmp_path: Path) -> None:
    """Run the refusal's spelling: re-save from the session, then load and apply."""

    assert "tl.io.save_intervention(fork, path, level='portable')" in error.fields["remedy"]
    session = _session(tl.intervention.set_affine_coordinates(_W, bias=_B))
    staged = session.push()
    assert torch.equal(
        _apply(_save_load(session, tmp_path / "resaved.tlspec"))[_SITE].out, staged[_SITE].out
    )


@pytest.mark.parametrize("case", sorted(_MALFORMED))
def test_malformed_mask_refuses_at_load_and_the_remedy_runs(tmp_path: Path, case: str) -> None:
    field, mask, digest = _MALFORMED[case]
    value = mask()
    with pytest.raises(InvalidArgumentError) as excinfo:
        rebuild_builtin_helper("subspace_edit", (tl.scale(2.0),), _stored(value, digest(value)))
    assert excinfo.value.fields["code"] == "subspace_spec_invalid"
    assert excinfo.value.fields["argument"] == field
    _remedy_runs(excinfo.value, tmp_path)


def test_tampered_mask_refuses_at_load_and_the_remedy_runs(tmp_path: Path) -> None:
    session = _session(tl.intervention.set_affine_coordinates(_W, bias=_B))
    path = tmp_path / "tampered.tlspec"
    tl.io.save_intervention(session, path, level="portable")
    spec_json = path / "spec.json"
    text = spec_json.read_text(encoding="utf-8")
    digest = _mask_digest(_mask())
    assert digest in text
    spec_json.write_text(text.replace(digest, _mask_digest(_mask(1))), encoding="utf-8")
    with pytest.raises(Exception) as excinfo:
        tl.io.load_intervention_spec(path)
    error = _coded(excinfo.value)
    assert error.fields["code"] == "subspace_spec_mask_tampered"
    _remedy_runs(error, tmp_path)


def test_unknown_stored_keyword_refuses_typed(tmp_path: Path) -> None:
    """A keyword this loader does not know never loads with its meaning dropped."""

    stored = _stored(_mask(), _mask_digest(_mask()))
    stored["positions_rule"] = "last"
    with pytest.raises(InvalidArgumentError) as excinfo:
        rebuild_builtin_helper("subspace_edit", (tl.scale(2.0),), stored)
    assert excinfo.value.fields["code"] == "subspace_spec_invalid"
    assert excinfo.value.fields["argument"] == "positions_rule"
    _remedy_runs(excinfo.value, tmp_path)


def _coded(error: BaseException) -> Any:
    current: BaseException | None = error
    while current is not None:
        if getattr(current, "fields", {}).get("code"):
            return current
        current = current.__cause__ or current.__context__
    raise AssertionError(f"no coded error in the chain of {error!r}")
