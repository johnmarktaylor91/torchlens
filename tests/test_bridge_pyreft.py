"""``tl.bridge.pyreft.read``: a pyreft LoReFT artifact applied as a TorchLens subspace edit.

The artifact here is a stand-in written by the test in pyreft's file layout (``config.json``
plus ``intkey_<key>.bin`` holding ``weight``, ``bias`` and ``rotate_layer``, with the
float32 rotation and bfloat16 source pyreft saves); nothing is downloaded and pyreft is not
imported. The reader's tensors and provenance must equal what the test wrote, the applied
edit must equal the hand LoReFT formula in float64, and a malformed artifact refuses with a
code whose remedy runs.
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
from torchlens._selection.subspace import _basis_digest

_SITE = "linear_1_1"
_D = 6
_RANK = 4
_KEY = "layer.8.comp.block_output.unit.pos.nunit.1#0"
_POSITION = 3


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
# A float32 rotation, as pyreft trains it: orthonormal only to about 1e-7.
_ROTATION = torch.linalg.qr(
    torch.randn(_D, _RANK, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
)[0].float()
_WEIGHT = torch.randn(_RANK, _D, generator=torch.Generator().manual_seed(4)).to(torch.bfloat16)
_BIAS = torch.randn(_RANK, generator=torch.Generator().manual_seed(5)).to(torch.bfloat16)


def _config(**changes: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "intervention_constant_sources": [True],
        "intervention_dimensions": [_D],
        "intervention_types": ["<class 'pyreft.interventions.LoreftIntervention'>"],
        "mode": "parallel",
        "representations": [[8, "block_output", "pos", 1, _RANK] + [None] * 9],
        "sorted_keys": [_KEY],
        "transformers_version": "4.45.2",
    }
    config.update(changes)
    return config


def _write(directory: Path, *, underscored: bool = False, **changes: Any) -> Path:
    """Write a stand-in pyreft artifact; ``changes`` overrides config or tensors."""

    directory.mkdir(parents=True, exist_ok=True)
    tensors = {"weight": _WEIGHT, "bias": _BIAS, "rotate_layer": _ROTATION}
    tensors.update(changes.pop("tensors", {}))
    config = _config(**changes)
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    name = _KEY.replace(".", "_") if underscored else _KEY
    torch.save(tensors, directory / f"intkey_{name}.bin")
    return directory


def _ready() -> Any:
    return tl.trace(_MODEL, _X, capture=tl.options.CaptureOptions(replayable=True))


def _positions() -> torch.Tensor:
    positions = torch.zeros(2, 5, dtype=torch.bool)
    positions[:, _POSITION] = True
    return positions


def _remedy_runs(error: Any, tmp_path: Path) -> None:
    """Run the refusal's spelling: read the directory pyreft saved."""

    assert "tl.bridge.pyreft.read(directory)" in error.fields["remedy"]
    directory = _write(tmp_path / "remedy")
    assert tl.bridge.pyreft.read(directory).rank == _RANK


def _two_interventions(directory: Path) -> None:
    _write(directory, sorted_keys=[_KEY, _KEY + "x"])


def _not_loreft(directory: Path) -> None:
    _write(directory, intervention_types=["<class 'pyreft.interventions.NoreftIntervention'>"])


def _no_config(directory: Path) -> None:
    _write(directory)
    (directory / "config.json").unlink()


def _bad_json(directory: Path) -> None:
    _write(directory)
    (directory / "config.json").write_text("{not json", encoding="utf-8")


def _no_weight_file(directory: Path) -> None:
    _write(directory)
    (directory / f"intkey_{_KEY}.bin").unlink()


def _wrong_keys(directory: Path) -> None:
    _write(directory, tensors={"rotate_layer.original": _ROTATION})


def _wrong_shape(directory: Path) -> None:
    _write(directory, tensors={"weight": _WEIGHT[:2]})


def _not_a_weight_file(directory: Path) -> None:
    _write(directory)
    (directory / f"intkey_{_KEY}.bin").write_bytes(b"not a torch file")


def _not_orthonormal(directory: Path) -> None:
    _write(directory, tensors={"rotate_layer": _ROTATION * 1.5})


def _key_outside_the_directory(directory: Path) -> None:
    _write(directory, sorted_keys=["../" + _KEY])


def _another_class(directory: Path) -> None:
    _write(directory, intervention_types=["<class 'mine.MyLoreftIntervention'>"])


_MALFORMED = {
    "not-orthonormal": (_not_orthonormal, "pyreft_artifact_invalid"),
    "key-outside-the-directory": (_key_outside_the_directory, "pyreft_artifact_invalid"),
    "another-class": (_another_class, "pyreft_artifact_unsupported"),
    "two-interventions": (_two_interventions, "pyreft_artifact_unsupported"),
    "not-loreft": (_not_loreft, "pyreft_artifact_unsupported"),
    "no-config": (_no_config, "pyreft_artifact_invalid"),
    "bad-json": (_bad_json, "pyreft_artifact_invalid"),
    "no-weight-file": (_no_weight_file, "pyreft_artifact_invalid"),
    "wrong-keys": (_wrong_keys, "pyreft_artifact_invalid"),
    "wrong-shape": (_wrong_shape, "pyreft_artifact_invalid"),
    "not-a-weight-file": (_not_a_weight_file, "pyreft_artifact_invalid"),
}


def _reloaded_helper(fork: Any, tmp_path: Path) -> Any:
    """Return the staged edit as a portable save reloads it."""

    path = tmp_path / "loreft.tlspec"
    tl.io.save_intervention(fork, path, level="portable")
    return tl.io.load_intervention_spec(path).hook_specs[0].helper


@pytest.mark.parametrize("case", sorted(_MALFORMED))
def test_malformed_artifact_refuses_and_the_remedy_runs(tmp_path: Path, case: str) -> None:
    build, code = _MALFORMED[case]
    directory = tmp_path / "artifact"
    build(directory)
    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.bridge.pyreft.read(directory)
    assert excinfo.value.fields["code"] == code
    _remedy_runs(excinfo.value, tmp_path)


def test_positions_that_are_not_a_bool_mask_refuse(tmp_path: Path) -> None:
    artifact = tl.bridge.pyreft.read(_write(tmp_path / "artifact"))
    with pytest.raises(InvalidArgumentError) as excinfo:
        artifact.subspace(_SITE, positions=torch.zeros(2, 5))
    assert excinfo.value.fields["code"] == "pyreft_positions_invalid"
    positions = torch.zeros(2, 5, dtype=torch.bool)
    positions[:, -1] = True
    assert artifact.subspace(_SITE, positions=positions) is not None


def test_applied_edit_equals_the_hand_loreft_formula(tmp_path: Path) -> None:
    """``h + Q^T (W h + b - Q h)`` with ``Q`` the float64 orthonormalised ``R^T``."""

    artifact = tl.bridge.pyreft.read(_write(tmp_path / "artifact"))
    fork = _ready().fork()
    fork.do(artifact.subspace(_SITE, positions=_positions()), artifact.edit)
    result = fork[_SITE].out
    helper = _reloaded_helper(fork, tmp_path)
    kwargs = dict(helper.kwargs)
    rows = kwargs["basis"]
    assert kwargs["orthonormalised"] is True and kwargs["source_dtype"] == "float32"
    assert kwargs["origin_sha256"] == _basis_digest(_ROTATION.T.double().contiguous())
    assert kwargs["basis_sha256"] == _basis_digest(rows)
    assert kwargs["method"] == "loreft"
    weight, bias = _WEIGHT.double(), _BIAS.double()
    with torch.no_grad():
        h = _MODEL.lin1(_X)
        hand = h + (h @ weight.T + bias - h @ rows.T) @ rows
    mask = _positions().unsqueeze(-1).expand(2, 5, _D)
    assert torch.allclose(result, torch.where(mask, hand, h), rtol=0.0, atol=1e-12)
    assert torch.equal(result[~mask], h[~mask])
    # Orthonormalising the float32 rotation moves the rows by float32 rounding only.
    assert float((rows - _ROTATION.T.double()).abs().max()) < 1e-6
