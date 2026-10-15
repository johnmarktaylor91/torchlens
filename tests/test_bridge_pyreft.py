"""``tl.bridge.pyreft.read``: a pyreft LoReFT artifact applied as a TorchLens subspace edit.

The artifact here is a stand-in written by the test in pyreft's file layout (``config.json``
plus ``intkey_<key>.bin`` holding ``weight``, ``bias`` and ``rotate_layer``, with the
float32 rotation and bfloat16 source pyreft saves); nothing is downloaded and pyreft is not
imported. The reader's tensors and provenance must equal what the test wrote, the applied
edit must equal the hand LoReFT formula in float64, and a malformed artifact refuses with a
code whose remedy runs.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.selection_subspace import _basis_digest

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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _float64_sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.double().contiguous().numpy().tobytes()).hexdigest()


@pytest.mark.parametrize("underscored", [False, True], ids=["dotted", "underscored"])
def test_reader_returns_what_the_artifact_holds(tmp_path: Path, underscored: bool) -> None:
    directory = _write(tmp_path / "artifact", underscored=underscored)
    artifact = tl.bridge.pyreft.read(directory, repo="example/loreft", revision="abc123")
    assert torch.equal(artifact.rotation, _ROTATION) and artifact.rotation.dtype == torch.float32
    assert torch.equal(artifact.weight, _WEIGHT) and artifact.weight.dtype == torch.bfloat16
    assert torch.equal(artifact.bias, _BIAS) and artifact.bias.dtype == torch.bfloat16
    assert (artifact.layer, artifact.component, artifact.unit) == (8, "block_output", "pos")
    assert artifact.units_per_example == 1 and artifact.rank == _RANK
    provenance = artifact.provenance
    weight_name = f"intkey_{_KEY.replace('.', '_') if underscored else _KEY}.bin"
    assert provenance["weight_file"] == weight_name
    assert provenance["repo"] == "example/loreft" and provenance["revision"] == "abc123"
    assert provenance["files_sha256"] == {
        "config.json": _sha(directory / "config.json"),
        weight_name: _sha(directory / weight_name),
    }
    assert provenance["source_dtype"] == {
        "rotation": "float32",
        "weight": "bfloat16",
        "bias": "bfloat16",
    }
    assert provenance["param_sha256"] == {
        "R": _float64_sha(_ROTATION),
        "W": _float64_sha(_WEIGHT),
        "b": _float64_sha(_BIAS),
    }
    assert "example/loreft@abc123" in artifact.origin


def _ready() -> Any:
    return tl.trace(_MODEL, _X, capture=tl.options.CaptureOptions(intervention_ready=True))


def _positions() -> torch.Tensor:
    positions = torch.zeros(2, 5, dtype=torch.bool)
    positions[:, _POSITION] = True
    return positions


def test_applied_edit_equals_the_hand_loreft_formula(tmp_path: Path) -> None:
    """``h + Q^T (W h + b - Q h)`` with ``Q`` the float64 orthonormalised ``R^T``."""

    artifact = tl.bridge.pyreft.read(_write(tmp_path / "artifact"))
    fork = _ready().fork()
    fork.do(artifact.subspace(_SITE, positions=_positions()), artifact.edit)
    result = fork.push()[_SITE].out
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


def _reloaded_helper(fork: Any, tmp_path: Path) -> Any:
    """Return the staged edit as a portable save reloads it."""

    path = tmp_path / "loreft.tlspec"
    tl.io.save_intervention(fork, path, level="portable")
    return tl.io.load_intervention_spec(path).hook_specs[0].helper


def test_one_position_edit_saves_portable_and_reapplies_bit_identically(tmp_path: Path) -> None:
    artifact = tl.bridge.pyreft.read(_write(tmp_path / "artifact"))
    fork = _ready().fork()
    fork.do(artifact.subspace(_SITE, positions=_positions()), artifact.edit)
    staged = fork.push()
    path = tmp_path / "one_position.tlspec"
    tl.io.save_intervention(fork, path, level="portable")
    spec = tl.io.load_intervention_spec(path)
    fresh = _ready().fork()
    for hook_spec in spec.hook_specs:
        fresh.attach_hooks(hook_spec.site_target, hook_spec.helper, confirm_mutation=True)
    assert torch.equal(fresh.push()[_SITE].out, staged[_SITE].out)


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


def test_bridge_module_imports_no_peer() -> None:
    """Resolving ``tl.bridge.pyreft`` imports neither pyreft nor pyvene."""

    assert tl.bridge.pyreft.read is not None
    assert "pyreft" not in sys.modules and "pyvene" not in sys.modules
