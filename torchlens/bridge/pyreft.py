"""Read a pyreft LoReFT artifact and apply it as a TorchLens subspace edit.

A directory saved by pyreft (``ReftModel.save``) holds ``config.json`` and
one weight file per intervention, ``intkey_<key>.bin``. For a
``LoreftIntervention`` the weight file holds the rotation ``R`` (``[d, k]``,
orthonormal columns), and the learned source ``W`` (``[k, d]``) and ``b``
(``[k]``); the edit is ``h' = h + R (W h + b - R^T h)``. In TorchLens terms
that is::

    artifact = tl.bridge.pyreft.read(directory)
    fork.do(artifact.subspace(site, positions=mask), artifact.edit)

which is ``tl.subspace(tl.units(site, ...), R.T, origin=..., method="loreft")``
with ``tl.intervention.affine_source(W, bias=b)``. :func:`read` never fits
anything and never imports pyreft or pyvene: it reads the files with
``torch.load(weights_only=True)``.

Precision: pyreft trains ``R`` in float32, so its columns are orthonormal only
to about ``1e-7``, above the ``1e-10`` rule a subspace basis is held to. The
basis is therefore orthonormalised once, in float64, by the same sign-keeping
Gram-Schmidt every non-orthonormal basis gets; the saved edit records the
digest of ``R`` as read (``origin_sha256``), the digest of the rows applied
(``basis_sha256``), ``orthonormalised=True`` and ``source_dtype="float32"``.
``provenance`` records the repo and revision the caller names, every file's
sha256, each tensor's dtype as read and the float64 sha256 of ``R``, ``W``
and ``b``.

The weight file name is accepted with the key's dots (older pyvene) or with
underscores (pyvene 0.1.8). A directory that is not one LoReFT intervention
refuses ``pyreft_artifact_invalid`` or ``pyreft_artifact_unsupported``.

DOCUMENTED-UNSTABLE.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch

from .._errors import InvalidArgumentError

__all__ = ["LoreftArtifact", "read"]

_LOREFT = "LoreftIntervention"
#: pyreft trains R through an orthogonal parametrization, so its columns are
#: orthonormal to float32 rounding (about 1e-7); a Gram error above this is
#: not a LoReFT rotation as trained.
_ROTATION_GRAM_LIMIT = 1e-4
_REMEDY = (
    "point read() at the directory pyreft saved with reft_model.save(directory), "
    "which holds config.json and intkey_<key>.bin for one LoreftIntervention: "
    "tl.bridge.pyreft.read(directory)"
)


def _refusal(
    why: str, *, code: str = "pyreft_artifact_invalid", **fields: Any
) -> InvalidArgumentError:
    """Return the refusal for an artifact this reader cannot read.

    ``code`` is spelled at each raise site so the refusal census sees it.
    """

    return InvalidArgumentError(
        f"tl.bridge.pyreft.read: {why}",
        code=code,
        remedy=f"{_REMEDY} (see docs/migration/from_pyvene.md)",
        argument="directory",
        **fields,
    )


def _float64_sha256(tensor: torch.Tensor) -> str:
    """Return the sha256 of a tensor's float64 bytes (row-major, no header)."""

    data = tensor.detach().to(torch.float64).cpu().contiguous()
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()


@dataclasses.dataclass(frozen=True)
class LoreftArtifact:
    """One pyreft LoReFT intervention, as read from its saved directory.

    ``rotation`` (``R``, ``[d, k]``), ``weight`` (``W``, ``[k, d]``) and
    ``bias`` (``b``, ``[k]``) are the tensors exactly as read, in their saved
    dtypes. ``layer``, ``component`` and ``unit`` are the site pyreft names
    (for example layer 8, ``"block_output"``, ``"pos"``);
    ``units_per_example`` is how many positions per example pyreft edits.
    ``provenance`` records the repo, revision, file digests, dtypes and
    parameter digests.
    """

    rotation: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor
    layer: int
    component: str
    unit: str
    units_per_example: int
    provenance: Mapping[str, Any]

    @property
    def rank(self) -> int:
        """Return ``k``, the number of subspace coordinates."""

        return int(self.weight.shape[0])

    @property
    def origin(self) -> str:
        """Return the ``origin=`` text the subspace records."""

        where = self.provenance.get("repo") or "local directory"
        revision = self.provenance.get("revision") or "unrecorded revision"
        weight_sha = str(self.provenance["files_sha256"][self.provenance["weight_file"]])
        return (
            f"pyreft LoReFT {where}@{revision}, intervention {self.provenance['key']}, "
            f"weight file sha256 {weight_sha[:16]}"
        )

    def subspace(self, site: str, *, positions: torch.Tensor | None = None) -> Any:
        """Return the subspace selection the edit acts through at ``site``.

        Parameters
        ----------
        site:
            The TorchLens site label of the value pyreft edits (for
            ``block_output`` of layer ``n``, the output of decoder layer ``n``).
        positions:
            Optional bool mask over the site's axes except the trailing feature
            axis (``[batch, sequence]`` for a transformer block), True where
            pyreft edits (LoReFT's ``unit_locations``). ``None`` edits every
            position.

        Returns
        -------
        Selection
            ``tl.subspace(within, R.T, origin=self.origin, method="loreft")``.
        """

        from ..selection import units
        from ..selection_subspace import subspace

        within: Any = site
        if positions is not None:
            if not isinstance(positions, torch.Tensor) or positions.dtype != torch.bool:
                raise InvalidArgumentError(
                    f"LoreftArtifact.subspace positions= must be a bool tensor, got "
                    f"{type(positions).__name__}",
                    code="pyreft_positions_invalid",
                    remedy=(
                        "pass a bool mask over the site's batch and sequence axes, e.g. "
                        "positions = torch.zeros(batch, seq, dtype=torch.bool); "
                        "positions[:, -1] = True"
                    ),
                    argument="positions",
                )
            width = int(self.rotation.shape[0])
            mask = positions.unsqueeze(-1).expand(*positions.shape, width).contiguous()
            within = units(site, mask)
        return subspace(within, self.rotation.T, origin=self.origin, method="loreft")

    @property
    def edit(self) -> Any:
        """Return ``tl.intervention.affine_source(W, bias=b)`` in float64."""

        from ..intervention._affine_source import affine_source

        return affine_source(self.weight.to(torch.float64), bias=self.bias.to(torch.float64))


def _config(directory: Path) -> tuple[dict[str, Any], str]:
    """Return the parsed ``config.json`` and its sha256, refusing a missing or unreadable one."""

    path = directory / "config.json"
    if not path.is_file():
        raise _refusal(
            f"{directory.name!r} has no config.json",
            code="pyreft_artifact_invalid",
            file="config.json",
        )
    try:
        data = path.read_bytes()
        config = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _refusal(
            f"config.json is not readable JSON ({exc})",
            code="pyreft_artifact_invalid",
            file="config.json",
        ) from exc
    if not isinstance(config, dict):
        raise _refusal(
            "config.json is not a JSON object", code="pyreft_artifact_invalid", file="config.json"
        )
    return config, hashlib.sha256(data).hexdigest()


def _one_loreft(config: Mapping[str, Any]) -> tuple[str, list[Any], int]:
    """Return ``(key, representation, dimension)`` of the one LoReFT intervention."""

    fields = [
        config.get(name)
        for name in (
            "sorted_keys",
            "intervention_types",
            "representations",
            "intervention_dimensions",
        )
    ]
    if not all(isinstance(value, list) for value in fields):
        raise _refusal(
            "config.json lacks the sorted_keys, intervention_types, representations "
            "and intervention_dimensions lists pyreft writes",
            code="pyreft_artifact_invalid",
            file="config.json",
        )
    keys, kinds, representations, dimensions = (cast("list[Any]", value) for value in fields)
    if len(keys) != 1 or not (len(kinds) == len(representations) == len(dimensions) == 1):
        raise _refusal(
            f"the artifact holds {len(keys)} interventions; this reader reads exactly one",
            code="pyreft_artifact_unsupported",
            file="config.json",
        )
    kind = str(kinds[0]).rstrip("'>")
    if kind != _LOREFT and not kind.endswith("." + _LOREFT):
        raise _refusal(
            f"the intervention is {kinds[0]!r}, not a {_LOREFT}",
            code="pyreft_artifact_unsupported",
            file="config.json",
        )
    representation = representations[0]
    shape_ok = isinstance(representation, list) and len(representation) >= 5
    if not shape_ok or not isinstance(keys[0], str) or isinstance(dimensions[0], bool):
        raise _refusal(
            "config.json's representation is not pyreft's [layer, component, unit, "
            "units, rank, ...] list",
            code="pyreft_artifact_invalid",
            file="config.json",
        )
    if not isinstance(dimensions[0], int):
        raise _refusal(
            f"intervention_dimensions holds {dimensions[0]!r}, not an int",
            code="pyreft_artifact_invalid",
            file="config.json",
        )
    return keys[0], representation, int(dimensions[0])


def _weight_file(directory: Path, key: str) -> Path:
    """Return the intervention's weight file, under the dotted or underscored name."""

    names = (f"intkey_{key}.bin", f"intkey_{key.replace('.', '_')}.bin")
    if any(Path(name).name != name or "\\" in name for name in names) or ".." in key:
        raise _refusal(
            f"the intervention key {key!r} is not a plain file-name part",
            code="pyreft_artifact_invalid",
            file="config.json",
        )
    for name in names:
        if (directory / name).is_file():
            return directory / name
    raise _refusal(
        f"no weight file for intervention {key!r} (looked for {names[0]!r} and {names[1]!r})",
        code="pyreft_artifact_invalid",
        file=names[0],
    )


def _tensors(path: Path, data: bytes, rank: int, dimension: int) -> dict[str, torch.Tensor]:
    """Return ``weight``, ``bias`` and ``rotate_layer`` from ``data``, refusing a malformed file."""

    try:
        state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as exc:  # torch raises several unrelated types for a bad file
        raise _refusal(
            f"{path.name!r} is not a torch weight file ({type(exc).__name__}: {exc})",
            code="pyreft_artifact_invalid",
            file=path.name,
        ) from exc
    expected = {"weight": (rank, dimension), "bias": (rank,), "rotate_layer": (dimension, rank)}
    if not isinstance(state, Mapping) or set(state) != set(expected):
        found = sorted(map(str, state)) if isinstance(state, Mapping) else type(state).__name__
        raise _refusal(
            f"{path.name!r} holds {found!r}, not LoReFT's weight, bias and rotate_layer",
            code="pyreft_artifact_invalid",
            file=path.name,
        )
    for name, shape in expected.items():
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            got = tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else type(tensor).__name__
            raise _refusal(
                f"{path.name!r} {name} is {got!r}, expected {shape!r} from config.json",
                code="pyreft_artifact_invalid",
                file=path.name,
            )
        if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
            raise _refusal(
                f"{path.name!r} {name} is not finite real values",
                code="pyreft_artifact_invalid",
                file=path.name,
            )
    return {name: state[name].detach().clone() for name in expected}


def read(
    directory: str | Path, *, repo: str | None = None, revision: str | None = None
) -> LoreftArtifact:
    """Read a pyreft LoReFT artifact directory.

    Parameters
    ----------
    directory:
        The directory pyreft saved (``config.json`` plus ``intkey_<key>.bin``),
        for example a Hugging Face snapshot.
    repo, revision:
        Optional names of where the directory came from (a Hub repo id and
        commit); recorded in ``provenance`` and the subspace ``origin``.

    Returns
    -------
    LoreftArtifact
        ``R``, ``W``, ``b`` as read, the site pyreft names and the provenance.

    Raises
    ------
    InvalidArgumentError
        ``pyreft_artifact_invalid`` for a missing or malformed file,
        ``pyreft_artifact_unsupported`` for anything but one LoReFT intervention.
    """

    root = Path(directory)
    if not root.is_dir():
        raise _refusal(f"{str(directory)!r} is not a directory", code="pyreft_artifact_invalid")
    config, config_sha256 = _config(root)
    key, representation, dimension = _one_loreft(config)
    layer, component, unit, units_per_example, rank = representation[:5]
    if not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in (layer, units_per_example, rank)
    ):
        raise _refusal(
            f"the representation {representation[:5]!r} has no int layer, unit count and rank",
            code="pyreft_artifact_invalid",
            file="config.json",
        )
    weight_path = _weight_file(root, key)
    weight_bytes = weight_path.read_bytes()
    tensors = _tensors(weight_path, weight_bytes, int(rank), dimension)
    rotation = tensors["rotate_layer"]
    gram = rotation.to(torch.float64).T @ rotation.to(torch.float64)
    gram_error = float((gram - torch.eye(int(rank), dtype=torch.float64)).abs().max())
    if gram_error > _ROTATION_GRAM_LIMIT:
        raise _refusal(
            f"{weight_path.name!r} rotate_layer columns are not orthonormal (largest Gram "
            f"error {gram_error:.3g}); pyreft trains an orthogonal rotation, so this is not "
            "a LoReFT artifact as saved",
            code="pyreft_artifact_invalid",
            file=weight_path.name,
        )
    files = {
        "config.json": config_sha256,
        weight_path.name: hashlib.sha256(weight_bytes).hexdigest(),
    }
    provenance = {
        "repo": repo,
        "revision": revision,
        "key": key,
        "intervention_type": _LOREFT,
        "weight_file": weight_path.name,
        "files_sha256": files,
        "source_dtype": {
            "rotation": str(rotation.dtype).removeprefix("torch."),
            "weight": str(tensors["weight"].dtype).removeprefix("torch."),
            "bias": str(tensors["bias"].dtype).removeprefix("torch."),
        },
        "param_sha256": {
            "R": _float64_sha256(rotation),
            "W": _float64_sha256(tensors["weight"]),
            "b": _float64_sha256(tensors["bias"]),
        },
        "orthonormality_max_abs": gram_error,
    }
    return LoreftArtifact(
        rotation=rotation,
        weight=tensors["weight"],
        bias=tensors["bias"],
        layer=int(layer),
        component=str(component),
        unit=str(unit),
        units_per_example=int(units_per_example),
        provenance=provenance,
    )
