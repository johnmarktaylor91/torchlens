"""Round-7 security regression: torch I/O / serialization TYPE construction.

The :class:`SafeBundleUnpickler` torch-``type`` branch historically admitted ANY
resolved ``torch.*`` ``type`` on the premise that "constructing a torch data type
executes no attacker code". That premise is FALSE for torch's I/O-capable types:
constructing one opens/reads/writes a filesystem PATH taken from the pickle stream
at ``tl.load()`` time, BEFORE any resolver / scrub / rehydrate defense fires --

* ``torch._C.PyTorchFileReader('/etc/passwd')`` -- opens/reads an arbitrary victim
  path (existence/type oracle, zip parse, zip-bomb DoS).
* ``torch._C.PyTorchFileWriter(path)`` -- CREATES/opens an arbitrary path.
* ``torch.package.PackageImporter(dir)`` -- ``DirectoryReader`` traverses/reads an
  attacker-named directory.

HIGH: arbitrary file access + DoS, not RCE (``os.system`` / ``eval`` /
``torch.serialization.load`` stayed blocked). The fix adds a module-prefix +
structural type-name guard (``_torch_type_denied``) to the torch-``type`` branch,
denying the file-reader / file-writer / packaging / serialization / jit types --
aligning with torch's own ``weights_only`` allowlist which excludes them. These
tests pin the door shut (no file opened/created + no construction), prove the
audited siblings are denied, prove legit torch DATA types still resolve, and prove
all five save levels still load.
"""

from __future__ import annotations

import io
import pickle
import struct
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io._safe_unpickle import SafeBundleUnpickler, _torch_type_denied
from torchlens.options import CaptureOptions

_CAP = CaptureOptions(
    replayable=True,
    capture_container_structure=True,
    cache=False,
)


class _ControlFlow(nn.Module):
    """Control-flow graph; a portable save embeds a predicate tensor + method refs."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.sum() > 0:
            return torch.relu(value) * 2
        return torch.sigmoid(value) - 1


class _StatefulLinear(nn.Module):
    """Runnable-eligible parameterized graph with a persistent buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.register_buffer("scale", torch.ones(4))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(value)) * self.scale


class _CNN(nn.Module):
    """A real conv/BN/linear model exercising the metadata type surface."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn(self.conv(x)))
        x = x.mean(dim=(2, 3))
        return self.fc(x)


def _reduce_pickle(module: str, name: str, args: tuple[str, ...]) -> bytes:
    """Hand-assemble a ``GLOBAL(module,name) + REDUCE(args)`` pickle stream.

    This is exactly the construction gadget a malicious ``metadata.pkl`` would
    carry: resolve the named global then CALL it with attacker string args.
    """

    out = (
        pickle.PROTO
        + bytes([2])
        + pickle.GLOBAL
        + (module + "\n" + name + "\n").encode()
        + pickle.MARK
    )
    for arg in args:
        raw = arg.encode()
        out += pickle.BINUNICODE + struct.pack("<I", len(raw)) + raw
    return out + pickle.TUPLE + pickle.REDUCE + pickle.STOP


def _save_all_levels(tmp_path: Path) -> dict[str, Path]:
    """Save one bundle at each required level and return their paths."""

    x = torch.randn(2, 4)
    paths: dict[str, Path] = {}
    keep_alive: list[nn.Module] = []  # runnable weight/state payload reads live model

    audit = tmp_path / "audit"
    cf_a = _ControlFlow()
    keep_alive.append(cf_a)
    tl.trace(cf_a, x, save="all").save(audit, level="audit")
    paths["analysis"] = audit

    portable = tmp_path / "portable"
    cf_p = _ControlFlow()
    keep_alive.append(cf_p)
    tl.trace(cf_p, x, save="all").save(portable, level="portable")
    paths["portable"] = portable

    run = tmp_path / "runnable"
    lin = _StatefulLinear()
    keep_alive.append(lin)
    tl.trace(lin, x, save=tl.func("relu"), capture=_CAP).save(run, level="runnable")
    paths["runnable"] = run

    run_w = tmp_path / "runnable_weights"
    lin_w = _StatefulLinear()
    keep_alive.append(lin_w)
    tl.trace(lin_w, x, save=tl.func("relu"), capture=_CAP).save(
        run_w, level="runnable", include_weights=True
    )
    paths["runnable_weights"] = run_w

    run_a = tmp_path / "runnable_activations"
    lin_a = _StatefulLinear()
    keep_alive.append(lin_a)
    tl.trace(lin_a, x, save=tl.func("relu"), capture=_CAP).save(
        run_a, level="runnable", include_activations=True
    )
    paths["runnable_activations"] = run_a

    assert len(keep_alive) == 5
    return paths


# --------------------------------------------------------------------------- #
# The exploit: torch I/O / serialization types are denied (no file touched).
# --------------------------------------------------------------------------- #


def test_pytorch_file_writer_type_denied_no_file_created(tmp_path: Path) -> None:
    """``PyTorchFileWriter(path)`` is denied and CREATES no file (proof of no ctor)."""

    sentinel = tmp_path / "created_by_writer.zip"
    payload = _reduce_pickle("torch._C", "PyTorchFileWriter", (str(sentinel),))
    with pytest.raises(pickle.UnpicklingError, match="torch I/O / serialization type"):
        SafeBundleUnpickler(io.BytesIO(payload)).load()
    assert not sentinel.exists(), "PyTorchFileWriter was constructed and created a file"


# --------------------------------------------------------------------------- #
# The policy helper: denied I/O types vs admitted data types.
# --------------------------------------------------------------------------- #


def test_torch_type_denied_helper_classifies_io_types() -> None:
    """``_torch_type_denied`` denies the I/O / serialization / packaging types."""

    denied = [
        ("torch._C", "PyTorchFileReader", torch._C.PyTorchFileReader),
        ("torch._C", "PyTorchFileWriter", torch._C.PyTorchFileWriter),
        ("torch._C", "FileCheck", torch._C.FileCheck),
        ("torch.package.package_importer", "PackageImporter", torch.package.PackageImporter),
    ]
    for module, name, obj in denied:
        assert _torch_type_denied(module, name, obj), f"{module}.{name} should be denied"


# --------------------------------------------------------------------------- #
# Legitimate bundles at every save level (+ real models) still load.
# --------------------------------------------------------------------------- #
