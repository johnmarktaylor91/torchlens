"""Round-5 security regression: load-time / run-time ``torch.from_file`` file-I/O.

``torch.from_file`` is a ``_VariableFunctionsClass`` factory that CREATES/TRUNCATES
an arbitrary file (``shared=True``) or READS an arbitrary file into a tensor
(``shared=False``). Its real ``__module__`` is plainly ``"torch"``, so the
module-granular callable-safety policy admitted it, and it was reachable through
BOTH untrusted surfaces:

* VECTOR 1 (unpickler): ``_safe_getattr`` admitted
  ``getattr(torch._C._VariableFunctionsClass, "from_file")`` for a following pickle
  ``REDUCE`` -- arbitrary file creation/read DURING ``tl.load()`` with no trust
  opt-in.
* VECTOR 2 (resolver): ``is_pure_forward_callable(torch.from_file)`` returned
  ``True``, so a bundle op keyed ``("torch", "from_file")`` resolved and was CALLED
  at ``Trace.run()`` with recorded literal args.

The fix closes the CLASS: a qualname-level guard (audited denylist + structural
file/import/serialization name guard) layered on the module gate, consulted by the
unpickler ``_safe_getattr`` AND the shared ``is_pure_forward_callable`` policy the
resolvers use. These tests pin both vectors shut, prove the audited siblings are
denied, prove genuine pure forward ops still resolve/run/VERIFY, and prove all five
save levels still load.
"""

from __future__ import annotations

import io
import json
import operator
import os
import pickle
import shutil
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io._safe_unpickle import SafeBundleUnpickler, _safe_getattr
from torchlens.intervention.errors import UntrustedCallableError
from torchlens.intervention.resolver import resolve_function_registry_key
from torchlens.intervention.types import FunctionRegistryKey
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness
from torchlens.utils._callable_safety import is_pure_forward_callable

_CAP = CaptureOptions(
    intervention_ready=True,
    capture_container_structure=True,
    cache=False,
)


# --------------------------------------------------------------------------- #
# Hand-written pickle opcodes reproducing the exact exploit stream.
# --------------------------------------------------------------------------- #


def _from_file_reduce_stream(target: str, *, shared: bool, size: int) -> bytes:
    """Build the round-5 exploit pickle: getattr(VFC, from_file) then REDUCE it.

    Mirrors ``/tmp/tl_round5/repro_from_file_unpickle.py``: resolve the file-I/O
    builtin via the admitted ``getattr`` wrapper, then call it on attacker args.
    """

    proto = pickle.PROTO + bytes([2])
    op = b"c" + b"builtins\ngetattr\n"
    op += b"c" + b"torch._C\n_VariableFunctionsClass\n"
    op += b"\x8c" + bytes([len("from_file")]) + b"from_file"
    op += b"\x86R"  # TUPLE2 ; REDUCE -> _safe_getattr(VFC, "from_file")
    op += b"("  # MARK
    op += b"\x8c" + bytes([len(target)]) + target.encode()
    op += b"\x88" if shared else b"\x89"  # NEWTRUE / NEWFALSE -> shared
    op += b"K" + bytes([size])  # BININT1 size
    op += b"tR."  # TUPLE ; REDUCE ; STOP
    return proto + op


@pytest.mark.smoke
def test_from_file_unpickle_vector_creation_denied(tmp_path: Path) -> None:
    """VECTOR 1a: the unpickler refuses from_file and creates no file (shared=True)."""

    target = tmp_path / "PWNED_BY_UNPICKLE"
    stream = _from_file_reduce_stream(str(target), shared=True, size=64)

    with pytest.raises(pickle.UnpicklingError):
        SafeBundleUnpickler(io.BytesIO(stream)).load()
    assert not target.exists(), "from_file created a file during unpickle"


@pytest.mark.smoke
def test_from_file_unpickle_vector_read_denied(tmp_path: Path) -> None:
    """VECTOR 1b: the unpickler refuses from_file and reads no file (shared=False)."""

    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")  # pragma: allowlist secret
    stream = _from_file_reduce_stream(str(secret), shared=False, size=11)

    with pytest.raises(pickle.UnpicklingError):
        SafeBundleUnpickler(io.BytesIO(stream)).load()


def test_safe_getattr_denies_from_file() -> None:
    """The getattr wrapper refuses the file-I/O builtin but keeps pure refs."""

    with pytest.raises(pickle.UnpicklingError):
        _safe_getattr(torch._C._VariableFunctionsClass, "from_file")
    # Pure factory / method references off the same holder still resolve.
    assert callable(_safe_getattr(torch._C._VariableFunctionsClass, "from_numpy"))
    assert callable(_safe_getattr(torch._C._VariableFunctionsClass, "frombuffer"))
    assert callable(_safe_getattr(torch._C.TensorBase, "sum"))


# --------------------------------------------------------------------------- #
# VECTOR 2: resolver refuses from_file at run() time (both key shapes).
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.parametrize(
    "key",
    [
        FunctionRegistryKey("torch", "from_file", "function"),
        FunctionRegistryKey(
            "custom",
            "from_file",
            "function",
            import_path="torch._C._VariableFunctionsClass:from_file",
        ),
    ],
    ids=["torch_namespace", "internal_builtin"],
)
def test_resolver_denies_from_file(key: FunctionRegistryKey) -> None:
    """torch.from_file is refused with a typed security error at both key shapes."""

    with pytest.raises(UntrustedCallableError):
        resolve_function_registry_key(key)


# --------------------------------------------------------------------------- #
# Close the CLASS: the audited side-effecting siblings are all denied.
# --------------------------------------------------------------------------- #


def _audited_dangerous_callables() -> list:
    """Return the audited side-effecting callables reachable in the allowed surface."""

    candidates = [
        torch.from_file,
        torch.import_ir_module,
        torch.import_ir_module_from_buffer,
        torch.PyTorchFileReader,
        torch.PyTorchFileWriter,
        torch.FileCheck,
        torch._load_global_deps,
        torch._preload_cuda_deps,
        torch._import_device_backends,
        torch._utils._import_dotted_name,
        torch.get_file_path,
        # Prior denials that must STAY closed.
        torch.load,
        torch.save,
        torch.jit.load,
        torch.jit.save,
        os.system,
        eval,
        pickle.loads,
        pickle.load,
    ]
    return [c for c in candidates if c is not None]


def test_audited_side_effecting_callables_denied() -> None:
    """Every audited file-I/O / serialization / import / process callable is denied."""

    for func in _audited_dangerous_callables():
        assert is_pure_forward_callable(func) is False, func


def test_pure_forward_ops_still_admitted() -> None:
    """Genuine pure forward ops -- including from_numpy/frombuffer -- stay resolvable.

    ``from_numpy`` / ``frombuffer`` are pure tensor factories whose names begin with
    ``from`` yet must NOT be swept up by the structural file-I/O name guard.
    """

    import torch.nn.functional as functional

    for func in (
        torch.matmul,
        torch.add,
        torch.einsum,
        torch.conv2d,
        torch.layer_norm,
        torch.relu,
        torch.sigmoid,
        torch.addbmm,
        torch.baddbmm,
        torch.nan_to_num,
        torch.from_numpy,
        torch.frombuffer,
        torch.fft.fft,
        functional.relu,
        functional.linear,
        functional.huber_loss,
        torch.Tensor.view,
        torch.Tensor.reshape,
        operator.add,
        operator.mul,
    ):
        assert is_pure_forward_callable(func) is True, func


@pytest.mark.parametrize("qualname", ["from_numpy", "frombuffer", "matmul", "conv2d"])
def test_resolver_admits_pure_torch_ops(qualname: str) -> None:
    """Pure torch factories/ops resolve through the intervention resolver."""

    resolved = resolve_function_registry_key(FunctionRegistryKey("torch", qualname, "function"))
    assert callable(resolved)


# --------------------------------------------------------------------------- #
# End-to-end: repointing a runnable op to torch.from_file must not touch a file.
# --------------------------------------------------------------------------- #


class _TinyLinear(nn.Module):
    """Minimal parameterized graph whose forward has a relu op to repoint."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(value))


def _relu_registry_index(run: dict) -> int:
    for index, entry in enumerate(run["callable_registry"]):
        if entry["key"]["qualname"] == "relu":
            return index
    raise AssertionError("expected a relu op in the tiny model")


@pytest.mark.smoke
def test_runnable_load_blocks_from_file_creation(tmp_path: Path) -> None:
    """Repointing a run op to torch.from_file must not create an attacker file."""

    model = (
        _TinyLinear().eval()
    )  # retain: include_weights needs the live source model (survive full-suite GC)
    log = tl.trace(model, torch.ones(1, 4), capture=_CAP)
    clean = tmp_path / "clean.tlspec"
    tl.save(log, clean, level="runnable", include_weights=True)
    assert model is not None

    evil = tmp_path / "evil.tlspec"
    shutil.copytree(clean, evil)
    target = tmp_path / "PWNED_BY_RUN"

    manifest_path = evil / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    run = manifest["run"]
    index = _relu_registry_index(run)
    registry_id = str(run["callable_registry"][index]["registry_id"])
    key = run["callable_registry"][index]["key"]
    key["namespace"] = "torch"
    key["qualname"] = "from_file"
    key["import_path"] = None
    call = next(c for c in run["calls"] if c["registry_id"] == registry_id)
    call["argument_names"] = ["filename", "shared", "size"]
    call["num_positional_args"] = 3
    call["num_keyword_args"] = 0
    call["tensor_arguments"] = []
    call["literal_arguments"] = [
        {"argument_path": ["args", 0], "value": {"kind": "str", "value": str(target)}},
        {"argument_path": ["args", 1], "value": {"kind": "bool", "value": True}},
        {"argument_path": ["args", 2], "value": {"kind": "int", "value": 64}},
    ]
    manifest_path.write_text(json.dumps(manifest))

    loaded = tl.load(evil)
    with pytest.raises(Exception):
        loaded.run(inputs=torch.ones(1, 4))
    assert not target.exists(), "from_file created a file during run()"


# --------------------------------------------------------------------------- #
# Regression: every save level still loads; genuine models still VERIFY.
# --------------------------------------------------------------------------- #


class _ControlFlow(nn.Module):
    """Control-flow graph; a portable save embeds method refs via getattr."""

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


class _TinyCNN(nn.Module):
    """Small convolutional network exercising pure forward ops end to end."""

    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.fc = nn.Linear(8, 10)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.relu(self.c1(value))
        value = torch.nn.functional.adaptive_avg_pool2d(value, 1).flatten(1)
        return self.fc(value)


@pytest.mark.smoke
def test_all_five_save_levels_still_load(tmp_path: Path) -> None:
    """Every legitimately-saved bundle at every level loads under the tightened gate."""

    x = torch.randn(2, 4)
    keep_alive: list[nn.Module] = []

    audit = tmp_path / "audit"
    cf_a = _ControlFlow()
    keep_alive.append(cf_a)
    tl.trace(cf_a, x, layers_to_save="all").save(audit, level="audit")

    portable = tmp_path / "portable"
    cf_p = _ControlFlow()
    keep_alive.append(cf_p)
    tl.trace(cf_p, x, layers_to_save="all").save(portable, level="portable")

    run = tmp_path / "runnable"
    lin = _StatefulLinear()
    keep_alive.append(lin)
    tl.trace(lin, x, save=tl.func("relu"), capture=_CAP).save(run, level="runnable")

    run_w = tmp_path / "runnable_weights"
    lin_w = _StatefulLinear()
    keep_alive.append(lin_w)
    tl.trace(lin_w, x, save=tl.func("relu"), capture=_CAP).save(
        run_w, level="runnable", include_weights=True
    )

    run_a = tmp_path / "runnable_activations"
    lin_a = _StatefulLinear()
    keep_alive.append(lin_a)
    tl.trace(lin_a, x, save=tl.func("relu"), capture=_CAP).save(
        run_a, level="runnable", include_activations=True
    )

    assert len(keep_alive) == 5
    for name, bundle_path in (
        ("audit", audit),
        ("portable", portable),
        ("runnable", run),
        ("runnable_weights", run_w),
        ("runnable_activations", run_a),
    ):
        assert tl.load(bundle_path) is not None, name


@pytest.mark.smoke
def test_genuine_cnn_runnable_still_verified(tmp_path: Path) -> None:
    """A real CNN round-trips through the runnable path and VERIFIES after the fix."""

    inputs = torch.randn(1, 3, 16, 16)
    model = _TinyCNN().eval()  # retain a strong ref: include_weights needs the live source model
    log = tl.trace(model, inputs, capture=_CAP)
    path = tmp_path / "cnn.tlspec"
    tl.save(log, path, level="runnable", include_weights=True)
    assert (
        model is not None
    )  # keep the model alive across save (defeats weakref GC under full-suite pressure)
    result = tl.load(path).run(inputs=inputs)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
