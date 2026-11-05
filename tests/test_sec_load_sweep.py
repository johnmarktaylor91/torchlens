"""Untrusted-artifact load sweep: each crafted input refuses before anything acts on it.

Every test hands TorchLens a hostile artifact piece (a ledger row naming a file
outside the artifact, a symlinked shard, a ``.pt`` shard on a torch whose
``weights_only`` loader is bypassable, a non-PNG/JPEG image payload, an unlisted
runnable descriptor, a FIFO named like a JSON file, a bounds-unchecked kernel named
as a replay callable) and asserts the typed refusal. No test dereferences memory or
reads a file outside its temp directory: the kernels are only resolved, never called
on real data, and the FIFO tests run in a daemon thread with a timeout.
"""

from __future__ import annotations

import io
import json
import os
import threading
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from _format3_envelope import mutate_runnable_descriptor
from test_tlspec_runnable_alloc_bomb import _recompute_runtime_fingerprints

import torchlens as tl
from torchlens._data_substrate.artifact import ExtractionArtifactError, read_trusted_rows
from torchlens._io import _json
from torchlens._io._small_raw import _RAW_IMAGE_SENTINEL, _rehydrate_small_raw_image
from torchlens._runnable import ReadinessStatus
from torchlens._utils._callable_safety import is_pure_forward_callable
from torchlens._utils._torch_symbols import (
    ALLOWED_EXACT_ROOTS,
    ENUMERATED_TORCH_NAMESPACES,
    getattr_allowlisted,
    torch_namespace,
)
from torchlens._utils._unchecked_kernel_denial import (
    UNCHECKED_NAME_PREFIXES,
    is_unchecked_memory_kernel,
    unchecked_switch_kernel_names,
)
from torchlens.errors import RunCapabilityUnavailableError, TorchLensError

# --------------------------------------------------------------------------- #
# (1) extraction ledger rows stay inside the artifact                         #
# --------------------------------------------------------------------------- #


def _ledger(container: Path, rows: list[dict[str, Any]]) -> None:
    """Write ``rows`` as the container's ``ledger.jsonl``."""

    container.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row) + "\n" for row in rows)
    (container / "ledger.jsonl").write_text(text, encoding="utf-8")


@pytest.mark.smoke
@pytest.mark.parametrize("kind", ["absolute", "dotdot", "subdir"])
def test_ledger_row_naming_a_file_outside_the_artifact_refuses(tmp_path: Path, kind: str) -> None:
    """A ``file`` cell naming an absolute path or ``..`` refuses before any read."""

    outside = tmp_path / "victim.pt"
    outside.write_bytes(b"not yours")
    container = tmp_path / "art"
    name = {
        "absolute": str(outside),
        "dotdot": "../victim.pt",
        "subdir": "sub/batch_00000.pt",
    }[kind]
    _ledger(container, [{"index": 0, "file": name, "row_start": 0, "n_rows": 1}])
    with pytest.raises(ExtractionArtifactError) as info:
        read_trusted_rows(container)
    assert info.value.fields["code"] == "extraction_ledger_invalid"


def test_symlinked_shard_and_symlinked_ledger_refuse(tmp_path: Path) -> None:
    """A writer-shaped shard name that is a symlink, or a symlinked ledger, refuses."""

    outside = tmp_path / "victim.pt"
    outside.write_bytes(b"not yours")
    container = tmp_path / "art"
    _ledger(container, [{"index": 0, "file": "batch_00000.pt", "row_start": 0, "n_rows": 1}])
    (container / "batch_00000.pt").symlink_to(outside)
    with pytest.raises(ExtractionArtifactError):
        read_trusted_rows(container)

    other = tmp_path / "art2"
    other.mkdir()
    target = tmp_path / "rows.jsonl"
    target.write_text("{}\n", encoding="utf-8")
    (other / "ledger.jsonl").symlink_to(target)
    with pytest.raises(ExtractionArtifactError):
        read_trusted_rows(other)


def test_writer_shaped_ledger_rows_still_verify(tmp_path: Path) -> None:
    """No regression: a regular ``batch_NNNNN`` shard inside the artifact verifies."""

    container = tmp_path / "art"
    _ledger(
        container,
        [{"index": 0, "file": "batch_00000.safetensors", "row_start": 0, "n_rows": 1}],
    )
    (container / "batch_00000.safetensors").write_bytes(b"x")
    assert [row["file"] for row in read_trusted_rows(container)] == ["batch_00000.safetensors"]


# --------------------------------------------------------------------------- #
# (2) torch.load(weights_only=True) only on a torch where it is safe          #
# --------------------------------------------------------------------------- #


def test_pt_shard_refuses_on_a_bypassable_weights_only_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CVE-2025-32434: with the safe-loader capability absent, no shard is unpickled."""

    from torchlens._errors import InvalidArgumentError
    from torchlens._extraction import shards

    def _never(*_args: Any, **_kwargs: Any) -> Any:
        """Fail the test if the shard bytes ever reach torch.load."""

        raise AssertionError("torch.load reached on an unsafe torch")

    monkeypatch.setattr(shards, "HAS_SAFE_WEIGHTS_ONLY_LOAD", False)
    monkeypatch.setattr(shards.torch, "load", _never)
    shard = tmp_path / "batch_00000.pt"
    shard.write_bytes(b"crafted")
    with pytest.raises(InvalidArgumentError) as info:
        shards.read_shard(shard, "pt")
    assert info.value.fields["code"] == "extraction_pt_shard_unsafe_torch"


def test_pyreft_weights_refuse_on_unsafe_torch_and_symlinked_weight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pyreft bridge holds the same torch.load rule and never follows a weight symlink."""

    from torchlens._errors import InvalidArgumentError
    from torchlens.bridge import pyreft

    monkeypatch.setattr(pyreft, "HAS_SAFE_WEIGHTS_ONLY_LOAD", False)
    with pytest.raises(InvalidArgumentError):
        pyreft._tensors(tmp_path / "intkey_k.bin", b"crafted", 1, 2)

    outside = tmp_path / "victim.bin"
    outside.write_bytes(b"not yours")
    folder = tmp_path / "reft"
    folder.mkdir()
    (folder / "intkey_k.bin").symlink_to(outside)
    with pytest.raises(InvalidArgumentError):
        pyreft._weight_file(folder, "k")


# --------------------------------------------------------------------------- #
# (3) image payloads decode only as the formats the writer emits              #
# --------------------------------------------------------------------------- #


def _encoded(fmt: str) -> bytes:
    """Return a tiny RGB image encoded as ``fmt``."""

    image_module = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    image_module.new("RGB", (4, 4), (10, 20, 30)).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.mark.smoke
@pytest.mark.parametrize("fmt", ["GIF", "BMP", "TIFF"])
def test_image_payload_outside_png_jpeg_is_not_decoded(fmt: str) -> None:
    """Only PNG and JPEG reach a decoder; any other format stays the inert raw dict."""

    value = {_RAW_IMAGE_SENTINEL: True, "data": _encoded(fmt)}
    assert _rehydrate_small_raw_image(value) is value


def test_png_and_jpeg_payloads_still_decode() -> None:
    """No regression: the writer's own formats round-trip to an image."""

    for fmt in ("PNG", "JPEG"):
        value = {_RAW_IMAGE_SENTINEL: True, "data": _encoded(fmt)}
        assert _rehydrate_small_raw_image(value) is not value, fmt


# --------------------------------------------------------------------------- #
# (4) the runnable descriptor is read only when the manifest lists it         #
# --------------------------------------------------------------------------- #


class _Tiny(nn.Module):
    """One linear plus a softmax: the smallest replayable capture."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a softmax over the linear output."""

        return torch.softmax(self.fc(x), dim=1)


def _bundle(tmp_path: Path, name: str) -> Path:
    """Save one runnable bundle for tampering."""

    trace = tl.trace(
        _Tiny().eval(), torch.randn(2, 4), capture=tl.options.CaptureOptions(replayable=True)
    )
    path = tmp_path / name
    trace.save(str(path), level="runnable", include_weights=True)
    return path


def test_unlisted_runnable_descriptor_is_never_read(tmp_path: Path) -> None:
    """Dropping the descriptor from the file table refuses typed instead of reading it."""

    bundle = _bundle(tmp_path, "unlisted.tlspec")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rel = next(name for name in manifest["files"] if name.endswith("descriptor.json"))
    del manifest["files"][rel]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(Exception, match="not listed"), warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the now-unlisted file is named in an advisory
        tl.load(str(bundle))


# --------------------------------------------------------------------------- #
# (5) a FIFO named like a JSON file refuses instead of hanging the load       #
# --------------------------------------------------------------------------- #


def _finishes(call: Any, timeout: float = 20.0) -> tuple[bool, BaseException | None]:
    """Run ``call`` in a daemon thread; return (finished, raised exception)."""

    outcome: dict[str, BaseException | None] = {"error": None}

    def _target() -> None:
        """Record the call's exception, if any."""

        try:
            call()
        except BaseException as exc:  # noqa: BLE001 - the test inspects it
            outcome["error"] = exc

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout)
    return (not worker.is_alive()), outcome["error"]


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs POSIX FIFOs")
def test_fifo_json_refuses_without_blocking(tmp_path: Path) -> None:
    """The bounded JSON readers and ``tl.load`` refuse a FIFO ``manifest.json`` promptly."""

    fifo = tmp_path / "pipe.json"
    os.mkfifo(fifo)
    finished, error = _finishes(lambda: _json.read_bytes_bounded(fifo))
    assert finished and isinstance(error, OSError)
    finished, error = _finishes(lambda: _json.read_bounded(fifo))
    assert finished and isinstance(error, OSError)

    bundle = tmp_path / "fifo.tlspec"
    bundle.mkdir()
    os.mkfifo(bundle / "manifest.json")
    finished, error = _finishes(lambda: tl.load(str(bundle)))
    assert finished, "tl.load blocked on a FIFO manifest"
    assert error is not None and not isinstance(error, AssertionError)


# --------------------------------------------------------------------------- #
# (6) bounds-unchecked kernels never resolve as replay callables              #
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.parametrize(
    "path",
    [
        "torch._C._VariableFunctions._unsafe_index",
        "torch._C._VariableFunctions._unsafe_index_put",
        "torch._C._VariableFunctions._unsafe_masked_index",
        "torch._C._VariableFunctions._unsafe_masked_index_put_accumulate",
        "torch._C._VariableFunctions._index_put_impl_",
        "torch._C._VariableFunctions.segment_reduce",
        "torch._C._VariableFunctions._nested_view_from_buffer",
        "torch._C._VariableFunctions._nested_view_from_buffer_copy",
        "torch._C._VariableFunctions._reshape_alias_copy",
    ],
)
def test_bounds_unchecked_kernels_are_denied(path: str) -> None:
    """Each named kernel skips a bounds check on its arguments; the gate refuses it."""

    namespace, _, name = path.rpartition(".")
    value = getattr(torch_namespace(namespace), name, None)
    if value is None:
        pytest.skip(f"{path} is absent on this torch")
    assert is_unchecked_memory_kernel(value), path
    assert not is_pure_forward_callable(value), f"{path} admitted"


def test_no_admitted_callable_is_bounds_unchecked() -> None:
    """RUNTIME enumeration of the replay roots: nothing admitted matches the rule."""

    switch = unchecked_switch_kernel_names()
    assert {"segment_reduce", "_index_put_impl_"} <= switch
    offenders: list[str] = []
    for namespace in (*ALLOWED_EXACT_ROOTS, *ENUMERATED_TORCH_NAMESPACES):
        root = ALLOWED_EXACT_ROOTS.get(namespace) or torch_namespace(namespace)
        if root is None:
            continue
        for name in dir(root):
            if not (name.startswith(UNCHECKED_NAME_PREFIXES) or name in switch):
                continue
            value = getattr_allowlisted(namespace, name)
            if value is not None and is_pure_forward_callable(value):
                offenders.append(f"{namespace}.{name}")
    assert not offenders, f"bounds-unchecked kernels admitted: {offenders}"


def test_checked_neighbours_still_resolve() -> None:
    """No regression: the bounds-checked public spellings stay admitted."""

    for value in (
        torch.reshape,
        torch.as_strided,
        torch.Tensor.unsafe_split,
        torch.softmax,
    ):
        assert is_pure_forward_callable(value), value


def test_crafted_bundle_naming_an_unchecked_kernel_never_runs(tmp_path: Path) -> None:
    """End to end: a descriptor whose callable is renamed to ``_unsafe_index`` refuses."""

    bundle = _bundle(tmp_path, "unsafe.tlspec")

    def _craft(descriptor: dict[str, Any]) -> None:
        """Point every softmax registry key at the bounds-unchecked index kernel."""

        for entry in descriptor["callable_registry"]:
            if entry["key"]["qualname"].endswith("softmax"):
                entry["key"]["qualname"] = "_unsafe_index"
        _recompute_runtime_fingerprints(descriptor)

    mutate_runnable_descriptor(bundle, _craft)
    try:
        loaded = tl.load(str(bundle))
    except TorchLensError:
        return  # refused at load: nothing was resolved or run
    assert loaded._runnable.readiness.status is ReadinessStatus.UNAVAILABLE
    with pytest.raises(RunCapabilityUnavailableError):
        loaded.run_transaction(inputs=torch.randn(2, 4))


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("import_path", ["torch:_unsafe_index", "torch:segment_reduce"])
def test_registry_resolver_refuses_unchecked_kernels(import_path: str, trust: bool) -> None:
    """The resolver behind validation replay of a loaded bundle refuses them, even trusted."""

    from torchlens.errors import UntrustedCallableError
    from torchlens.intervention._resolver_registry import resolve_function_registry_key
    from torchlens.intervention.types import FunctionRegistryKey

    key = FunctionRegistryKey(
        "custom", import_path.split(":")[1], "function", import_path=import_path
    )
    with pytest.raises(UntrustedCallableError):
        resolve_function_registry_key(key, trust_custom_callables=trust)


# --------------------------------------------------------------------------- #
# (7) hostile numbers refuse typed instead of crashing                        #
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_json_huge_integer_and_float_overflow_refuse_typed() -> None:
    """A 5000-digit integer and ``1e999`` refuse as ``JSONDecodeError``, not a raw crash."""

    with pytest.raises(json.JSONDecodeError):
        _json.loads_bounded('{"a": 1' + "0" * 5000 + "}")
    with pytest.raises(json.JSONDecodeError):
        _json.loads_bounded("[1e999]")
    assert _json.loads_bounded('[1.5, 2, {"k": -3e10}]') == [1.5, 2, {"k": -3e10}]


def test_zero_dimension_cannot_hide_an_oversized_dimension() -> None:
    """``[0, 10**30]`` has zero elements but refuses at the cell check, not in torch."""

    from torchlens._io.format3 import reader

    cell = {"shape": [0, 10**30], "device": "cpu", "dtype": "float32", "requires_grad": False}
    with pytest.raises(Exception, match="per dim"):
        reader._tensor_geometry(cell, "probe")
