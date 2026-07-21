"""r55 CLASS 1 + r57 immunizer -- portable device strings are closed-gate data.

r54 ``sec_2`` (HIGH): a tinygrad ``.tlspec`` carries per-blob ``logical_device``
/ ``device_at_save`` strings in plaintext ``manifest.json``; the codec passed
them straight into ``tinygrad.Tensor(device=<attacker str>)`` where the
``disk:<path>`` backend writes an attacker-named file on a default
``tl.load(path)``. r56 ``corr_1`` (MED): the r55 fix used a hand-maintained
positive allowlist that omitted legit tinygrad accelerators (``NV``/``AMD``/
``WEBGPU``/``QCOM``/``DSP``/``NPY``) and silently relocated their payloads to
the runtime default. r57 deletes BOTH static-list directions: admission is the
closed grammar, then (tinygrad) an I/O-sink exclude checked BEFORE membership
in the RUNTIME-DERIVED ``tinygrad.Device._devices`` vocabulary -- fail-closed
to the runtime default on unknown bases and when tinygrad is absent.

Whole-class coverage, enumeration-free where the class is open-ended:

* a filesystem-sentinel end-to-end test (tinygrad ``disk:`` token creates NO
  file on default ``tl.load()`` -- the r55 C1 close, kept verbatim);
* accelerator round-trip derived from the LIVE runtime inventory (minus the
  sink set), so a future tinygrad accelerator is auto-covered;
* sink refusal (``disk``/``tinyfs``/``rdma``/``ext``/``shm``) including the
  ordering pin that ``disk`` is refused even though the runtime names it;
* fail-closed refusal of grammar-valid garbage and of every token when
  tinygrad is not importable (no crash, no open-ended admission);
* a source-scan tripwire: every non-torch ``from_numpy`` either ignores the
  saved device string or routes it through the shared resolver -- so a future
  codec inherits the refusal by construction.
"""

from __future__ import annotations

import inspect
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import torchlens as tl
from torchlens._io import _artifact_strings, payload_codec
from torchlens._io._artifact_strings import (
    _TINYGRAD_IO_SINK_BASES,
    _resolve_portable_device,
    _sanitize_artifact_device_token,
    _tinygrad_runtime_device_bases,
)

pytestmark = pytest.mark.smoke


# --------------------------------------------------------------------------- #
# (a) the static accelerator allowlist is GONE (r56 corr_1 class close)        #
# --------------------------------------------------------------------------- #


def test_static_device_allowlist_deleted() -> None:
    """The hand-maintained positive device vocabulary must not come back."""

    assert not hasattr(_artifact_strings, "_ALLOWED_DEVICE_BASES")


# --------------------------------------------------------------------------- #
# (b) token grammar: legit admitted, attacker path/url/scheme refused          #
# --------------------------------------------------------------------------- #

_LEGIT_GRAMMAR_TOKENS = [
    "cpu",
    "gpu",
    "cuda",
    "cuda:0",
    "cuda:3",
    "mps",
    "CPU",
    "GPU",
    "CLANG",
    "METAL",
    "CUDA",
    "CUDA:0",
    "LLVM",
    "PYTHON",
    "npu",
    "xpu",
    "tpu",
    "Device(cpu)",
    "Device(gpu)",
    "DeviceType.gpu",
    "Place(cpu)",
    "Place(gpu:0)",
    "gpu:0",
    "cuda:1",
]

_ATTACKER_TOKENS = [
    "disk:/tmp/tl_sentinel",
    "disk:./rel",
    "ext:/dev/sda",
    "file:///etc/shadow",
    "/etc/passwd",
    "..\\..\\win",
    "cpu:/../../x",
    "https://evil.com/x",
    "\\\\unc\\share",
    "disk:C:\\Windows",
    "device(disk:/tmp/x)",
    "cpu; rm -rf /",
    "cuda:notanint",
]


@pytest.mark.parametrize("backend_name", ["paddle", "mlx", "jax", "tf"])
@pytest.mark.parametrize("token", _LEGIT_GRAMMAR_TOKENS)
def test_grammar_only_backends_admit_device_tokens_unchanged(backend_name: str, token: str) -> None:
    """Backends with no filesystem/network device admit every grammar-valid token.

    mlx/paddle/jax/tf expose no path-bearing device, so the closed grammar alone
    is the whole gate there -- and no shared accelerator list can over-refuse a
    legitimate token for them (the r56 corr_1 failure mode).
    """

    assert _sanitize_artifact_device_token(token, backend_name) == token


@pytest.mark.parametrize("backend_name", ["tinygrad", "paddle"])
@pytest.mark.parametrize("token", _ATTACKER_TOKENS)
def test_attacker_device_tokens_refused_to_default(backend_name: str, token: str) -> None:
    """Every path/url/scheme/malformed token resolves to the runtime default (None)."""

    assert _sanitize_artifact_device_token(token, backend_name) is None


def test_unknown_and_empty_tokens_resolve_default() -> None:
    """The sentinel ``unknown``/empty strings and non-strings resolve to default."""

    for backend_name in ("tinygrad", "paddle"):
        assert _sanitize_artifact_device_token("unknown", backend_name) is None
        assert _sanitize_artifact_device_token("", backend_name) is None
        assert _sanitize_artifact_device_token(None, backend_name) is None
        assert _sanitize_artifact_device_token(123, backend_name) is None


def test_resolve_portable_device_trusts_caller_map_location() -> None:
    """A caller ``map_location`` is trusted input and passes through unchanged."""

    assert _resolve_portable_device("tinygrad", "disk:/tmp/x", "cuda") == "cuda"
    # Artifact token is gated only when no trusted caller value is supplied.
    assert _resolve_portable_device("tinygrad", "disk:/tmp/x", None) is None


# --------------------------------------------------------------------------- #
# (c) tinygrad: runtime-derived accelerator vocabulary (r56 corr_1 fix)        #
# --------------------------------------------------------------------------- #


def _runtime_accelerator_bases() -> frozenset[str]:
    """The live tinygrad inventory minus the I/O-sink exclude, derived at test time."""

    return _tinygrad_runtime_device_bases() - _TINYGRAD_IO_SINK_BASES


def test_every_runtime_accelerator_round_trips_admitted() -> None:
    """Every non-sink base the INSTALLED runtime names is admitted unchanged.

    Derived from ``Device._devices`` at test time, so a future tinygrad
    accelerator is auto-covered without touching this test (enumeration-free).
    """

    pytest.importorskip("tinygrad")
    bases = _runtime_accelerator_bases()
    assert bases, "installed tinygrad runtime reported no accelerator bases"
    for base in bases:
        for token in (base, base.upper(), f"{base.upper()}:0"):
            assert _sanitize_artifact_device_token(token, "tinygrad") == token
            assert _resolve_portable_device("tinygrad", token, None) == token


@pytest.mark.parametrize("token", ["NV", "AMD", "WEBGPU", "QCOM", "DSP", "NPY", "NV:0"])
def test_corr1_victim_accelerators_admitted(token: str) -> None:
    """The exact accelerators the r55 static allowlist over-refused now round-trip."""

    pytest.importorskip("tinygrad")
    base = token.split(":")[0].lower()
    if base not in _tinygrad_runtime_device_bases():
        pytest.skip(f"installed tinygrad runtime no longer names {base!r}")
    assert _sanitize_artifact_device_token(token, "tinygrad") == token
    assert _resolve_portable_device("tinygrad", token, None) == token


def test_codec_round_trips_runtime_accelerator_device() -> None:
    """End-to-end corr_1 pin: a legit recorded accelerator token is NOT silently
    relocated to the runtime default by the codec.

    Uses ``NPY`` (numpy-backed, realizable on any host, and a base the r55
    allowlist refused) so the pin needs no accelerator hardware.
    """

    pytest.importorskip("tinygrad")
    if "npy" not in _runtime_accelerator_bases():
        pytest.skip("installed tinygrad runtime no longer ships an NPY device")
    entry = SimpleNamespace(logical_dtype="float32", logical_device="NPY", device_at_save="NPY")
    loaded = payload_codec.TinygradPayloadCodec().from_numpy(
        np.asarray([1.0, 2.0], dtype=np.float32),
        entry,
        map_location=None,
    )
    assert str(loaded.device).split(":")[0].upper() == "NPY", (
        "portable codec silently relocated an NPY payload to the runtime default"
    )


# --------------------------------------------------------------------------- #
# (d) tinygrad: I/O-sink refusal (r55 C1 preserved) + fail-closed unknowns     #
# --------------------------------------------------------------------------- #

_SINK_TOKENS = [
    "disk",
    "disk:64",
    "DISK:64",
    "ext",
    "ext:0",
    "tinyfs",
    "tinyfs:6767",
    "TINYFS:64",
    "rdma",
    "rdma:0",
    "shm",
]


@pytest.mark.parametrize("token", _SINK_TOKENS)
def test_tinygrad_io_sink_bases_refused(token: str) -> None:
    """Every file/socket/fabric sink base is refused, index suffix or not.

    ``disk:64`` is grammar-valid, so only the sink exclude stands between it and
    a DISK file write -- this is the r55 C1 close carried into the r57 gate.
    """

    assert _sanitize_artifact_device_token(token, "tinygrad") is None
    assert _resolve_portable_device("tinygrad", token, None) is None


def test_sink_exclude_precedes_runtime_membership() -> None:
    """Ordering pin: ``disk`` is refused even though the runtime NAMES it.

    ``DISK``/``TINYFS``/``RDMA`` appear in ``Device._devices``, so a
    membership-first gate would admit them; the sink exclude must run first.
    """

    pytest.importorskip("tinygrad")
    runtime_bases = _tinygrad_runtime_device_bases()
    named_sinks = runtime_bases & _TINYGRAD_IO_SINK_BASES
    assert "disk" in named_sinks, "expected the installed runtime to name DISK"
    for base in named_sinks:
        assert _sanitize_artifact_device_token(base, "tinygrad") is None
        assert _sanitize_artifact_device_token(f"{base.upper()}:0", "tinygrad") is None


@pytest.mark.parametrize("token", ["foobar", "foobar:0", "zzz", "notadevice:3"])
def test_unknown_grammar_valid_bases_fail_closed(token: str) -> None:
    """A grammar-valid base the runtime does not name refuses to the default."""

    assert _sanitize_artifact_device_token(token, "tinygrad") is None
    assert _resolve_portable_device("tinygrad", token, None) is None


def test_tinygrad_absent_fails_closed_without_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """With tinygrad unimportable the vocabulary is empty: refuse-to-default, no crash."""

    monkeypatch.setitem(sys.modules, "tinygrad.device", None)
    monkeypatch.setitem(sys.modules, "tinygrad", None)
    assert _tinygrad_runtime_device_bases() == frozenset()
    assert _sanitize_artifact_device_token("NV", "tinygrad") is None
    assert _resolve_portable_device("tinygrad", "CPU", None) is None
    # Trusted caller map_location is unaffected by runtime absence.
    assert _resolve_portable_device("tinygrad", "NV", "CPU") == "CPU"


# --------------------------------------------------------------------------- #
# (e) filesystem-sentinel end-to-end (tinygrad -- the live sec_2 hole)         #
# --------------------------------------------------------------------------- #


def _build_tinygrad_bundle(tmp_path: Path) -> Path:
    tinygrad = pytest.importorskip("tinygrad")
    Tensor = tinygrad.Tensor

    def model(x: object) -> object:
        return (x * 2.0 + 1.0).relu()

    x = Tensor(np.random.randn(2, 4).astype(np.float32))
    trace = tl.trace(model, x, backend="tinygrad")
    bundle = tmp_path / "tiny.tlspec"
    tl.save(trace, str(bundle))
    return bundle


def test_tinygrad_disk_device_creates_no_file(tmp_path: Path) -> None:
    """A ``disk:<abs path>`` device token on default ``tl.load()`` writes NO file."""

    base = _build_tinygrad_bundle(tmp_path)
    att = tmp_path / "att.tlspec"
    shutil.copytree(base, att)

    sentinel = tmp_path / "CREATED_BY_LOAD"
    victim = tmp_path / "victim.txt"
    original = b"IMPORTANT-VICTIM-DATA-" * 8
    victim.write_bytes(original)

    manifest_path = att / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    tensors = manifest["tensors"]
    tensors[0]["logical_device"] = f"disk:{sentinel}"
    tensors[0]["device_at_save"] = f"disk:{sentinel}"
    if len(tensors) > 1:
        tensors[1]["logical_device"] = f"disk:{victim}"
        tensors[1]["device_at_save"] = f"disk:{victim}"
    manifest_path.write_text(json.dumps(manifest))

    trace = tl.load(str(att))  # default action, no .run(), no flags

    assert trace is not None
    assert not sentinel.exists(), "attacker disk: device token created a file on load"
    assert victim.read_bytes() == original, "attacker disk: device token clobbered a file"


# --------------------------------------------------------------------------- #
# (f) source-scan tripwire: every non-torch codec routes the saved device      #
# --------------------------------------------------------------------------- #

_NON_TORCH_CODECS = [
    payload_codec.TinygradPayloadCodec,
    payload_codec.MlxPayloadCodec,
    payload_codec.PaddlePayloadCodec,
    payload_codec.JaxPayloadCodec,
    payload_codec.TFPayloadCodec,
]


@pytest.mark.parametrize("codec_cls", _NON_TORCH_CODECS, ids=lambda c: c.__name__)
def test_non_torch_from_numpy_routes_or_ignores_saved_device(codec_cls: type) -> None:
    """Every non-torch ``from_numpy`` ignores the saved device or routes the resolver.

    A codec that reads the artifact ``logical_device`` / ``device_at_save`` string
    MUST launder it through ``_resolve_portable_device``; a codec that never reads
    it is trivially safe. This bans a NEW codec from re-opening the sec_2 hole.
    """

    source = inspect.getsource(codec_cls.from_numpy)
    reads_artifact_device = '"logical_device"' in source or '"device_at_save"' in source
    if reads_artifact_device:
        assert "_resolve_portable_device" in source, (
            f"{codec_cls.__name__}.from_numpy reads the artifact device string without "
            "routing it through the shared closed-grammar resolver (sec_2 regression risk)"
        )
