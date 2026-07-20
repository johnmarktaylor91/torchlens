"""r55 CLASS 1 immunizer -- portable device strings are closed-grammar data, not sinks.

r54 ``sec_2`` (HIGH): a tinygrad ``.tlspec`` carries per-blob ``logical_device`` /
``device_at_save`` strings in plaintext ``manifest.json``; the codec passed them
straight into ``tinygrad.Tensor(device=<attacker str>)`` where the ``disk:<path>``
backend writes an attacker-named file on a default ``tl.load(path)``. The whole
class is closed by ONE shared closed-grammar allowlist
(``_io/_artifact_strings._resolve_portable_device``) every non-torch codec routes
an artifact-supplied device token through:

* a filesystem-sentinel end-to-end test (tinygrad ``disk:`` token creates NO file);
* a token-grammar table (every legit backend device string admitted unchanged,
  every path/url/scheme token refused to the runtime default);
* a source-scan tripwire: every non-torch ``from_numpy`` either ignores the saved
  device string or routes it through the shared resolver -- so a future codec
  inherits the refusal by construction.
"""

from __future__ import annotations

import inspect
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import torchlens as tl
from torchlens._io import payload_codec
from torchlens._io._artifact_strings import (
    _resolve_portable_device,
    _sanitize_artifact_device_token,
)

pytestmark = pytest.mark.smoke


# --------------------------------------------------------------------------- #
# (a) token grammar: legit admitted, attacker path/url/scheme refused          #
# --------------------------------------------------------------------------- #

_LEGIT_TOKENS = [
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


@pytest.mark.parametrize("token", _LEGIT_TOKENS)
def test_legit_device_tokens_admitted_unchanged(token: str) -> None:
    """Every legitimate backend device string is admitted and returned unchanged."""

    assert _sanitize_artifact_device_token(token) == token


@pytest.mark.parametrize("token", _ATTACKER_TOKENS)
def test_attacker_device_tokens_refused_to_default(token: str) -> None:
    """Every path/url/scheme/malformed token resolves to the runtime default (None)."""

    assert _sanitize_artifact_device_token(token) is None


def test_unknown_and_empty_tokens_resolve_default() -> None:
    """The sentinel ``unknown``/empty strings and non-strings resolve to default."""

    assert _sanitize_artifact_device_token("unknown") is None
    assert _sanitize_artifact_device_token("") is None
    assert _sanitize_artifact_device_token(None) is None
    assert _sanitize_artifact_device_token(123) is None


def test_resolve_portable_device_trusts_caller_map_location() -> None:
    """A caller ``map_location`` is trusted input and passes through unchanged."""

    assert _resolve_portable_device("tinygrad", "disk:/tmp/x", "cuda") == "cuda"
    # Artifact token is gated only when no trusted caller value is supplied.
    assert _resolve_portable_device("tinygrad", "disk:/tmp/x", None) is None
    assert _resolve_portable_device("tinygrad", "CUDA:0", None) == "CUDA:0"


# --------------------------------------------------------------------------- #
# (b) filesystem-sentinel end-to-end (tinygrad -- the live sec_2 hole)         #
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
# (c) source-scan tripwire: every non-torch codec routes the saved device      #
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
