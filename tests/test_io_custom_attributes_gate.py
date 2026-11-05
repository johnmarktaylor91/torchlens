"""R62: gate + disclose the harvested module instance-attribute channel.

``model_prep`` harvests every public, non-callable module instance attribute into
``Module.custom_attributes`` and portable saves persisted the whole channel
verbatim with no opt-out and no disclosure — tokens, private paths, usernames,
and large containers stored as module attributes rode along in the shareable
artifact invisibly (disputed-r2 b8). The converged fix is additive: an
``include_custom_attributes=`` kwarg on ``Trace.save`` defaulting to ``True``
(historical behavior preserved), a manifest disclosure naming the channel, and
NEVER rewriting user values. These canaries plant instance attributes — the
exact channel the earlier canary sweeps missed — and assert both halves.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl

_CANARY_TOKEN = "hf_fake_canary_token_2468"
_CANARY_PATH = "/home/fake-owner/secret-project/config.yaml"
_CANARY_OWNER = "fake-owner-name"


class _AttrModel(nn.Module):
    """Model whose submodule carries planted public instance attributes."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 3)
        self.api_token = _CANARY_TOKEN
        self.config_path = _CANARY_PATH
        self.owner = _CANARY_OWNER
        self.big_list = list(range(500))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def _capture() -> tl.Trace:
    return tl.trace(_AttrModel().eval(), torch.randn(2, 4))


def _bundle_bytes(bundle_path: Path) -> bytes:
    return b"".join(
        member.read_bytes() for member in sorted(bundle_path.rglob("*")) if member.is_file()
    )


def _manifest_disclosure(bundle_path: Path) -> dict:
    manifest = json.loads((bundle_path / "manifest.json").read_text())
    assert "custom_attributes_disclosure" in manifest, (
        "every save must disclose the custom_attributes channel in the manifest"
    )
    return manifest["custom_attributes_disclosure"]


def test_streaming_save_warns_and_discloses(tmp_path: Path) -> None:
    """R62 REOPEN: streaming to_disk goes through the same belt as Trace.save.

    Fail-before (sol, round 6): ``storage=tl.to_disk(...)`` hardcoded
    ``included=True``, emitted ZERO warnings, and offered no opt-out -- the
    canary token shipped in the streamed bundle silently, reopening the exact
    class the round-5 Trace.save fix closed.
    """

    from torchlens.errors._base import TorchLensWarning

    bundle = tmp_path / "streamed.tl"
    with pytest.warns(TorchLensWarning, match="custom module attribute"):
        tl.trace(_AttrModel().eval(), torch.randn(2, 4), storage=tl.to_disk(bundle))

    disclosure = _manifest_disclosure(bundle)
    assert disclosure["included"] is True
    assert "api_token" in disclosure["top_level_keys"]
    assert _CANARY_TOKEN.encode() in _bundle_bytes(bundle)


def test_streaming_optout_withholds_values(tmp_path: Path) -> None:
    """``to_disk(..., include_custom_attributes=False)`` withholds the channel."""

    bundle = tmp_path / "held.tl"
    tl.trace(
        _AttrModel().eval(),
        torch.randn(2, 4),
        storage=tl.to_disk(bundle, include_custom_attributes=False),
    )

    raw = _bundle_bytes(bundle)
    for canary in (_CANARY_TOKEN, _CANARY_PATH, _CANARY_OWNER):
        assert canary.encode() not in raw, f"canary {canary!r} leaked past the streaming opt-out"

    disclosure = _manifest_disclosure(bundle)
    assert disclosure["included"] is False
    assert "api_token" in disclosure["top_level_keys"]
