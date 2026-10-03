"""B8 R62: a captured function signature must not leak host paths or, under
``include_source=False``, source-derived default VALUES.

``inspect.Signature.__str__`` renders every parameter default via
``repr(default)``, so a default like ``cfg='/home/user/x.yaml'`` embedded an
absolute host path verbatim and a ``token='SECRET'`` default shipped a
source-derived value -- both surviving ``include_source=False``, the one channel
that defeated the shipped "no ``$HOME``/username ever reaches the bundle" guarantee.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io._source_privacy import _scrub_signature_string

_CANARY_DEFAULT = "canary-sig-default-value-r62"
_HOME = os.path.expanduser("~")
_CFG = f"{_HOME}/private/torchlens_r62_config.yaml"


class _SignatureModel(nn.Module):
    """A forward whose defaults embed a secret string and an absolute host path."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor, opt: str = _CANARY_DEFAULT, cfg: str = _CFG) -> torch.Tensor:
        return self.lin(x)


def _bundle_bytes(bundle: Path) -> bytes:
    return (bundle / "metadata.pkl").read_bytes() + (bundle / "manifest.json").read_bytes()


def test_home_path_never_reaches_signature_regardless_of_flag(tmp_path: Path) -> None:
    trace = tl.trace(_SignatureModel(), torch.randn(2, 4))
    for flag in (True, False):
        bundle = tmp_path / f"sig_{flag}.tlspec"
        tl.save(trace, bundle, overwrite=True, include_source=flag)
        blob = _bundle_bytes(bundle)
        assert _HOME.encode() not in blob, f"$HOME leaked at include_source={flag}"
        assert _CFG.encode() not in blob, f"absolute config path leaked at include_source={flag}"


def test_source_derived_default_dropped_when_source_excluded(tmp_path: Path) -> None:
    trace = tl.trace(_SignatureModel(), torch.randn(2, 4))
    bundle = tmp_path / "no_source.tlspec"
    tl.save(trace, bundle, overwrite=True, include_source=False)
    blob = _bundle_bytes(bundle)
    assert _CANARY_DEFAULT.encode() not in blob, (
        "source-derived signature default survived include_source=False"
    )


def test_signature_shape_preserved_when_defaults_stubbed() -> None:
    sig = "(self, x, opt='canary-sig-default-value-r62', cfg='/home/u/x.yaml', n=5)"
    stubbed = _scrub_signature_string(sig, include_source=False)
    # Parameter names / shape survive; every default becomes '...'.
    assert stubbed == "(self, x, opt=..., cfg=..., n=...)"


def test_signature_abs_path_relativized_with_source_kept() -> None:
    sig = "(self, cfg='/home/u/secret/config.yaml', url='https://example.com/x')"
    scrubbed = _scrub_signature_string(sig, include_source=True)
    assert "/home/u/secret" not in scrubbed
    assert "config.yaml" in scrubbed
    # A non-path literal (a URL) is left intact.
    assert "https://example.com/x" in scrubbed


def test_signature_nested_default_bracket_is_not_split() -> None:
    sig = "(self, opts={'a': 1, 'b': 2}, xs=(1, 2, 3))"
    stubbed = _scrub_signature_string(sig, include_source=False)
    assert stubbed == "(self, opts=..., xs=...)"


@pytest.mark.smoke
def test_func_signature_scrubbed_in_manifest_sites(tmp_path: Path) -> None:
    trace = tl.trace(_SignatureModel(), torch.randn(2, 4))
    bundle = tmp_path / "sites.tlspec"
    tl.save(trace, bundle, overwrite=True, include_source=False)
    manifest = json.loads((bundle / "manifest.json").read_text())
    text = json.dumps(manifest)
    assert _CANARY_DEFAULT not in text
    assert _HOME not in text
