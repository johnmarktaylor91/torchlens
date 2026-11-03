"""Regression tests for r18o (A5/M1): mapping-KEY tensors must not bypass the
scrub/blobify tensor-policy + blob-inventory belt.

Before the fix, a tensor embedded in a mapping KEY was copied verbatim into
``metadata.pkl`` while the identical tensor in a VALUE became a manifest-indexed
``BlobRef``. That produced two failure modes:

* a dense tensor key -> a bundle whose ``tensors`` / ``body_index`` manifest is
  empty yet whose metadata carries a raw tensor payload (silent inventory
  contradiction); and
* a policy-rejected tensor key (e.g. sparse-COO) -> a bundle that
  ``validate_tlspec`` accepts but ``tl.load`` refuses (``ArtifactError``),
  i.e. validate and load DISAGREE.

The producer-side fix refuses tensor-payload keys at save so that
``validate_tlspec`` and ``tl.load`` always agree: no asymmetric bundle can be
produced in the first place.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import ArtifactError


class _Add1(nn.Module):
    """Minimal traceable model."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return value + 1


def _trace_with_custom_attrs(custom_attributes: dict) -> tl.Trace:
    """Trace ``_Add1`` and stamp ``custom_attributes`` on its self-module log."""

    trace = tl.trace(_Add1(), torch.tensor([1.0]))
    trace._module_logs._dict["self"].custom_attributes = custom_attributes
    return trace


def _save(trace: tl.Trace, path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    trace.save(path, include_outs=False, include_grads=False)


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    "key",
    [
        pytest.param(frozenset({torch.tensor([2.0])}), id="tensor_in_frozenset_key"),
    ],
)
def test_tensor_payload_key_refused_at_save(tmp_path: Path, key) -> None:
    """A tensor embedded in (or inside) a mapping key is refused producer-side."""

    trace = _trace_with_custom_attrs({key: "key-payload"})
    path = tmp_path / "payload_key.tlspec"
    with pytest.raises(ArtifactError, match="as .or inside. a key"):
        _save(trace, path)
    # The asymmetric artifact must never be produced: nothing to validate/load.
    assert not path.exists()
