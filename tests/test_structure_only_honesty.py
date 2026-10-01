"""Honesty guards for structure-only capture (L7a memo sec 3.4/9.5-9.7).

G2 digest domain separation (meta can never collide with real), G3 render
honesty (banners on every human surface), the default-path teaching
enrichment of unsupported_tensor_variant (ships regardless of D8), and the
baseline entry-gate pins that become the D8 carve-through tests.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._robustness import UnsupportedTensorVariantError
from torchlens.options import CaptureOptions

smoke = pytest.mark.smoke


class TwoLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


def _structure_trace():
    return tl.trace(TwoLayer(), torch.randn(2, 4), capture=CaptureOptions(structure_only=True))


# ---------------------------------------------------------------------------
# G2: digest domain separation
# ---------------------------------------------------------------------------


@smoke
def test_meta_and_real_tensors_of_identical_geometry_never_collide() -> None:
    """hash.py digests meta under a DISTINCT domain tag: a structure-only
    artifact can never masquerade as a value-bearing one BY CONSTRUCTION."""

    real = torch.zeros(3, 4)
    with torch.device("meta"):
        meta = torch.zeros(3, 4)
    assert tl.hash.content(real) != tl.hash.content(meta)
    # And two meta tensors of identical geometry DO agree (the digest is a
    # real digest of the geometry, never None/absent).
    with torch.device("meta"):
        meta_twin = torch.zeros(3, 4)
    assert tl.hash.content(meta) == tl.hash.content(meta_twin)


@smoke
def test_structure_digests_always_computed() -> None:
    log = _structure_trace()
    digest = tl.hash.trace(log)
    assert isinstance(digest, str) and len(digest) == 64


# ---------------------------------------------------------------------------
# G3: render honesty banners
# ---------------------------------------------------------------------------


@smoke
def test_summary_carries_the_structure_only_banner() -> None:
    log = _structure_trace()
    text = log.summary()
    assert "structure-only capture" in text
    assert "HYPOTHESES" in text
    # Default path: no banner.
    plain = tl.trace(TwoLayer(), torch.randn(2, 4)).summary()
    assert "structure-only" not in plain


@smoke
def test_profile_repr_carries_the_banner() -> None:
    log = _structure_trace()
    profile = log.profile()
    assert profile.structure_only is True
    assert "structure-only capture" in repr(profile)
    # Measured-value columns render honestly: nothing presents a hypothesis
    # figure as "measured". Boundary pseudo-rows read not_applicable (A07:
    # they own no compute by the identity partition) -- also never "measured".
    honesty = profile.honesty()
    # F09 (costreport D9): structure-only shape-derived cells are labeled
    # HYPOTHESIS -- the exact banner-vs-labels mismatch fix. Still never
    # "measured", never "estimated"-presented-as-fact.
    assert set(honesty["flops"].unique()) <= {"hypothesis", "unknown", "not_applicable"}
    assert set(honesty["activation_memory"].unique()) <= {
        "hypothesis",
        "unknown",
        "not_applicable",
    }
    assert "measured" not in set(honesty["flops"].unique())
    assert "hypothesis" in set(honesty["flops"].unique())


@smoke
def test_explain_carries_the_hypothesis_line() -> None:
    log = _structure_trace()
    text = tl.report.explain(log)
    assert "Structure-only capture" in text
    assert "HYPOTHESES" in text


# ---------------------------------------------------------------------------
# Default-path teaching enrichment + baseline gate pins (memo 1.4-B / 9.7)
# ---------------------------------------------------------------------------


def _meta_model() -> nn.Module:
    with torch.device("meta"):
        return nn.Linear(4, 4)


@smoke
def test_meta_init_model_still_refuses_at_the_gate_with_enriched_teaching() -> None:
    """Baseline pin (becomes the D8 carve-through test): the DEFAULT path and
    the D8-default flag-on path both refuse meta state unchanged, now naming
    the user callsite and pointing at the structure-only contract."""

    with pytest.raises(UnsupportedTensorVariantError) as excinfo:
        tl.trace(_meta_model(), torch.randn(2, 4))
    err = excinfo.value
    assert err.fields["code"] == "unsupported_tensor_variant"
    # Structured offenses unchanged (the first-offense break stays).
    assert any("meta tensor" in offense["name"] for offense in err.fields["offenses"])
    # Enrichment: the USER callsite (this file) and the mode pointer.
    assert err.file_path == __file__
    assert "structure_only_capabilities.md" in str(err)
    assert f"{__file__}:" in str(err)


@smoke
def test_meta_model_real_input_mixed_cell_refuses_post_flip() -> None:
    """Entry-matrix E-3 under the D8 GRANT: structure_only=True admits meta
    only with a UNIFORM substrate — a meta model with a REAL input is a mixed
    cell and refuses typed with the substrate-mismatch code, naming sides."""

    with pytest.raises(Exception) as excinfo:
        tl.trace(
            _meta_model(),
            torch.randn(2, 4),
            capture=CaptureOptions(structure_only=True),
        )
    assert excinfo.value.fields["code"] == "structure_only_substrate_mismatch"
    assert excinfo.value.fields["meta_side"]
    assert excinfo.value.fields["real_side"]


@smoke
def test_real_model_meta_input_mixed_cell_refuses_post_flip() -> None:
    """Entry-matrix E-4 under the D8 GRANT: meta INPUTS against real state
    are the other mixed direction — the same typed substrate-mismatch code."""

    with torch.device("meta"):
        meta_input = torch.zeros(2, 4)
    with pytest.raises(Exception) as excinfo:
        tl.trace(TwoLayer(), meta_input, capture=CaptureOptions(structure_only=True))
    assert excinfo.value.fields["code"] == "structure_only_substrate_mismatch"


@smoke
def test_fake_tensor_still_refuses_under_the_flag() -> None:
    """Tamper pin t2 (D8-default form): the flag narrows NOTHING at the
    variant gate — FakeTensor offenses refuse under structure_only=True."""

    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        fake_input = torch.empty(2, 4)
    with pytest.raises(UnsupportedTensorVariantError):
        tl.trace(TwoLayer(), fake_input, capture=CaptureOptions(structure_only=True))
