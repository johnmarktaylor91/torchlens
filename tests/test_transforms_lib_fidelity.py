"""Corpus-scoped fidelity oracles on REAL features (memo B10; lane F19).

O14 (fixture admissibility, discriminative in both directions), O1b (the one
theorem-shaped empirical bound), O1c (pinned per-cell baselines with
headroom, regenerated ONLY via tools/regen_transforms_baselines.py), O1d
(the self-baselining SRP-vs-subsampling comparative gate), and O1e
(construction equivalence). All run on real resnet50 IMAGENET1K_V2 features
of the vendored fixture — the O14 thresholds are calibrated to trained-model
features (raw-pixel and random-conv bases measured at eff-rank 6-12 on this
checkout, far below any usable bar) — and SKIP where the checkpoint is not
already cached (the realism tier never downloads).

Headroom policy (deliberate, documented): median/p95 within +/-15% relative
of the pin, max within +/-35%, Spearman rhos within +/-0.05 absolute; the
JL bound is asserted ABSOLUTELY (max distortion <= eps), not against a pin;
O1d asserts the ordering with at least half the pinned margin in the same
run (self-baselining, cannot go stale); O1e bounds the construction ratio in
[0.7, 1.43] -- generous, and still an order of magnitude tighter than any
construction bug the panel observed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.transforms_corpus.loader import (
    CORPUS_DIR,
    admissibility_stats,
    build_fixture,
    load_vendored,
    manifest_rows,
    resnet50_checkpoint_cached,
    resnet50_features,
)
from torchlens._io._json import loads_bounded
from torchlens.transforms import TransformContext, srp
from torchlens.transforms._helpers import _condensed_corr_rdm, _condensed_l2_rdm, spearman
from torchlens.transforms._srp_hash import draw_positions

# slow tier, honestly: one real resnet50 forward per fixture (two for the
# discriminative O14 pair) plus D=100352 projections measured 35-60s wall
# under parallel box load. The corpus-free O1a/O1e/O1f identities run in
# smoke via test_transforms_lib_srp.py.
pytestmark = [pytest.mark.slow, pytest.mark.real_model]

pytest.importorskip("PIL", reason="the corpus loader decodes JPEG via PIL")
pytest.importorskip("torchvision", reason="the fidelity basis is real resnet50")

if not resnet50_checkpoint_cached():  # pragma: no cover - cache-dependent venue
    pytest.skip(
        "resnet50 IMAGENET1K_V2 checkpoint not cached; fidelity gates run "
        "where the real checkpoint exists (never download)",
        allow_module_level=True,
    )

BASELINES = loads_bounded((CORPUS_DIR / "baselines.json").read_text())


@pytest.fixture(scope="module")
def fixture_features() -> dict[str, torch.Tensor]:
    """Real layer4/avgpool features of the admissible fixture (one forward)."""

    return resnet50_features(build_fixture(load_vendored()))


def _distortion_stats(projected: torch.Tensor, base: torch.Tensor) -> tuple[float, float, float]:
    """(median, p95, max) relative pairwise-L2 distortion vs the base RDM."""

    distances = _condensed_l2_rdm(projected)
    keep = base > 0
    relative = (distances[keep] - base[keep]).abs() / base[keep]
    return (
        float(relative.median()),
        float(torch.quantile(relative, 0.95)),
        float(relative.max()),
    )


def test_o14_fixture_is_admissible_on_the_pinned_basis(
    fixture_features: dict[str, torch.Tensor],
) -> None:
    """The shipped fixture passes its own O14 gate with margin."""

    stats = admissibility_stats(fixture_features["layer4"], n_distinct=len(manifest_rows()))
    assert stats["admissible"], stats
    assert stats["eff_rank"] >= 40.0
    assert stats["corr_cov"] <= 0.12


def test_o14_degenerate_corpus_fails_the_gate() -> None:
    """128 augmentations of ONE photograph fail eff-rank AND CoV.

    The retracted one-photograph table's corpus scored 15.2 / 0.265 on the
    panel's basis; this control reproduces the failure (measured 11.6 /
    0.293 at calibration) — the gate discriminates, it does not decorate.
    """

    degenerate = build_fixture(load_vendored()[0:1])
    features = resnet50_features(degenerate)["layer4"]
    stats = admissibility_stats(features, n_distinct=1)
    assert not stats["admissible"]
    assert stats["eff_rank"] < 40.0
    assert stats["corr_cov"] > 0.12


def test_o1b_and_o1c_pinned_cells_hold(fixture_features: dict[str, torch.Tensor]) -> None:
    """Every pinned cell: JL bound holds absolutely; stats within headroom."""

    layer4 = fixture_features["layer4"]
    l2_full = _condensed_l2_rdm(layer4)
    corr_full = _condensed_corr_rdm(layer4, context="o1c")
    for cell in BASELINES["cells"]:
        k, seed = cell["n_components"], cell["seed"]
        projected = srp(k, seed=seed).apply(
            layer4, TransformContext(site_label=f"cal:layer4:{k}:{seed}")
        )
        median, p95, maximum = _distortion_stats(projected, l2_full)
        # O1b: the theorem-shaped empirical bound, asserted absolutely. Its
        # docstring caveat: this is a high-probability bound, not almost-sure
        # -- a rare exceedance under FRESH seeds is not automatically a code
        # bug; under the PINNED seeds it is deterministic.
        assert maximum <= cell["jl_eps"], (k, seed, maximum, cell["jl_eps"])
        assert median == pytest.approx(cell["median_distortion"], rel=0.15)
        assert p95 == pytest.approx(cell["p95_distortion"], rel=0.15)
        assert maximum == pytest.approx(cell["max_distortion"], rel=0.35)
        l2_rho = spearman(l2_full, _condensed_l2_rdm(projected))
        corr_rho = spearman(corr_full, _condensed_corr_rdm(projected, context="o1c"))
        assert l2_rho == pytest.approx(cell["l2_rdm_spearman"], abs=0.05)
        assert corr_rho == pytest.approx(cell["corr_rdm_spearman"], abs=0.05)


def test_o1d_srp_beats_coordinate_subsampling_at_strong_compression(
    fixture_features: dict[str, torch.Tensor],
) -> None:
    """Self-baselining comparative gate: SRP's corr-RDM fidelity wins.

    Computed from the same data in the same run, so it cannot go stale; the
    pinned margins only set the floor (half the calibrated margin).
    """

    layer4 = fixture_features["layer4"]
    extent = int(layer4.shape[1])
    corr_full = _condensed_corr_rdm(layer4, context="o1d")
    for row in BASELINES["o1d"]:
        k = row["n_components"]
        assert k <= extent // 8  # the strong-compression regime the claim is scoped to
        candidates = draw_positions(
            row["subsample_seed"],
            torch.zeros(3 * k, dtype=torch.int64),
            torch.arange(3 * k, dtype=torch.int64),
            extent,
        )
        coords = torch.unique(candidates)[:k]
        sub_rho = spearman(corr_full, _condensed_corr_rdm(layer4[:, coords], context="o1d"))
        projected = srp(k, seed=0).apply(layer4, TransformContext(site_label=f"o1d:{k}"))
        srp_rho = spearman(corr_full, _condensed_corr_rdm(projected, context="o1d"))
        assert srp_rho > sub_rho + 0.5 * row["margin"], (k, srp_rho, sub_rho)


def test_o1e_constructions_are_geometrically_equivalent(
    fixture_features: dict[str, torch.Tensor],
) -> None:
    """vsf and iid distortion percentiles agree within the tolerance band.

    Each construction is the other's oracle with no calibrated constant
    beyond the generous [0.7, 1.43] ratio band; a broken construction moves
    these by integer factors.
    """

    avgpool = fixture_features["avgpool"]
    l2_full = _condensed_l2_rdm(avgpool)
    k = BASELINES["o1e"]["n_components"]
    medians: dict[str, list[float]] = {}
    p95s: dict[str, list[float]] = {}
    for construction in ("very_sparse_fixed", "iid_bernoulli"):
        medians[construction] = []
        p95s[construction] = []
        for run in BASELINES["o1e"]["runs"][construction]:
            projected = srp(k, seed=run["seed"], construction=construction).apply(
                avgpool, TransformContext(site_label=f"o1e:{construction}:{run['seed']}")
            )
            median, p95, _maximum = _distortion_stats(projected, l2_full)
            assert median == pytest.approx(run["median"], rel=0.15)
            assert p95 == pytest.approx(run["p95"], rel=0.15)
            medians[construction].append(median)
            p95s[construction].append(p95)
    for stats in (medians, p95s):
        fixed = sum(stats["very_sparse_fixed"]) / len(stats["very_sparse_fixed"])
        bernoulli = sum(stats["iid_bernoulli"]) / len(stats["iid_bernoulli"])
        ratio = bernoulli / fixed
        assert 0.7 <= ratio <= 1.43, stats


def test_baselines_were_generated_from_this_fixture() -> None:
    """The pinned baselines name THIS manifest (regeneration is deliberate)."""

    import hashlib

    manifest_digest = hashlib.sha256((CORPUS_DIR / "MANIFEST.jsonl").read_bytes()).hexdigest()
    assert BASELINES["fixture"]["manifest_sha256"] == manifest_digest, (
        "baselines.json was generated from a different vendored manifest; "
        "regenerate deliberately via tools/regen_transforms_baselines.py"
    )
    assert BASELINES["admissibility"]["admissible"] is True
    assert Path(CORPUS_DIR / "LICENSES.md").exists()
