"""THE named calibration script for the transforms fidelity baselines (memo B10).

Regenerates ``tests/transforms_corpus/baselines.json`` from the vendored
fixture corpus under the SHIPPED SRP algorithm (never a prototype): O14
admissibility on the real resnet50 layer4 basis, per-cell L2-distortion
percentiles + RDM fidelities with the JL bound, the O1d SRP-vs-subsampling
comparison, and the O1e construction-equivalence cells. Pinned baselines are
regenerated ONLY through this script (a deliberate, reviewed change) — never
inside a merge gate.

Run from the repo root:

    python tools/regen_transforms_baselines.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tests.transforms_corpus.loader import (  # noqa: E402
    CORPUS_DIR,
    admissibility_stats,
    build_fixture,
    load_vendored,
    manifest_rows,
    resnet50_checkpoint_cached,
    resnet50_features,
    verify_vendored_integrity,
)
from torchlens.transforms import TransformContext, srp  # noqa: E402
from torchlens.transforms._helpers import (  # noqa: E402
    _condensed_corr_rdm,
    _condensed_l2_rdm,
    spearman,
)
from torchlens.transforms._srp_hash import ALGORITHM_VERSION, draw_positions  # noqa: E402

#: Pinned per-case cells: (site, n_components, seed).
CELLS = [
    ("layer4", 256, 0),
    ("layer4", 256, 1),
    ("layer4", 1024, 0),
    ("layer4", 1024, 1),
    ("layer4", 4096, 0),
]

#: O1d comparison widths (both <= D/8 at layer4's D=100352).
O1D_WIDTHS = (1024, 4096)

#: Deterministic coordinate-subsample seed (part of the gate definition).
SUBSAMPLE_SEED = 4242

#: O1e construction-equivalence cells run on avgpool (D=2048) so the
#: O(D*k) iid generation stays cheap; three seeds per construction.
O1E_SEEDS = (0, 1, 2)
O1E_WIDTH = 256


def jl_eps(n_rows: int, k: int) -> float:
    """Invert the dense-JL sizing bound: the eps that k rows buy for n rows."""

    low, high = 1e-4, 0.9999
    target = 4.0 * math.log(n_rows) / k
    for _ in range(80):
        mid = (low + high) / 2
        if mid * mid / 2 - mid**3 / 3 < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def distortion_stats(projected: torch.Tensor, base: torch.Tensor) -> tuple[float, float, float]:
    """(median, p95, max) relative pairwise-L2 distortion vs the base RDM."""

    distances = _condensed_l2_rdm(projected)
    keep = base > 0
    relative = (distances[keep] - base[keep]).abs() / base[keep]
    return (
        float(relative.median()),
        float(torch.quantile(relative, 0.95)),
        float(relative.max()),
    )


def subsample_coordinates(extent: int, k: int) -> torch.Tensor:
    """Deterministic distinct coordinate subsample (the O1d comparator)."""

    candidates = draw_positions(
        SUBSAMPLE_SEED,
        torch.zeros(3 * k, dtype=torch.int64),
        torch.arange(3 * k, dtype=torch.int64),
        extent,
    )
    return torch.unique(candidates)[:k]


def main() -> None:
    """Regenerate baselines.json from the fixture under the shipped algorithm."""

    if not resnet50_checkpoint_cached():
        raise SystemExit(
            "resnet50 IMAGENET1K_V2 checkpoint is not cached; the calibration "
            "runs where the real checkpoint exists (never downloads)."
        )
    verify_vendored_integrity()
    rows = manifest_rows()
    fixture = build_fixture(load_vendored())
    features = resnet50_features(fixture)
    layer4 = features["layer4"]
    avgpool = features["avgpool"]
    n = int(layer4.shape[0])

    admissibility = admissibility_stats(layer4, n_distinct=len(rows))
    if not admissibility["admissible"]:
        raise SystemExit(f"fixture fails its own O14 admissibility gate: {admissibility}")

    l2_full = _condensed_l2_rdm(layer4)
    corr_full = _condensed_corr_rdm(layer4, context="calibration")

    cells = []
    for site, k, seed in CELLS:
        spec = srp(k, seed=seed)
        projected = spec.apply(layer4, TransformContext(site_label=f"cal:{site}:{k}:{seed}"))
        median, p95, maximum = distortion_stats(projected, l2_full)
        cells.append(
            {
                "site": site,
                "n_components": k,
                "seed": seed,
                "construction": "very_sparse_fixed",
                "algorithm_version": ALGORITHM_VERSION,
                "median_distortion": median,
                "p95_distortion": p95,
                "max_distortion": maximum,
                "jl_eps": jl_eps(n, k),
                "l2_rdm_spearman": spearman(l2_full, _condensed_l2_rdm(projected)),
                "corr_rdm_spearman": spearman(
                    corr_full, _condensed_corr_rdm(projected, context="calibration")
                ),
            }
        )

    o1d = []
    extent = int(layer4.shape[1])
    for k in O1D_WIDTHS:
        coords = subsample_coordinates(extent, k)
        subsampled = layer4[:, coords]
        srp_projected = srp(k, seed=0).apply(layer4, TransformContext(site_label=f"o1d:{k}"))
        srp_rho = spearman(corr_full, _condensed_corr_rdm(srp_projected, context="calibration"))
        sub_rho = spearman(corr_full, _condensed_corr_rdm(subsampled, context="calibration"))
        o1d.append(
            {
                "n_components": k,
                "srp_corr_rdm_spearman": srp_rho,
                "subsample_corr_rdm_spearman": sub_rho,
                "margin": srp_rho - sub_rho,
                "subsample_seed": SUBSAMPLE_SEED,
            }
        )

    l2_avgpool = _condensed_l2_rdm(avgpool)
    o1e: dict[str, list[dict[str, float]]] = {}
    for construction in ("very_sparse_fixed", "iid_bernoulli"):
        entries = []
        for seed in O1E_SEEDS:
            spec = srp(O1E_WIDTH, seed=seed, construction=construction)
            projected = spec.apply(
                avgpool, TransformContext(site_label=f"o1e:{construction}:{seed}")
            )
            median, p95, _maximum = distortion_stats(projected, l2_avgpool)
            entries.append({"seed": seed, "median": median, "p95": p95})
        o1e[construction] = entries

    manifest_digest = hashlib.sha256((CORPUS_DIR / "MANIFEST.jsonl").read_bytes()).hexdigest()
    payload = {
        "schema": "tl_transforms_fidelity_baselines_v1",
        "fixture": {
            "n_distinct": len(rows),
            "rows": n,
            "manifest_sha256": manifest_digest,
            "builder": "loader.build_fixture defaults",
        },
        "feature_basis": "torchvision resnet50 IMAGENET1K_V2; bilinear+antialias 224; "
        "ImageNet normalization; layer4 flatten (O1e cells: avgpool)",
        "environment_disclosure": {
            "torch": torch.__version__,
            "note": "pins carry headroom; regenerate ONLY via this script",
        },
        "admissibility": admissibility,
        "cells": cells,
        "o1d": o1d,
        "o1e": {"n_components": O1E_WIDTH, "site": "avgpool", "runs": o1e},
    }
    out = CORPUS_DIR / "baselines.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    print(json.dumps(payload["admissibility"], indent=2))
    for cell in cells:
        print(
            f"k={cell['n_components']} seed={cell['seed']}: med "
            f"{cell['median_distortion']:.4f} p95 {cell['p95_distortion']:.4f} "
            f"max {cell['max_distortion']:.4f} (eps {cell['jl_eps']:.3f}) "
            f"l2rho {cell['l2_rdm_spearman']:.4f} corr_rho {cell['corr_rdm_spearman']:.4f}"
        )
    for row in o1d:
        print(
            f"O1d k={row['n_components']}: srp {row['srp_corr_rdm_spearman']:.4f} "
            f"vs subsample {row['subsample_corr_rdm_spearman']:.4f} "
            f"margin {row['margin']:+.4f}"
        )


if __name__ == "__main__":
    main()
