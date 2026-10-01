"""Corpus fixture loaders + the O14 admissibility instrument (memo B9/B10).

The vendored fast path is 32 distinct CC0/public-domain photographs
(licensing verified per image; see ``MANIFEST.jsonl`` + ``LICENSES.md``),
deterministically expanded to a 128-row fixture through seeded crops, flips,
and channel gains. The precision path is the panel's exact 128-image COCO
test2017 corpus as a download-on-demand pinned manifest
(``coco_manifest.jsonl``: URL + sha256 per image) — COCO's Flickr-varied
licensing is exactly why it cannot be vendored; the download runs in the
release job, never in PR smoke.

O14 admissibility (transforms memo decision 19): pinned corr-RDM baselines
are trusted only when the fixture ITSELF passes ``n_distinct >= 16``,
``effective rank >= 40``, and ``corr-distance CoV <= 0.12`` — measured on
the SAME feature basis the pinned comparisons use (real resnet50 layer4;
the thresholds are calibrated to trained-model features, and this checkout
measured raw-pixel and random-conv bases at eff-rank 6-12, far below any
usable bar). The one-photograph corpus behind the memo's retracted table
fails the gate by a wide margin, which is the point.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from torchlens._io._json import loads_bounded

CORPUS_DIR = Path(__file__).resolve().parent
IMAGES_DIR = CORPUS_DIR / "images"

#: Deterministic fixture-builder constants (part of the gate definition).
FIXTURE_ROWS = 128
FIXTURE_CROP = 96
FIXTURE_SEED = 7
LOAD_SIZE = 128

#: O14 admissibility thresholds (memo decision 19; calibrated on real
#: resnet50 layer4 features of this fixture: good 87.5 / 0.060, the
#: single-image control 11.6 / 0.293).
O14_MIN_DISTINCT = 16
O14_MIN_EFF_RANK = 40.0
O14_MAX_CORR_COV = 0.12


def manifest_rows() -> list[dict[str, Any]]:
    """Parse the vendored-image manifest (through the guarded JSON reader).

    Returns
    -------
    list[dict[str, Any]]
        One provenance row per vendored image.
    """

    rows = []
    for line in (CORPUS_DIR / "MANIFEST.jsonl").read_text().splitlines():
        if line.strip():
            rows.append(loads_bounded(line))
    return rows


def coco_manifest_rows() -> list[dict[str, Any]]:
    """Parse the pinned COCO precision-path manifest.

    Returns
    -------
    list[dict[str, Any]]
        128 rows of ``{file, url, sha256, bytes}``.
    """

    rows = []
    for line in (CORPUS_DIR / "coco_manifest.jsonl").read_text().splitlines():
        if line.strip():
            rows.append(loads_bounded(line))
    return rows


def verify_vendored_integrity() -> None:
    """Assert every vendored image matches its manifest sha256 (drift refuses)."""

    for row in manifest_rows():
        blob = (IMAGES_DIR / row["file"]).read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        assert digest == row["sha256"], (
            f"vendored corpus drift: {row['file']} hashes to {digest}, "
            f"manifest records {row['sha256']} -- regenerate deliberately or "
            "restore the file"
        )


def load_vendored(size: int = LOAD_SIZE) -> torch.Tensor:
    """Load the vendored images: center square crop, resize, [0, 1] floats.

    Parameters
    ----------
    size:
        Output side length.

    Returns
    -------
    torch.Tensor
        ``(n_images, 3, size, size)`` float32 batch in manifest order.
    """

    from PIL import Image

    out = []
    for row in manifest_rows():
        with Image.open(IMAGES_DIR / row["file"]) as image:
            image = image.convert("RGB")
            width, height = image.size
            side = min(width, height)
            image = image.crop(
                (
                    (width - side) // 2,
                    (height - side) // 2,
                    (width + side) // 2,
                    (height + side) // 2,
                )
            ).resize((size, size), Image.LANCZOS)
            out.append(
                torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(
                    size, size, 3
                )
            )
    return torch.stack(out).permute(0, 3, 1, 2).float() / 255.0


def build_fixture(
    source: torch.Tensor,
    n_rows: int = FIXTURE_ROWS,
    crop: int = FIXTURE_CROP,
    seed: int = FIXTURE_SEED,
) -> torch.Tensor:
    """Expand source images to the deterministic augmented fixture.

    Row ``r`` takes image ``r % n`` with a seeded crop offset, horizontal
    flip bit, and per-channel gain in ``[0.8, 1.2]`` — the expansion is what
    makes the 128-row admissibility statistics computable from 16-32
    distinct photographs (memo decision 19's ladder measured fidelity FLAT
    from 128 down to 16 distinct images and out of band at 8).

    Parameters
    ----------
    source:
        ``(n, 3, H, W)`` image batch (H, W >= crop).
    n_rows:
        Fixture row count.
    crop:
        Square crop side.
    seed:
        Augmentation seed.

    Returns
    -------
    torch.Tensor
        ``(n_rows, 3, crop, crop)`` float32 fixture.
    """

    generator = torch.Generator().manual_seed(seed)
    n = source.shape[0]
    height, width = int(source.shape[2]), int(source.shape[3])
    rows_out = []
    for row in range(n_rows):
        image = source[row % n]
        offset_y = int(torch.randint(0, height - crop + 1, (1,), generator=generator))
        offset_x = int(torch.randint(0, width - crop + 1, (1,), generator=generator))
        patch = image[:, offset_y : offset_y + crop, offset_x : offset_x + crop]
        if int(torch.randint(0, 2, (1,), generator=generator)):
            patch = patch.flip(-1)
        gains = 0.8 + 0.4 * torch.rand(3, 1, 1, generator=generator)
        rows_out.append((patch * gains).clamp(0.0, 1.0))
    return torch.stack(rows_out)


def effective_rank(features: torch.Tensor) -> float:
    """Participation-ratio effective rank of a row-feature matrix.

    Parameters
    ----------
    features:
        ``(n, d)`` feature matrix.

    Returns
    -------
    float
        ``(sum ev)^2 / sum(ev^2)`` over the centered row-gram eigenvalues.
    """

    centered = (features - features.mean(dim=0, keepdim=True)).double()
    eigenvalues = torch.linalg.eigvalsh(centered @ centered.T).clamp_min(0)
    return float((eigenvalues.sum() ** 2 / (eigenvalues**2).sum()).item())


def corr_distance_cov(features: torch.Tensor) -> float:
    """Coefficient of variation of the condensed correlation-distance RDM.

    Parameters
    ----------
    features:
        ``(n, d)`` feature matrix.

    Returns
    -------
    float
        ``std / mean`` of the pairwise correlation distances. Diverse
        corpora concentrate (low CoV); augmentations of one photograph
        split into near-0 same-source and near-1 cross pairs (high CoV).
    """

    centered = features.double() - features.double().mean(dim=1, keepdim=True)
    normalized = centered / torch.linalg.vector_norm(centered, dim=1, keepdim=True)
    similarity = normalized @ normalized.T
    n = features.shape[0]
    rows, cols = torch.triu_indices(n, n, offset=1)
    distances = 1.0 - similarity[rows, cols]
    return float(distances.std() / distances.mean())


def admissibility_stats(features: torch.Tensor, n_distinct: int) -> dict[str, Any]:
    """O14: the fixture's own statistics plus per-threshold verdicts.

    Parameters
    ----------
    features:
        ``(n, d)`` fixture features ON THE BASIS the pinned comparisons use.
    n_distinct:
        Number of distinct source photographs behind the fixture.

    Returns
    -------
    dict[str, Any]
        Stats, thresholds, and the combined ``admissible`` verdict.
    """

    rank = effective_rank(features)
    cov = corr_distance_cov(features)
    return {
        "n_distinct": n_distinct,
        "eff_rank": rank,
        "corr_cov": cov,
        "thresholds": {
            "min_distinct": O14_MIN_DISTINCT,
            "min_eff_rank": O14_MIN_EFF_RANK,
            "max_corr_cov": O14_MAX_CORR_COV,
        },
        "admissible": (
            n_distinct >= O14_MIN_DISTINCT and rank >= O14_MIN_EFF_RANK and cov <= O14_MAX_CORR_COV
        ),
    }


def resnet50_checkpoint_cached() -> bool:
    """Whether the real resnet50 IMAGENET1K_V2 checkpoint is already local.

    The fidelity gates never download: they run where the checkpoint cache
    exists (this is the realism tier's offline discipline).

    Returns
    -------
    bool
        True when the torchvision checkpoint file is present.
    """

    hub = Path(torch.hub.get_dir()) / "checkpoints" / "resnet50-11ad3fa6.pth"
    return hub.exists()


def resnet50_features(fixture: torch.Tensor) -> dict[str, torch.Tensor]:
    """Real resnet50 (IMAGENET1K_V2) layer4 + avgpool features of a fixture.

    The fixture is bilinearly resized (antialiased) to 224 and normalized
    with the ImageNet statistics — the deterministic stand-in for the
    weight's official preprocessing on an in-memory batch.

    Parameters
    ----------
    fixture:
        ``(n, 3, H, W)`` float batch in ``[0, 1]``.

    Returns
    -------
    dict[str, torch.Tensor]
        ``{"layer4": (n, 100352), "avgpool": (n, 2048)}`` float32 features.
    """

    import torch.nn.functional as functional
    from torchvision.models import ResNet50_Weights, resnet50

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    resized = functional.interpolate(
        fixture, size=(224, 224), mode="bilinear", align_corners=False, antialias=True
    )
    batch = (resized - mean) / std
    store: dict[str, torch.Tensor] = {}
    modules = dict(model.named_modules())
    handles = [
        modules["layer4"].register_forward_hook(
            lambda module, args, output: store.__setitem__("layer4", output.detach())
        ),
        modules["avgpool"].register_forward_hook(
            lambda module, args, output: store.__setitem__("avgpool", output.detach())
        ),
    ]
    layer4, avgpool = [], []
    with torch.no_grad():
        for start in range(0, batch.shape[0], 16):
            model(batch[start : start + 16])
            layer4.append(store["layer4"].flatten(1).float())
            avgpool.append(store["avgpool"].flatten(1).float())
    for handle in handles:
        handle.remove()
    return {"layer4": torch.cat(layer4), "avgpool": torch.cat(avgpool)}
