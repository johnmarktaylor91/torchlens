"""Subprocess entry point for isolated surface-oracle snapshot generation.

The surface goldens freeze CONSTRUCTION-TIME behavior too: a model built
after torch has been wrapped can bake live wrappers into its state (the
SF-53 class — ctor-read ``F.<op>`` defaults, string-activation resolution),
which forks the snapshot depending on what ran earlier in the pytest
session. This worker runs in a fresh interpreter and constructs EVERY
requested model before the first capture, so no ctor ever sees wrapped
torch — for the regen path and the enforce path alike (b10 R78-1, the
capture-oracle isolation pattern batched to one process for the whole axis
family).
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence

# Pin the CPU kernel paths BEFORE torch loads (ATen reads its capability and
# MKL its reproducibility branch at first use), exactly as the sibling
# capture-oracle worker does. The dumps hash raw float bytes, and conv/linear
# results otherwise follow the host's ISA: goldens recorded on AVX-512 Xeons
# failed the enforcing Tests row on GitHub's runners with identical inputs and
# code. AVX2 is the common floor of every x86 runner; MKL_CBWR=COMPATIBLE is
# MKL's cross-processor reproducible branch.
os.environ["ATEN_CPU_CAPABILITY"] = "avx2"
os.environ["MKL_CBWR"] = "COMPATIBLE"


def main(argv: Sequence[str] | None = None) -> int:
    """Print a JSON mapping of axis -> canonical surface dump on stdout.

    Parameters
    ----------
    argv:
        Optional explicit argument sequence (one or more model axes).

    Returns
    -------
    int
        Process exit status.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_axes", nargs="+")
    args = parser.parse_args(argv)

    # Pin the execution environment the fixed seeds alone do not cover:
    # thread count and kernel selection both steer float reduction order,
    # and the surface dumps embed sha256 of raw tensor bytes. The sibling
    # capture worker pins for exactly this reason; the documented-blind env
    # fingerprint makes this worker the only closure point (b10 R78 round-4).
    # Pinning lives HERE so regen and enforce run identically by construction.
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    # oneDNN JIT-selects its ISA per host and ignores ATEN_CPU_CAPABILITY;
    # route conv through the ATen/MKL path the pins above make reproducible.
    torch.backends.mkldnn.enabled = False

    from surface_oracle._snapshot import canonical_dump
    from surface_oracle._stages import build_stage_snapshots, prebuild_model_cases

    prebuilt = prebuild_model_cases(tuple(args.model_axes))
    dumps = {
        axis: canonical_dump(build_stage_snapshots(axis, prebuilt=prebuilt[axis]))
        for axis in args.model_axes
    }
    print(json.dumps(dumps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
