"""Subprocess entry point for isolated capture-oracle characterization."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence

# Pin the CPU kernel paths BEFORE torch loads (ATen reads its capability and
# MKL its reproducibility branch at first use). The goldens hash raw float
# bytes, and conv/linear results otherwise follow the host's ISA: goldens
# recorded on AVX-512 Xeons failed on GitHub's Xeon 6973P-C runners with
# identical inputs and code. AVX2 is the common floor of every x86 runner;
# MKL_CBWR=COMPATIBLE is MKL's cross-processor reproducible branch.
os.environ["ATEN_CPU_CAPABILITY"] = "avx2"
os.environ["MKL_CBWR"] = "COMPATIBLE"

from ._characterize import characterize_case  # noqa: E402 -- after the kernel pins


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the worker command line.

    Parameters
    ----------
    argv:
        Optional explicit argument sequence.

    Returns
    -------
    argparse.Namespace
        Parsed case identifier.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate one JSON characterization on standard output.

    Parameters
    ----------
    argv:
        Optional explicit argument sequence.

    Returns
    -------
    int
        Process exit status.
    """

    args = _parse_args(argv)

    # Pin the execution environment the seeds alone do not cover: thread
    # count and kernel selection both steer float bytes, and the goldens
    # compare raw sha256 chunks (b10 R78-7). Pinning lives HERE so the regen
    # path and the enforce path run under identical settings by construction.
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    # oneDNN JIT-selects its ISA per host and ignores ATEN_CPU_CAPABILITY;
    # route conv through the ATen/MKL path the pins above make reproducible.
    torch.backends.mkldnn.enabled = False

    print(json.dumps(characterize_case(args.case), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
