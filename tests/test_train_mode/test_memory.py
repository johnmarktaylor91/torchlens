"""Memory regression test for repeated train-mode captures."""

from __future__ import annotations

import gc
import resource

import pytest
import torch

import torchlens as tl
from .conftest import TwoLayerMlp


def _rss_bytes() -> int:
    """Return the current resident set size in bytes."""

    try:
        import psutil
    except ImportError:
        usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return usage_kb * 1024
    process = psutil.Process()
    return int(process.memory_info().rss)


@pytest.mark.slow
def test_train_mode_repeated_ops_rss_bounded(two_layer_mlp: TwoLayerMlp) -> None:
    """Repeated train-mode ops do not grow RSS beyond a bounded steady-state ratio."""

    rss_samples: list[int] = []
    for step in range(20):
        trace = tl.trace(
            two_layer_mlp,
            torch.randn(8, 4, requires_grad=True),
            train_mode=True,
            random_seed=step,
        )
        saved = trace[trace.output_layers[0]].out
        two_layer_mlp.zero_grad(set_to_none=True)
        saved.sum().backward()
        trace.cleanup()
        del trace, saved
        gc.collect()
        rss_samples.append(_rss_bytes())

    steady_state = max(rss_samples[4:8])
    assert max(rss_samples[8:]) <= steady_state * 1.5
