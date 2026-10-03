"""The BFS completeness census counts PASS-QUALIFIED ops, not bare layers.

Lost-redundancy regression (2026-08-19): the census compared bare
``layer_label`` sets, so a phantom extra pass of a legitimate multi-pass
layer was invisible whenever ``validate_metadata=False`` -- the reached
sibling passes vouched for the bare label. The metadata invariants backstop
this on the default path, but redundancy is the point of two independent
tripwires: the value-rooted census must catch it alone.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

import torchlens as tl
from torchlens.validation.diagnostics import CHECK_COMPLETENESS, TRACE_FAILURE_ATTR


class Loop(nn.Module):
    """Three applications of one shared Linear+ReLU cell."""

    def __init__(self) -> None:
        """Build the shared cell."""

        super().__init__()
        self.cell = nn.Linear(4, 4, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the cell three times."""

        h = x
        for _ in range(3):
            h = torch.relu(self.cell(h))
        return h


def _looped_trace() -> tuple[tl.Trace, list[torch.Tensor]]:
    """Return a finished 3-pass trace and its ground-truth outputs."""

    torch.manual_seed(0)
    model = Loop()
    log = tl.trace(
        model,
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )
    log._tl_test_model_keepalive = model
    outputs = [log.layer_dict_all_keys[label].out for label in log.output_layers]
    return log, outputs


def test_clean_multipass_trace_validates_without_metadata_invariants() -> None:
    """Baseline: the untampered loop trace passes the value-rooted sweep."""

    log, outputs = _looped_trace()
    assert log.validate_forward_pass(outputs, validate_metadata=False)


def test_phantom_extra_pass_fails_the_bfs_census_alone() -> None:
    """A forged unreachable pass of a real layer fails with metadata OFF."""

    log, outputs = _looped_trace()
    victim = next(
        op for op in log.layer_list if op.num_passes > 1 and op.pass_index == op.num_passes
    )
    phantom = copy.copy(victim)
    phantom_label = f"{victim.layer_label}:{victim.num_passes + 1}"
    phantom._internal_set("label", phantom_label)
    phantom._internal_set("label_short", phantom_label)
    phantom._internal_set("pass_index", victim.num_passes + 1)
    # Unreachable on purpose: no seed lists it, no real op names it as parent.
    log.layer_list.append(phantom)

    status = log.validate_forward_pass(outputs, validate_metadata=False)
    assert not status, (
        "a phantom extra pass validated clean through the value-rooted sweep: "
        "the BFS census is still counting bare layer labels"
    )
    failure = getattr(log, TRACE_FAILURE_ATTR, None)
    assert failure is not None, "validation failed without recording a failure"
    assert failure.check == CHECK_COMPLETENESS
    assert phantom_label in failure.message, failure.message
