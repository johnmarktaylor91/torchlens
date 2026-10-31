"""Key-forgery plants for the loop-detection invariants, one per grouper.

Moved out of ``tests/test_validation.py`` (its size ceiling). The grouper_v1
plant forges param barcodes so rule 1 must fire; the callsite_v1 plant forges a
shared call-site key so contract (ii) must fire.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from support.structural_grouper import legacy_structural_grouper  # noqa: F401

import torchlens as tl
from torchlens import trace as trace_fn
from torchlens.errors import MetadataInvariantError
from torchlens.validation.invariants import _check_loop_detection_invariants


class _TwoLinearChain(nn.Module):
    """Two distinct Linears, for param-key forgery plants."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(5, 4)
        self.fc2 = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.mark.usefixtures("legacy_structural_grouper")
def test_corruption_arm_param_sharing_key_forgery() -> None:
    """Killer for loop_detection_invariants#a11: param-sharing violation.

    Rule 1 (layers sharing func_name, param barcodes and equivalence_class are
    one layer) is the grouper_v1 contract, so the plant runs under the legacy
    structural grouper. Under callsite_v1, contract (ii) replaces it; the
    equivalent forgery there is
    ``test_corruption_arm_call_site_key_forgery_fails_contract_ii``.
    """

    import copy as _copy

    log = trace_fn(
        _TwoLinearChain(),
        torch.randn(2, 5),
        capture=tl.options.CaptureOptions(seed=42),
    )
    try:
        first, second = [lay for lay in log.layer_list if lay.layer_label.startswith("linear")][:2]
        donor = first.ops[0]
        for op in second.ops:
            op._param_barcodes = _copy.copy(donor._param_barcodes)
            op.equivalence_class = donor.equivalence_class
        with pytest.raises(MetadataInvariantError, match="Param sharing violation"):
            _check_loop_detection_invariants(log)
    finally:
        log.cleanup()


def test_corruption_arm_call_site_key_forgery_fails_contract_ii() -> None:
    """callsite_v1 analogue of the param-sharing forgery: two layers, one key.

    Under call-site grouping the evidence that two ops are one layer is the
    call-site key, not param barcodes (RULE.md section 3, row 51). Forging
    fc2's key to equal fc1's leaves two layers sharing one key, which contract
    (ii) must refuse.
    """

    from torchlens.postprocess import _call_site_contract as contract

    log = trace_fn(
        _TwoLinearChain(),
        torch.randn(2, 5),
        capture=tl.options.CaptureOptions(seed=42),
    )
    try:
        assert log.grouping_policy["detector"] == "callsite_v1"
        classes = dict(contract.call_site_classes_for(log) or {})
        first, second = [op for op in log.layer_list if op.func_name == "linear"][:2]
        assert classes[first._label_raw] != classes[second._label_raw]
        classes[second._label_raw] = classes[first._label_raw]
        contract.remember_call_site_classes(log, classes)
        with pytest.raises(MetadataInvariantError, match=r"\(ii\)"):
            _check_loop_detection_invariants(log)
    finally:
        log.cleanup()
