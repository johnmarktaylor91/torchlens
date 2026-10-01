"""Training sanity checks -- the torcheck niche, honestly redefined (F23).

A standalone, CAPTURE-FREE check registry on one shared loop-session
chassis (checks memo D1): the chassis owns the step axis, the three hook
sites (S-A per-parameter counting hooks, S-B optimizer pre-step, S-C
post-step), the skip/scale/clip ledgers, and the batched tensor scan;
trackers subscribe to its bounded event stream (the C06
``torchlens.observability`` chassis). The registry constructs and runs with
TorchLens capture entirely off, and the step-check family runs on
``torch.compile``'d models.

The honestly-redefined flagship checks (memo section 1, all measured):

- "Params changing" ships as a FACT vocabulary whose primary detector is
  the GRADIENT FACT -- under default AdamW a dead network "changes" every
  step, and movement statistics lag a real death by 22 to >32 steps.
- "Exploding gradients" cannot be checked at the optimizer site at all
  (clipping pins the post-clip norm to max_norm); magnitude lives only at
  the pre-clip S-A site, and asking where clipping censored the evidence
  returns a typed ``unavailable``, never a silent pass.
- Nonfinite parameter gradients COLLECT at S-A (under a GradScaler they
  are the mechanism working) and RAISE at S-B before the weights are
  written -- the only pre-write tripwire under fp32 and bf16.

One-shot parameter-space auditing is ``audit_params`` (also served as
``tl.debug.audit_params``). Access spelling is ``import torchlens.checks``
for now; root-facade routing is an F35 registration fragment. Every
spelling here is DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

from ._adapters import OptimizerFacts, optimizer_facts
from ._audit import ParamAudit, audit_params
from ._constants import (
    MAX_FRACTION_DEFAULT,
    SUBNORMAL_FRACTION_THRESHOLD_DEFAULT,
    WARN_WINDOW_DEFAULT,
    WATCHDOG_FACTOR,
)
from ._errors import CheckConfigError, CheckLifecycleError, CheckViolationError
from ._ledgers import (
    ClipLedger,
    ClipLedgerSnapshot,
    ScaleLedger,
    ScaleLedgerSnapshot,
    Watchdog,
    WatchdogSnapshot,
)
from ._range import LivenessProbe, RangeProbe
from ._records import (
    ACTIONS,
    CHECK_REPORT_SCHEMA_VERSION,
    EVIDENCE_KINDS,
    SCALE_PROVENANCE,
    SEVERITIES,
    STAGES,
    CheckFinding,
    CheckReport,
)
from ._scan import ScanRow, named_entries_from_target, scan_named_tensors, tensor_digest
from ._session import HAS_MULTI_GRAD_HOOK, MAX_STORED_FINDINGS, ChecksSession

__tl_layer__ = "L5"

__all__ = [
    "ACTIONS",
    "CHECK_REPORT_SCHEMA_VERSION",
    "EVIDENCE_KINDS",
    "HAS_MULTI_GRAD_HOOK",
    "MAX_FRACTION_DEFAULT",
    "MAX_STORED_FINDINGS",
    "SCALE_PROVENANCE",
    "SEVERITIES",
    "STAGES",
    "SUBNORMAL_FRACTION_THRESHOLD_DEFAULT",
    "WARN_WINDOW_DEFAULT",
    "WATCHDOG_FACTOR",
    "CheckConfigError",
    "CheckFinding",
    "CheckLifecycleError",
    "CheckReport",
    "CheckViolationError",
    "ChecksSession",
    "ClipLedger",
    "ClipLedgerSnapshot",
    "LivenessProbe",
    "OptimizerFacts",
    "ParamAudit",
    "RangeProbe",
    "ScaleLedger",
    "ScaleLedgerSnapshot",
    "ScanRow",
    "Watchdog",
    "WatchdogSnapshot",
    "audit_params",
    "named_entries_from_target",
    "optimizer_facts",
    "scan_named_tensors",
    "tensor_digest",
]
