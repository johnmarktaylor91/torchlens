"""Ledger of tracked Weekly slow-tier failures (mostly real-model validation).

JMT's ruling (round 2, lane-L8-ci-fix / lane-L17-integrate, 2026-10-02/03): the
Weekly slow tier must not stay red as a durable state. A failure here is
either fixed (its entry is removed in the same change) or tracked here as a
strict ``xfail`` -- never silently skipped, and never loosened at the
``validation/`` layer itself (AGENTS.md "Validation Integrity (LOCKED
PRINCIPLE)" still applies in full: nothing here touches a tolerance, a check,
or an invariant).

This is the ONE place these tests are marked ``xfail``. ``tests/conftest.py``
reads :data:`KNOWN_FAILURES` during collection and applies
``pytest.mark.xfail(strict=True, reason=...)`` to each listed node id
directly -- no test file decorates ``xfail`` itself
(``tests/test_known_failures_ledger.py`` enforces both halves: every entry
matches a real collected node id, and no ledger-named file declares an
``xfail`` of its own). ``strict=True`` means an unexpected PASS fails the
run, so fixing one of these forces removing its row here in the same change
-- the ledger can never silently drift from reality in the fixed direction
either. ``pyproject.toml`` already sets ``xfail_strict = true`` suite-wide;
the per-entry ``strict=True`` below is kept explicit so this file's own
intent does not depend on that ini default.

Each entry's ``reason`` is the failure class actually observed on a real run
of the Weekly environment (torch 2.7.1+cpu / torchvision 0.22.1+cpu), not a
guess from memory or from an older round's notes -- see
tests/test_known_failures_ledger.py for how that is checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


@dataclass(frozen=True)
class KnownFailure:
    """One tracked, strict-xfail real-model test.

    Attributes
    ----------
    nodeid:
        Exact pytest node id, e.g.
        ``"tests/test_real_world_models.py::test_timm_beit_base_patch16_224"``.
    reason:
        The failure class observed on a real Weekly-environment run.
    tracking:
        Where the round-2 follow-up for this failure is recorded.
    """

    nodeid: str
    reason: str
    tracking: str


#: Populated from a real run of the Weekly slow tier (one test per process,
#: torch 2.7.1+cpu / torchvision 0.22.1+cpu); see tests/AGENTS.md "Testing
#: Tiers" for how to reproduce that environment.
KNOWN_FAILURES: tuple[KnownFailure, ...] = (
    KnownFailure(
        nodeid="tests/test_real_world_models.py::test_audio_encodec",
        reason="assert False: validate_forward_pass(...) returned False",
        tracking="round-2 (lane-L8-ci-fix / lane-L17-integrate)",
    ),
    KnownFailure(
        nodeid="tests/test_real_world_models.py::test_audio_vits",
        reason="assert False: validate_forward_pass(...) returned False",
        tracking="round-2 (lane-L8-ci-fix / lane-L17-integrate)",
    ),
    KnownFailure(
        nodeid="tests/test_real_world_models.py::test_blip2",
        reason="assert False: validate_forward_pass(...) returned False",
        tracking="round-2 (lane-L8-ci-fix / lane-L17-integrate)",
    ),
    KnownFailure(
        nodeid="tests/test_real_world_models.py::test_detr",
        reason="assert False: validate_forward_pass(...) returned False",
        tracking="round-2 (lane-L8-ci-fix / lane-L17-integrate)",
    ),
    KnownFailure(
        nodeid="tests/test_real_world_models.py::test_fasterrcnn_mobilenet_eval",
        reason="assert False: validate_forward_pass(...) returned False",
        tracking="round-2 (lane-L8-ci-fix / lane-L17-integrate)",
    ),
    KnownFailure(
        nodeid="tests/test_real_world_models.py::test_fasterrcnn_mobilenet_train",
        reason="assert False: validate_forward_pass(...) returned False",
        tracking="round-2 (lane-L8-ci-fix / lane-L17-integrate)",
    ),
    KnownFailure(
        nodeid="tests/test_real_world_models.py::test_fcos_resnet50_eval",
        reason="assert False: validate_forward_pass(...) returned False",
        tracking="round-2 (lane-L8-ci-fix / lane-L17-integrate)",
    ),
    KnownFailure(
        nodeid="tests/test_real_world_models.py::test_fcos_resnet50_train",
        reason="assert False: validate_forward_pass(...) returned False",
        tracking="round-2 (lane-L8-ci-fix / lane-L17-integrate)",
    ),
    KnownFailure(
        nodeid="tests/test_real_world_models.py::test_funnel_transformer",
        reason="assert False: validate_forward_pass(...) returned False",
        tracking="round-2 (lane-L8-ci-fix / lane-L17-integrate)",
    ),
    KnownFailure(
        nodeid="tests/test_real_world_models.py::test_gatv2_pyg",
        reason="assert False: validate_forward_pass(...) returned False",
        tracking="round-2 (lane-L8-ci-fix / lane-L17-integrate)",
    ),
    KnownFailure(
        nodeid="tests/test_real_world_models.py::test_gptj",
        reason="ValueError: A transposed window map requires a positive output extent.",
        tracking="round-2 (lane-L8-ci-fix / lane-L17-integrate)",
    ),
    KnownFailure(
        nodeid="tests/test_real_world_models.py::test_informer",
        reason="assert False: validate_forward_pass(...) returned False",
        tracking="round-2 (lane-L8-ci-fix / lane-L17-integrate)",
    ),
    KnownFailure(
        nodeid="tests/test_real_world_models.py::test_keypointrcnn_resnet50_eval",
        reason="assert False: validate_forward_pass(...) returned False",
        tracking="round-2 (lane-L8-ci-fix / lane-L17-integrate)",
    ),
    KnownFailure(
        nodeid="tests/test_real_world_models.py::test_keypointrcnn_resnet50_train",
        reason="assert False: validate_forward_pass(...) returned False",
        tracking="round-2 (lane-L8-ci-fix / lane-L17-integrate)",
    ),
    KnownFailure(
        nodeid="tests/test_real_world_models.py::test_led",
        reason="assert False: validate_forward_pass(...) returned False",
        tracking="round-2 (lane-L8-ci-fix / lane-L17-integrate)",
    ),
    KnownFailure(
        nodeid="tests/test_real_world_models.py::test_mamba2",
        reason="assert False: validate_forward_pass(...) returned False",
        tracking="round-2 (lane-L8-ci-fix / lane-L17-integrate)",
    ),
    KnownFailure(
        nodeid="tests/test_real_world_models.py::test_maskrcnn_resnet50_eval",
        reason="assert False: validate_forward_pass(...) returned False",
        tracking="round-2 (lane-L8-ci-fix / lane-L17-integrate)",
    ),
    KnownFailure(
        nodeid="tests/test_real_world_models.py::test_maskrcnn_resnet50_train",
        reason="assert False: validate_forward_pass(...) returned False",
        tracking="round-2 (lane-L8-ci-fix / lane-L17-integrate)",
    ),
    KnownFailure(
        nodeid="tests/test_real_world_models.py::test_mobilebert",
        reason="RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x128 and 64x64)",
        tracking="round-2 (lane-L8-ci-fix / lane-L17-integrate)",
    ),
    KnownFailure(
        nodeid="tests/test_real_world_models.py::test_opticflow_raftlarge",
        reason="validate_forward_pass fail-closed: deepcopy cannot snapshot non-registered plain attribute(s) (CorrBlock[0].corr_pyramid), so model-state restoration cannot be proven",
        tracking="round-2 (lane-L8-ci-fix / lane-L17-integrate)",
    ),
    KnownFailure(
        nodeid="tests/test_real_world_models.py::test_opticflow_raftsmall",
        reason="validate_forward_pass fail-closed: deepcopy cannot snapshot non-registered plain attribute(s) (CorrBlock[0].corr_pyramid), so model-state restoration cannot be proven",
        tracking="round-2 (lane-L8-ci-fix / lane-L17-integrate)",
    ),
    KnownFailure(
        nodeid="tests/test_real_world_models.py::test_reformer",
        reason="ValueError: input_axis must be non-negative and input_extent must be positive.",
        tracking="round-2 (lane-L8-ci-fix / lane-L17-integrate)",
    ),
    KnownFailure(
        nodeid="tests/test_real_world_models.py::test_retinanet_resnet50_eval",
        reason="assert False: validate_forward_pass(...) returned False",
        tracking="round-2 (lane-L8-ci-fix / lane-L17-integrate)",
    ),
    KnownFailure(
        nodeid="tests/test_real_world_models.py::test_retinanet_resnet50_train",
        reason="assert False: validate_forward_pass(...) returned False",
        tracking="round-2 (lane-L8-ci-fix / lane-L17-integrate)",
    ),
    KnownFailure(
        nodeid="tests/test_real_world_models.py::test_ssd300_vgg16_eval",
        reason="GraphvizRenderError: Graphviz render timed out after 120s for a 3318-node forward graph",
        tracking="round-2 (lane-L8-ci-fix / lane-L17-integrate)",
    ),
    KnownFailure(
        nodeid="tests/test_real_world_models.py::test_ssd300_vgg16_train",
        reason="assert False: validate_forward_pass(...) returned False",
        tracking="round-2 (lane-L8-ci-fix / lane-L17-integrate)",
    ),
    KnownFailure(
        nodeid="tests/test_real_world_models.py::test_timm_efficientformer",
        reason="validate_forward_pass fail-closed: deepcopy cannot snapshot non-registered plain attribute(s) (Attention[0].attention_bias_cache), so model-state restoration cannot be proven",
        tracking="round-2 (lane-L8-ci-fix / lane-L17-integrate)",
    ),
    KnownFailure(
        nodeid="tests/test_real_world_models.py::test_timm_levit_128",
        reason="validate_forward_pass fail-closed: deepcopy cannot snapshot non-registered plain attribute(s) (Attention[0].attention_bias_cache and 9 more plain attributes), so model-state restoration cannot be proven",
        tracking="round-2 (lane-L8-ci-fix / lane-L17-integrate)",
    ),
    KnownFailure(
        nodeid="tests/test_real_world_models.py::test_timm_xcit_tiny_24_p8_224",
        reason="assert False: validate_forward_pass(...) returned False",
        tracking="round-2 (lane-L8-ci-fix / lane-L17-integrate)",
    ),
    KnownFailure(
        nodeid="tests/bench/test_capture_bench.py::test_capture_bench_matrix",
        reason=(
            "UserWarning promoted to error: tensor arguments with no graph/source "
            "provenance, adopted at module entry blocks.layers.N.dropout of the "
            "nn.TransformerEncoder GPT-block workload (also on torch 2.14.1)"
        ),
        tracking="Weekly 37176706721 triage (2026-10-04)",
    ),
    KnownFailure(
        nodeid="tests/test_weightsfree_order.py::test_realpre_cell_refuses_typed_never_refutes",
        reason=(
            "structure-only capture subprocess exits 1: first import of torch._dynamo "
            "runs under the installed wrappers and hits a circular import (torch 2.7.1 "
            "only; passes on torch 2.14.1)"
        ),
        tracking="Weekly 37176706721 triage (2026-10-04)",
    ),
)


def duplicate_nodeids() -> list[str]:
    """Return node ids that appear more than once in :data:`KNOWN_FAILURES`."""

    seen: set[str] = set()
    duplicates: list[str] = []
    for entry in KNOWN_FAILURES:
        if entry.nodeid in seen:
            duplicates.append(entry.nodeid)
        seen.add(entry.nodeid)
    return duplicates


def stale_entries(collected_nodeids: set[str]) -> list[KnownFailure]:
    """Return ledger entries whose node id was not actually collected.

    Pure helper (red-capable without a real pytest session): a stale entry
    means the test was renamed, removed, or never existed under this id --
    the ledger must never carry a row pytest cannot resolve.

    Parameters
    ----------
    collected_nodeids:
        Node ids pytest actually collected for the files this ledger names.

    Returns
    -------
    list[KnownFailure]
        Entries with no matching collected node id.
    """

    return [entry for entry in KNOWN_FAILURES if entry.nodeid not in collected_nodeids]


def ledger_by_nodeid() -> dict[str, KnownFailure]:
    """Return the ledger indexed by node id (duplicates keep the last entry)."""

    return {entry.nodeid: entry for entry in KNOWN_FAILURES}


def apply_xfail_marks(items: list[pytest.Item]) -> None:
    """Mark every collected item named in :data:`KNOWN_FAILURES` ``xfail``.

    Called from ``tests/conftest.py::pytest_collection_modifyitems`` so this
    stays the single place that turns a ledger row into a live pytest marker.

    Parameters
    ----------
    items:
        Collected pytest items for this session.
    """

    import pytest as _pytest

    ledger = ledger_by_nodeid()
    if not ledger:
        return
    for item in items:
        entry = ledger.get(item.nodeid)
        if entry is not None:
            item.add_marker(_pytest.mark.xfail(reason=entry.reason, strict=True))
