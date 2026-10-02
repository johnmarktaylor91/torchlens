"""Snapshot equality tests for module-containment metadata."""

from __future__ import annotations

import inspect
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest
import torch.nn.functional
from _module_containment_snapshot import build_snapshot
from _oracle_env import (
    flag_armed,
    guard_wrap_state_for_golden_update,
    require_update_reason,
    write_provenance,
)
from fixtures.module_containment_models import ALL_FIXTURES, FixtureBuilder

import torchlens as tl
from torchlens.backends.torch._tl import get_module_meta

SNAPSHOT_DIR = Path(__file__).parent / "snapshots" / "module_containment"
#: Update flag for the module-containment golden corpus (b10 R78-1 round 3:
#: baselines are generated deliberately, never silently on first run).
_UPDATE_ENV = "TORCHLENS_UPDATE_MODULE_CONTAINMENT"
# Synthetic hook replacement is intentionally snapshotted with hook-stack semantics:
# downstream ops stay in the dynamic call stack instead of inheriting a replaced module.
# Hook-stack IS the only containment engine (the `_module_containment_engine` selector
# was a validated-but-never-read knob, deleted in fixwave-2 R47-5), so these fixtures
# no longer need any capture-option opt-in.
HOOK_STACK_FIXTURES = {
    "raw_hook_replacement_synthetic",
}
# Fixtures whose op sequence depends on which ops UPSTREAM TORCH itself calls, not on
# anything TorchLens decides. Across the declared torch support range (2.1 -> 2.12+),
# ``F.multi_head_attention_forward`` spells its output reshape two different ways:
#   older: attn_output.transpose(0, 1).contiguous().view(...)   -> contiguous + view
#   newer: attn_output.transpose(0, 1).reshape(...)             -> a single reshape
# TorchLens faithfully records whichever ops torch actually calls, so one op fewer
# appears and every later op ordinal shifts by one. Both spellings are pinned
# BYTE-EXACTLY in their own golden rather than either snapshot being loosened, so the
# equality tripwire stays fully armed on both sides of the torch matrix.
TORCH_VARIANT_FIXTURES = {
    "multihead_attention_demo": "torch_fused_reshape",
}


def _torch_fuses_mha_output_reshape() -> bool:
    """Report whether torch fuses the MHA output reshape into one ``reshape`` call.

    This is a capability probe against torch's own source, deliberately NOT a
    ``torch.__version__`` comparison. An unrecognized spelling fails loudly rather
    than silently selecting a golden, because silently picking the wrong golden is
    exactly how a real capture regression would be mistaken for torch drift.

    Returns
    -------
    bool
        True when the output path calls a single fused ``reshape``.
    """

    try:
        source = inspect.getsource(torch.nn.functional.multi_head_attention_forward)
    except (OSError, TypeError) as exc:  # pragma: no cover - source always available on CPython
        raise AssertionError(
            "cannot read torch.nn.functional.multi_head_attention_forward source, so the "
            "multihead_attention_demo golden variant cannot be selected; re-audit the fixture"
        ) from exc
    # torch 2.7.1 wraps the RHS in parens for line length
    # ("attn_output = (\n    attn_output.transpose(...)...\n)") -- an upstream
    # formatting-only change, so the optional "(" is tolerated here rather
    # than treated as a new, unrecognized spelling.
    transpose_prefix = r"attn_output\s*=\s*\(?\s*attn_output\.transpose\(\s*0\s*,\s*1\s*\)"
    output_extent = r"\(\s*tgt_len\s*\*\s*bsz\s*,\s*embed_dim\s*\)"
    if re.search(rf"{transpose_prefix}\.reshape{output_extent}", source):
        return True
    if re.search(rf"{transpose_prefix}\.contiguous\(\s*\)\.view{output_extent}", source):
        return False
    raise AssertionError(
        "torch.nn.functional.multi_head_attention_forward spells its output reshape in a "
        "way this probe does not recognize (neither the fused reshape nor the exact legacy "
        "transpose/contiguous/view chain); "
        "re-audit the multihead_attention_demo golden instead of trusting either variant"
    )


def test_mha_probe_recognizes_exact_legacy_output_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Select the legacy golden only for the exact historical output expression."""

    source = """
def multi_head_attention_forward():
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
"""
    monkeypatch.setattr(inspect, "getsource", lambda _object: source)

    assert _torch_fuses_mha_output_reshape() is False


def test_mha_probe_recognizes_paren_wrapped_legacy_output_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """torch 2.7.1 wraps the legacy RHS in parens for line length; still legacy."""

    source = """
def multi_head_attention_forward():
    attn_output = (
        attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    )
"""
    monkeypatch.setattr(inspect, "getsource", lambda _object: source)

    assert _torch_fuses_mha_output_reshape() is False


def test_mha_probe_rejects_unrelated_contiguous_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed when an unknown output spelling merely contains ``contiguous``."""

    source = """
def multi_head_attention_forward():
    unrelated = query.contiguous()
    attn_output = attn_output.transpose(0, 1).flatten(0, 1)
"""
    monkeypatch.setattr(inspect, "getsource", lambda _object: source)

    with pytest.raises(AssertionError, match="exact legacy transpose/contiguous/view chain"):
        _torch_fuses_mha_output_reshape()


def _snapshot_path(fixture_name: str) -> Path:
    """Resolve the golden path for a fixture, honoring torch-spelling variants.

    Parameters
    ----------
    fixture_name:
        Name of the current fixture.

    Returns
    -------
    Path
        Path of the golden this environment must match exactly.
    """

    suffix = TORCH_VARIANT_FIXTURES.get(fixture_name)
    if suffix is not None and _torch_fuses_mha_output_reshape():
        return SNAPSHOT_DIR / f"{fixture_name}.{suffix}.json"
    return SNAPSHOT_DIR / f"{fixture_name}.json"


def _unpack_fixture(result: tuple[Any, ...]) -> tuple[Any, Any, str, Any | None]:
    """Unpack fixture results with optional hook handles.

    Parameters
    ----------
    result:
        Fixture builder return tuple.

    Returns
    -------
    tuple[Any, Any, str, Any | None]
        Model, input args, fixture name, and optional hook handle.
    """

    if len(result) == 4:
        model, input_args, fixture_name, hook_handle = result
        return model, input_args, fixture_name, hook_handle
    model, input_args, fixture_name = result
    return model, input_args, fixture_name, None


def _assert_lazy_address_stable(model: Any, fixture_name: str) -> None:
    """Assert LazyLinear address survives first-call materialization.

    Parameters
    ----------
    model:
        Fixture model after tracing.
    fixture_name:
        Name of the current fixture.
    """

    if fixture_name != "lazy_linear_demo":
        return
    lazy_layer = model[0]
    module_meta = get_module_meta(lazy_layer)
    assert module_meta is not None
    assert module_meta.address == "0"


def _assert_synthetic_replacement_present(actual: dict[str, Any], fixture_name: str) -> None:
    """Assert synthetic raw-hook fixture includes an interventionreplacement op.

    Parameters
    ----------
    actual:
        Built snapshot dictionary.
    fixture_name:
        Name of the current fixture.
    """

    if fixture_name != "raw_hook_replacement_synthetic":
        return
    func_names = {op["func_name"] for op in actual["ops"]}
    assert "interventionreplacement" in func_names


def test_torch_variant_goldens_are_all_present_and_distinct() -> None:
    """Both spellings of every torch-variant fixture are pinned on disk.

    A missing variant golden would fail closed at comparison time (the
    generate path is flag-gated), but pinning both files exist and genuinely
    differ keeps the variant registry honest: an identical pair means the
    variant is pointless and its registry row is stale.
    """

    for fixture_name, suffix in TORCH_VARIANT_FIXTURES.items():
        base = SNAPSHOT_DIR / f"{fixture_name}.json"
        variant = SNAPSHOT_DIR / f"{fixture_name}.{suffix}.json"
        assert base.exists(), f"missing baseline golden {base}"
        assert variant.exists(), f"missing torch-variant golden {variant}"
        assert json.loads(base.read_text()) != json.loads(variant.read_text()), (
            f"{fixture_name} variant goldens are identical; the variant is pointless "
            "and should be deleted along with its registry entry"
        )
        assert _snapshot_path(fixture_name) in {base, variant}


@pytest.mark.parametrize("builder", ALL_FIXTURES, ids=lambda builder: builder.__name__)
def test_module_containment_snapshot(builder: FixtureBuilder) -> None:
    """Compare module-containment snapshot for one fixture."""

    if flag_armed(os.environ, _UPDATE_ENV) and not os.environ.get("CI"):
        # Regeneration is in-process, and this family's fixtures include the
        # MHA/`multi_head_attention_forward` path whose op DECOMPOSITION is
        # exactly what wrap state changed in the sf-fastpath incident: refuse
        # to freeze bytes on a torch some earlier test already wrapped
        # (SF-53), BEFORE the fixture constructs or captures anything.
        guard_wrap_state_for_golden_update(_UPDATE_ENV)

    model, input_args, fixture_name, hook_handle = _unpack_fixture(builder())
    try:
        trace = tl.trace(model, input_args)
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    _assert_lazy_address_stable(model, fixture_name)
    actual = build_snapshot(trace, fixture_name)
    _assert_synthetic_replacement_present(actual, fixture_name)

    snapshot_path = _snapshot_path(fixture_name)
    if not snapshot_path.exists():
        # FLAG-GATED baseline generation (b10 R78-1 round 3): the historical
        # unconditional write-then-skip was the exact silent self-baselining
        # require_env_golden was written to eliminate — a fresh clone would
        # bless whatever the current build produced, no flag, no provenance,
        # no review trail.
        if flag_armed(os.environ, _UPDATE_ENV) and not os.environ.get("CI"):
            reason = require_update_reason(_UPDATE_ENV)
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_path.write_text(json.dumps(actual, indent=2, sort_keys=True, default=str))
            write_provenance(snapshot_path.parent, Path(__file__).name, _UPDATE_ENV, reason)
            pytest.skip(
                f"baseline snapshot generated: {snapshot_path}; re-run without "
                f"{_UPDATE_ENV} to verify"
            )
        pytest.fail(
            f"missing module-containment golden {snapshot_path}. Refusing to "
            f"self-baseline: generate deliberately with {_UPDATE_ENV}=1 (plus "
            "TORCHLENS_GOLDEN_REASON), review the diff, and commit it."
        )

    expected = json.loads(snapshot_path.read_text())
    assert actual == expected, (
        f"snapshot drift for {fixture_name}; if intentional, delete the file and "
        f"regenerate deliberately with {_UPDATE_ENV}=1 (plus "
        "TORCHLENS_GOLDEN_REASON), then re-run to verify"
    )
