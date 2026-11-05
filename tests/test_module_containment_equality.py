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
# appears and every later op ordinal shifts by one. The weights path also scales the
# query two ways:
#   older: q_scaled = q / math.sqrt(E)                 -> __truediv__ (the 2.1 floor)
#   newer: q_scaled = q * math.sqrt(1.0 / float(E))    -> __mul__
# Each spelling torch has shipped is pinned BYTE-EXACTLY in its own golden rather than
# any snapshot being loosened, so the equality tripwire stays fully armed across the
# torch matrix. The base golden is the legacy reshape with the multiplied scale.
TORCH_VARIANT_FIXTURES = {
    "multihead_attention_demo": ("torch_fused_reshape", "torch_div_scale"),
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


def _torch_divides_mha_query_scale() -> bool:
    """Report whether torch scales the MHA weights-path query by dividing by sqrt(E).

    Like :func:`_torch_fuses_mha_output_reshape`, a probe of torch's own source rather
    than a version check, failing loudly on a spelling it does not recognize.

    Returns
    -------
    bool
        True for ``q / math.sqrt(E)``, False for ``q * math.sqrt(1.0 / float(E))``.
    """

    source = inspect.getsource(torch.nn.functional.multi_head_attention_forward)
    if re.search(r"q_scaled\s*=\s*q\s*/\s*math\.sqrt\(\s*E\s*\)", source):
        return True
    if re.search(
        r"q_scaled\s*=\s*q\s*\*\s*math\.sqrt\(\s*1\.0\s*/\s*float\(\s*E\s*\)\s*\)", source
    ):
        return False
    raise AssertionError(
        "torch.nn.functional.multi_head_attention_forward scales its query in a way this "
        "probe does not recognize (neither q / math.sqrt(E) nor "
        "q * math.sqrt(1.0 / float(E))); re-audit the multihead_attention_demo golden"
    )


def _mha_variant_suffix() -> str | None:
    """Return the golden suffix for this torch's MHA spelling (``None`` for the base).

    Raises
    ------
    AssertionError
        For a combination no golden pins (the fused reshape with the divided scale has
        never shipped), so a new torch spelling is audited instead of guessed.
    """

    fused, divided = _torch_fuses_mha_output_reshape(), _torch_divides_mha_query_scale()
    if fused and divided:
        raise AssertionError(
            "torch fuses the MHA output reshape AND divides the query scale; no "
            "multihead_attention_demo golden pins that combination, re-audit the fixture"
        )
    return "torch_fused_reshape" if fused else "torch_div_scale" if divided else None


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


_LEGACY_OUTPUT = (
    "    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)\n"
)
_FUSED_OUTPUT = "    attn_output = attn_output.transpose(0, 1).reshape(tgt_len * bsz, embed_dim)\n"
_DIVIDED_SCALE = "    q_scaled = q / math.sqrt(E)\n"
_MULTIPLIED_SCALE = "    q_scaled = q * math.sqrt(1.0 / float(E))\n"


@pytest.mark.parametrize(
    ("scale", "output", "expected"),
    [
        (_MULTIPLIED_SCALE, _LEGACY_OUTPUT, None),
        (_MULTIPLIED_SCALE, _FUSED_OUTPUT, "torch_fused_reshape"),
        (_DIVIDED_SCALE, _LEGACY_OUTPUT, "torch_div_scale"),
    ],
    ids=["legacy-reshape-mul", "fused-reshape-mul", "legacy-reshape-div"],
)
def test_mha_variant_follows_both_spellings(
    monkeypatch: pytest.MonkeyPatch, scale: str, output: str, expected: str | None
) -> None:
    """Each shipped pair of spellings selects its own golden."""

    source = "def multi_head_attention_forward():\n" + scale + output
    monkeypatch.setattr(inspect, "getsource", lambda _object: source)

    assert _mha_variant_suffix() == expected


def test_mha_variant_refuses_unpinned_and_unknown_scales(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fused reshape with a divided scale, or an unknown scale, fails closed."""

    sources = {
        "pins that combination": _DIVIDED_SCALE + _FUSED_OUTPUT,
        "scales its query in a way": "    q_scaled = q * (E ** -0.5)\n" + _LEGACY_OUTPUT,
    }
    for match, body in sources.items():
        source = "def multi_head_attention_forward():\n" + body
        monkeypatch.setattr(inspect, "getsource", lambda _object, source=source: source)
        with pytest.raises(AssertionError, match=match):
            _mha_variant_suffix()


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

    suffix = _mha_variant_suffix() if fixture_name in TORCH_VARIANT_FIXTURES else None
    if suffix is not None:
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
    """Every spelling of every torch-variant fixture is pinned on disk.

    A missing variant golden would fail closed at comparison time (the
    generate path is flag-gated), but pinning that all files exist and
    pairwise differ keeps the variant registry honest: an identical pair
    means a variant is pointless and its registry row is stale.
    """

    for fixture_name, suffixes in TORCH_VARIANT_FIXTURES.items():
        base = SNAPSHOT_DIR / f"{fixture_name}.json"
        variants = [SNAPSHOT_DIR / f"{fixture_name}.{suffix}.json" for suffix in suffixes]
        for path in (base, *variants):
            assert path.exists(), f"missing torch-variant golden {path}"
        contents = [json.loads(path.read_text()) for path in (base, *variants)]
        for index, content in enumerate(contents):
            assert content not in contents[index + 1 :], (
                f"{fixture_name} variant goldens are identical; the variant is pointless "
                "and should be deleted along with its registry entry"
            )
        assert _snapshot_path(fixture_name) in {base, *variants}


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
