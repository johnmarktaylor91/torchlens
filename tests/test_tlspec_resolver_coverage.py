"""Stage 9 representative-corpus resolver release gate."""

from __future__ import annotations

from collections import Counter
import gc
import json
from pathlib import Path
import runpy
from typing import Any

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import torchlens as tl
from torchlens._io.runnable import build_sparse_run_descriptor, preflight_sparse_run_descriptor
from torchlens.intervention.types import FunctionRegistryKey
from torchlens.options import CaptureOptions
from torchlens.runnable import ResolverRecord, ResolverStatus, RunnableErrorCode


RELEASE_MAX_UNRESOLVED_TORCH_KEYS = 0
"""Release threshold for the representative corpus's unique torch registry keys."""


class _ConvFamily(nn.Module):
    """Convolution, normalization, pooling, and activation family."""

    def __init__(self) -> None:
        """Initialize compact image-family state."""

        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.norm = nn.BatchNorm2d(4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Run the compact convolution family."""

        return F.adaptive_avg_pool2d(F.gelu(self.norm(self.conv(value))), (2, 2))


class _EmbeddingFamily(nn.Module):
    """Embedding and reduction family."""

    def __init__(self) -> None:
        """Initialize a compact embedding table."""

        super().__init__()
        self.embedding = nn.Embedding(17, 8)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Embed tokens and reduce their sequence axis."""

        return self.embedding(value).mean(dim=1)


class _RecurrentFamily(nn.Module):
    """Recurrent multi-output family."""

    def __init__(self) -> None:
        """Initialize a compact GRU."""

        super().__init__()
        self.gru = nn.GRU(5, 7, batch_first=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Run the recurrent layer and select its sequence output."""

        output, _hidden = self.gru(value)
        return output


class _AttentionFamily(nn.Module):
    """Attention, layer normalization, and residual family."""

    def __init__(self) -> None:
        """Initialize compact attention state."""

        super().__init__()
        self.query = nn.Linear(8, 8)
        self.key = nn.Linear(8, 8)
        self.value = nn.Linear(8, 8)
        self.norm = nn.LayerNorm(8)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Run self-attention followed by a normalized residual."""

        query = self.query(value).reshape(2, 4, 2, 4).transpose(1, 2)
        key = self.key(value).reshape(2, 4, 2, 4).transpose(1, 2)
        projected_value = self.value(value).reshape(2, 4, 2, 4).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, projected_value)
        attended = attended.transpose(1, 2).reshape(2, 4, 8)
        return self.norm(attended + value)


class _TensorMethodFamily(nn.Module):
    """Tensor method, dunder, shape, and operator family."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Exercise stable public Tensor methods and arithmetic wrappers."""

        flattened = value.transpose(0, 1).contiguous().reshape(3, -1)
        return flattened.clamp(min=-1.0, max=1.0) + flattened.square()


class _SpecialFamily(nn.Module):
    """Public special-function namespace family."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Exercise special functions represented in cross-version fixtures."""

        return torch.special.erf(value) + torch.special.gammaln(value.abs() + 1.0)


class _LinearFamily(nn.Module):
    """Linear, dropout, softmax, and matrix arithmetic family."""

    def __init__(self) -> None:
        """Initialize compact affine state."""

        super().__init__()
        self.first = nn.Linear(6, 9)
        self.second = nn.Linear(9, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Run two affine calls and public functional wrappers."""

        hidden = F.dropout(F.relu(self.first(value)), p=0.0, training=self.training)
        return F.softmax(self.second(hidden), dim=-1)


def _representative_cases() -> tuple[tuple[str, nn.Module, Any], ...]:
    """Build the economical cross-family model corpus.

    Returns
    -------
    tuple[tuple[str, nn.Module, Any], ...]
        Named model/input cases spanning the test-suite runnable families.
    """

    return (
        ("linear_mlp", _LinearFamily().eval(), torch.randn(2, 6)),
        ("conv_norm_pool", _ConvFamily().eval(), torch.randn(1, 3, 8, 8)),
        ("embedding", _EmbeddingFamily().eval(), torch.randint(0, 17, (2, 5))),
        ("recurrent", _RecurrentFamily().eval(), torch.randn(2, 4, 5)),
        ("attention", _AttentionFamily().eval(), torch.randn(2, 4, 8)),
        ("tensor_methods", _TensorMethodFamily().eval(), torch.randn(2, 3, 4)),
        ("special", _SpecialFamily().eval(), torch.rand(2, 4)),
    )


def _key_label(key: FunctionRegistryKey) -> str:
    """Return one stable registry-key display label.

    Parameters
    ----------
    key:
        Unique callable registry key.

    Returns
    -------
    str
        Namespace-qualified key plus dispatch kind.
    """

    path = key.import_path or f"{key.namespace}.{key.qualname}"
    return f"{path} [{key.dispatch_kind}; schema={key.version}]"


def resolver_coverage_report() -> dict[str, Any]:
    """Count every unique resolver outcome in the representative corpus.

    Returns
    -------
    dict[str, Any]
        JSON-ready counts plus the complete unresolved and ambiguous key lists.
    """

    records_by_key: dict[FunctionRegistryKey, ResolverRecord] = {}
    model_names: list[str] = []
    registry_occurrences = 0
    for name, model, inputs in _representative_cases():
        model_names.append(name)
        trace = tl.trace(
            model,
            inputs,
            capture=CaptureOptions(
                intervention_ready=True,
                capture_container_structure=True,
                cache=False,
            ),
        )
        descriptor = build_sparse_run_descriptor(trace)
        report, attachments = preflight_sparse_run_descriptor(descriptor)
        registry_occurrences += len(descriptor.callable_registry)
        assert attachments is not None, (name, report.diagnostics)
        for record in report.resolver_records:
            previous = records_by_key.setdefault(record.recorded_key, record)
            assert previous.status is record.status

    counts = Counter(record.status.value for record in records_by_key.values())
    ambiguous: list[str] = []
    unresolved: list[str] = []
    for key, record in sorted(records_by_key.items(), key=lambda item: _key_label(item[0])):
        codes = {diagnostic.code for diagnostic in record.diagnostics}
        if RunnableErrorCode.AMBIGUOUS_QUALNAME in codes:
            ambiguous.append(_key_label(key))
        elif record.status is ResolverStatus.UNAVAILABLE:
            unresolved.append(_key_label(key))

    return {
        "corpus": model_names,
        "model_count": len(model_names),
        "registry_occurrences": registry_occurrences,
        "unique_registry_keys": len(records_by_key),
        "resolved_exact": counts[ResolverStatus.RESOLVED_EXACT.value],
        "resolved_alias": counts[ResolverStatus.RESOLVED_ALIAS.value],
        "resolved_exact_keys": sorted(
            _key_label(key)
            for key, record in records_by_key.items()
            if record.status is ResolverStatus.RESOLVED_EXACT
        ),
        "resolved_alias_keys": sorted(
            _key_label(key)
            for key, record in records_by_key.items()
            if record.status is ResolverStatus.RESOLVED_ALIAS
        ),
        "unresolved_count": len(unresolved),
        "ambiguous_count": len(ambiguous),
        "unresolved_keys": unresolved,
        "ambiguous_keys": ambiguous,
        "release_max_unresolved_torch_keys": RELEASE_MAX_UNRESOLVED_TORCH_KEYS,
    }


def classics_resolver_coverage_report(
    max_models: int = 300,
    start_index: int = 0,
) -> dict[str, Any]:
    """Run reattachment readiness over direct-build menagerie classics.

    Parameters
    ----------
    max_models:
        Maximum number of sorted classic source modules to attempt. Every
        attempted failure is retained in the returned report.
    start_index:
        Zero-based sorted candidate offset. This permits process-isolated
        shards so one hostile legacy module cannot terminate the corpus run.

    Returns
    -------
    dict[str, Any]
        Counts, complete unavailable-key lists, and every model failure.
    """

    classic_root = Path(__file__).parents[1] / "menagerie" / "classics"
    candidates = []
    for path in sorted(classic_root.glob("*.py")):
        source = path.read_text(encoding="utf-8", errors="replace")
        if "def build(" in source and "def example_input(" in source:
            candidates.append(path)
    selected = candidates[start_index : start_index + max_models]
    records_by_key: dict[FunctionRegistryKey, ResolverRecord] = {}
    failures: list[dict[str, str]] = []
    registry_occurrences = 0
    successful_models = 0
    for path in selected:
        try:
            namespace = runpy.run_path(str(path))
            build = namespace["build"]
            example_input = namespace["example_input"]
            model = build().eval()
            inputs = example_input()
            trace = tl.trace(
                model,
                inputs,
                capture=CaptureOptions(
                    intervention_ready=True,
                    capture_container_structure=True,
                    cache=False,
                ),
            )
            descriptor = build_sparse_run_descriptor(trace)
            report, attachments = preflight_sparse_run_descriptor(descriptor)
            registry_occurrences += len(descriptor.callable_registry)
            if attachments is None:
                details = "; ".join(
                    f"{diagnostic.code.value}:{diagnostic.message}"
                    for diagnostic in report.diagnostics
                )
                failures.append({"model": path.stem, "error": details})
            else:
                successful_models += 1
            for record in report.resolver_records:
                previous = records_by_key.setdefault(record.recorded_key, record)
                if previous.status is not record.status:
                    failures.append(
                        {
                            "model": path.stem,
                            "error": f"inconsistent resolver status for {_key_label(record.recorded_key)}",
                        }
                    )
        except Exception as exc:  # noqa: BLE001 - corpus report must retain every failed case
            failures.append(
                {
                    "model": path.stem,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            gc.collect()

    counts = Counter(record.status.value for record in records_by_key.values())
    ambiguous: list[str] = []
    unresolved: list[str] = []
    for key, record in sorted(records_by_key.items(), key=lambda item: _key_label(item[0])):
        codes = {diagnostic.code for diagnostic in record.diagnostics}
        if RunnableErrorCode.AMBIGUOUS_QUALNAME in codes:
            ambiguous.append(_key_label(key))
        elif record.status is ResolverStatus.UNAVAILABLE:
            unresolved.append(_key_label(key))
    return {
        "candidate_count": len(candidates),
        "attempted_models": len(selected),
        "successful_models": successful_models,
        "model_failure_count": len(failures),
        "model_failures": failures,
        "registry_occurrences": registry_occurrences,
        "unique_registry_keys": len(records_by_key),
        "resolved_exact": counts[ResolverStatus.RESOLVED_EXACT.value],
        "resolved_alias": counts[ResolverStatus.RESOLVED_ALIAS.value],
        "resolved_exact_keys": sorted(
            _key_label(key)
            for key, record in records_by_key.items()
            if record.status is ResolverStatus.RESOLVED_EXACT
        ),
        "resolved_alias_keys": sorted(
            _key_label(key)
            for key, record in records_by_key.items()
            if record.status is ResolverStatus.RESOLVED_ALIAS
        ),
        "unresolved_count": len(unresolved),
        "ambiguous_count": len(ambiguous),
        "unresolved_keys": unresolved,
        "ambiguous_keys": ambiguous,
        "release_max_unresolved_torch_keys": RELEASE_MAX_UNRESOLVED_TORCH_KEYS,
    }


@pytest.mark.smoke
def test_representative_resolver_coverage_release_threshold(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Count and print every unique unresolved or ambiguous corpus key."""

    coverage = resolver_coverage_report()
    print("RESOLVER_COVERAGE=" + json.dumps(coverage, sort_keys=True))
    captured = capsys.readouterr()
    assert "RESOLVER_COVERAGE=" in captured.out
    assert (
        coverage["unresolved_count"] + coverage["ambiguous_count"]
        <= RELEASE_MAX_UNRESOLVED_TORCH_KEYS
    ), coverage
