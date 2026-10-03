"""grind-r5 b7 R21 / P4: persisted keys must be address-free and seed-stable.

``_append_arg_hash``'s tail stringified default-repr objects (a memory
address) into ``equivalence_class``, so cross-process keys diverged for any
op holding an object arg (``torch.randn(..., generator=g)``), colliding for
distinct objects after address reuse, and splitting recurrence grouping for
per-call fresh objects. Sets enumerated in hash order; frozensets fell to the
address tail. The BackwardPass row's ``root_grad_fn_ids`` persisted raw
grad_fn memory addresses the a169e886 dense-ordinal fix never reached.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch.tensor_tracking import _get_hash_from_args


def test_object_arg_hash_is_address_free() -> None:
    """Two semantically identical fresh objects must hash identically."""

    # Hold BOTH generators alive: sequential construction lets CPython reuse
    # the freed address, which would make an address-derived hash pass this
    # test vacuously (the collision half of the same defect).
    first_generator, second_generator = torch.Generator(), torch.Generator()
    first = _get_hash_from_args((first_generator,), {})
    second = _get_hash_from_args((second_generator,), {})
    assert first == second, "the arg-hash tail still folds the object's memory address"


def test_object_arg_hash_distinguishes_types() -> None:
    """The address-free token still separates different argument types."""

    class _Marker:
        pass

    generator_hash = _get_hash_from_args((torch.Generator(),), {})
    marker_hash = _get_hash_from_args((_Marker(),), {})
    assert generator_hash != marker_hash


@pytest.mark.smoke
def test_set_and_frozenset_args_hash_identically() -> None:
    """Equal members must produce one fingerprint regardless of container."""

    members = {"alpha", "beta", "gamma", "delta"}
    assert _get_hash_from_args((set(members),), {}) == _get_hash_from_args(
        (frozenset(members),), {}
    ), "frozenset still falls through to the str() tail"


@pytest.mark.heavy
def test_set_arg_hash_is_hashseed_independent() -> None:
    """The set fingerprint must not depend on PYTHONHASHSEED."""

    script = (
        "from torchlens.backends.torch.tensor_tracking import _get_hash_from_args;"
        "print(_get_hash_from_args(({'alpha','beta','gamma','delta'},), {}))"
    )
    digests = []
    for seed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        env.setdefault("PYTHONPATH", os.getcwd())
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        digests.append(result.stdout.strip())
    assert digests[0] == digests[1], f"set fingerprint is PYTHONHASHSEED-dependent: {digests!r}"


class _BackwardModel(nn.Module):
    """Tiny model for the backward root-ordinal persistence pin."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


def test_backward_pass_root_ids_persist_as_dense_ordinals(tmp_path) -> None:
    """Loaded root_grad_fn_ids must be trace-local ordinals inside
    grad_fn_logs, never raw process addresses."""

    trace = tl.trace(
        _BackwardModel(),
        torch.randn(2, 4).requires_grad_(True),
        capture=tl.options.CaptureOptions(backward_ready=True),
        save_mode="reference",
    )
    trace.log_backward(trace[trace.output_layers[0]].out.sum())

    bundle = tmp_path / "backward.tlspec"
    tl.save(trace, str(bundle))
    loaded = tl.load(str(bundle))

    backward_passes = list(loaded.backward_pass_logs.values())
    assert backward_passes, "expected at least one persisted backward pass"
    all_roots = [root_id for record in backward_passes for root_id in record.root_grad_fn_ids]
    assert all_roots, "expected persisted root_grad_fn_ids"
    for root_id in all_roots:
        assert root_id in loaded.grad_fn_logs, (
            f"persisted root id {root_id} dangles outside the remapped grad_fn_logs"
        )
        assert root_id < 1_000_000, f"persisted root id {root_id} looks like a raw memory address"
