"""model_prep hardening (fw2 b4-P slice): SF-45 namespace walk + B1-13a disclosures.

SF-45: ``torch.distributed.reduce_op`` is a deprecation singleton whose
``__getattribute__`` emits a ``FutureWarning`` on ANY attribute access,
including the ``__class__`` read every ``isinstance()`` performs. The session
cleanup walks reach it through model-owned helper objects that reference
broadly-imported modules, so an initialized distributed process running under
``-W error::FutureWarning`` had its capture CLEANUP aborted.

B1-13a: a failed buffer-provenance stamp is capture-evidence loss (the
buffer's reads may log as internal sources instead of buffer versions); it
was silently swallowed with ``except Exception: pass``.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl


class _Holder:
    """Plain model-owned helper object carrying a module reference."""


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed unavailable")
class TestNamespaceWalkSkipsDeprecationShim:
    """SF-45: the cleanup walks never touch the reduce_op deprecation singleton."""

    # Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
    def test_object_walk_is_silent_under_error_promotion(self) -> None:
        """Fail-before: the isinstance chain warned at the walk's type checks."""

        from torchlens.backends.torch.model_prep import _clear_session_tensor_metadata

        holder = _Holder()
        holder.reduce_op = torch.distributed.reduce_op  # type: ignore[attr-defined]
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            _clear_session_tensor_metadata(holder, set())

    # Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
    def test_module_namespace_walk_is_silent_under_error_promotion(self) -> None:
        """The ModuleType slot scan skips the singleton before isinstance."""

        from torchlens.backends.torch.model_prep import _clear_session_tensor_metadata

        holder = _Holder()
        holder.dist = torch.distributed  # type: ignore[attr-defined]
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            _clear_session_tensor_metadata(holder, set())

    # Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/_session_cleanup.py, campaign mutant kills.
    def test_container_walk_is_silent_under_error_promotion(self) -> None:
        """The pure container-tree descent skips the singleton too."""

        from torchlens.backends.torch.model_prep import (
            _clear_container_tree_tensor_metadata,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            _clear_container_tree_tensor_metadata(
                [torch.distributed.reduce_op, torch.ones(2)], set(), 0
            )


class TestBufferStampFailureIsDisclosed:
    """B1-13a: a failed buffer-provenance stamp warns instead of vanishing."""

    # Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/model_prep.py, campaign mutant kills.
    def test_unstampable_buffer_warns_with_address(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fail-before: ``except Exception: pass`` hid the evidence loss."""

        import torchlens.backends.torch.buffer_writes as buffer_writes

        real_stamp = buffer_writes.register_session_buffer_stamp
        fired = []

        def failing_stamp(trace, tensor, address):  # type: ignore[no-untyped-def]
            # Fail only the prep-time scan's stamp; the same helper is routed
            # by mid-forward stamping paths that have their own handling.
            if address.endswith("mask") and not fired:
                fired.append(address)
                raise AttributeError("simulated unstampable tensor subclass")
            return real_stamp(trace, tensor, address)

        monkeypatch.setattr(buffer_writes, "register_session_buffer_stamp", failing_stamp)

        class WithBuffer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("mask", torch.ones(4))
                self.fc = nn.Linear(4, 4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x * self.mask)

        with pytest.warns(UserWarning, match="buffer provenance.*mask"):
            log = tl.trace(WithBuffer(), torch.randn(2, 4))
        log.cleanup()
