"""Run torch captures through the structural grouper (``grouper_v1``).

Torch exhaustive captures group layers by call site (``callsite_v1``). The
backend-neutral structural grouper still serves every capture without call-site
evidence (other backends, predicate-mode recordings), and its own hardening
suites drive it through torch models. This fixture makes the capture skip the
call-site equivalence class, so step 7 sees exactly the legacy fingerprints and
runs ``grouper_v1`` as those suites expect.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from torchlens.backends.torch import _ops_activations, ops


@pytest.fixture
def legacy_structural_grouper(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Capture with legacy equivalence classes so step 7 runs ``grouper_v1``."""

    def _no_call_site(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(ops, "_call_site_equivalence_class", _no_call_site)
    monkeypatch.setattr(_ops_activations, "_call_site_equivalence_class", _no_call_site)
    yield
