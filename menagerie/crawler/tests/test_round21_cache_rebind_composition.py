"""Round-21 VS8 active-authority cache-rebind composition proof."""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.crawler.authority import AuthorityDerivationError, EnvironmentAuthorityCache
from menagerie.crawler.tests import test_round17_structural_inventories as structural
from menagerie.crawler.tests.conftest import RealEnvironmentFixture
from menagerie.crawler.tests.test_round21_environment_matrix_composition import (
    _adapter_source,
    _assert_award,
    _run_composition,
)


def _cache_counters(cache: EnvironmentAuthorityCache) -> dict[str, int]:
    """Return counters that a rejected rebind must not change.

    Parameters
    ----------
    cache:
        Active lifecycle-owned environment authority cache.

    Returns
    -------
    dict[str, int]
        Snapshot of every pre-VS8 validation, seal, spawn, and invalidation counter.
    """

    return {
        name: getattr(cache, name)
        for name in (
            "full_seals",
            "cheap_validations",
            "cheap_tree_walks",
            "lstat_tree_walks",
            "currentness_passes",
            "spawn_validations",
            "real_spawns",
            "invalidations",
            "rehashes",
        )
    }


def test_round21_mismatched_rebind_preserves_active_authority_and_awards(
    tmp_path: Path,
    real_environment_fixture: RealEnvironmentFixture,
) -> None:
    """Rejected rebinds preserve one real seal and its later shipped award.

    Parameters
    ----------
    tmp_path:
        Isolated real-composition campaign root and mismatched-prefix location.
    real_environment_fixture:
        Strictly bound lock-selected hardlink clone used by the shipped compiler.
    """

    binding = real_environment_fixture.binding
    cache = binding.environment_authority_cache
    authority = binding.environment_authority
    assert cache is not None
    assert authority is not None
    manifest = cache.manifest
    assert manifest is authority.content_manifest
    initial_epoch = cache._epoch  # noqa: SLF001
    initial_counters = _cache_counters(cache)
    assert getattr(cache, "rejected_rebinds", 0) == 0

    mismatched_prefix = tmp_path / "mismatched-prefix"
    mismatched_prefix.mkdir()
    mismatched_inputs: tuple[tuple[Path, Path, str], ...] = (
        (
            real_environment_fixture.prefix,
            binding.python_executable,
            "sha256:" + "0" * 64,
        ),
        (
            real_environment_fixture.prefix,
            real_environment_fixture.prefix / "bin/python-mismatched",
            authority.base_environment_generation,
        ),
        (
            mismatched_prefix,
            mismatched_prefix / "bin/python",
            authority.base_environment_generation,
        ),
    )
    for rejected_count, (prefix, interpreter, base_generation) in enumerate(
        mismatched_inputs, start=1
    ):
        with pytest.raises(
            AuthorityDerivationError,
            match="active environment authority cache cannot be rebound to different inputs",
        ):
            cache.bind(
                prefix=prefix,
                selected_interpreter=interpreter,
                base_environment_generation=base_generation,
            )
        cache.assert_active(authority)
        assert cache.authority is authority
        assert cache.manifest is manifest
        assert cache._epoch == initial_epoch  # noqa: SLF001
        assert _cache_counters(cache) == initial_counters
        assert cache.rejected_rebinds == rejected_count

    rebound = cache.bind(
        prefix=real_environment_fixture.prefix,
        selected_interpreter=binding.python_executable,
        base_environment_generation=authority.base_environment_generation,
        validate_active=False,
    )
    assert rebound is authority
    assert cache.manifest is manifest
    assert cache.full_seals == 1
    assert cache.invalidations == 0

    observation = _run_composition(
        tmp_path / "award",
        real_environment_fixture,
        _adapter_source(
            "assert os.environ.get('MENAGERIE_ROUND19_PTH_SENTINEL') == 'sealed-startup'"
        ),
    )
    _assert_award(observation)
    cache.assert_active(authority)
    assert cache.authority is authority
    assert cache.full_seals == 1
    assert cache.invalidations == 0
    assert cache.rejected_rebinds == 3
    assert set(structural.ROUND21_VS8_PROOF_REGISTRY) == {
        "P01",
        "P02",
        "P03",
        "P04",
        "P05",
        "P06",
        "P07",
        "P08",
        "P12",
        "P13",
        "P14",
        "P17",
        "P19",
        "T01",
        "T01-CI",
        "T02",
        "T03",
    }
