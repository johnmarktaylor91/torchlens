"""Red-capable gates for security and runnable wire-format single sources."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest


def authority_drift(values: Sequence[Any], *, require_identity: bool) -> tuple[int, ...]:
    """Return indexes whose value diverges from the first authority.

    Parameters
    ----------
    values:
        Ordered authority values with the canonical value first.
    require_identity:
        Whether aliases must be the same object as well as equal.

    Returns
    -------
    tuple[int, ...]
        Indexes of divergent values.
    """

    if not values:
        return ()
    canonical = values[0]
    return tuple(
        index
        for index, value in enumerate(values[1:], start=1)
        if value != canonical or (require_identity and value is not canonical)
    )


def test_security_policy_uses_one_callable_safety_authority() -> None:
    """The unpickler aliases every shared security policy from callable safety."""

    from torchlens._io import _safe_unpickle
    from torchlens.utils import _callable_safety

    authority_pairs = (
        (_callable_safety._DENIED_MODULES, _safe_unpickle._DENIED_FOREIGN_MODULES),
        (_callable_safety._ALLOWED_STDLIB_ROOTS, _safe_unpickle._ALLOWED_STDLIB_ROOTS),
        (
            _callable_safety._STDLIB_AND_BUILTIN_TOP_LEVEL,
            _safe_unpickle._STDLIB_AND_BUILTIN_TOP_LEVEL,
        ),
        (
            _callable_safety.is_denied_stdlib_or_builtin_module,
            _safe_unpickle._stdlib_or_builtin_denied,
        ),
        (_callable_safety._APPLIANCE_MODULES, _safe_unpickle._TORCHLENS_APPLIANCE_MODULES),
    )
    for values in authority_pairs:
        assert not authority_drift(values, require_identity=True)


def test_fenced_appliance_pin_matches_the_canonical_authority() -> None:
    """The resolver's fenced appliance copy stays equal to the canonical set."""

    from torchlens.intervention import resolver
    from torchlens.utils import _callable_safety

    assert not authority_drift(
        (
            _callable_safety._APPLIANCE_MODULES,
            resolver._TORCHLENS_APPLIANCE_MODULES,
        ),
        require_identity=False,
    )


class TestSecurityDriftGateIsRedCapable:
    """Plant policy divergence and prove each comparison can fail."""

    def test_value_drift_is_detected(self) -> None:
        """A one-sided security-set addition is reported."""

        canonical = frozenset({"os", "pickle"})
        planted = canonical | {"subprocess"}
        assert authority_drift((canonical, planted), require_identity=False) == (1,)

    def test_copy_drift_is_detected_when_identity_is_required(self) -> None:
        """An equal-but-independent replacement fails a single-source gate."""

        canonical = frozenset({"os", "pickle"})
        copied = frozenset(value for value in canonical)
        assert copied == canonical
        assert copied is not canonical
        assert authority_drift((canonical, copied), require_identity=True) == (1,)

    def test_callable_drift_is_detected(self) -> None:
        """A reimplemented predicate fails the identity gate."""

        def canonical(value: str) -> bool:
            """Return whether the planted value is denied."""

            return value == "os"

        def planted(value: str) -> bool:
            """Return a deliberately divergent planted decision."""

            return value in {"os", "sys"}

        assert authority_drift((canonical, planted), require_identity=True) == (1,)


@pytest.mark.smoke
def test_runnable_wire_vocabulary_uses_one_authority_per_concept() -> None:
    """Writers and readers alias the same prefix and closed-vocabulary objects."""

    from torchlens import _input_walk, _runnable_execution, _runnable_state
    from torchlens._io import runnable_load
    from torchlens.backends.torch import ops
    from torchlens.ir import container
    from torchlens.utils import _callable_safety

    authority_groups = (
        (
            _runnable_state._INPUT_STRUCTURE_SITE_PREFIX,
            _runnable_execution._INPUT_STRUCTURE_SITE_PREFIX,
            runnable_load._INPUT_STRUCTURE_SITE_PREFIX,
        ),
        (
            _runnable_state._STATE_METADATA_FACT_SITE_PREFIX,
            _runnable_execution._STATE_METADATA_FACT_SITE_PREFIX,
            runnable_load._STATE_METADATA_FACT_SITE_PREFIX,
        ),
        (_input_walk.INPUT_CONTAINER_KINDS, runnable_load._INPUT_STRUCTURE_NODE_KINDS),
        (
            _callable_safety._PURE_TENSOR_PROPERTY_NAMES,
            ops._SAFE_TENSOR_PROPERTY_NAMES,
            runnable_load._SAFE_TENSOR_PROPERTY_NAMES,
        ),
        (container._SAFE_DEFAULT_FACTORIES, ops._SAFE_DEFAULT_FACTORIES),
    )
    for values in authority_groups:
        assert not authority_drift(values, require_identity=True)


class TestRunnableWireDriftGateIsRedCapable:
    """Plant wire-format divergence and prove each closed-set gate reports it."""

    def test_site_prefix_drift_is_detected(self) -> None:
        """A reader prefix that differs from its writer is reported."""

        assert authority_drift(
            ("input_structure:", "input_structure_v2:"), require_identity=False
        ) == (1,)

    def test_container_kind_drift_is_detected(self) -> None:
        """A parse-only container kind is reported."""

        canonical = frozenset({"tensor", "mapping", "leaf"})
        planted = canonical | {"planted"}
        assert authority_drift((canonical, planted), require_identity=False) == (1,)

    def test_safe_property_drift_is_detected(self) -> None:
        """A capture-only safe tensor property is reported."""

        canonical = frozenset({"T", "real"})
        planted = canonical | {"data"}
        assert authority_drift((canonical, planted), require_identity=False) == (1,)

    def test_default_factory_drift_is_detected(self) -> None:
        """A capture-only default factory is reported."""

        canonical = {"list": list, "dict": dict}
        planted = {**canonical, "set": set}
        assert authority_drift((canonical, planted), require_identity=False) == (1,)
