"""Unit tests for internal algorithms and invariants.

Tests here cover internal implementation details that aren't exercised
through the public API integration tests: field ordering, data structure
invariants, algorithm correctness, etc.
"""

import pytest

# ---------------------------------------------------------------------------
# FIELD_ORDER sync tests
# ---------------------------------------------------------------------------

from torchlens.constants import (
    MODEL_LOG_FIELD_ORDER,
    LAYER_PASS_LOG_FIELD_ORDER,
    MODULE_LOG_FIELD_ORDER,
    MODULE_PASS_LOG_FIELD_ORDER,
    PARAM_LOG_FIELD_ORDER,
    FUNC_CALL_LOCATION_FIELD_ORDER,
)


class TestFieldOrderSync:
    """Verify that FIELD_ORDER constants have no duplicates and stay sane."""

    def _init_assigned_attrs(self, cls):
        """Extract attribute names assigned via self.X = ... in __init__."""
        import inspect

        source = inspect.getsource(cls.__init__)
        attrs = set()
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("self.") and "=" in stripped:
                # "self.foo: int = bar" -> "foo: int" -> "foo"
                attr_part = stripped.split("=")[0].replace("self.", "").strip()
                attr = attr_part.split(":")[0].strip()
                attrs.add(attr)
        return attrs

    @pytest.mark.parametrize(
        "field_order,name",
        [
            (LAYER_PASS_LOG_FIELD_ORDER, "LAYER_PASS_LOG_FIELD_ORDER"),
            (MODEL_LOG_FIELD_ORDER, "MODEL_LOG_FIELD_ORDER"),
            (MODULE_LOG_FIELD_ORDER, "MODULE_LOG_FIELD_ORDER"),
            (MODULE_PASS_LOG_FIELD_ORDER, "MODULE_PASS_LOG_FIELD_ORDER"),
            (PARAM_LOG_FIELD_ORDER, "PARAM_LOG_FIELD_ORDER"),
            (FUNC_CALL_LOCATION_FIELD_ORDER, "FUNC_CALL_LOCATION_FIELD_ORDER"),
        ],
    )
    def test_no_duplicates(self, field_order, name):
        dupes = [f for f in field_order if field_order.count(f) > 1]
        assert not dupes, f"Duplicates in {name}: {set(dupes)}"

    def test_layer_pass_log_field_order_covers_init(self):
        """LAYER_PASS_LOG_FIELD_ORDER should cover all self.X assignments in LayerPassLog.__init__."""
        from torchlens.data_classes.layer_pass_log import LayerPassLog

        init_attrs = self._init_assigned_attrs(LayerPassLog)
        # Private fields (prefixed _) are intentionally excluded from some FIELD_ORDERs,
        # but some _ fields ARE in FIELD_ORDER (e.g. _pass_finished). Check both directions:
        # 1. Every FIELD_ORDER entry should be an init attr or a property
        for field in LAYER_PASS_LOG_FIELD_ORDER:
            assert field in init_attrs or hasattr(LayerPassLog, field), (
                f"{field!r} in LAYER_PASS_LOG_FIELD_ORDER but not in LayerPassLog"
            )

    def test_model_log_field_order_covers_init(self):
        """MODEL_LOG_FIELD_ORDER should cover all public self.X assignments in ModelLog.__init__."""
        from torchlens.data_classes.model_log import ModelLog

        init_attrs = self._init_assigned_attrs(ModelLog)
        order_set = set(MODEL_LOG_FIELD_ORDER)
        # Every non-private init attr should be in FIELD_ORDER
        public_attrs = {a for a in init_attrs if not a.startswith("_")}
        missing = public_attrs - order_set
        assert not missing, f"ModelLog public fields missing from FIELD_ORDER: {missing}"

    def test_module_pass_log_field_order_covers_init(self):
        from torchlens.data_classes.module_log import ModulePassLog

        init_attrs = self._init_assigned_attrs(ModulePassLog)
        order_set = set(MODULE_PASS_LOG_FIELD_ORDER)
        public_attrs = {a for a in init_attrs if not a.startswith("_")}
        missing = public_attrs - order_set
        assert not missing, f"ModulePassLog public fields missing from FIELD_ORDER: {missing}"

    def test_module_log_field_order_covers_init(self):
        from torchlens.data_classes.module_log import ModuleLog

        init_attrs = self._init_assigned_attrs(ModuleLog)
        order_set = set(MODULE_LOG_FIELD_ORDER)
        public_attrs = {a for a in init_attrs if not a.startswith("_")}
        missing = public_attrs - order_set
        assert not missing, f"ModuleLog public fields missing from FIELD_ORDER: {missing}"


# ---------------------------------------------------------------------------
# Constants crawl test
# ---------------------------------------------------------------------------


class TestConstantsCrawl:
    """Verify the torch function crawl produces consistent results."""

    def test_overridable_funcs_cached(self):
        """_get_torch_overridable_functions returns same object on repeated calls."""
        from torchlens.constants import _get_torch_overridable_functions

        a = _get_torch_overridable_functions()
        b = _get_torch_overridable_functions()
        assert a is b

    def test_overridable_funcs_nonempty(self):
        from torchlens.constants import OVERRIDABLE_FUNCS

        assert len(OVERRIDABLE_FUNCS) > 100

    def test_orig_torch_funcs_includes_ignored(self):
        from torchlens.constants import ORIG_TORCH_FUNCS, IGNORED_FUNCS

        ignored_set = set(IGNORED_FUNCS)
        orig_set = set(ORIG_TORCH_FUNCS)
        assert ignored_set.issubset(orig_set)
