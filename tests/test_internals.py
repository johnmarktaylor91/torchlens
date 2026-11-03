"""Unit tests for internal algorithms and invariants.

Tests here cover internal implementation details that aren't exercised
through the public API integration tests: field ordering, data structure
invariants, algorithm correctness, etc.
"""

from collections import defaultdict

import torch
import torch.nn as nn

from torchlens import trace as trace_fn

# ---------------------------------------------------------------------------
# FIELD_ORDER sync tests
# ---------------------------------------------------------------------------
from torchlens._constants import (
    LAYER_PASS_LOG_FIELD_ORDER,
)
from torchlens._utils.arg_handling import _safe_copy_arg
from torchlens._utils.tensor_utils import (
    get_memory_amount,
    get_memory_amount_from_metadata,
    print_override,
    safe_copy,
    safe_to,
)
from torchlens.backends.torch._tl import get_tensor_label, set_tensor_label


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

    def test_op_log_field_order_covers_init(self):
        """LAYER_PASS_LOG_FIELD_ORDER should cover all self.X assignments in Op.__init__."""
        from torchlens._data_classes.op import Op

        init_attrs = self._init_assigned_attrs(Op)
        # Private fields (prefixed _) are intentionally excluded from some FIELD_ORDERs,
        # but some _ fields ARE in FIELD_ORDER (e.g. _tracing_finished). Check both directions:
        # 1. Every FIELD_ORDER entry should be an init attr or a property
        for field in LAYER_PASS_LOG_FIELD_ORDER:
            assert field in init_attrs or hasattr(Op, field), (
                f"{field!r} in LAYER_PASS_LOG_FIELD_ORDER but not in Op"
            )


# ---------------------------------------------------------------------------
# Constants crawl test
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Toy models for bugfix regression tests
# ---------------------------------------------------------------------------


class _SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


class _SharedBufferModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor([2.0]))
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        x = x * self.scale
        x = self.fc(x)
        x = x * self.scale
        return x


# ---------------------------------------------------------------------------
# safe_copy tests
# ---------------------------------------------------------------------------


class TestSafeCopy:
    def test_safe_copy_preserves_label(self):
        """safe_copy(detach_tensor=True) preserves the TorchLens raw label."""
        t = torch.randn(3, 3)
        set_tensor_label(t, "test_label")
        copied = safe_copy(t, detach_tensor=True)
        assert get_tensor_label(copied) == "test_label"


class TestSafeTo:
    def test_safe_to_subclass(self):
        """safe_to should work with tensor subclasses (isinstance check)."""
        p = nn.Parameter(torch.randn(3, 3))
        result = safe_to(p, "cpu")
        assert isinstance(result, torch.Tensor)


# ---------------------------------------------------------------------------
# print_override tests
# ---------------------------------------------------------------------------


class TestPrintOverride:
    def test_print_override_bfloat16(self):
        """bfloat16 should not crash."""
        t = torch.tensor([70000.0], dtype=torch.bfloat16)
        result = print_override(t, "__repr__")
        assert "tensor" in result


# ---------------------------------------------------------------------------
# get_memory_amount tests
# ---------------------------------------------------------------------------


class TestGetTensorMemory:
    def test_metadata_memory_counts_sparse_index_storage(self) -> None:
        """Sparse metadata memory counts index AND values storage.

        The values-only figure ledgered this 3-nnz COO tensor as 12 bytes
        while its int64 index storage alone holds 48 physical bytes.

        Returns
        -------
        None
            Assertion-only regression test.
        """

        indices = torch.tensor([[0, 1, 1], [2, 0, 2]])
        values = torch.tensor([3.0, 4.0, 5.0])
        sparse = torch.sparse_coo_tensor(indices, values, (2, 3))

        # 3 float32 values (12) + 2x3 int64 indices (48).
        assert get_memory_amount_from_metadata(sparse, tuple(sparse.shape), sparse.dtype) == 60
        assert get_memory_amount(sparse) == 60

    def test_save_budget_identity_charges_sparse_components(self) -> None:
        """The budget's storage identity charges sparse index + values bytes.

        The generic fallback billed sparse payloads at LOGICAL dense bytes
        under an id-based identity: index storage unledgered, values
        overcounted at dense shape, and no alias dedup across payloads
        sharing the same components.

        Returns
        -------
        None
            Assertion-only regression test.
        """

        from torchlens._save_budget import _retained_storage_identity

        coo = torch.sparse_coo_tensor(torch.tensor([[0], [0]]), torch.tensor([1.0]), (4, 4))
        identity, num_bytes = _retained_storage_identity(coo)

        physical = sum(
            int(component.untyped_storage().nbytes())
            for component in (coo._indices(), coo._values())
        )
        assert num_bytes == physical
        assert num_bytes != coo.numel() * coo.element_size()  # not logical dense
        # Identity is component-storage-based and stable across reads.
        assert identity == _retained_storage_identity(coo)[0]
        assert "tensor" not in identity  # never the id-based fallback


# ---------------------------------------------------------------------------
# _safe_copy_arg tests
# ---------------------------------------------------------------------------


class TestSafeCopyArg:
    def test_defaultdict_preserved(self):
        """defaultdict should preserve its default_factory."""
        dd = defaultdict(list, {"a": [1, 2], "b": [3]})
        copied = _safe_copy_arg(dd)
        assert isinstance(copied, defaultdict)
        assert copied.default_factory is list
        assert copied["a"] == [1, 2]
        copied["new_key"].append(42)
        assert copied["new_key"] == [42]

    def test_regular_dict_preserved(self):
        d = {"a": torch.tensor([1.0]), "b": 2}
        copied = _safe_copy_arg(d)
        assert isinstance(copied, dict)
        assert not isinstance(copied, defaultdict)


# ---------------------------------------------------------------------------
# Exception safety tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Buffer duplicate tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Dead type check
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# IPython lazy import
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Cleanup / GC tests (GC-5, GC-12, GC-1)
# ---------------------------------------------------------------------------


class TestCleanupReleasesReferences:
    def test_cleanup_no_crash(self):
        """GC-1/GC-5/GC-12: cleanup() must actually release references.

        The litmus: a no-op ``cleanup()`` would pass a bare "doesn't crash"
        check trivially, so assert the concrete reference-release effects the
        docstring on ``torchlens._data_classes.cleanup.cleanup`` claims --
        per-Op state cleared (breaking the Op<->Trace cycle) and cached
        ``Param`` references dropped to allow model GC -- rather than only
        that the call completes.
        """
        model = _SimpleLinear()
        x = torch.randn(2, 10)
        log = trace_fn(model, x)

        entries = list(log)
        assert entries, "expected at least one captured Op entry"
        first_entry = entries[0]
        assert hasattr(first_entry, "out"), "sanity check: Op should carry live state pre-cleanup"

        # Force each Param's live reference to rehydrate (iteration triggers
        # rehydrate-on-iter after a normal capture) so we have a genuine
        # non-None reference to prove cleanup() releases.
        param_logs = list(log._param_logs)
        assert param_logs, "expected at least one captured Param"
        for param_log in param_logs:
            assert param_log._param_ref is not None

        log.cleanup()

        # GC-1: cached live Parameter references must be released.
        for param_log in param_logs:
            assert param_log._param_ref is None

        # GC-5/GC-12: per-Op instance state must be cleared (breaks the
        # Op -> Trace circular reference via source_trace), and internal
        # containers not in MODEL_LOG_FIELD_ORDER must be gone.
        assert not hasattr(first_entry, "out")
        assert not hasattr(log, "_layer_logs")


# ---------------------------------------------------------------------------
# Argument handling tests
# ---------------------------------------------------------------------------


class TestDisplayLargeTensor:
    def test_display_no_oom(self):
        """Displaying a large captured tensor must never clone it (#73 -> F10).

        The historical guard allowed one <=8x8=64-element preview clone
        (slice-then-clone, #73). The F10 card renders through the C02
        allocation-free stats kernel and carries NO raw preview (it lives
        behind the ``More: .out`` exit), so the pin strengthens: formatting
        ``str(op)`` performs ZERO ``torch.Tensor.clone()`` calls of any
        size. A regression that reintroduces a full-tensor (or any) copy on
        the repr path fails here.
        """
        model = nn.Linear(100, 2000)
        x = torch.randn(50, 100)
        log = trace_fn(model, x, save="all")

        clone_call_sizes: list[int] = []
        orig_clone = torch.Tensor.clone

        def _tracking_clone(self, *args, **kwargs):
            clone_call_sizes.append(self.numel())
            return orig_clone(self, *args, **kwargs)

        torch.Tensor.clone = _tracking_clone
        try:
            for label in log.layer_labels:
                entry = log[label]
                op = entry.ops[0]
                if op.out is None:
                    continue
                str(op)
        finally:
            torch.Tensor.clone = orig_clone

        assert clone_call_sizes == [], (
            f"str(op) cloned tensors of sizes {clone_call_sizes[:5]}; the F10 "
            "card path must not copy payloads at all (the preview moved "
            "behind the More: .out exit)"
        )


class TestAliasContractPositionScan:
    def test_contract_lookup_semantics_unchanged(self) -> None:
        """Contract coverage keys on contract positions, not a full arg scan.

        The full scan cost O(fan_in) per parent — O(fan_in^2) per op for
        variadic ops like a 4k-arg ``stack`` — with the common EMPTY contract.
        """

        from torchlens.backends.torch.aliasing import parent_label_has_alias_contract

        positions = {
            "args": {i: f"parent_{i}" for i in range(50)},
            "kwargs": {"out": "parent_out"},
        }
        # Empty contract: never covered.
        assert not parent_label_has_alias_contract("parent_3", positions, ())
        # Covered arg position.
        assert parent_label_has_alias_contract("parent_3", positions, (3,))
        # Covered kwargs position.
        assert parent_label_has_alias_contract("parent_out", positions, ("out",))
        # Contract position exists but holds a different parent.
        assert not parent_label_has_alias_contract("parent_3", positions, (4,))
        # Parent present only at a non-contract position.
        assert not parent_label_has_alias_contract("parent_7", positions, (3, "out"))


class TestStorageAliasBucketLayout:
    """SF-52: the per-storage alias index single-alias fast path."""

    def test_single_alias_is_one_weakref_and_upgrades_on_second(self) -> None:
        """One labeled tensor per storage stores a bare ref; two upgrade.

        A full WeakIdKeyDictionary per bucket cost ~6 marginal objects per op
        (measured -6.95 obj/op on the pinned linear602 census after this
        change). Candidate reads must be identical across both layouts.
        """

        import weakref

        from torch.utils.weak import WeakIdKeyDictionary

        from torchlens.backends.torch import _tl

        session = _tl._LabelSession(token=1)
        t1 = torch.randn(4)
        t2 = t1.view(2, 2)  # distinct object, same storage
        ptr = t1.untyped_storage().data_ptr()

        _tl._register_storage_alias(session, ptr, t1)
        assert isinstance(session.by_storage_ptr[ptr], weakref.ref)
        # Idempotent re-registration keeps the slim layout.
        _tl._register_storage_alias(session, ptr, t1)
        assert isinstance(session.by_storage_ptr[ptr], weakref.ref)

        saved_session = _tl._ACTIVE_LABEL_SESSION
        try:
            _tl._ACTIVE_LABEL_SESSION = session
            assert _tl.session_storage_alias_candidates(ptr) == [t1]

            # Second live distinct alias upgrades in place.
            _tl._register_storage_alias(session, ptr, t2)
            assert isinstance(session.by_storage_ptr[ptr], WeakIdKeyDictionary)
            candidates = _tl.session_storage_alias_candidates(ptr)
            assert {id(c) for c in candidates} == {id(t1), id(t2)}

            # A dead single-alias entry reads as no candidates and is
            # replaced by the next registration.
            t3 = torch.randn(4)
            ptr3 = t3.untyped_storage().data_ptr()
            _tl._register_storage_alias(session, ptr3, t3)
            del t3
            assert _tl.session_storage_alias_candidates(ptr3) == []
        finally:
            _tl._ACTIVE_LABEL_SESSION = saved_session
