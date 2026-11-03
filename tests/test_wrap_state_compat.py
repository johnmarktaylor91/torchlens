"""Wrap-history compatibility batteries: torch meta-APIs and identity bases.

Regression batteries for the wrap-state identity class (R55/R56/R78 family):

1. ``R55.defaults`` scan — walk every loaded ``torch.*`` module for function
   defaults frozen to a wrapped original at torch-import time. The class is
   closed (the transformer trio, shimmed; ``torch.jit._trace.verify``,
   irrelevant); a new torch release adding a fourth site fails here.
2. Installed-tree identity grep gate — every ``is F.<name>`` / ``is
   torch.*.<name>`` comparison in torch's eager runtime source that resolves
   to a WRAPPED callable must be a reviewed site (shimmed or a documented
   residual). A new torch release adding one fails here.
3. ``R55.eco`` meta-API smoke — torch meta-API entry points that historically
   broke (or could break) while wrappers are installed, run post-wrap.
4. Override-table coherence (B8-4) — both ``functools.cache``'d torch
   introspection tables must be keyed by ORIGINALS: no torchlens path may
   first-call one while wrapped (the poisoned table survived
   ``unwrap_torch()`` and made belt derivation order-dependent).
5. Wrap-history construction corpus — the transformer trio built under wrap
   must match an unwrapped construction: state bytes, stored activation
   object, fastpath flag, pickle-ability, and legacy ``__setstate__``.
"""

from __future__ import annotations

import io
import pickle
import re
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchlens as tl
from torchlens import _state


def _ensure_wrapped() -> None:
    """Force the lazy torch wrap through the public capture path."""

    tl.trace(nn.Linear(2, 2), torch.randn(1, 2))


def _resolve(fn):
    """Follow the wrapper ledger to the original callable."""

    seen: set[int] = set()
    while id(fn) in _state._decorated_to_orig and id(fn) not in seen:
        seen.add(id(fn))
        fn = _state._decorated_to_orig[id(fn)]
    return fn


# ---------------------------------------------------------------------------
# 1. R55.defaults scan
# ---------------------------------------------------------------------------

# (function __module__, __qualname__) pairs allowed to hold a wrapped original
# in __defaults__/__kwdefaults__. Transformer.__init__ delegates activation to
# the SHIMMED layer ctors; torch.jit._trace.verify is outside capture scope.
_DEFAULTS_ALLOWLIST = {
    ("torch.nn.modules.transformer", "Transformer.__init__"),
    ("torch.nn.modules.transformer", "TransformerEncoderLayer.__init__"),
    ("torch.nn.modules.transformer", "TransformerDecoderLayer.__init__"),
    ("torch.jit._trace", "verify"),
}


# ---------------------------------------------------------------------------
# 2. Installed-tree identity grep gate
# ---------------------------------------------------------------------------

# Namespaces outside eager capture scope by the shim census contract
# (compiler/export/testing stacks, distributed, dispatch-level machinery).
_GREP_SKIP_PREFIXES = (
    "test",
    "testing",
    "_dynamo",
    "_inductor",
    "_export",
    "export",
    "_lazy",
    "_subclasses",
    "_functorch",
    "func",
    "fx",
    "onnx",
    "jit",
    "ao",
    "quantization",
    "distributed",
    "distributions",
    "masked",
    "signal",
    "special",
    "backends",
    "cuda",
    "xpu",
    "mtia",
    "compiler",
    "profiler",
    "package",
    "monitor",
    "overrides",
    "_refs",
    "_decomp",
    "_prims",
    "_meta_registrations",
    "_utils/_sympy",
)

# (relative source path, attribute name) pairs reviewed 2026-08-14:
# transformer/bias/expanded-weights sites are SHIMMED; the nested/_internal
# NJT identity reads are a documented unshimmed residual (nested jagged
# tensors are not supported capture inputs).
_GREP_ALLOWLIST = {
    ("nn/modules/transformer.py", "relu"),
    ("nn/modules/transformer.py", "gelu"),
    ("nn/attention/bias.py", "scaled_dot_product_attention"),
    ("nn/utils/_expanded_weights/conv_utils.py", "conv1d"),
    ("nn/utils/_expanded_weights/conv_utils.py", "conv2d"),
    ("nn/utils/_expanded_weights/conv_utils.py", "conv3d"),
    ("nn/utils/_expanded_weights/expanded_weights_impl.py", "_cudnn_rnn_flatten_weight"),
    ("nested/_internal/nested_tensor.py", "size"),
    ("nested/_internal/nested_tensor.py", "dim"),
    ("nested/_internal/ops.py", "scaled_dot_product_attention"),
}

_IDENTITY_PATTERN = re.compile(
    r"\bis\s+(?:not\s+)?(F|torch(?:\.[a-zA-Z_][\w.]*)?)\.([a-zA-Z_]\w*)\b"
)


# ---------------------------------------------------------------------------
# 3. R55.eco meta-API smoke
# ---------------------------------------------------------------------------


class TestMetaApiSmokePostWrap:
    # Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/identity_shims.py, campaign mutant kills.
    def test_resolve_name_matches_unwrapped_answer(self):
        _ensure_wrapped()
        assert torch.overrides.resolve_name(F.relu) == "torch.nn.functional.relu"
        assert torch.overrides.resolve_name(_resolve(F.relu)) == "torch.nn.functional.relu"


# ---------------------------------------------------------------------------
# 4. Override-table coherence (B8-4)
# ---------------------------------------------------------------------------


class TestDerivedCacheCensus:
    """R56 derived-cache census (b8-fable/b8-opus round 3).

    Torch's DERIVED caches memoize sets/maps keyed by the live callables
    resolved from the namespace at materialization time. The attribute-identity
    restore census structurally cannot see them: a table materialized while
    torchlens wrappers are installed holds wrappers, misses identity tests
    against originals, and (for tables built mid-epoch) survives
    ``unwrap_torch()``. Every identity-keyed derived cache torchlens knows
    about must be keyed by ORIGINALS during the wrapped epoch, or dropped at
    unwrap so torch re-derives it from the restored originals.
    """

    # Kept by shrink-r4 (per-id subset rule): sole coverage of planted fault catches.
    def test_device_constructor_cache_keyed_by_originals(self):
        # b8-fable-R56-1: decorate_all_once used to cache_clear+re-materialize
        # torch's _device_constructors() AFTER installing wrappers, so the
        # memoized set held torchlens wrappers for the whole epoch.
        _ensure_wrapped()
        from torch.utils._device import _device_constructors

        wrapper_ids = set(_state._decorated_to_orig.keys())
        poisoned = [
            getattr(fn, "__name__", repr(fn))
            for fn in _device_constructors()
            if id(fn) in wrapper_ids
        ]
        assert not poisoned, (
            f"_device_constructors() holds {len(poisoned)} torchlens wrappers "
            f"({poisoned[:5]}...): DeviceContext.__torch_function__ receives "
            "ORIGINALS, so C-level device injection misses for stale pre-wrap "
            "factory references"
        )

    # Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
    def test_dynamo_rule_map_built_mid_epoch_does_not_survive_unwrap(self):
        # b8-opus-R56-1: a rule map materialized while wrapped is keyed by
        # wrappers; without the unwrap-time clear it survives unwrap_torch()
        # and torch.compile(fullgraph=True) fails (lookup(torch.cos) degrades
        # to SkipFunctionVariable).
        trace_rules = pytest.importorskip("torch._dynamo.trace_rules")
        from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

        _ensure_wrapped()
        try:
            # Simulate the user materializing the tables mid-epoch.
            trace_rules.get_torch_obj_rule_map.cache_clear()
            trace_rules.get_tensor_method.cache_clear()
            wrapped_map = trace_rules.get_torch_obj_rule_map()
            wrapped_methods = trace_rules.get_tensor_method()
            unwrap_torch()
            fresh_map = trace_rules.get_torch_obj_rule_map()
            fresh_methods = trace_rules.get_tensor_method()
            assert fresh_map is not wrapped_map, (
                "get_torch_obj_rule_map() built during the wrapped epoch survived unwrap_torch()"
            )
            assert fresh_methods is not wrapped_methods, (
                "get_tensor_method() built during the wrapped epoch survived unwrap_torch()"
            )
            assert torch.cos in fresh_map, (
                "restored torch.cos missing from the re-derived dynamo rule "
                "map: post-unwrap torch.compile(fullgraph=True) would fail"
            )
        finally:
            wrap_torch()

    # Kept by shrink-r4 (per-id subset rule): sole coverage of _constants.py.
    def test_submodule_alias_imported_mid_epoch_is_restored_by_unwrap(self):
        # torch.onnx.operators re-exports torch._shape_as_tensor under a
        # second name ("from torch import _shape_as_tensor as shape_as_tensor"),
        # lazily imported by torch itself rather than eagerly by `import
        # torch` -- so the one-time ORIG_TORCH_FUNCS scan at `import
        # torchlens` time can miss it. If that submodule's FIRST import lands
        # while wrappers are installed (simulated here by importing it
        # mid-epoch, exactly as torch._dynamo.trace_rules's own rule-table
        # build does when resolving "torch.onnx.operators.shape_as_tensor"),
        # its module-level alias binds to the WRAPPER; without tracking this
        # alias, unwrap_torch() never restores it and dynamo's identity-keyed
        # rule map stays permanently poisoned for the rest of the process
        # (round-2 CI triage, 2026-10-01).
        import importlib

        from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

        sys.modules.pop("torch.onnx.operators", None)
        _ensure_wrapped()
        try:
            onnx_operators = importlib.import_module("torch.onnx.operators")
            if id(onnx_operators.shape_as_tensor) not in _state._decorated_to_orig:
                # On some torch builds torch.onnx.operators.shape_as_tensor is
                # its own plain function (or was already eagerly imported into
                # the baseline ORIG_TORCH_FUNCS scan before this test ever
                # ran), not a lazily-bound alias of the currently wrapped
                # torch._shape_as_tensor -- the exact precondition this
                # regression targets does not reproduce on this torch build.
                pytest.skip(
                    "torch.onnx.operators.shape_as_tensor did not bind to the "
                    "current wrapper on mid-epoch import on this torch build "
                    "(the aliasing precondition this test targets is absent here)"
                )
            unwrap_torch()
            assert id(onnx_operators.shape_as_tensor) not in _state._decorated_to_orig, (
                "torch.onnx.operators.shape_as_tensor survived unwrap_torch() as a "
                "torchlens wrapper: the submodule-alias inventory did not track it"
            )
            assert onnx_operators.shape_as_tensor is torch._shape_as_tensor
        finally:
            if not _state._is_decorated:
                wrap_torch()


# ---------------------------------------------------------------------------
# 4b. Unwrap safety: R54 admission-lock atomicity + B8-6 burial diagnostic
# ---------------------------------------------------------------------------


class TestUnwrapSafety:
    # Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/_completeness_finalize.py, backends/torch/backend.py, backends/torch/escape_detection.py ....
    def test_admission_epoch_check_refuses_unwrapped_process(self):
        # R54 second half: a capture admitted AFTER a concurrent unwrap
        # completes must refuse typed rather than run an unlogged forward.
        from torchlens._errors import CaptureNotActiveError
        from torchlens.backends.torch.backend import TorchBackend

        _ensure_wrapped()
        session = tl.trace(nn.Linear(2, 2), torch.randn(1, 2))
        saved = _state._is_decorated
        try:
            _state._is_decorated = False
            with pytest.raises(CaptureNotActiveError) as exc_info:
                with TorchBackend().active_logging(session):
                    pass
            assert exc_info.value.fields["code"] == "wrappers_removed_before_capture"
        finally:
            _state._is_decorated = saved
        # The refusal must have unwound admission cleanly.
        assert _state._active_trace is None
        assert not _state._logging_enabled

    # Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/wrappers.py, campaign mutant kills.
    def test_unwrap_warns_on_buried_wrapper(self):
        # B8-6: a third-party wrapper installed on top of a torchlens wrapper
        # is (correctly) not clobbered at teardown, but the burial must be
        # named instead of silent.
        import functools
        import warnings as warnings_module

        from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

        _ensure_wrapped()
        tl_wrapper = torch.cos
        assert id(tl_wrapper) in _state._decorated_to_orig, "torch.cos not wrapped"

        @functools.wraps(tl_wrapper)
        def foreign(*args, **kwargs):
            return tl_wrapper(*args, **kwargs)

        torch.cos = foreign
        try:
            with warnings_module.catch_warnings(record=True) as records:
                warnings_module.simplefilter("always")
                unwrap_torch()
            buried_messages = [
                str(record.message) for record in records if "buried" in str(record.message)
            ]
            assert len(buried_messages) == 1, buried_messages
            assert "torch.cos" in buried_messages[0]
            # Drift tolerance: the foreign wrapper is preserved, never clobbered.
            assert torch.cos is foreign
        finally:
            torch.cos = _resolve(tl_wrapper)
            wrap_torch()


# ---------------------------------------------------------------------------
# 5. Wrap-history construction corpus (transformer trio + __setstate__)
# ---------------------------------------------------------------------------


def _has_fastpath_flag() -> bool:
    from torchlens._utils import _torch_compat

    return bool(getattr(_torch_compat, "HAS_TRANSFORMER_ACTIVATION_FASTPATH_FLAG", False))


@pytest.mark.skipif(
    not _has_fastpath_flag(),
    reason="torch build lacks the transformer activation fastpath flag",
)
class TestWrapHistoryConstructionCorpus:
    def _build_trio(self):
        torch.manual_seed(99)
        encoder_layer = nn.TransformerEncoderLayer(d_model=8, nhead=2)
        torch.manual_seed(99)
        decoder_layer = nn.TransformerDecoderLayer(d_model=8, nhead=2)
        torch.manual_seed(99)
        transformer = nn.Transformer(d_model=8, nhead=2, num_encoder_layers=1, num_decoder_layers=1)
        return encoder_layer, decoder_layer, transformer

    # Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/identity_shims.py, campaign mutant kills.
    def test_string_activation_construction_pickles_while_wrapped(self):
        # R78-2 sibling: the STRING spelling reads F.gelu at ctor time -- under
        # wrap that read returns the live wrapper, which then poisoned
        # whole-model pickle (and survived unwrap). The ctor shim stores the
        # original instead.
        _ensure_wrapped()
        layer = nn.TransformerEncoderLayer(d_model=8, nhead=2, activation="gelu")
        assert layer.activation is _resolve(F.gelu)
        assert layer.activation_relu_or_gelu == 2
        pickle.dump(layer, io.BytesIO())

    # Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/identity_shims.py, campaign mutant kills.
    def test_setstate_legacy_unpickle_is_wrap_invariant(self):
        # R78-2 sibling: __setstate__ injects the CURRENT F.relu (the live
        # wrapper while wrapped) when legacy state lacks ``activation``.
        _ensure_wrapped()
        for cls in (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer):
            layer = cls(d_model=8, nhead=2)
            state = layer.__dict__.copy()
            state.pop("activation", None)
            revived = cls(d_model=8, nhead=2)
            revived.__setstate__(state)
            assert revived.activation is _resolve(F.relu), cls.__name__


# ---------------------------------------------------------------------------
# 6. Import-callback vs unwrap lifecycle race (R21/R55 b7 barrier probe)
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
def test_import_callback_unwrap_race_cannot_corrupt_shim_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A causal-bias import callback racing ``unwrap_torch()`` must never leave
    a shim installed with wrappers off, and a lone late-appended record must
    never stand in for the full shim family on the next wrap."""

    import threading

    from torchlens._utils import _torch_compat
    from torchlens.backends.torch import identity_shims
    from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

    if not _torch_compat.HAS_ATTENTION_CAUSAL_BIAS:
        pytest.skip("causal-bias site absent on this torch")
    pytest.importorskip("torch.nn.attention.bias")
    import torch.nn.attention.bias as bias_module

    unwrap_torch()
    wrap_torch()
    assert identity_shims.identity_shims_installed()

    unwrap_started = threading.Event()
    unwrap_done = threading.Event()
    real_install = identity_shims._install_causal_bias_shim

    def paused_install(records: list) -> None:
        # Pause the callback tail between the lifecycle check and the install,
        # exactly where the b7 barrier probe parked the loader. The bounded
        # wait lets the LOCKED (fixed) tail proceed while unwrap blocks on the
        # lifecycle lock; the UNLOCKED (buggy) tail instead lets unwrap finish
        # first and then installs into torn-down state.
        unwrap_started.set()
        unwrap_done.wait(timeout=2.0)
        real_install(records)

    monkeypatch.setattr(identity_shims, "_install_causal_bias_shim", paused_install)

    class _NoopLoader:
        def exec_module(self, module: object) -> None:
            return None

    loader = identity_shims._ShimOnExecLoader(_NoopLoader())
    callback = threading.Thread(target=loader.exec_module, args=(bias_module,))
    callback.start()
    assert unwrap_started.wait(timeout=5.0)
    unwrap_torch()
    unwrap_done.set()
    callback.join(timeout=10.0)
    assert not callback.is_alive()
    monkeypatch.setattr(identity_shims, "_install_causal_bias_shim", real_install)

    # Wrappers are off: no shim record may survive and CausalBias must be
    # pristine (final-state equality, the b7 probe's failing assertion).
    assert not identity_shims.identity_shims_installed()
    causal_tf = vars(bias_module.CausalBias).get("__torch_function__")
    assert not identity_shims._is_shimmed(causal_tf)

    # The next wrap must perform the FULL family install: a default-built
    # encoder layer keeps its fused-fastpath classification.
    wrap_torch()
    try:
        assert identity_shims.identity_shims_installed()
        if _torch_compat.HAS_TRANSFORMER_ACTIVATION_FASTPATH_FLAG:
            layer = nn.TransformerEncoderLayer(d_model=8, nhead=2)
            assert layer.activation_relu_or_gelu == 1
    finally:
        unwrap_torch()
        wrap_torch()


# ---------------------------------------------------------------------------
# 7. R55 membership-form census — import-time CONTAINER tables (b7 blind spot)
# ---------------------------------------------------------------------------

# (module name, attribute name) -> reviewed rationale. The defaults scan (1)
# and the `is`-form grep gate (2) cannot see the THIRD wrap-state shape: a
# container built at IMPORT time and consulted by MEMBERSHIP at call time
# (exactly the shape of the shimmed expanded-weights handler tables). Every
# import-time container holding a WRAPPED-ORIGINAL torch callable must be a
# reviewed entry here; a new torch release adding one fails this gate.
_MEMBERSHIP_TABLE_REVIEWED: dict[tuple[str, str], str] = {
    ("torch._library.utils", "_RANDOM_FUNCTIONS"): (
        "is_impure()/fx DCE authority; eagerly imported with torch so keys are "
        "pre-wrap originals. fx records the protocol-supplied ORIGINAL as the "
        "node target for C functions (verified); directly-called Python "
        "functionals recorded the WRAPPER until the Tracer.trace identity shim "
        "began remapping targets through the ledger (grind-r5 b8 R56), so "
        "membership answers stay correct on both paths."
    ),
    ("torch.masked.maskedtensor.reductions", "TORCH_REDUCE_MAP"): (
        "MaskedTensor reduction dispatch keyed on torch.* ORIGINALS; eagerly "
        "imported with torch, and the C-level protocol reads a torch.* function "
        "from the unpatched _VariableFunctions table, so the handler sees the "
        "original."
    ),
    ("torch.masked.maskedtensor.reductions", "TENSOR_REDUCE_MAP"): (
        "MaskedTensor reduction dispatch keyed on Tensor METHOD originals. The "
        "C parser presents getattr(torch.Tensor, name) read at CALL time (the "
        "wrapper during the wrap epoch), so a method call misses this table and "
        "falls through to MaskedTensor's aten dispatch table, which computes the "
        "same reduction (mt.sum() observed equal). Documented unshimmed residual: "
        "the function table itself lives in the lazily imported _ops_refs, so a "
        "fixed identity basis cannot be chosen at wrap time."
    ),
    ("torch._jit_internal", "boolean_dispatched"): (
        "TorchScript boolean-dispatch table; deliberately DUAL-KEYED by "
        "torchlens (each wrapper registered alongside its original sharing one "
        "dispatch record), so membership holds under either alias."
    ),
    ("torch.masked.maskedtensor.reductions", "TORCH_REDUCE_FNS"): (
        "Source list the reduce MAPs are built from; same eager-import basis."
    ),
    ("torch.masked.maskedtensor.reductions", "TENSOR_REDUCE_FNS"): (
        "Source list the reduce MAPs are built from; same eager-import basis."
    ),
    ("torch.nn.parameter", "UninitializedTensorMixin._allowed_methods"): (
        "Lazy-module materialization allowlist consulted from "
        "UninitializedTensorMixin.__torch_function__. For TENSOR METHODS the "
        "C-level protocol hands the handler the CURRENT torch.Tensor class "
        "attribute (the wrapper during the wrap epoch), so the list is shimmed "
        "with a ledger-resolving view while wrappers are installed "
        "(identity_shims._install_lazy_module_allowlist_shim; wave-3 stress F2, "
        "tests/test_identity_shims.py::TestLazyModuleAllowlist)."
    ),
    # The two torch._export sym-op tables surface only when a suite test
    # imports torch._export pre-wrap (lazy, not eagerly imported with torch).
    # Both are consulted exclusively inside torch.export pass/serde machinery,
    # which is out of capture scope by contract (the same class as the
    # _MEMBERSHIP_SCAN_SKIP_PREFIXES export/compiler namespaces; TorchLens
    # never logs torch.export artifacts and capture forces eager). Reviewed
    # per-table rather than prefix-skipped so a NEW torch._export table still
    # fails here for review.
    ("torch._export.pass_base", "_TORCH_SYM_OPS"): (
        "Export-pass sym-op classifier; consulted only while running "
        "torch.export passes, out of eager capture scope by contract."
    ),
    ("torch._export.serde.serialize", "_SYM_OPS"): (
        "Export serde sym-op table; consulted only while serializing an "
        "ExportedProgram, out of eager capture scope by contract."
    ),
    # torch 2.14 additions.
    ("torch._higher_order_ops.flex_gemm", "FLEX_GEMM_OP_ALIASES"): (
        "FlexGEMM decomposition-pass alias table keyed by torch.mm/addmm/bmm/"
        "baddbmm; eagerly imported with torch (torch/__init__.py imports "
        "torch._higher_order_ops, whose __init__ imports flex_gemm), so keys "
        "are pre-wrap originals. Consulted only while torch.compile/Inductor "
        "lowers a FlexGEMM higher-order op to its underlying aten overload -- "
        "out of eager capture scope by contract, the same class as the "
        "torch._export sym-op tables above."
    ),
    ("torch.utils.dlpack", "ReadOnlyTensorWrapper._DLPACK_ALLOWED"): (
        "DLPack read-only-export allowlist keyed by torch.Tensor.__dlpack__ / "
        "__dlpack_device__; eagerly imported with torch (torch/__init__.py "
        "imports torch.utils.dlpack), so keys are pre-wrap originals. "
        "Membership is checked inside ReadOnlyTensorWrapper.__torch_function__; "
        "TorchLens never wraps Tensor.__dlpack__ / __dlpack_device__ on the "
        "class (the function-boundary export patch is capture-scoped), so the "
        "func operand the protocol hands the handler is the pre-wrap original "
        "outside capture -- same basis as the MaskedTensor reduce maps above."
    ),
}

# Compiler/export/quantization namespaces are out of capture scope by contract
# (the identity-shim census covers eager runtime paths only).
_MEMBERSHIP_SCAN_SKIP_PREFIXES = (
    "torch._dynamo",
    "torch._inductor",
    "torch._prims",
    "torch._refs",
    "torch._decomp",
    "torch.ao",
    "torch.quantization",
    "torch.fx",
    "torch.jit",
    "torch.onnx",
    "torch.testing",
    "torch.distributed",
)


# ---------------------------------------------------------------------------
# 8. R02 per-SITE post-wrap audit — object-keyed inventory blindness (b3-opus)
# ---------------------------------------------------------------------------

# Public module-namespace attribute sites that legitimately keep their ORIGINAL
# callable while wrapped: torch.functional re-exports outside torch.__all__,
# so only the torch.functional.<name> twin is repointed. Each is a pure-Python
# COMPOSITE over wrapped interiors -- traces are byte-identical under either
# spelling (verified b3-opus /tmp/p3.py). A NEW name appearing here (e.g. a
# torch release re-exporting a LEAF op into torch outside __all__) must be
# reviewed, not silently unwrapped.
_ORIGINAL_HOLDING_SITE_ALLOWLIST = {
    # The four public composites: torch.functional re-exports outside
    # torch.__all__, so only the torch.functional.<name> twin is repointed.
    # Each is a pure-Python COMPOSITE over wrapped interiors -- traces are
    # byte-identical under either spelling (verified, b3-opus /tmp/p3.py).
    ("torch", "unique"),
    ("torch", "pca_lowrank"),
    ("torch", "svd_lowrank"),
    ("torch", "lu"),
    # Private-underscore alias spellings of wrapped objects: the public /
    # canonical site is repointed; these private twins are not called by the
    # eager public surface.
    ("torch", "_segment_reduce"),
    ("torch", "_sym_sqrt"),
    ("torch.functional", "_add_docstr"),
    ("torch.sparse", "_add_docstr"),
    ("torch.functional", "overload"),
    # torch 2.1.2 floor only (the CI 2.1.2/2.2.2 rows): ``torch.obj`` is an
    # undocumented module attribute whose underlying object IS, by identity,
    # ``torch.zeros_like``'s own raw C callable (verified: id(torch.obj) ==
    # id(torch.zeros_like) pre-wrap; absent from get_orig_torch_funcs()'s
    # roster and from torch.overrides.get_testing_overrides(), so no torch
    # release documents or exercises it as a real entry point -- a torch
    # 2.1.2 build quirk, not a torchlens gap). Calling it post-wrap runs the
    # identical raw zeros_like implementation either spelling would have run
    # pre-wrap: a stronger composite-over-wrapped-interiors case than the
    # entries above (not just interior-composite -- the SAME object).
    ("torch", "obj"),
}


# Roster rows whose torch attribute is REVIEWED-dead on current torch: the
# spelling no longer exists, so the row wraps nothing and the decoration
# loop's hasattr-continue is the correct behavior FOR THESE ROWS ONLY. Each
# entry needs a reason; an entry that RESOLVES again must be removed (the
# gate below fails in both directions).
_KNOWN_DEAD_ROSTER_ROWS = {
    # Removed upstream (absent on torch 2.13); kept in IGNORED_FUNCS for the
    # torch releases that still expose it. R3-B3-R02-1 evidence row.
    ("torch", "_sparse_csr_tensor"),
}


class TestProtocolArgWrapperIdentity:
    """grind-r6 b8 R56: pure-Python functionals and the shim residues."""

    # Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/_identity_shim_views.py, campaign mutant kills.
    def test_overridable_functions_view_preserves_defaultdict_semantics(self):
        # opus wave-introduced residue: the r5 membership view rebuilt
        # get_overridable_functions()'s defaultdict(list) as a plain dict, so
        # indexing a namespace with no recorded entries raised KeyError where
        # upstream auto-vivifies an empty list.
        _ensure_wrapped()
        table = torch.overrides.get_overridable_functions()
        sentinel_key = object()
        try:
            assert table[sentinel_key] == [], "defaultdict auto-vivification lost"
        finally:
            # The view is cached and shared; leave no sentinel row behind.
            table.pop(sentinel_key, None)
        # Membership resolution through the ledger still holds.
        assert F.relu in table.get(F, []) or any(F.relu in members for members in table.values())
