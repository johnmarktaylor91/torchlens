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
import types
from pathlib import Path
from typing import Any

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


def test_no_unreviewed_wrapped_original_in_import_time_defaults():
    import sys

    _ensure_wrapped()
    orig_ids = {id(v) for v in _state._decorated_to_orig.values()}
    hits: set[tuple[str, str]] = set()
    for mod_name, mod in list(sys.modules.items()):
        if not mod_name.startswith("torch") or mod is None:
            continue
        candidates: list[types.FunctionType] = []
        for attr in list(vars(mod).values()):
            if isinstance(attr, types.FunctionType):
                candidates.append(attr)
            elif isinstance(attr, type):
                for method_name in ("__init__", "__new__", "forward", "__call__"):
                    method = vars(attr).get(method_name)
                    if isinstance(method, types.FunctionType):
                        candidates.append(method)
        for fn in candidates:
            defaults = list(fn.__defaults__ or ())
            defaults.extend((fn.__kwdefaults__ or {}).values())
            if any(callable(d) and id(d) in orig_ids for d in defaults):
                hits.add((fn.__module__, fn.__qualname__))
    unreviewed = hits - _DEFAULTS_ALLOWLIST
    assert not unreviewed, (
        f"New import-time default(s) frozen to a wrapped original: {sorted(unreviewed)}. "
        "Each needs an identity shim (see backends/torch/identity_shims.py) or a "
        "reviewed allowlist entry."
    )


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
    "utils/_sympy",
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


def _grep_skipped(rel_path: str) -> bool:
    return any(
        rel_path.startswith(prefix) or f"/{prefix}/" in rel_path for prefix in _GREP_SKIP_PREFIXES
    )


def test_installed_torch_tree_identity_checks_are_reviewed():
    _ensure_wrapped()
    orig_ids = {id(v) for v in _state._decorated_to_orig.values()}
    wrapper_ids = set(_state._decorated_to_orig.keys())
    torch_root = Path(torch.__file__).parent
    unreviewed: set[tuple[str, str]] = set()
    for source_path in torch_root.rglob("*.py"):
        rel_path = source_path.relative_to(torch_root).as_posix()
        if _grep_skipped(rel_path):
            continue
        try:
            text = source_path.read_text(errors="ignore")
        except OSError:
            continue
        for match in _IDENTITY_PATTERN.finditer(text):
            base, attr = match.group(1), match.group(2)
            namespace = torch.nn.functional if base == "F" else torch
            if base != "F":
                for part in base.split(".")[1:]:
                    namespace = getattr(namespace, part, None)
                    if namespace is None:
                        break
            if namespace is None:
                continue
            target = getattr(namespace, attr, None)
            if target is None or not callable(target):
                continue
            wrapped_wrapper = getattr(target, "__wrapped__", None)
            if (
                id(target) in wrapper_ids
                or id(target) in orig_ids
                or (wrapped_wrapper is not None and id(wrapped_wrapper) in orig_ids)
            ):
                if (rel_path, attr) not in _GREP_ALLOWLIST:
                    unreviewed.add((rel_path, attr))
    assert not unreviewed, (
        f"New torch-tree identity check(s) against wrapped callables: "
        f"{sorted(unreviewed)}. Each needs an identity shim or a reviewed "
        "allowlist entry (see backends/torch/identity_shims.py)."
    )


# ---------------------------------------------------------------------------
# 3. R55.eco meta-API smoke
# ---------------------------------------------------------------------------


class TestMetaApiSmokePostWrap:
    def test_resolve_name_matches_unwrapped_answer(self):
        _ensure_wrapped()
        assert torch.overrides.resolve_name(F.relu) == "torch.nn.functional.relu"
        assert torch.overrides.resolve_name(_resolve(F.relu)) == "torch.nn.functional.relu"

    def test_functional_call_and_grad(self):
        _ensure_wrapped()
        module = nn.Linear(3, 2)
        x = torch.randn(1, 3)
        params = dict(module.named_parameters())
        out = torch.func.functional_call(module, params, (x,))
        assert out.shape == (1, 2)
        grad_fn = torch.func.grad(lambda t: (t * t).sum())
        assert torch.allclose(grad_fn(torch.tensor(3.0)), torch.tensor(6.0))

    def test_vmap(self):
        _ensure_wrapped()
        batched = torch.vmap(torch.dot)(torch.randn(4, 3), torch.randn(4, 3))
        assert batched.shape == (4,)

    @pytest.mark.smoke
    def test_flop_counter_mode(self):
        from torch.utils.flop_counter import FlopCounterMode

        _ensure_wrapped()
        module = nn.Linear(8, 8)
        with FlopCounterMode(display=False) as counter:
            module(torch.randn(2, 8))
        assert counter.get_total_flops() > 0

    def test_gradcheck(self):
        _ensure_wrapped()
        x = torch.randn(3, dtype=torch.double, requires_grad=True)
        assert torch.autograd.gradcheck(lambda t: (t * t).sum(), (x,))

    def test_checkpoint(self):
        from torch.utils.checkpoint import checkpoint

        _ensure_wrapped()
        module = nn.Linear(4, 4)
        x = torch.randn(2, 4, requires_grad=True)
        out = checkpoint(module, x, use_reentrant=False)
        out.sum().backward()
        assert x.grad is not None

    def test_parametrize(self):
        import torch.nn.utils.parametrize as parametrize

        _ensure_wrapped()
        module = nn.Linear(3, 3)

        class Sym(nn.Module):
            def forward(self, weight):
                return weight.triu() + weight.triu(1).transpose(-1, -2)

        parametrize.register_parametrization(module, "weight", Sym())
        weight = module.weight
        assert torch.allclose(weight, weight.transpose(-1, -2))

    def test_pickle_fresh_module(self):
        _ensure_wrapped()
        payload = pickle.dumps(nn.Linear(2, 2))
        assert pickle.loads(payload).weight.shape == (2, 2)

    def test_jit_script(self):
        _ensure_wrapped()

        @torch.jit.script
        def scripted(x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x) + 1

        assert torch.equal(scripted(torch.tensor([-1.0, 1.0])), torch.tensor([1.0, 2.0]))


class _ActModel(nn.Module):
    """Module-level so pickle can save it by reference."""

    def __init__(self):
        super().__init__()
        self.act = F.relu

    def forward(self, x):
        return self.act(x)


class TestWrapperPickleLadder:
    # B8-1: @wraps copied the original C descriptor's __qualname__
    # (_VariableFunctionsClass.cos) onto the wrapper, so pickling a BARE
    # wrapped callable failed on attribute lookup. Install-site stamping makes
    # every rung pickle by reference to the public torch name while wrapped.

    def test_bare_wrapped_function_pickles_while_wrapped(self):
        _ensure_wrapped()
        for target in (torch.cos, F.relu, torch.mean):
            loaded = pickle.loads(pickle.dumps(target))
            assert _resolve(loaded) is _resolve(target)
        result = pickle.loads(pickle.dumps(torch.cos))(torch.zeros(2))
        assert torch.equal(result, torch.ones(2))

    def test_model_holding_namespace_read_pickles_while_wrapped(self):
        _ensure_wrapped()
        loaded = pickle.loads(pickle.dumps(_ActModel()))
        assert torch.equal(loaded(torch.tensor([-1.0, 2.0])), torch.tensor([0.0, 2.0]))

    def test_wrapper_introspection_module_fidelity(self):
        # B8-5 (module namespaces only): wrapped module-namespace functions
        # report the install site. CLASS-namespace wrappers (tensor methods)
        # deliberately KEEP the honest torchlens wrappers module: stamping
        # them "torch" would make a wrapped storage-unsafe method claim torch
        # purity to string-based safety gates — the r36 smuggling defense
        # (tests/test_r36_tensor_method_smuggling.py, LOCKED) pins that
        # surface. Security disclosure wins over introspection fidelity here.
        _ensure_wrapped()
        assert torch.cos.__module__ == "torch"
        assert torch.cos.__qualname__ == "cos"
        assert torch.Tensor.add.__module__ == "torchlens.backends.torch.wrappers"
        assert F.relu.__module__ == "torch.nn.functional"

    def test_signature_fabrication_residual_shape(self):
        # DISCLOSED RESIDUAL (B8-5 half): inspect.signature on a wrapped C
        # builtin reports the wrapper's (*args, **kwargs) instead of the
        # honest ValueError -- __wrapped__ must stay deleted for JIT
        # compatibility and a function attribute cannot raise. Pinned so a
        # silent change gets noticed.
        import inspect

        _ensure_wrapped()
        if id(torch.mean) not in _state._decorated_to_orig:
            pytest.skip("torch.mean not wrapped on this build")
        parameters = inspect.signature(torch.mean).parameters
        assert set(parameters) == {"args", "kwargs"}


def test_wrapped_functional_warning_attribution_residual_shape():
    """DISCLOSED RESIDUAL (b8-sol R56-4): warning attribution under wrappers.

    The wrapper adds one Python frame, so every ``warnings.warn(...,
    stacklevel=N)`` inside a wrapped Python functional (``F.softmax``
    implicit-dim is torch's canonical case) is attributed to torch internals
    (``functional.py``) instead of the user's call site. Because Python's
    default-filter ``__warningregistry__`` dedup keys on the ATTRIBUTED
    location, distinct user call sites additionally collapse into ONE warning
    per process while wrappers are installed. The frame is inherent to
    Python-level wrapping (no trivial fix); the shape is pinned here and
    documented in ``docs/migration/scoped_detached_patching.md`` so a silent
    change in either direction gets noticed.
    """

    import warnings

    _ensure_wrapped()
    if id(F.softmax) not in _state._decorated_to_orig:
        pytest.skip("F.softmax not wrapped on this build")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        F.softmax(torch.randn(2, 3))
    implicit_dim = [w for w in caught if "implicit dimension" in str(w.message).lower()]
    if not implicit_dim:
        pytest.skip("this torch no longer warns on implicit softmax dim")
    attributed = implicit_dim[-1].filename
    assert attributed.endswith("functional.py"), (
        f"implicit-dim warning attributed to {attributed}: the wrapped-epoch "
        "stacklevel residual changed shape -- if it now points at the caller, "
        "the residual healed; update the scoped_detached_patching.md row and "
        "this pin together"
    )
    assert not attributed.endswith("test_wrap_state_compat.py")


# ---------------------------------------------------------------------------
# 4. Override-table coherence (B8-4)
# ---------------------------------------------------------------------------


class TestOverrideTableCoherence:
    def test_both_cached_tables_keyed_by_originals(self):
        from torch.overrides import get_overridable_functions, get_testing_overrides

        _ensure_wrapped()
        # Force the belt derivation (the historical first-caller-while-wrapped).
        from torchlens.backends.torch import belt

        assert belt.belt_report() is not None
        wrapper_ids = set(_state._decorated_to_orig.keys())
        overridable_poisoned = sum(
            1
            for functions in get_overridable_functions().values()
            for fn in functions
            if id(fn) in wrapper_ids
        )
        testing_poisoned = sum(1 for fn in get_testing_overrides() if id(fn) in wrapper_ids)
        assert overridable_poisoned == 0, (
            f"{overridable_poisoned} wrapper entries in get_overridable_functions(): "
            "a torchlens path first-called a cached torch introspection table while wrapped"
        )
        assert testing_poisoned == 0

    def test_belt_discloses_unprobed_candidates_by_name(self):
        from torchlens.backends.torch import belt

        _ensure_wrapped()
        report = belt.belt_report()
        assert report is not None
        assert report.unprobed_candidate_count == len(report.unprobed_candidates)
        if report.unprobed_candidates:
            namespace_name, func_name = report.unprobed_candidates[0]
            assert isinstance(namespace_name, str) and isinstance(func_name, str)


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

    def test_stale_prewrap_factory_ref_gets_context_device(self):
        # The user-visible failure: `from torch import zeros` held from before
        # the first capture, called under `with torch.device('meta')` while
        # wrappers are installed, must still land on meta (C-level injection).
        _ensure_wrapped()
        stale_zeros = _resolve(torch.zeros)
        assert stale_zeros is not torch.zeros, "expected torch.zeros to be wrapped"
        with torch.device("meta"):
            out = stale_zeros(2, 2)
        assert out.device.type == "meta", (
            f"stale pre-wrap zeros landed on {out.device}: torch's "
            "_device_constructors() cache is not keyed by originals"
        )

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

    def test_dynamo_rule_caches_prewarmed_before_wrapping(self):
        # The other half of the fix: when dynamo is already imported at wrap
        # time, the tables are warmed BEFORE the first wrapper setattr, so a
        # torch.compile during the wrapped epoch reads originals-keyed rules.
        trace_rules = pytest.importorskip("torch._dynamo.trace_rules")
        from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

        _ensure_wrapped()
        try:
            unwrap_torch()
            trace_rules.get_torch_obj_rule_map.cache_clear()
            trace_rules.get_tensor_method.cache_clear()
            wrap_torch()
            wrapper_ids = set(_state._decorated_to_orig.keys())
            rule_map = trace_rules.get_torch_obj_rule_map()
            assert _resolve(torch.cos) in rule_map, (
                "original torch.cos missing from the dynamo rule map after "
                "wrap_torch(): the pre-warm did not run before decoration"
            )
            poisoned = sum(1 for fn in rule_map if id(fn) in wrapper_ids)
            assert poisoned == 0, (
                f"{poisoned} torchlens wrappers keyed into the dynamo rule map "
                "despite the pre-wrap warm"
            )
        finally:
            if not _state._is_decorated:
                wrap_torch()

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
    def test_unwrap_holds_capture_admission_lock(self):
        # R54: the mid-capture refusal reads state published under
        # _capture_admission_lock; without holding it, a capture admitted
        # between the refusal check and the uninstall was silently truncated
        # (reproduced with a deterministic barrier). Pin the lock coverage:
        # unwrap_torch must block while the admission lock is held elsewhere.
        import threading

        from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

        _ensure_wrapped()
        assert _state._capture_admission_lock.acquire(timeout=5)
        done = threading.Event()

        def do_unwrap():
            unwrap_torch()
            done.set()

        worker = threading.Thread(target=do_unwrap)
        try:
            worker.start()
            assert not done.wait(0.3), "unwrap_torch() proceeded without the capture admission lock"
        finally:
            _state._capture_admission_lock.release()
            worker.join(10)
        assert done.is_set()
        wrap_torch()

    def test_admission_epoch_check_refuses_unwrapped_process(self):
        # R54 second half: a capture admitted AFTER a concurrent unwrap
        # completes must refuse typed rather than run an unlogged forward.
        from torchlens._errors import CaptureContextError
        from torchlens.backends.torch.backend import TorchBackend

        _ensure_wrapped()
        session = tl.trace(nn.Linear(2, 2), torch.randn(1, 2))
        saved = _state._is_decorated
        try:
            _state._is_decorated = False
            with pytest.raises(CaptureContextError) as exc_info:
                with TorchBackend().active_logging(session):
                    pass
            assert exc_info.value.fields["code"] == "wrappers_removed_before_capture"
        finally:
            _state._is_decorated = saved
        # The refusal must have unwound admission cleanly.
        assert _state._active_trace is None
        assert not _state._logging_enabled

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

    def test_clean_unwrap_emits_no_burial_warning(self):
        import warnings as warnings_module

        from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

        _ensure_wrapped()
        try:
            with warnings_module.catch_warnings(record=True) as records:
                warnings_module.simplefilter("always")
                unwrap_torch()
            assert not [r for r in records if "buried" in str(r.message)]
        finally:
            wrap_torch()


# ---------------------------------------------------------------------------
# 5. Wrap-history construction corpus (transformer trio + __setstate__)
# ---------------------------------------------------------------------------


def _has_fastpath_flag() -> bool:
    from torchlens.utils import _torch_compat

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

    def test_trio_construction_matches_unwrapped(self):
        from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

        _ensure_wrapped()
        try:
            unwrap_torch()
            reference = self._build_trio()
        finally:
            wrap_torch()
        wrapped_builds = self._build_trio()
        for ref_model, wrapped_model in zip(reference, wrapped_builds):
            for (name_r, p_r), (name_w, p_w) in zip(
                ref_model.state_dict().items(), wrapped_model.state_dict().items()
            ):
                assert name_r == name_w
                assert torch.equal(p_r, p_w), f"parameter drift in {name_r}"
            # Stored callables must be wrap-invariant (never a torchlens wrapper).
            for module_r, module_w in zip(ref_model.modules(), wrapped_model.modules()):
                act_r = getattr(module_r, "activation", None)
                act_w = getattr(module_w, "activation", None)
                if callable(act_r) or callable(act_w):
                    assert act_w is act_r
        # The wrapped-build trio must pickle exactly like the reference once
        # wrappers are removed (pickling a stored-ORIGINAL while the namespace
        # holds the wrapper is the pinned disclosed residual — see
        # test_identity_shims.TestDisclosedResiduals).
        try:
            unwrap_torch()
            for wrapped_model in wrapped_builds:
                pickle.dump(wrapped_model, io.BytesIO())
        finally:
            wrap_torch()

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


def test_import_callback_unwrap_race_cannot_corrupt_shim_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A causal-bias import callback racing ``unwrap_torch()`` must never leave
    a shim installed with wrappers off, and a lone late-appended record must
    never stand in for the full shim family on the next wrap."""

    import threading

    from torchlens.backends.torch import identity_shims
    from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch
    from torchlens.utils import _torch_compat

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
        "MaskedTensor reduction dispatch; eagerly imported with torch, and the "
        "C-level __torch_function__ protocol supplies the ORIGINAL func operand."
    ),
    ("torch.masked.maskedtensor.reductions", "TENSOR_REDUCE_MAP"): (
        "MaskedTensor reduction dispatch; same basis as TORCH_REDUCE_MAP."
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
        "UninitializedTensorMixin.__torch_function__, where the C-level "
        "protocol supplies the ORIGINAL func operand; eagerly imported."
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
        "Membership is checked inside ReadOnlyTensorWrapper.__torch_function__, "
        "where the C-level __torch_function__ protocol supplies the ORIGINAL "
        "func operand -- same basis as UninitializedTensorMixin._allowed_methods "
        "and the MaskedTensor reduce maps above."
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


def _iter_membership_hits() -> list[tuple[str, str]]:
    """Scan loaded torch modules for containers holding wrapped originals."""

    import sys as sys_module

    wrapped_original_ids = set(_state._orig_to_decorated.keys())
    hits: set[tuple[str, str]] = set()

    def _container_members(value) -> list:
        try:
            if isinstance(value, dict):
                return list(value.keys())
            if isinstance(value, (set, frozenset, tuple, list)):
                return list(value)
            if type(value).__name__ == "WeakKeyDictionary":
                return list(value.keys())
        except Exception:
            return []
        return []

    def _scan_namespace(mod_name: str, holder_name: str, namespace: dict) -> None:
        for attr_name, value in list(namespace.items()):
            members = _container_members(value)
            if not members:
                continue
            if any(id(member) in wrapped_original_ids for member in members):
                hits.add((mod_name, f"{holder_name}{attr_name}"))

    for mod_name, module in list(sys_module.modules.items()):
        if module is None or not mod_name.startswith("torch"):
            continue
        if mod_name.startswith(_MEMBERSHIP_SCAN_SKIP_PREFIXES):
            continue
        module_vars = getattr(module, "__dict__", None)
        if not isinstance(module_vars, dict):
            continue
        _scan_namespace(mod_name, "", module_vars)
        for cls_name, value in list(module_vars.items()):
            if isinstance(value, type) and getattr(value, "__module__", None) == mod_name:
                _scan_namespace(mod_name, f"{cls_name}.", dict(vars(value)))
    return sorted(hits)


def test_import_time_membership_tables_holding_wrapped_originals_are_reviewed() -> None:
    """Every import-time membership table keyed by wrapped originals is reviewed.

    The wrap-state gates covered function DEFAULTS and ``is``-form source
    comparisons but were structurally blind to import-time membership
    CONTAINERS (b7-opus + b7-fable, independently corroborated) — the exact
    shape of the already-shimmed expanded-weights tables. Their safety today
    rests on eager import (keys are pre-wrap originals) and protocol-supplied
    original operands; a torch release that adds a NEW such table, or an
    unreviewed family, must fail here for review rather than flip silently.

    BOUNDARY (grind-r5 b8 R56): this census scans ``torch*`` modules only.
    The eager-import rationale structurally cannot hold for EXTENSION
    libraries (the official "Extending torch" ``@implements`` pattern)
    imported after the first capture — their handler tables key on wrappers
    and silently miss dispatch. That user-side class is a declared residual
    in ``docs/migration/scoped_detached_patching.md`` ("Honest boundaries"),
    not something an in-repo census can gate.
    """

    _ensure_wrapped()
    unreviewed = [
        (mod_name, attr_name)
        for mod_name, attr_name in _iter_membership_hits()
        if (mod_name, attr_name) not in _MEMBERSHIP_TABLE_REVIEWED
        and not mod_name.startswith("torch.nn.utils._expanded_weights")
        and not mod_name.startswith("torchlens")
    ]
    assert unreviewed == [], (
        "Unreviewed import-time membership tables hold wrapped-original torch "
        f"callables: {unreviewed}. Review each (shim it like the expanded-weights "
        "tables, or add a reasoned entry to _MEMBERSHIP_TABLE_REVIEWED)."
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


def _inventory_namespaces() -> list[tuple[str, Any]]:
    """Every namespace named by the wrap inventory, resolved live.

    R3-B3-R02-2: the audit's namespace list is DERIVED from
    ``get_orig_torch_funcs()`` instead of hardcoded, so the audit is
    structurally co-extensive with the roster it certifies. The historical
    7-namespace list omitted ``torch.Tensor`` (the bulk of the roster, where
    torch adds method aliases), ``torch._VF`` (the interior those aliases
    delegate to), ``torch.utils.dlpack``, and ``torch.signal.windows`` -- for
    those the gate was structurally unable to fail.
    """

    from torchlens.constants import get_orig_torch_funcs
    from torchlens.utils._torch_compat import get_optional_torch_namespace

    resolved: list[tuple[str, Any]] = []
    for ns_name in sorted({ns for ns, _ in get_orig_torch_funcs()}):
        ns = get_optional_torch_namespace(ns_name)
        if ns is not None:
            resolved.append((ns_name, ns))
    return resolved


def test_every_public_module_site_holding_a_wrapped_original_is_reviewed() -> None:
    """Post-wrap, no UNREVIEWED public module attribute may hold an original.

    The wrap-inventory completeness gate was OBJECT-keyed (torch's override
    registry is keyed by function object and built from ``torch.__all__``),
    so an attribute SITE keeping its original callable was structurally
    invisible to it -- a disarmed tripwire for the exact
    "invisible capture-gap generator on a version boundary" class (b3-opus
    R02-3). This audit is per (namespace, attribute) SITE: every public
    callable attr whose OBJECT has a wrapper must be repointed or reviewed.
    The namespace list is derived from the inventory (R3-B3-R02-2), never
    hardcoded.
    """

    _ensure_wrapped()
    unreviewed: list[tuple[str, str]] = []
    for ns_name, ns in _inventory_namespaces():
        for attr in dir(ns):
            try:
                obj = getattr(ns, attr)
            except (AttributeError, RuntimeError):
                continue
            if id(obj) in _state._orig_to_decorated and (
                (ns_name, attr) not in _ORIGINAL_HOLDING_SITE_ALLOWLIST
            ):
                unreviewed.append((ns_name, attr))
    assert unreviewed == [], (
        "Public module attribute sites hold a wrapped ORIGINAL callable "
        f"(unwrapped spelling of a wrapped op): {unreviewed}. Repoint the site "
        "in decoration, or review it into _ORIGINAL_HOLDING_SITE_ALLOWLIST "
        "with a composite-over-wrapped-interiors verification."
    )


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


def test_curated_roster_rows_resolve_to_live_sites() -> None:
    """R3-B3-R02-1 liveness gate: no roster row may go dead SILENTLY.

    ``IGNORED_FUNCS`` is the hand-curated re-add list of ops torch's override
    registries omit -- exactly the ops whose absence from the roster
    previously produced silent unattributed-literal capture gaps. The
    decoration loop ``continue``s on an unresolvable pair with no diagnostic,
    so a torch release that renames, privatizes, or moves ANY re-added
    spelling would reopen the precise gap the row exists to close, with zero
    signal. This gate makes every dead row a REVIEWED fact: unexpected dead
    rows fail, and known-dead rows that resurrect fail until the ledger entry
    is removed.
    """

    from torchlens.constants import get_orig_torch_funcs
    from torchlens.utils._torch_compat import get_optional_torch_namespace

    _ensure_wrapped()
    dead: set[tuple[str, str]] = set()
    for ns_name, func_name in get_orig_torch_funcs():
        ns = get_optional_torch_namespace(ns_name)
        if ns is None or not hasattr(ns, func_name):
            dead.add((ns_name, func_name))
    unexpected_dead = dead - _KNOWN_DEAD_ROSTER_ROWS
    resurrected = _KNOWN_DEAD_ROSTER_ROWS - dead
    assert not unexpected_dead, (
        f"Wrap-inventory rows silently wrap NOTHING on this torch: "
        f"{sorted(unexpected_dead)}. If the spelling moved, update the roster "
        "(the silent gap the row closes is back); if it was removed upstream, "
        "review it into _KNOWN_DEAD_ROSTER_ROWS with the torch version."
    )
    assert not resurrected, (
        f"_KNOWN_DEAD_ROSTER_ROWS entries resolve again on this torch: "
        f"{sorted(resurrected)}. Remove them from the ledger so the liveness "
        "gate re-arms for those rows."
    )


class TestProtocolArgWrapperIdentity:
    """grind-r6 b8 R56: pure-Python functionals and the shim residues."""

    def test_pure_python_functional_presents_original_to_user_handler(self):
        # sol HIGH / fable MED, same root: F.relu's body dispatches
        # handle_torch_function(relu, ...) where `relu` resolves from module
        # globals -- the torchlens WRAPPER during the wrap epoch -- so every
        # user handler keyed on originals (the documented import-time
        # HANDLED_FUNCTIONS shape) silently missed, process-wide, after the
        # first capture. C builtins present the original; pure-Python must
        # match.
        _ensure_wrapped()
        wrapped_relu = F.relu
        original_relu = _state._decorated_to_orig.get(id(wrapped_relu))
        assert original_relu is not None, "F.relu is not wrapped; test premise broken"
        assert isinstance(original_relu, types.FunctionType)

        seen: list[Any] = []

        class _Probe(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types_, args=(), kwargs=None):
                seen.append(func)
                return super().__torch_function__(func, types_, args, kwargs or {})

        F.relu(torch.randn(4).as_subclass(_Probe))
        assert seen, "handler never fired"
        assert seen[0] is original_relu, (
            "user __torch_function__ handler saw the torchlens wrapper as func "
            "for a pure-Python functional; original-keyed handler tables miss"
        )

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

    def test_overrides_accessor_cache_api_survives_shimming(self):
        # opus wave-introduced residue: the upstream accessors may be
        # lru_cache functions; functools.wraps copies __dict__ only, so
        # cache_clear / cache_info vanished from the shimmed surface. Assert
        # PARITY with the pristine upstream original (on this build
        # get_testing_overrides is lru_cache'd; get_overridable_functions
        # caches in a private helper and exposes no cache API to lose).
        from torchlens.backends.torch import identity_shims

        _ensure_wrapped()
        checked = 0
        for holder, name, original in identity_shims._installed:
            if holder is not torch.overrides or name not in (
                "get_testing_overrides",
                "get_overridable_functions",
            ):
                continue
            shimmed = getattr(torch.overrides, name)
            for cache_attr in ("cache_clear", "cache_info"):
                if callable(getattr(original, cache_attr, None)):
                    checked += 1
                    assert callable(getattr(shimmed, cache_attr, None)), (
                        f"{name}.{cache_attr} stripped by the membership shim"
                    )
            # NOTE: cache_clear() is deliberately NOT invoked here -- clearing
            # and rebuilding the table mid-epoch would key it by WRAPPERS,
            # recreating the exact poisoning the coherence gate above guards.
            if callable(getattr(shimmed, "cache_info", None)):
                shimmed.cache_info()  # usable, not merely present
        assert checked, "no upstream cache API found on either accessor; premise drifted"

    def test_teardown_never_clobbers_spoofed_marker_site(self):
        # sol MED: _restore keyed on the spoofable _SHIM_MARKER attribute, so
        # a foreign monkeypatch carrying the marker was CLOBBERED with our
        # stored original at unwrap. Teardown must key on exact shim object
        # identity.
        from torchlens.backends.torch import identity_shims
        from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

        _ensure_wrapped()
        overrides_module = torch.overrides
        pristine = None
        for holder, name, original in identity_shims._installed:
            if holder is overrides_module and name == "resolve_name":
                pristine = original
                break
        assert pristine is not None, "resolve_name shim record missing; test premise broken"

        def foreign_resolve_name(f):
            return "foreign"

        foreign_resolve_name._torchlens_identity_shim = True  # the spoof
        overrides_module.resolve_name = foreign_resolve_name
        try:
            unwrap_torch()
            assert overrides_module.resolve_name is foreign_resolve_name, (
                "teardown clobbered a foreign monkeypatch that spoofed the shim marker"
            )
        finally:
            overrides_module.resolve_name = pristine
            wrap_torch()
