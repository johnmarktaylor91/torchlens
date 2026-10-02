"""Wrap-state identity shims: torch-internal ``x is F.y`` checks stay truthful.

TorchLens wrapping replaces public torch callables with wrapper functions, so
a torch-internal identity check whose two operands were read at different wrap
epochs (a class-def-time default or a protocol-passed original on one side, a
post-wrap namespace read on the other) silently changes answer once wrappers
are installed. Census 2026-08-14 over the supported eager range found exactly
these runtime sites:

1. ``nn.TransformerEncoderLayer.__init__`` -- ``activation is F.relu/F.gelu``
   decides ``activation_relu_or_gelu`` (fused fastpath + nested-tensor path).
   Post-wrap construction with the DEFAULT activation silently got flag 0.
2. ``nn.attention.bias.CausalBias.__torch_function__`` -- ``func is F.sdpa``
   decides mask dispatch; a miss silently DROPPED the causal mask (wrong
   numbers, broken result subclass).
3. ``nn.utils._expanded_weights`` -- ``conv_picker`` namespace reads and the
   ``ExpandedWeight.__torch_function__`` mixed table/namespace bases broke
   per-sample-grads loudly.

These tests pin the identity shims that keep each site behaving exactly as
unwrapped eager torch. Flag gates use ``getattr`` so the module also imports
(and fails RED) on a pre-fix tree without the ``HAS_*`` flags.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchlens as tl
from torchlens import _state
from torchlens.utils import _torch_compat

pytestmark = pytest.mark.smoke


def _flag(name: str) -> bool:
    """Read a capability flag, defaulting to True on pre-fix trees."""

    value = getattr(_torch_compat, name, None)
    return True if value is None else bool(value)


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


@pytest.mark.skipif(
    not _flag("HAS_TRANSFORMER_ACTIVATION_FASTPATH_FLAG"),
    reason="torch build lacks the transformer activation fastpath flag",
)
class TestTransformerFastpathFlag:
    def test_default_activation_flag_survives_wrap(self):
        # The original defect repro: the class-def-time default activation is
        # the pre-wrap F.relu; the ctor identity check reads the post-wrap
        # namespace. Unfixed, the flag silently drops to 0.
        pre = nn.TransformerEncoderLayer(d_model=8, nhead=2)
        _ensure_wrapped()
        post = nn.TransformerEncoderLayer(d_model=8, nhead=2)
        assert post.activation_relu_or_gelu == pre.activation_relu_or_gelu == 1

    def test_explicit_prewrap_relu_and_gelu_refs(self):
        _ensure_wrapped()
        orig_relu = _resolve(F.relu)
        orig_gelu = _resolve(F.gelu)
        assert orig_relu is not F.relu, "wrap must be installed for this test"
        relu_layer = nn.TransformerEncoderLayer(d_model=8, nhead=2, activation=orig_relu)
        gelu_layer = nn.TransformerEncoderLayer(d_model=8, nhead=2, activation=orig_gelu)
        assert relu_layer.activation_relu_or_gelu == 1
        assert gelu_layer.activation_relu_or_gelu == 2

    def test_explicit_wrapped_namespace_ref(self):
        _ensure_wrapped()
        layer = nn.TransformerEncoderLayer(d_model=8, nhead=2, activation=F.relu)
        assert layer.activation_relu_or_gelu == 1

    def test_positional_activation_spelling(self):
        _ensure_wrapped()
        layer = nn.TransformerEncoderLayer(8, 2, 16, 0.1, _resolve(F.gelu))
        assert layer.activation_relu_or_gelu == 2

    def test_non_relu_gelu_activation_keeps_flag_zero(self):
        _ensure_wrapped()
        layer = nn.TransformerEncoderLayer(d_model=8, nhead=2, activation=F.silu)
        assert layer.activation_relu_or_gelu == 0

    def test_stored_activation_is_never_a_torchlens_wrapper(self):
        # Wrap-state invariance of constructed module state: whatever spelling
        # the user picks, the stored attribute must be the ORIGINAL torch
        # function (what an unwrapped construction stores), never a wrapper.
        _ensure_wrapped()
        orig_relu = _resolve(F.relu)
        for kwargs in (
            {},
            {"activation": "relu"},
            {"activation": F.relu},
            {"activation": orig_relu},
        ):
            layer = nn.TransformerEncoderLayer(d_model=8, nhead=2, **kwargs)
            assert layer.activation is orig_relu, f"leak for kwargs={kwargs}"
        decoder = nn.TransformerDecoderLayer(d_model=8, nhead=2, activation="gelu")
        assert decoder.activation is _resolve(F.gelu)

    def test_construction_bytes_invariant_across_wrap_state(self):
        # Same seed, wrappers OFF vs ON: identical parameters, identical flag,
        # identical stored activation object.
        from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

        _ensure_wrapped()
        try:
            unwrap_torch()
            torch.manual_seed(1234)
            before = nn.TransformerEncoderLayer(d_model=8, nhead=2)
        finally:
            wrap_torch()
        torch.manual_seed(1234)
        after = nn.TransformerEncoderLayer(d_model=8, nhead=2)
        assert after.activation_relu_or_gelu == before.activation_relu_or_gelu
        assert after.activation is before.activation
        for (name_b, p_b), (name_a, p_a) in zip(
            before.state_dict().items(), after.state_dict().items()
        ):
            assert name_b == name_a
            assert torch.equal(p_b, p_a), f"parameter drift in {name_b}"

    def test_transformer_encoder_nested_tensor_path_stays_enabled(self):
        # TransformerEncoder(enable_nested_tensor=True) downgrades with a
        # warning when the layer flag is 0 -- post-wrap construction must not
        # trigger that downgrade.
        _ensure_wrapped()
        layer = nn.TransformerEncoderLayer(d_model=8, nhead=2, batch_first=True)
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error", UserWarning)
            encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=True)
        assert encoder.use_nested_tensor


@pytest.mark.skipif(
    not _flag("HAS_ATTENTION_CAUSAL_BIAS"),
    reason="torch build lacks torch.nn.attention.bias.CausalBias",
)
class TestCausalBiasDispatch:
    def test_causal_bias_sdpa_matches_materialized_truth(self):
        from torch.nn.attention.bias import causal_lower_right

        _ensure_wrapped()
        torch.manual_seed(0)
        q = torch.randn(1, 2, 8, 4)
        k = torch.randn(1, 2, 8, 4)
        v = torch.randn(1, 2, 8, 4)
        bias = causal_lower_right(8, 8)
        truth = F.scaled_dot_product_attention(q, k, v, attn_mask=bias._materialize(q.device))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        # Unfixed, the identity miss silently dropped the mask AND returned a
        # broken CausalBias-typed result.
        assert type(out) is torch.Tensor
        assert torch.allclose(out, truth, atol=1e-6)

    def test_causal_bias_other_funcs_still_delegate(self):
        from torch.nn.attention.bias import causal_lower_right

        _ensure_wrapped()
        bias = causal_lower_right(4, 4)
        materialized = bias._materialize(torch.device("cpu"))
        assert materialized.shape == (4, 4)


@pytest.mark.skipif(
    not _flag("HAS_EXPANDED_WEIGHTS_CONV_PICKER"),
    reason="torch build lacks the private expanded-weights machinery",
)
class TestExpandedWeightsDispatch:
    def test_per_sample_grads_conv_post_wrap(self):
        from torch.nn.utils._per_sample_grad import call_for_per_sample_grads

        _ensure_wrapped()
        torch.manual_seed(0)
        module = nn.Conv2d(3, 4, 3)
        x = torch.randn(2, 3, 8, 8)
        call_for_per_sample_grads(module, batch_size=2)(x).sum().backward()
        grad_sample = module.weight.grad_sample
        assert grad_sample.shape == (2, 4, 3, 3, 3)
        # Cross-check per-sample grads against a plain per-sample autograd loop.
        for i in range(2):
            ref = nn.Conv2d(3, 4, 3)
            ref.load_state_dict(module.state_dict())
            ref(x[i : i + 1]).sum().backward()
            assert torch.allclose(grad_sample[i], ref.weight.grad, atol=1e-5)

    def test_flatten_weight_special_case_still_short_circuits(self):
        # The protocol passes the ORIGINAL torch._cudnn_rnn_flatten_weight;
        # torch's special case compares against the (wrapped) namespace read.
        # Unfixed, the miss fell through to the loud RuntimeError path.
        from torch.nn.utils._expanded_weights.expanded_weights_impl import (
            ExpandedWeight,
        )

        _ensure_wrapped()
        orig_flatten = _resolve(torch._cudnn_rnn_flatten_weight)
        result = ExpandedWeight.__torch_function__(orig_flatten, (), (), None)
        assert result is None


class TestDisclosedResiduals:
    def test_pickle_while_wrapped_residual_shape(self):
        # DISCLOSED RESIDUAL (same namespace-identity root, USER-side check):
        # pickle's save_global identity-compares a stored original function
        # against the (wrapped) namespace read, so pickling a module holding
        # F.relu fails WHILE wrappers are installed. Pre-existing before the
        # identity shims (default-constructed and pre-wrap-constructed layers
        # always stored the original); fixing it would require restoring the
        # namespace between captures -- a wrapper-lifecycle design change, not
        # a shim. For THIS stored-ORIGINAL shape, unwrap_torch() or a fresh
        # process pickles fine (the namespace read matches the original
        # again). The MIRROR shape -- a user-held plain attribute read taken
        # WHILE wrapped, which stores the WRAPPER -- is NOT recovered by
        # unwrap_torch(); see test_user_held_wrapper_survives_unwrap below.
        # This test pins the residual's shape so a silent change gets noticed.
        import io
        import pickle

        if not _flag("HAS_TRANSFORMER_ACTIVATION_FASTPATH_FLAG"):
            pytest.skip("no transformer fastpath flag on this torch build")
        _ensure_wrapped()
        layer = nn.TransformerEncoderLayer(d_model=8, nhead=2)
        assert layer.activation is _resolve(F.relu)
        with pytest.raises(pickle.PicklingError, match="relu"):
            pickle.dump(layer, io.BytesIO())

    def test_pickle_after_unwrap_succeeds(self):
        # Recovery claim scoped to the stored-ORIGINAL shape only: the shim
        # stores the original torch function, so once unwrap_torch() restores
        # the namespace the identity comparison matches again. This does NOT
        # generalize to user-held wrapper references (next test).
        import io
        import pickle

        from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

        if not _flag("HAS_TRANSFORMER_ACTIVATION_FASTPATH_FLAG"):
            pytest.skip("no transformer fastpath flag on this torch build")
        _ensure_wrapped()
        layer = nn.TransformerEncoderLayer(d_model=8, nhead=2)
        try:
            unwrap_torch()
            buffer = io.BytesIO()
            pickle.dump(layer, buffer)
            assert buffer.getvalue()
        finally:
            wrap_torch()

    def test_user_held_wrapper_survives_unwrap(self):
        # DISCLOSED RESIDUAL (honest shape): a plain attribute read of a
        # wrapped function taken WHILE wrappers are installed (``held =
        # F.relu``) hands the user the WRAPPER object, and unwrap_torch()
        # cannot repair it -- TorchLens never crawls or mutates user objects
        # (the sys.modules crawler is deleted by design). After unwrap the
        # held reference stays callable (it delegates to the original) but is
        # identity-poisoned: ``held is F.relu`` is False and pickling it (or
        # any object holding it) fails, because save_global resolves the
        # qualname to the restored ORIGINAL and identity-compares. Recovery
        # requires re-reading the attribute (or a fresh process), never
        # unwrap alone. The migration doc must disclose this exactly.
        import pickle

        from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

        _ensure_wrapped()
        held = F.relu
        assert _resolve(held) is not held, "namespace read while wrapped must be the wrapper"
        try:
            unwrap_torch()
            assert held is not F.relu, "unwrap_torch() does not repair user-held references"
            assert torch.equal(held(torch.tensor([-1.0, 1.0])), torch.tensor([0.0, 1.0]))
            with pytest.raises(pickle.PicklingError, match="relu"):
                pickle.dumps(held)
        finally:
            wrap_torch()
        doc = (
            Path(__file__).resolve().parents[1]
            / "docs"
            / "migration"
            / "scoped_detached_patching.md"
        )
        text = doc.read_text(encoding="utf-8")
        assert "does not repair user-held wrapper references" in text, (
            "the migration doc must disclose that unwrap_torch() cannot recover "
            "plain attribute reads taken while wrapped"
        )


class TestCausalBiasImportWindow:
    # A torch module first imported WHILE wrappers are installed used to
    # escape the shims until the NEXT capture entry re-ran
    # install_identity_shims -- and in that window a CausalBias sdpa OUTSIDE
    # any capture silently dropped the causal mask (wrong numbers). The
    # import hook closes the window: the shim installs the moment the module
    # executes.

    @pytest.mark.skipif(
        not _flag("HAS_ATTENTION_CAUSAL_BIAS"),
        reason="torch build lacks torch.nn.attention.bias.CausalBias",
    )
    def test_post_wrap_import_is_shimmed_immediately(self):
        import sys

        _ensure_wrapped()
        module_name = "torch.nn.attention.bias"
        saved_module = sys.modules.pop(module_name, None)
        parent = sys.modules.get("torch.nn.attention")
        saved_attr = getattr(parent, "bias", None) if parent is not None else None
        if parent is not None and saved_attr is not None:
            delattr(parent, "bias")
        try:
            # Fresh import under wrap, with NO capture entry in between:
            # exactly the historical coverage window.
            import torch.nn.attention.bias as bias_module

            tf_method = vars(bias_module.CausalBias).get("__torch_function__")
            shimmed = bool(
                getattr(
                    getattr(tf_method, "__func__", tf_method),
                    "_torchlens_identity_shim",
                    False,
                )
            )
            assert shimmed, "the shim must install at import time, not at the next capture"

            # And the numbers must be right OUTSIDE any capture: the sdpa
            # dispatch must apply the causal mask, matching an explicit mask.
            torch.manual_seed(0)
            q = torch.randn(1, 2, 6, 8)
            k = torch.randn(1, 2, 6, 8)
            v = torch.randn(1, 2, 6, 8)
            bias = bias_module.causal_lower_right(q.shape[-2], k.shape[-2])
            explicit = torch.tril(torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool))
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
            reference = F.scaled_dot_product_attention(q, k, v, attn_mask=explicit)
            assert torch.allclose(out, reference, atol=1e-6), (
                "CausalBias sdpa outside a capture dropped the causal mask"
            )
        finally:
            if saved_module is not None:
                sys.modules[module_name] = saved_module
            if parent is not None and saved_attr is not None:
                parent.bias = saved_attr


class TestSubclassCtorUnderWitness:
    # ``TensorBase.__new__`` with a strict Tensor SUBCLASS cls crashes whenever
    # ANY python TorchDispatchMode is active (torch materializes the interior
    # tensor's python object as plain ``Tensor`` before the subclass
    # association runs -- reproduced on stock torch with a no-op mode). The
    # completeness witness is TorchLens's own dispatch mode, armed for
    # validation and runnable-eligible captures, so without the wrapper-side
    # carve-out a mainstream SDPA+CausalBias model captured fine but could
    # never be VALIDATED (the R55-1 capture-time residual, root-caused).

    def test_user_tensor_subclass_ctor_survives_witnessed_capture(self):
        class PlainSubclass(torch.Tensor):
            pass

        class SubclassCtorModel(nn.Module):
            def forward(self, x):
                scratch = PlainSubclass(2, 3)
                return x + scratch.sum() * 0

        torch.manual_seed(0)
        x = torch.randn(2, 3)
        assert tl.validation.validate_forward_pass(SubclassCtorModel(), x)

    @pytest.mark.skipif(
        not _flag("HAS_ATTENTION_CAUSAL_BIAS"),
        reason="torch build lacks torch.nn.attention.bias.CausalBias",
    )
    def test_causal_bias_model_validates(self):
        from torch.nn.attention.bias import causal_lower_right

        class SDPAModel(nn.Module):
            def forward(self, q, k, v):
                bias = causal_lower_right(q.shape[-2], k.shape[-2])
                return F.scaled_dot_product_attention(q, k, v, attn_mask=bias)

        torch.manual_seed(0)
        q = torch.randn(1, 2, 6, 8)
        k = torch.randn(1, 2, 6, 8)
        v = torch.randn(1, 2, 6, 8)
        assert tl.validation.validate_forward_pass(SDPAModel(), (q, k, v))

    def test_plain_tensor_ctor_keeps_witness_view(self):
        # cls == torch.Tensor exactly must NOT pop the witness: the plain
        # legacy ctor works under a dispatch mode, so the census keeps its
        # full view there.
        class PlainCtorModel(nn.Module):
            def forward(self, x):
                scratch = torch.Tensor(2, 3)
                return x + scratch.sum() * 0

        torch.manual_seed(0)
        x = torch.randn(2, 3)
        assert tl.validation.validate_forward_pass(PlainCtorModel(), x)

    def test_all_top_contiguous_torchlens_modes_are_restored(self) -> None:
        """The constructor bracket exits and restores every removable owned mode."""
        from torchlens.backends.torch._modes import (
            _TorchLensDispatchMode,
            pause_own_dispatch_modes,
        )

        events: list[str] = []

        class SyntheticMode(_TorchLensDispatchMode):
            def __init__(self, name: str) -> None:
                super().__init__()
                self.name = name

            def __enter__(self) -> Any:
                events.append(f"enter:{self.name}")
                return super().__enter__()

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc_value: BaseException | None,
                traceback: Any,
            ) -> bool | None:
                events.append(f"exit:{self.name}")
                return super().__exit__(exc_type, exc_value, traceback)

        outer = SyntheticMode("outer")
        inner = SyntheticMode("inner")
        with outer, inner:
            events.clear()
            with pause_own_dispatch_modes() as exited:
                assert exited == (inner, outer)
            assert events == ["exit:inner", "exit:outer", "enter:outer", "enter:inner"]

    def test_foreign_mode_blocks_owned_mode_pop(self) -> None:
        """A foreign top mode and every owned mode below it remain untouched."""
        from torch.utils._python_dispatch import TorchDispatchMode

        from torchlens.backends.torch._modes import (
            _TorchLensDispatchMode,
            pause_own_dispatch_modes,
        )

        class OwnedMode(_TorchLensDispatchMode):
            pass

        class ForeignMode(TorchDispatchMode):
            pass

        owned = OwnedMode()
        foreign = ForeignMode()
        with owned, foreign, pause_own_dispatch_modes() as exited:
            assert exited == ()

    def test_owned_modes_restore_when_paused_call_raises(self) -> None:
        """The pause bracket restores owned modes before propagating an exception."""
        from torchlens.backends.torch._modes import (
            _TorchLensDispatchMode,
            pause_own_dispatch_modes,
        )

        owned = _TorchLensDispatchMode()
        with owned:
            with (
                pytest.raises(RuntimeError, match="ctor failed"),
                pause_own_dispatch_modes() as exited,
            ):
                assert exited == (owned,)
                raise RuntimeError("ctor failed")
            from torchlens.utils._torch_compat import get_current_dispatch_mode_stack

            stack = get_current_dispatch_mode_stack()
            assert stack is not None and stack[-1] is owned


class _PlainTensorSubclass(torch.Tensor):
    pass


class TestAsSubclassOpIdentity:
    # R16-5: torch's default __torch_function__ return conversion calls
    # ret.as_subclass(cls) INSIDE the enclosing wrapped call, which used to
    # steal its bottom-level barcode: the real op (tanh) never logged and the
    # trace showed a parentless bookkeeping as_subclass node flagged only by
    # the provenance heuristic. as_subclass is now barcode-transparent.

    def test_ops_on_subclass_tensors_keep_their_identity(self):
        class Model(nn.Module):
            def forward(self, x):
                s = torch.sigmoid(x).as_subclass(_PlainTensorSubclass)
                return torch.tanh(s)

        torch.manual_seed(0)
        x = torch.randn(2, 3)
        log = tl.trace(Model(), x)
        names = [op.func_name for op in log.ops]
        assert "tanh" in names, names
        tanh_op = log["tanh"]
        assert tanh_op.parents, "tanh must be connected to the graph"
        assert "assubclass" in tanh_op.parents[0]
        assert log.rescue_rerun is None, "a clean capture must not need a rescue"
        expected = torch.tanh(torch.sigmoid(x))
        assert torch.allclose(log["tanh"].out, expected, atol=1e-6)

    def test_subclass_model_validates(self):
        class Model(nn.Module):
            def forward(self, x):
                s = torch.sigmoid(x).as_subclass(_PlainTensorSubclass)
                return torch.tanh(s)

        torch.manual_seed(0)
        if not _flag("HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE"):
            # Floor-torch capability gap (torch 2.1/2.2): see
            # SubclassConstructionUnderDispatchModeError.
            from torchlens.backends.torch._modes import (
                SubclassConstructionUnderDispatchModeError,
            )

            with pytest.raises(SubclassConstructionUnderDispatchModeError) as excinfo:
                tl.validation.validate_forward_pass(Model(), torch.randn(2, 3))
            assert excinfo.value.fields["code"] == "subclass_ctor_under_dispatch_mode_unsupported"
            return
        assert tl.validation.validate_forward_pass(Model(), torch.randn(2, 3))


class TestOverloadShimLedgerKeying:
    def test_overload_shim_keys_on_ledger_identity_not_marker_presence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """b8-fable R56 (attribute-vs-identity anti-pattern): a foreign
        ``@functools.wraps(F.relu)`` wrapper inherits the torchlens
        ``__dict__`` markers, and the marker-keyed jit-overload shim silently
        swapped it for the pristine original -- dropping the foreign behavior
        from overload resolution. The shim must key on LEDGER identity: real
        torchlens wrappers resolve to their originals, foreign wrappers pass
        through untouched even with copied markers.
        """

        import functools

        from torchlens.backends.torch import identity_shims

        _ensure_wrapped()
        module = _torch_compat.get_jit_overload_resolver_module()
        if module is None or not hasattr(module, "_get_overloads"):
            pytest.skip("no jit overload resolver on this torch")

        wrapped_relu = F.relu
        original_relu = _state._decorated_to_orig.get(id(wrapped_relu))
        assert original_relu is not None, "expected F.relu to be wrapped"

        @functools.wraps(wrapped_relu)
        def foreign_wrapper(*args: object, **kwargs: object) -> object:
            return wrapped_relu(*args, **kwargs)

        # functools.wraps copied the torchlens markers -- the spoof surface.
        assert foreign_wrapper.__dict__.get("__tl_wrapper_name__") is not None

        seen: list[object] = []

        def spy(obj: object) -> list[object]:
            seen.append(obj)
            return []

        records: list[tuple[object, str, object]] = []
        monkeypatch.setattr(module, "_get_overloads", spy)
        identity_shims._install_jit_overload_shim(records)  # shim over the spy
        shim = module._get_overloads
        assert shim is not spy, "installer did not shim the spy"

        shim(wrapped_relu)
        shim(foreign_wrapper)
        assert seen[0] is original_relu, "torchlens wrapper must resolve to its original"
        assert seen[1] is foreign_wrapper, (
            "foreign wrapper with copied __tl_* markers was swapped for the "
            "pristine original: overload resolution drops the foreign behavior"
        )
        # monkeypatch teardown restores the real shim over the real resolver.


class TestShimLifecycle:
    def test_shims_removed_on_unwrap_and_reinstalled_on_wrap(self):
        from torchlens.backends.torch import identity_shims
        from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

        _ensure_wrapped()
        init = vars(nn.TransformerEncoderLayer)["__init__"]
        assert getattr(init, "_torchlens_identity_shim", False)
        try:
            unwrap_torch()
            init = vars(nn.TransformerEncoderLayer)["__init__"]
            assert not getattr(init, "_torchlens_identity_shim", False)
            assert not identity_shims.identity_shims_installed()
        finally:
            wrap_torch()
        init = vars(nn.TransformerEncoderLayer)["__init__"]
        assert getattr(init, "_torchlens_identity_shim", False)
        assert identity_shims.identity_shims_installed()

    def test_repeated_wrap_installs_a_single_shim_layer(self):
        from torchlens.backends.torch.wrappers import wrap_torch

        _ensure_wrapped()
        wrap_torch()
        wrap_torch()
        init = vars(nn.TransformerEncoderLayer)["__init__"]
        assert getattr(init, "_torchlens_identity_shim", False)
        inner = getattr(init, "__wrapped__", None)
        assert inner is not None
        assert not getattr(inner, "_torchlens_identity_shim", False)

    def test_capability_flags_reported_in_snapshot(self):
        from torchlens.utils._torch_compat import get_torch_capability_snapshot

        snapshot = get_torch_capability_snapshot()
        for name in (
            "HAS_TRANSFORMER_ACTIVATION_FASTPATH_FLAG",
            "HAS_ATTENTION_CAUSAL_BIAS",
            "HAS_EXPANDED_WEIGHTS_CONV_PICKER",
        ):
            assert name in snapshot
