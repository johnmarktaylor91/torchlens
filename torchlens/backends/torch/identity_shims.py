"""Wrap-state identity shims: keep torch-internal identity checks truthful.

TorchLens wrapping replaces public torch callables with wrapper functions, so
a torch-internal ``x is F.y`` check whose two operands were read at different
wrap epochs silently changes answer once wrappers are installed. The two
broken operand pairings are:

* a HELD reference (a class-def-time default argument, or a user variable
  bound before the first capture) compared against a post-wrap namespace
  read -- the ``TransformerEncoderLayer`` ctor fastpath flag;
* the ORIGINAL function object passed to ``__torch_function__`` by torch's
  C-level protocol compared against a call-time namespace read -- the
  ``CausalBias`` sdpa dispatch and the expanded-weights per-sample-grads
  machinery (which additionally keys handler TABLES at import time, so its
  basis depends on whether the module was first imported before or after the
  wrap).

Census (2026-08-14, installed-source grep over the supported eager range,
identity/equality forms against wrappable callables, runtime paths only)
found exactly these sites for the ``is``-comparison FORM; compiler/export/
testing namespaces are out of capture scope by contract. The grep cannot see
the third wrap-state shape -- a container built at import time and consulted
by MEMBERSHIP at call time (the shimmed expanded-weights tables are exactly
that shape). The runtime membership-container census and its reviewed
allowlist (``torch._library.utils._RANDOM_FUNCTIONS``, the MaskedTensor
reduce maps, the lazy-module ``_allowed_methods`` allowlist -- each safe on
an eager-import or protocol-supplied-original basis) live in
``tests/test_wrap_state_compat.py``; an unreviewed new table fails that gate. A fourth normalization rides along:
``torch.overrides.resolve_name`` keys its cached index by the pre-warm
originals, so a wrapper argument resolved to ``None`` -- the shim retries a
miss with the ledger original. A fifth normalizes TorchScript's overload
resolver (``torch.jit._script._get_overloads``): it is the one recursive
compilation entry that skips ``__prepare_scriptable__``, so a wrapped
overloaded functional compiled its original source against the wrapper's
globals. The standing installed-tree grep gate lives
in ``tests/test_wrap_state_compat.py``; the ``nested/_internal`` NJT
identity reads it surfaces are a documented unshimmed residual (nested
jagged tensors are not supported capture inputs). An eighth normalizes the
PROTOCOL-ARG identity for pure-Python functionals (grind-r6 b8 R56): their
bodies dispatch ``handle_torch_function(<module-global self-reference>,
...)``, which resolves to the torchlens wrapper during the wrap epoch, so
every host module's ``handle_torch_function`` global gets a translating
shim presenting the ledger ORIGINAL to user ``__torch_function__`` handlers
-- the same identity basis C builtins and unwrapped eager torch present.

Strategy: NEVER re-implement torch's decision logic. Each shim normalizes
the identity operand to the basis the immediately-following torch comparison
uses (the call-time namespace read, or the import-time table key), then
delegates to the original torch code, so the decision itself always runs
upstream logic. Shims install with ``wrap_torch()`` and are removed by
``unwrap_torch()``; with wrappers absent every normalization is an identity
no-op. The one lazily-importable site (causal bias) is additionally covered
by a meta-path import hook, so a module first imported WHILE wrappers are
installed is shimmed the moment it executes — never left broken until the
next capture entry. Site availability is feature-detected in
``torchlens.utils._torch_compat`` (``HAS_TRANSFORMER_ACTIVATION_FASTPATH_FLAG``,
``HAS_ATTENTION_CAUSAL_BIAS``, ``HAS_EXPANDED_WEIGHTS_CONV_PICKER``) and is
visible through the doctor/compat capability snapshot.
"""

from __future__ import annotations

import collections
import functools
import importlib.util
import inspect
import sys
import threading
import types
import warnings
from collections.abc import Callable
from typing import Any

import torch

from ... import _state
from ...utils import _torch_compat

__all__ = [
    "identity_shims_installed",
    "install_identity_shims",
    "remove_identity_shims",
]

_SHIM_MARKER = "_torchlens_identity_shim"
_MISSING = object()

# (holder, attribute name, original attribute value) for every installed shim.
_installed: list[tuple[Any, str, Any]] = []

_live_shims: dict[int, Any] = {}
"""id -> the exact shim objects this module installed (teardown authority).

Restore keys on THIS identity registry, never on the ``_SHIM_MARKER``
attribute alone: the marker is spoofable (any foreign function can set
``_torchlens_identity_shim = True``), and a marker-keyed teardown would then
CLOBBER the user's monkeypatched site with our stored original (grind-r6 b8
R56, sol). The strong references also prevent id reuse for the registry's
lifetime.
"""

# True ONLY between a successful FULL family install and the matching remove.
# A non-empty ``_installed`` list must never stand in for family completeness:
# a raced import callback could append one causal-bias record into a
# post-teardown empty list, and the next ``install_identity_shims`` would then
# skip the full install, leaving the transformer/expanded-weights shims absent
# (the SF-53 fastpath bug re-created through the lifecycle seam).
_family_installed = False

# Live import hook covering the lazily-importable causal-bias site, or None.
_import_hook: _CausalBiasShimImportHook | None = None

_import_hook_local = threading.local()


class _CausalBiasShimImportHook:
    """Meta-path finder shimming CausalBias the moment its module executes.

    The causal-bias site is the ONE census entry that resolves lazily through
    ``sys.modules`` (importing it drags the dynamo tree into every wrap), so a
    user import of ``torch.nn.attention.bias`` WHILE wrappers are installed
    used to leave the fresh class unshimmed until the next capture entry
    re-ran ``install_identity_shims`` — and in that window a CausalBias sdpa
    OUTSIDE any capture silently dropped the causal mask (the C-level
    protocol's original-``func`` identity miss). This finder wraps the
    module's loader so the shim installs immediately after module execution,
    closing the window; the capture-entry re-pickup stays as the belt.
    """

    _WATCHED = "torch.nn.attention.bias"

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        """Return the watched module's spec with a shim-installing loader."""

        if fullname != self._WATCHED or getattr(_import_hook_local, "busy", False):
            return None
        # find_spec below walks sys.meta_path again (including this finder);
        # the thread-local busy flag breaks the recursion so the real finders
        # answer.
        _import_hook_local.busy = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            _import_hook_local.busy = False
        if spec is None or spec.loader is None:
            return None
        # The proxy duck-types the Loader protocol (create_module/exec_module
        # delegate; everything else forwards via __getattr__).
        spec.loader = _ShimOnExecLoader(spec.loader)  # type: ignore[assignment]
        return spec


class _ShimOnExecLoader:
    """Loader proxy: run the real module exec, then install the shim."""

    def __init__(self, loader: Any) -> None:
        self._loader = loader

    def create_module(self, spec: Any) -> Any:
        """Delegate module creation to the real loader."""

        return self._loader.create_module(spec)

    def exec_module(self, module: Any) -> None:
        """Execute the module, then shim the freshly-defined CausalBias.

        Mirrors ``install_identity_shims``'s failure contract: an error while
        shimming restores what this call patched and re-raises loudly — a
        silently unshimmed CausalBias is exactly the wrong-numbers bug this
        hook exists to close.

        The shim-install tail participates in the wrapper lifecycle lock:
        unlocked, the check-then-install sequence raced ``unwrap_torch()``
        (remove could run between the completeness check and the append,
        leaving one causal-bias record installed with wrappers off and the
        next wrap short-circuiting the full family install). The lock is NOT
        held across the real module exec — only around the shim tail — so a
        module import cannot deadlock against a concurrent wrap/unwrap.
        """

        self._loader.exec_module(module)
        from .wrappers import _wrapper_install_lock

        with _wrapper_install_lock:
            if not _family_installed:
                # Shims were removed between find_spec and this tail (unwrap
                # raced the import): with wrappers gone every normalization is
                # a no-op and nothing must be left patched.
                return
            records: list[tuple[Any, str, Any]] = []
            try:
                _install_causal_bias_shim(records)
            except Exception:
                _restore(records)
                raise
            _installed.extend(records)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)


def _ensure_import_hook() -> None:
    """Install the causal-bias import hook once; no-op when the site is absent."""

    global _import_hook
    if not _torch_compat.HAS_ATTENTION_CAUSAL_BIAS:
        return
    if _import_hook is not None and _import_hook in sys.meta_path:
        return
    _import_hook = _CausalBiasShimImportHook()
    sys.meta_path.insert(0, _import_hook)


def _remove_import_hook() -> None:
    """Remove the causal-bias import hook if installed."""

    global _import_hook
    if _import_hook is not None:
        with_hook = [finder for finder in sys.meta_path if finder is not _import_hook]
        if len(with_hook) != len(sys.meta_path):
            sys.meta_path[:] = with_hook
        _import_hook = None


def _resolve(fn: Any) -> Any:
    """Follow the wrapper ledger from a torchlens wrapper to its original.

    Parameters
    ----------
    fn:
        Any object; non-wrappers resolve to themselves.

    Returns
    -------
    Any
        The original torch callable for a torchlens wrapper, else ``fn``.
    """

    seen: set[int] = set()
    while id(fn) in _state._decorated_to_orig and id(fn) not in seen:
        seen.add(id(fn))
        fn = _state._decorated_to_orig[id(fn)]
    return fn


def _register_shim(fn: Any) -> Any:
    """Mark ``fn`` as a torchlens shim and enroll it in the identity registry."""

    setattr(fn, _SHIM_MARKER, True)
    _live_shims[id(fn)] = fn
    return fn


def _is_shimmed(value: Any) -> bool:
    """Return whether ``value`` (function or classmethod) is one of our shims."""

    fn = getattr(value, "__func__", value)
    return bool(getattr(fn, _SHIM_MARKER, False))


def _is_our_live_shim(value: Any) -> bool:
    """Identity check: ``value`` is an exact shim object THIS module installed.

    The spoof-resistant form of :func:`_is_shimmed`, used wherever the answer
    authorizes a MUTATION (teardown restore). A foreign callable carrying the
    marker attribute answers False here.
    """

    fn = getattr(value, "__func__", value)
    return id(fn) in _live_shims and _live_shims[id(fn)] is fn


def identity_shims_installed() -> bool:
    """Return whether the FULL identity-shim family is currently installed.

    Keyed on ``_family_installed``, never on ``_installed`` being non-empty
    (grind-r6 b7 R47-A2, 3rd round): a raced import callback can append one
    causal-bias record into a post-teardown empty list, and the non-empty
    proxy would then report "installed" with the transformer/
    expanded-weights shims absent.
    """

    return _family_installed


def install_identity_shims() -> None:
    """Install every census-listed identity shim; idempotent.

    Caller holds the wrapper install lock. A failure mid-install restores the
    already-patched sites and re-raises: a partially shimmed process would be
    a silent-protection lie, and an unexpected error here is a TorchLens bug
    that must surface loudly.
    """

    global _family_installed
    if _family_installed:
        # The causal-bias site resolves only through sys.modules (lazy-import
        # belt). The import hook shims a post-wrap import the moment the
        # module executes; this capture-entry re-pickup stays as the belt for
        # any import the hook missed. The install is a no-op when the site is
        # absent or already shimmed. Keyed on the explicit full-install flag,
        # never on ``_installed`` being non-empty: a lone late-appended record
        # must not stand in for the whole family.
        _ensure_import_hook()
        late_records: list[tuple[Any, str, Any]] = []
        try:
            _install_causal_bias_shim(late_records)
        except Exception:
            _restore(late_records)
            raise
        _installed.extend(late_records)
        return
    records: list[tuple[Any, str, Any]] = []
    try:
        _install_transformer_ctor_shims(records)
        _install_causal_bias_shim(records)
        _install_expanded_weights_shims(records)
        _install_resolve_name_shim(records)
        _install_jit_overload_shim(records)
        _install_fx_trace_shim(records)
        _install_overrides_membership_shims(records)
        _install_protocol_identity_shims(records)
    except Exception:
        _restore(records)
        raise
    _installed.extend(records)
    _family_installed = True
    _ensure_import_hook()


def remove_identity_shims() -> None:
    """Remove all installed identity shims; idempotent.

    Caller holds the wrapper install lock (the import-callback tail takes the
    same lock), so teardown can never interleave with a late causal-bias
    append: the callback either completes first (its record is restored here)
    or observes ``_family_installed`` False and installs nothing.
    """

    global _family_installed
    _family_installed = False
    _remove_import_hook()
    _restore(_installed)
    _installed.clear()
    _live_shims.clear()


def _restore(records: list[tuple[Any, str, Any]]) -> None:
    """Restore original attributes for ``records``, tolerating drift.

    A site whose current value is no longer our shim (user monkeypatching
    layered on top) is left untouched rather than clobbered, mirroring the
    namespace-drift tolerance of wrapper teardown. "Our shim" is decided by
    exact object IDENTITY against the live registry, never by the spoofable
    marker attribute (grind-r6 b8 R56, sol: a foreign function carrying
    ``_torchlens_identity_shim = True`` must not be clobbered at teardown).
    """

    for holder, name, original in reversed(records):
        current = vars(holder).get(name)
        if current is None or not _is_our_live_shim(current):
            continue
        try:
            setattr(holder, name, original)
        except (AttributeError, TypeError):
            continue
        # NOTE: the registry entry is NOT popped here -- one shim object can
        # be installed at several sites (the protocol-identity shim patches
        # every host module), so per-record removal would orphan the later
        # records' identity checks. ``remove_identity_shims`` clears the
        # registry after the full restore.


# ---------------------------------------------------------------------------
# Site 1: TransformerEncoderLayer / TransformerDecoderLayer constructors
# ---------------------------------------------------------------------------


def _install_transformer_ctor_shims(records: list[tuple[Any, str, Any]]) -> None:
    """Shim the transformer layer ctors' activation identity check.

    ``TransformerEncoderLayer.__init__`` decides ``activation_relu_or_gelu``
    (the fused fastpath + nested-tensor gate) with ``activation is F.relu``
    -- the DEFAULT argument is the pre-wrap original bound at class-def time,
    so every post-wrap default construction silently got flag 0 and a
    different forward kernel path. ``TransformerDecoderLayer`` has no flag
    but stores namespace-read activations (the string spelling), so it gets
    the same shim for stored-state wrap invariance.
    """

    if not _torch_compat.HAS_TRANSFORMER_ACTIVATION_FASTPATH_FLAG:
        return
    for cls_name in ("TransformerEncoderLayer", "TransformerDecoderLayer"):
        cls = getattr(torch.nn, cls_name, None)
        if cls is None:
            continue
        orig_init = vars(cls).get("__init__")
        if orig_init is None or _is_shimmed(orig_init):
            continue
        try:
            sig = inspect.signature(orig_init)
        except (TypeError, ValueError):
            continue
        if "activation" not in sig.parameters:
            continue
        cls.__init__ = _make_ctor_shim(orig_init, sig)
        records.append((cls, "__init__", orig_init))
        # ``__setstate__`` injects the CURRENT ``F.relu`` -- the live wrapper
        # while wrapped -- when unpickling legacy state that lacks
        # ``activation`` (encoder writes the attribute after delegating,
        # decoder patches the state dict before). Normalize the stored object
        # afterwards so legacy unpickles are wrap-invariant too.
        orig_setstate = vars(cls).get("__setstate__")
        if orig_setstate is not None and not _is_shimmed(orig_setstate):
            cls.__setstate__ = _make_setstate_shim(orig_setstate)
            records.append((cls, "__setstate__", orig_setstate))


def _make_ctor_shim(orig_init: Callable[..., None], sig: inspect.Signature) -> Callable[..., None]:
    """Build the ctor shim for one transformer layer class."""

    activation_default = sig.parameters["activation"].default

    @functools.wraps(orig_init)
    def ctor_shim(self: Any, *args: Any, **kwargs: Any) -> None:
        """Rebind a stale ``activation`` default to the live namespace callable."""

        import torch.nn.functional as F

        try:
            bound = sig.bind(self, *args, **kwargs)
        except TypeError:
            # Let torch's own signature error surface unchanged.
            orig_init(self, *args, **kwargs)
            return
        activation = bound.arguments.get("activation", _MISSING)
        effective = activation_default if activation is _MISSING else activation
        namespace_form = None
        if callable(effective):
            resolved = _resolve(effective)
            for name in ("relu", "gelu"):
                current = getattr(F, name, None)
                if current is not None and resolved is _resolve(current):
                    namespace_form = current
                    break
        if namespace_form is not None and effective is not namespace_form:
            # Pass the object the interior `activation is F.relu/F.gelu`
            # check reads, so torch's own logic decides the flag correctly.
            bound.arguments["activation"] = namespace_form
        orig_init(*bound.args, **bound.kwargs)
        stored = getattr(self, "activation", None)
        if callable(stored):
            resolved_stored = _resolve(stored)
            if resolved_stored is not stored:
                # Store what an unwrapped construction stores: the original
                # torch function, never a torchlens wrapper (keeps module
                # state byte-identical across wrap states and pickle-clean).
                self.activation = resolved_stored

    _register_shim(ctor_shim)
    return ctor_shim


def _make_setstate_shim(orig_setstate: Callable[..., None]) -> Callable[..., None]:
    """Build the ``__setstate__`` shim for one transformer layer class."""

    @functools.wraps(orig_setstate)
    def setstate_shim(self: Any, state: Any) -> None:
        """Run the original ``__setstate__``, then de-wrap a stored activation."""
        orig_setstate(self, state)
        stored = getattr(self, "activation", None)
        if callable(stored):
            resolved_stored = _resolve(stored)
            if resolved_stored is not stored:
                # Store what an unwrapped unpickle stores: the original torch
                # function, never a torchlens wrapper.
                self.activation = resolved_stored

    _register_shim(setstate_shim)
    return setstate_shim


# ---------------------------------------------------------------------------
# Site 2: torch.nn.attention.bias.CausalBias.__torch_function__
# ---------------------------------------------------------------------------


def _install_causal_bias_shim(records: list[tuple[Any, str, Any]]) -> None:
    """Shim CausalBias's sdpa identity dispatch.

    The C-level protocol passes the ORIGINAL sdpa as ``func``; torch compares
    it against the call-time (wrapped) namespace read. The miss silently fell
    through to the default Tensor path, DROPPING the causal mask and
    returning a broken ``CausalBias``-typed result.
    """

    if not _torch_compat.HAS_ATTENTION_CAUSAL_BIAS:
        return
    # Resolve ONLY through sys.modules (r45/r49 lazy-import belt): importing
    # torch.nn.attention.bias fires torch._dynamo.allow_in_graph at module
    # top level, dragging the _dynamo/_inductor tree into every wrap. A live
    # CausalBias can only exist after the USER imported the module, so an
    # absent module means there is nothing to shim; install_identity_shims
    # re-checks this site on every wrap so a post-wrap import is picked up
    # at the next capture entry.
    module = sys.modules.get("torch.nn.attention.bias")
    causal_bias = getattr(module, "CausalBias", None) if module is not None else None
    if causal_bias is None:
        return
    orig_classmethod = vars(causal_bias).get("__torch_function__")
    if orig_classmethod is None or _is_shimmed(orig_classmethod):
        return
    orig_tf = orig_classmethod.__func__

    @functools.wraps(orig_tf)
    def causal_bias_shim(
        cls: type,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Normalize a stale SDPA reference; defer everything else to Tensor.

        CausalBias's real ``__torch_function__`` raises ``NotImplementedError``
        for any ``func`` other than ``scaled_dot_product_attention`` -- by
        design, it supports nothing else when a USER calls it directly. But a
        CausalBias instance captured as a torchlens op argument or output
        (``causal_lower_right`` et al. are ordinary wrapped functions) is also
        touched by torchlens's OWN generic per-op bookkeeping -- autograd-stats
        partitioning reads ``.grad_fn``, payload cloning reads ``.numel()`` /
        ``.untyped_storage()``, and so on -- plain introspection CausalBias's
        restriction cannot distinguish from genuine misuse, since torch's
        ``__torch_function__`` protocol dispatches ALL of it here regardless
        of torchlens's own wrap/pause-logging state. Once the call is
        confirmed to NOT be (a stale-reference-normalized) sdpa, there is no
        reliable per-call signal left to tell "torchlens bookkeeping" apart
        from "user misuse"; deferring to ``torch.Tensor``'s own permissive
        handler trades away that one user-facing guardrail while a capture is
        instrumenting everything anyway, in exchange for CausalBias working
        as an sdpa argument under capture at all.
        """

        import torch.nn.functional as F

        target = getattr(F, "scaled_dot_product_attention", None)
        if target is not None and func is not target and _resolve(func) is _resolve(target):
            func = target
        if target is not None and func is target:
            return orig_tf(cls, func, types, args, kwargs)
        # torch.Tensor's own __torch_function__ self-checks
        # ``all(issubclass(cls, t) for t in types)``; binding it to
        # ``torch.Tensor`` (the ordinary ``Tensor.__torch_function__``
        # bound-classmethod spelling) fails that check against
        # ``types=(CausalBias,)`` and degrades to ``NotImplemented`` (a
        # multiple-dispatch TypeError). Call the underlying function with
        # the REAL ``cls`` received here instead of letting attribute
        # access rebind it.
        # ``torch.Tensor.__torch_function__`` is stub-typed as a plain bound
        # callable with no ``__func__``; it is a classmethod at runtime, and
        # ``__func__`` is how this unbinds it to call with the real ``cls``
        # above instead of the implicit ``torch.Tensor`` binding.
        unbound_torch_function = getattr(torch.Tensor.__torch_function__, "__func__")
        return unbound_torch_function(cls, func, types, args, kwargs)

    _register_shim(causal_bias_shim)
    causal_bias.__torch_function__ = classmethod(causal_bias_shim)
    records.append((causal_bias, "__torch_function__", orig_classmethod))


# ---------------------------------------------------------------------------
# Site 3: torch.nn.utils._expanded_weights (per-sample-grads machinery)
# ---------------------------------------------------------------------------


def _install_expanded_weights_shims(records: list[tuple[Any, str, Any]]) -> None:
    """Shim the expanded-weights dispatch bases.

    ``ExpandedWeight.__torch_function__`` mixes two comparison bases: handler
    TABLES keyed at module import time (original keys when imported pre-wrap,
    wrapper keys when imported post-wrap) and one call-time NAMESPACE read
    (``func is torch._cudnn_rnn_flatten_weight``). ``conv_picker`` then
    re-reads ``F.conv1d/2d/3d`` from the namespace. Each shim maps ``func``
    to the alias its next comparison actually uses.
    """

    if not _torch_compat.HAS_EXPANDED_WEIGHTS_CONV_PICKER:
        return
    import importlib

    try:
        conv_utils = importlib.import_module("torch.nn.utils._expanded_weights.conv_utils")
        conv_expanded = importlib.import_module(
            "torch.nn.utils._expanded_weights.conv_expanded_weights"
        )
        impl = importlib.import_module("torch.nn.utils._expanded_weights.expanded_weights_impl")
    except ImportError:
        return

    # conv_picker is imported BY VALUE into conv_expanded_weights at torch
    # import time, so both module attributes need the shim.
    for module in (conv_utils, conv_expanded):
        orig_picker = getattr(module, "conv_picker", None)
        if orig_picker is None or _is_shimmed(orig_picker):
            continue
        setattr(module, "conv_picker", _make_conv_picker_shim(orig_picker))  # noqa: B010
        records.append((module, "conv_picker", orig_picker))

    expanded_weight = getattr(impl, "ExpandedWeight", None)
    if expanded_weight is None:
        return
    orig_classmethod = vars(expanded_weight).get("__torch_function__")
    if orig_classmethod is None or _is_shimmed(orig_classmethod):
        return
    orig_tf = orig_classmethod.__func__

    @functools.wraps(orig_tf)
    def expanded_weight_shim(
        cls: type,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Normalize a stale cudnn flatten-weight reference to the namespace basis."""

        flatten = getattr(torch, "_cudnn_rnn_flatten_weight", None)
        if flatten is not None and func is not flatten and _resolve(func) is _resolve(flatten):
            # The special case reads the namespace at call time; hand it the
            # namespace object (it is in no handler table under either alias).
            func = flatten
        else:
            rnn_decomps = getattr(impl, "expanded_weights_rnn_decomps", {})
            handled = getattr(cls, "handled_functions", {})
            if func not in rnn_decomps and func not in handled:
                alias = _state._orig_to_decorated.get(id(func))
                if alias is None:
                    alias = _state._decorated_to_orig.get(id(func))
                if alias is not None and (alias in rnn_decomps or alias in handled):
                    func = alias
        return orig_tf(cls, func, types, args, kwargs)

    _register_shim(expanded_weight_shim)
    expanded_weight.__torch_function__ = classmethod(expanded_weight_shim)
    records.append((expanded_weight, "__torch_function__", orig_classmethod))


# ---------------------------------------------------------------------------
# Site 4: torch.overrides.resolve_name
# ---------------------------------------------------------------------------


def _install_resolve_name_shim(records: list[tuple[Any, str, Any]]) -> None:
    """Shim ``torch.overrides.resolve_name`` to the table's original-key basis.

    ``resolve_name`` looks the callable up in the cached overridable-functions
    index, which is keyed by the objects the namespaces held when the cache
    first materialized (the pre-wrap ORIGINALS once ``decorate_all_once``
    pre-warms both tables). A user or third-party tool passing the CURRENT
    namespace read -- the torchlens wrapper -- silently got ``None`` instead
    of the name. The shim retries a ``None`` miss with the ledger-resolved
    original, so the answer matches unwrapped eager torch under either alias.
    """

    overrides_module = getattr(torch, "overrides", None)
    if overrides_module is None:
        return
    orig_resolve = vars(overrides_module).get("resolve_name")
    if orig_resolve is None or _is_shimmed(orig_resolve):
        return

    @functools.wraps(orig_resolve)
    def resolve_name_shim(f: Any) -> Any:
        """Resolve a wrapper to its original before asking torch for the name."""
        result = orig_resolve(f)
        if result is None:
            original = _resolve(f)
            if original is not f:
                result = orig_resolve(original)
        return result

    _register_shim(resolve_name_shim)
    overrides_module.resolve_name = resolve_name_shim
    records.append((overrides_module, "resolve_name", orig_resolve))


# ---------------------------------------------------------------------------
# Site 5: torch.jit._script._get_overloads
# ---------------------------------------------------------------------------


def _install_jit_overload_shim(records: list[tuple[Any, str, Any]]) -> None:
    """Shim TorchScript's overload resolver to the wrapper's original basis.

    The C++ sugared-value layer resolves a called functional to the CURRENT
    namespace object (the torchlens wrapper) and hands it to
    ``torch.jit._script._get_overloads`` — the one recursive-compilation entry
    that does NOT honor ``__prepare_scriptable__``. It then compiled the
    ORIGINAL source (``inspect.unwrap`` follows ``__wrapped__``) against the
    WRAPPER's globals, so every overloaded pure-Python functional
    (``F.interpolate``, ``F.adaptive_avg_pool2d/3d``) failed to script with
    ``undefined value math`` while wrappers were installed. Normalizing a
    torchlens wrapper to its original before delegating hands torch a
    self-consistent (source, globals) pair; every other caller is untouched.
    """

    module = _torch_compat.get_jit_overload_resolver_module()
    if module is None:
        return
    orig_get_overloads = vars(module).get("_get_overloads")
    if orig_get_overloads is None or _is_shimmed(orig_get_overloads):
        return

    @functools.wraps(orig_get_overloads)
    def get_overloads_shim(obj: Any) -> Any:
        """Resolve a torchlens wrapper to its original before overload lookup.

        Keyed on LEDGER IDENTITY, never on ``__tl_*`` attribute presence: a
        foreign ``@functools.wraps(F.relu)`` wrapper inherits the torchlens
        ``__dict__`` markers (``__tl_wrapper_name__``,
        ``__prepare_scriptable__``), and the attribute-keyed check silently
        swapped such a wrapper for the pristine original -- dropping the
        foreign behavior from overload resolution (b8-fable R56,
        attribute-vs-identity anti-pattern).
        """
        original = _state._decorated_to_orig.get(id(obj))
        if original is not None:
            obj = original
        return orig_get_overloads(obj)

    _register_shim(get_overloads_shim)
    module._get_overloads = get_overloads_shim
    records.append((module, "_get_overloads", orig_get_overloads))


# ---------------------------------------------------------------------------
# Site 6: torch.fx.Tracer.trace -- wrapper-free graph artifacts
# ---------------------------------------------------------------------------


def _install_fx_trace_shim(records: list[tuple[Any, str, Any]]) -> None:
    """Shim ``torch.fx.Tracer.trace`` so node targets record ORIGINALS.

    fx's patcher reads Python functionals from the live namespace at trace
    time, so a ``symbolic_trace`` run during the wrapped epoch baked the
    torchlens WRAPPER object into ``call_function`` node targets (C functions
    correctly record the protocol-supplied original). Any identity/equality-
    keyed fx pass (``node.target == F.relu`` -- the standard torch.ao
    quantization matcher shape) then silently mismatched once wrappers were
    removed or in any other process, and the GraphModule artifact permanently
    embedded a torchlens object (grind-r5 b8 R56). Remapping targets through
    the wrapper ledger after the trace hands every consumer the same graph an
    unwrapped eager trace produces; subclassed tracers (HF-style) funnel
    through the same base method.
    """

    fx_module = getattr(torch, "fx", None)
    tracer_cls = getattr(fx_module, "Tracer", None)
    if tracer_cls is None:
        return
    orig_trace = vars(tracer_cls).get("trace")
    if orig_trace is None or _is_shimmed(orig_trace):
        return

    @functools.wraps(orig_trace)
    def trace_shim(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Trace, then re-point wrapper-valued call_function targets."""
        graph = orig_trace(self, *args, **kwargs)
        try:
            for node in graph.nodes:
                if node.op == "call_function":
                    original = _state._decorated_to_orig.get(id(node.target))
                    if original is not None:
                        node.target = original
        except Exception as error:
            from ..._errors import TorchLensWarning

            warnings.warn(
                "TorchLens could not normalize torchlens wrappers out of an "
                f"fx graph's node targets ({type(error).__name__}: {error}); "
                "the traced GraphModule may embed wrapper objects that break "
                "identity-keyed fx passes after unwrap_torch().",
                TorchLensWarning,
                stacklevel=2,
            )
        return graph

    _register_shim(trace_shim)
    tracer_cls.trace = trace_shim
    records.append((tracer_cls, "trace", orig_trace))


# ---------------------------------------------------------------------------
# Site 7: torch.overrides membership tables
# ---------------------------------------------------------------------------


class _LedgerResolvingTable(dict):
    """Dict view whose LOOKUPS resolve torchlens wrappers to originals.

    Iteration/keys stay exactly the underlying original-keyed contents (the
    wrapper-poisoning census gate keeps holding); only ``in``/``[]``/``get``
    additionally accept the live wrapper alias, so the documented
    ``func in torch.overrides.get_testing_overrides()`` membership check
    answers the same in both wrap epochs (grind-r5 b7 R55-A).
    """

    def __contains__(self, key: Any) -> bool:
        if super().__contains__(key):
            return True
        original = _resolve(key)
        return original is not key and super().__contains__(original)

    def __getitem__(self, key: Any) -> Any:
        try:
            return super().__getitem__(key)
        except KeyError:
            original = _resolve(key)
            if original is not key:
                return super().__getitem__(original)
            raise

    def get(self, key: Any, default: Any = None) -> Any:
        """Return the value for ``key`` (wrapper-resolving), else ``default``."""

        try:
            return self[key]
        except KeyError:
            return default


class _LedgerResolvingMembers(list):
    """List view whose membership test resolves torchlens wrappers."""

    def __contains__(self, item: Any) -> bool:
        if super().__contains__(item):
            return True
        original = _resolve(item)
        return original is not item and super().__contains__(original)


class _LedgerResolvingDefaultTable(collections.defaultdict):
    """Wrapper-resolving view that PRESERVES defaultdict semantics.

    ``get_overridable_functions()`` returns a ``defaultdict(list)``; the r5
    membership fix rebuilt it as a plain resolving dict, so a caller indexing
    a namespace with no recorded entries -- auto-vivification the upstream
    type guarantees -- started raising KeyError (grind-r6 b8 R56, opus
    wave-introduced residue). Ledger resolution runs BEFORE the default
    factory so a wrapper alias still finds its original's row rather than
    minting an empty one.
    """

    def __contains__(self, key: Any) -> bool:
        if dict.__contains__(self, key):
            return True
        original = _resolve(key)
        return original is not key and dict.__contains__(self, original)

    def __getitem__(self, key: Any) -> Any:
        if not dict.__contains__(self, key):
            original = _resolve(key)
            if original is not key and dict.__contains__(self, original):
                return dict.__getitem__(self, original)
        return super().__getitem__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        """Return the value for ``key`` (wrapper-resolving), never auto-creating."""

        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        original = _resolve(key)
        if original is not key and dict.__contains__(self, original):
            return dict.__getitem__(self, original)
        return default


def _install_overrides_membership_shims(records: list[tuple[Any, str, Any]]) -> None:
    """Shim the two cached ``torch.overrides`` table accessors for membership.

    ``decorate_all_once`` pre-warms both caches so their CONTENTS stay keyed
    by pristine originals (r4 F3, verified). But membership by the CURRENT
    namespace read -- ``F.relu in get_testing_overrides()`` or
    ``my_op in get_overridable_functions()[F]``, the documented
    ``__torch_function__`` author checks -- was False for every wrapped
    function during the wrap epoch. The shims hand back per-underlying-table
    cached views that resolve a wrapper argument through the ledger, exactly
    like the shipped ``resolve_name`` shim.
    """

    overrides_module = getattr(torch, "overrides", None)
    if overrides_module is None:
        return
    for accessor_name in ("get_testing_overrides", "get_overridable_functions"):
        orig_accessor = vars(overrides_module).get(accessor_name)
        if orig_accessor is None or _is_shimmed(orig_accessor):
            continue
        view_cache: dict[int, Any] = {}

        def _make_shim(orig: Callable[[], Any], cache: dict[int, Any]) -> Callable[[], Any]:
            """Bind one accessor's original and view cache into its shim closure."""

            @functools.wraps(orig)
            def accessor_shim() -> Any:
                """Return the accessor's table wrapped in a ledger-resolving view."""

                table = orig()
                view = cache.get(id(table))
                if view is None:
                    if isinstance(table, collections.defaultdict):
                        # Preserve the upstream auto-vivification contract
                        # (grind-r6 b8 R56 opus residue: the plain-dict view
                        # turned missing-namespace reads into KeyError).
                        view = _LedgerResolvingDefaultTable(table.default_factory)
                        view.update(
                            (key, _LedgerResolvingMembers(members))
                            if isinstance(members, list)
                            else (key, members)
                            for key, members in table.items()
                        )
                    elif table and isinstance(next(iter(table.values()), None), list):
                        view = _LedgerResolvingTable(
                            (key, _LedgerResolvingMembers(members))
                            for key, members in table.items()
                        )
                    else:
                        view = _LedgerResolvingTable(table)
                    cache.clear()  # underlying cache rebuilt: drop stale views
                    cache[id(table)] = view
                return view

            # The upstream accessors are @functools.lru_cache functions;
            # functools.wraps copies __dict__ only, so cache management
            # attributes vanished from the shimmed surface (grind-r6 b8 R56
            # opus residue: get_testing_overrides.cache_clear() raised
            # AttributeError during the wrap epoch). Forward them.
            for cache_attr in ("cache_clear", "cache_info", "cache_parameters"):
                upstream = getattr(orig, cache_attr, None)
                if upstream is not None:
                    setattr(accessor_shim, cache_attr, upstream)

            return accessor_shim

        shim = _make_shim(orig_accessor, view_cache)
        _register_shim(shim)
        setattr(overrides_module, accessor_name, shim)
        records.append((overrides_module, accessor_name, orig_accessor))


# ---------------------------------------------------------------------------
# Site 8: handle_torch_function host-module globals (pure-Python functionals)
# ---------------------------------------------------------------------------


def _install_protocol_identity_shims(records: list[tuple[Any, str, Any]]) -> None:
    """Present ORIGINALS to user ``__torch_function__`` handlers, both epochs.

    C builtins hand the protocol their own (original) identity, but a wrapped
    PURE-PYTHON torch functional dispatches ``handle_torch_function(relu,
    ...)`` where ``relu`` resolves from its module globals at call time --
    the torchlens WRAPPER once wrappers are installed. Every user handler
    keyed on originals (the documented import-time ``HANDLED_FUNCTIONS``
    table shape) then silently missed, process-wide, for as long as wrappers
    stayed installed after the first capture (grind-r6 b8 R56: sol HIGH,
    fable MED, same root; ~105+ pure-Python functionals).

    The shim patches the ``handle_torch_function`` global of every module
    hosting a wrapped pure-Python functional, translating a wrapper
    ``public_api`` to its ledger original before delegating. Dispatch TIMING
    is untouched (torch's own body still decides whether to dispatch); only
    the presented identity is normalized to the basis unwrapped eager torch
    presents, making pure-Python and C-builtin semantics consistent in both
    wrap epochs.
    """

    overrides_module = getattr(torch, "overrides", None)
    if overrides_module is None:
        return
    real_handle = vars(overrides_module).get("handle_torch_function")
    if real_handle is None or _is_shimmed(real_handle):
        return

    host_modules: dict[int, Any] = {}
    for obj in list(_state._decorated_func_mapper):
        if id(obj) not in _state._orig_to_decorated:
            continue  # not an original
        if not isinstance(obj, types.FunctionType):
            continue  # C originals already present themselves to the protocol
        module = sys.modules.get(getattr(obj, "__module__", "") or "")
        if module is not None:
            host_modules.setdefault(id(module), module)

    @functools.wraps(real_handle)
    def handle_torch_function_shim(
        public_api: Any, relevant_args: Any, *args: Any, **kwargs: Any
    ) -> Any:
        """Normalize a torchlens-wrapper ``public_api`` to its original."""

        original = _state._decorated_to_orig.get(id(public_api))
        if original is not None:
            public_api = original
        return real_handle(public_api, relevant_args, *args, **kwargs)

    _register_shim(handle_torch_function_shim)
    for module in host_modules.values():
        current = vars(module).get("handle_torch_function")
        if current is not real_handle:
            continue  # absent, already shimmed, or foreign-patched: hands off
        setattr(module, "handle_torch_function", handle_torch_function_shim)  # noqa: B010
        records.append((module, "handle_torch_function", real_handle))


def _make_conv_picker_shim(orig_picker: Callable[..., Any]) -> Callable[..., Any]:
    """Build a conv_picker shim normalizing ``func`` to the namespace basis."""

    @functools.wraps(orig_picker)
    def conv_picker_shim(func: Any, conv1d_opt: Any, conv2d_opt: Any, conv3d_opt: Any) -> Any:
        """Normalize a stale conv reference to the ``torch.nn.functional`` basis."""

        import torch.nn.functional as F

        for name in ("conv1d", "conv2d", "conv3d"):
            current = getattr(F, name, None)
            if current is not None and func is not current and _resolve(func) is _resolve(current):
                func = current
                break
        return orig_picker(func, conv1d_opt, conv2d_opt, conv3d_opt)

    _register_shim(conv_picker_shim)
    return conv_picker_shim
