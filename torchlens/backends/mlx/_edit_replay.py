"""Derived gradients of the edited or halted MLX program.

MLX derives gradients by running the model again under ``mx.value_and_grad``
(``MLXBackend._attach_derived_grads``). For a capture with ``intervene=`` or
``stop_after=``, that second run must be the program the capture recorded:
the same edits at the same sites, ending at the same halt frontier. This
module wraps the model for that run.

``edited_replay_model`` returns the model unchanged for a plain capture.
Otherwise it returns a proxy whose call installs an observer over the wrapped
MLX calls (composed with whatever observer the derived-gradient code already
installed, so the intermediate taps and the boundary-replacement oracle see
the edited values). The observer counts each call's output leaves per op name
exactly as capture's ``reserve_label_block`` does, and at each captured edit
site re-applies the capture's own applier to the leaf. A ``tl.grad`` site
wraps the leaf in an identity whose backward rule applies the action to the
gradient arriving at that op's output. At the halt frontier the proxy returns
the frontier call's output, which is the value a halted capture reports as its
output. Every edit site must be reached, and the caller still checks the
replayed output against the captured one, so a replay that drifts from the
capture refuses instead of returning gradients of a different program.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..registry import BackendUnsupportedError
from ._tap_observers import _replace_mlx_array_leaves

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace
    from .backend import MLXBackend

#: Refusal code shared with the capture-entry check (registered in ``_protocols.mlx``).
_REFUSAL_EDIT_GRADS = "intervention_with_derived_grads_unsupported"

SiteKey = tuple[str, int]
"""``(op name, per-op-name output index)``: the raw label without its global counter."""


class _ReplayHalt(Exception):  # noqa: N818 -- control-flow signal, not an error
    """Raised at the halt frontier; carries the frontier call's output."""

    def __init__(self, output: Any) -> None:
        super().__init__("mlx replay reached the halt frontier")
        self.output = output


def site_key(label_raw: str) -> SiteKey:
    """Return the ``(op name, type index)`` key of a capture raw label.

    Parameters
    ----------
    label_raw
        Raw label ``{op}_{type_index}_{raw_index}_raw``.

    Returns
    -------
    SiteKey
        The op name and its per-op-name index; independent of the global raw
        counter, which closure-constant and input labels also advance.
    """

    stem = label_raw[: -len("_raw")] if label_raw.endswith("_raw") else label_raw
    op_name, type_index, _raw_index = stem.rsplit("_", 2)
    return op_name, int(type_index)


def captured_edit_sites(trace: Trace) -> dict[SiteKey, Callable[[Any], Any]]:
    """Return each live-edited output leaf's site key and the applier capture used.

    Parameters
    ----------
    trace
        Finished MLX trace.

    Returns
    -------
    dict[SiteKey, Callable[[Any], Any]]
        One entry per edited leaf, read from the op capture records the
        validator also replays.
    """

    sites: dict[SiteKey, Callable[[Any], Any]] = {}
    for capture in getattr(trace, "_mlx_op_captures", None) or ():
        for index, apply in getattr(capture, "appliers", ()) or ():
            sites[site_key(capture.labels_raw[index])] = apply
    return sites


def check_edit_grad_options(plan: Any | None, grad_options: Any | None) -> None:
    """Refuse the edit and gradient combinations the replay cannot reproduce.

    Parameters
    ----------
    plan
        Resolved ``MLXInterventionPlan`` or ``None``.
    grad_options
        The capture's ``grad_options`` or ``None``.

    Raises
    ------
    BackendUnsupportedError
        For a ``tl.grad`` site without ``grad_options`` (MLX computes
        gradients only in the derived-gradient run) and for a module-exit edit
        with ``grad_options`` (its boundary op is emitted outside a wrapped
        call, which the replay does not reproduce yet).
    """

    if plan is None:
        return
    if getattr(plan, "gradient", False) and grad_options is None:
        raise BackendUnsupportedError(
            "On MLX a tl.grad(...) site edits the derived gradients, so it needs "
            "grad_options=tl.backends.mlx.GradOptions(...) in the same tl.trace call.",
            code=_REFUSAL_EDIT_GRADS,
        )
    if grad_options is not None and getattr(plan, "module_exit", False):
        raise BackendUnsupportedError(
            "MLX backend cannot combine grad_options with a tl.module(...) edit: the "
            "edit applies at the module's exit, outside any wrapped MLX call, and the "
            "derived-gradient run does not reproduce that boundary. Use an op-level "
            "selector (tl.func, tl.label, tl.in_module, tl.where) for the edit.",
            code=_REFUSAL_EDIT_GRADS,
        )


class _EditReplayObserver:
    """Re-apply captured edits, gradient edits and the halt in the AD replay."""

    def __init__(
        self,
        backend: MLXBackend,
        inner: Any | None,
        sites: Mapping[SiteKey, Callable[[Any], Any]],
        grad_sites: Mapping[SiteKey, Callable[[Any], Any]],
        halt_key: SiteKey | None,
    ) -> None:
        self.backend = backend
        self.inner = inner
        self.sites = sites
        self.grad_sites = grad_sites
        self.halt_key = halt_key
        self.depth = 0
        self.type_counters: dict[str, int] = {}
        self.hit: set[SiteKey] = set()

    def call(
        self,
        original: Callable[..., Any],
        op_name: str | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        module_address: str | None = None,
    ) -> Any:
        """Run one wrapped call; edit its leaves at captured sites; halt at the frontier.

        Forward edits apply to the call's own output, before any inner
        observer (an intermediate-gradient tap) sees it, so a tap records the
        gradient of the edited value. Gradient edits wrap the leaves the inner
        observer returns, so the gradient a tap records at a ``tl.grad`` site
        is the edited one, as torch saves it.
        """

        if op_name is None or self.depth > 0:
            return self._delegate(original, op_name, args, kwargs, module_address)
        keys: list[SiteKey] = []

        def run(*call_args: Any, **call_kwargs: Any) -> Any:
            """Call the original and apply the forward edits to its output."""

            return self._edit_output(op_name, original(*call_args, **call_kwargs), keys)

        self.depth += 1
        try:
            output = self._delegate(run, op_name, args, kwargs, module_address)
        finally:
            self.depth -= 1
        output = self._edit_gradients(output, keys)
        if self.halt_key is not None and self.halt_key in keys:
            raise _ReplayHalt(output)
        return output

    def _delegate(
        self,
        original: Callable[..., Any],
        op_name: str | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        module_address: str | None,
    ) -> Any:
        """Hand the call to the observer the derived-gradient code installed, if any."""

        if self.inner is None:
            return original(*args, **kwargs)
        return self.inner.call(original, op_name, args, kwargs, module_address=module_address)

    def _edit_output(self, op_name: str, output: Any, keys: list[SiteKey]) -> Any:
        """Count this call's leaves as capture does and apply the forward edits."""

        replacements: dict[int, Any] = {}
        for leaf in self.backend._iter_arrays(output):
            counter = self.type_counters.get(op_name, 0) + 1
            self.type_counters[op_name] = counter
            key = (op_name, counter)
            keys.append(key)
            apply = self.sites.get(key)
            if apply is not None:
                replacements[id(leaf)] = self._checked(apply(leaf), leaf, key)
                self.hit.add(key)
        if not replacements:
            return output
        return _replace_mlx_array_leaves(self.backend, output, replacements)

    def _edit_gradients(self, output: Any, keys: list[SiteKey]) -> Any:
        """Wrap the leaves at ``tl.grad`` sites in the gradient-editing identity."""

        positions = [index for index, key in enumerate(keys) if key in self.grad_sites]
        if not positions:
            return output
        leaves = list(self.backend._iter_arrays(output))
        if len(leaves) != len(keys):
            raise BackendUnsupportedError(
                "MLX derived-gradient replay: a call returned another number of arrays "
                "than at capture; refusing gradients of a different program.",
                code=_REFUSAL_EDIT_GRADS,
            )
        # The inner observer labels arrays by identity to match downstream
        # parents; the wrapped leaf carries its leaf's label.
        labels = getattr(self.inner, "labels_by_id", None)
        replacements: dict[int, Any] = {}
        for index in positions:
            leaf = leaves[index]
            wrapped = _gradient_edit(self.backend.mx, self.grad_sites[keys[index]], leaf)
            if isinstance(labels, dict) and id(leaf) in labels:
                labels[id(wrapped)] = labels[id(leaf)]
            replacements[id(leaf)] = wrapped
            self.hit.add(keys[index])
        return _replace_mlx_array_leaves(self.backend, output, replacements)

    def _checked(self, replacement: Any, leaf: Any, key: SiteKey) -> Any:
        """Hold the replay to capture's same-shape, same-dtype edit contract."""

        backend = self.backend
        if not backend.is_tensor(replacement) or (
            backend._shape(replacement) != backend._shape(leaf)
            or backend._dtype(replacement) != backend._dtype(leaf)
        ):
            raise BackendUnsupportedError(
                f"MLX derived-gradient replay: the edit at {key[0]} #{key[1]} returned a "
                "value of another type, shape or dtype than at capture.",
                code=_REFUSAL_EDIT_GRADS,
            )
        return replacement


def _gradient_edit(mx: Any, apply: Callable[[Any], Any], value: Any) -> Any:
    """Return ``value`` through an identity whose backward rule applies ``apply``.

    Parameters
    ----------
    mx
        Imported ``mlx.core``.
    apply
        The ``tl.grad`` site's action, applied to the arriving gradient.
    value
        The op's output leaf.

    Returns
    -------
    Any
        ``value`` unchanged in the forward; the gradient flowing back through
        it is ``apply(gradient)``.
    """

    @mx.custom_function
    def grad_site(input_value: Any) -> Any:
        """Return ``input_value`` unchanged."""

        return input_value

    @grad_site.vjp
    def grad_site_vjp(primals: tuple[Any, ...], cotangent: Any, output: Any) -> tuple[Any]:
        """Apply the site's action to the gradient arriving at the op's output."""

        del primals, output
        return (apply(cotangent),)

    return grad_site(value)


@dataclass(frozen=True)
class _ReplayEdits:
    """The captured forward edits, gradient edits and halt the replay re-applies."""

    sites: Mapping[SiteKey, Callable[[Any], Any]]
    grad_sites: Mapping[SiteKey, Callable[[Any], Any]]
    halt_key: SiteKey | None


class _EditedReplayModel:
    """Model proxy whose call replays the captured edits and halt."""

    def __init__(
        self, backend: MLXBackend, taps: Any, model: Callable[..., Any], edits: _ReplayEdits
    ) -> None:
        self._backend = backend
        self._taps = taps
        self._model = model
        self._sites = edits.sites
        self._grad_sites = edits.grad_sites
        self._halt_key = edits.halt_key

    def __getattr__(self, name: str) -> Any:
        """Read every other attribute (``update``, ``parameters``) from the model."""

        return getattr(self._model, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the model with the edits applied; return the halt frontier's output if halted."""

        taps = self._taps
        observer = _EditReplayObserver(
            self._backend,
            taps._ACTIVE_TAP_OBSERVER,
            self._sites,
            self._grad_sites,
            self._halt_key,
        )
        try:
            with taps.mlx_tap_observer(observer):
                output = self._model(*args, **kwargs)
        except _ReplayHalt as halt:
            output = halt.output
        else:
            if self._halt_key is not None:
                raise BackendUnsupportedError(
                    "MLX derived-gradient replay finished without reaching the captured "
                    "halt frontier; refusing gradients of a different program.",
                    code=_REFUSAL_EDIT_GRADS,
                )
        missing = (set(self._sites) | set(self._grad_sites)) - observer.hit
        if missing:
            raise BackendUnsupportedError(
                "MLX derived-gradient replay never reached the captured edit sites "
                f"{sorted(missing)}; refusing gradients of a different program.",
                code=_REFUSAL_EDIT_GRADS,
            )
        return output


def edited_replay_model(
    backend: MLXBackend,
    trace: Trace,
    model: Any,
    plan: Any | None,
    taps: Any,
) -> Callable[..., Any]:
    """Return the model the derived-gradient run should call.

    Parameters
    ----------
    backend
        Active MLX backend.
    trace
        Finished trace (its op captures hold the edit appliers; a halted trace
        names its frontier).
    model
        The captured model.
    plan
        The capture's ``MLXInterventionPlan``, whose ``grad_site_labels`` lists
        the ``tl.grad`` sites matched during the forward.
    taps
        The MLX wrappers module: its ``mlx_tap_observer`` installs the replay
        observer, composed over the ``_ACTIVE_TAP_OBSERVER`` already installed.
        Passed in so this module stays out of the backend's import cycle.

    Returns
    -------
    Callable[..., Any]
        ``model`` itself for a capture without edits or halt, else a proxy that
        replays them.
    """

    sites = captured_edit_sites(trace)
    grad_labels = tuple(getattr(plan, "grad_site_labels", ()) or ())
    grad_sites = (
        {} if plan is None else {site_key(label): plan.applier.apply for label in grad_labels}
    )
    frontier = trace._graph_stop_frontier if trace.is_stopped else None
    halt_key = site_key(frontier) if isinstance(frontier, str) else None
    if not sites and not grad_sites and halt_key is None:
        return model
    return _EditedReplayModel(backend, taps, model, _ReplayEdits(sites, grad_sites, halt_key))
