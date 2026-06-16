"""TensorFlow backend preview scaffold.

The P1 backend owns public-call normalization and eager/static mode selection only.
Live op-callback capture and FuncGraph materialization land in later phases.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from ..registry import BackendUnsupportedError


TFExecutionMode = Literal["eager", "graph_only"]


@dataclass(frozen=True)
class TFCallPlan:
    """Normalized TensorFlow capture call.

    Parameters
    ----------
    callable_obj:
        Single callable selected for the forward entry.
    args:
        Concrete positional values to pass to ``callable_obj``.
    call_kwargs:
        Concrete keyword values to pass to ``callable_obj``.
    mode:
        Selected execution mode.
    reason:
        Human-readable reason for the selected mode.
    """

    callable_obj: Any
    args: tuple[Any, ...]
    call_kwargs: dict[str, Any]
    mode: TFExecutionMode
    reason: str


class TFBackend:
    """TensorFlow backend preview shell."""

    def capture_trace(self, *args: Any, **kwargs: Any) -> Any:
        """Normalize a public trace call, select mode, then stop at the P2 boundary.

        Parameters
        ----------
        *args, **kwargs:
            Public ``trace`` arguments.

        Returns
        -------
        Any
            Never returns in P1.
        """

        plan = self.normalize_call(*args, **kwargs)
        if plan.mode == "graph_only":
            raise NotImplementedError(
                f"tf static graph capture lands after P1; selected graph-only mode: {plan.reason}."
            )
        raise BackendUnsupportedError("tf eager capture lands in P2")

    def validate_entry(self, *args: Any, **kwargs: Any) -> bool:
        """Normalize a validation entry and report P1 unsupported status.

        Parameters
        ----------
        *args, **kwargs:
            Public validation arguments.

        Returns
        -------
        bool
            Never returns in P1.
        """

        self.normalize_call(*args, **kwargs)
        raise BackendUnsupportedError("tf validation replay lands after P1")

    def validate_trace(self, *args: Any, **kwargs: Any) -> Any:
        """Report P1 unsupported status for trace replay validation.

        Parameters
        ----------
        *args, **kwargs:
            Trace validation arguments.

        Returns
        -------
        Any
            Never returns in P1.
        """

        del args, kwargs
        raise BackendUnsupportedError("tf trace replay validation lands after P1")

    def normalize_call(self, *args: Any, **kwargs: Any) -> TFCallPlan:
        """Normalize public or direct backend arguments into a TensorFlow call plan.

        Parameters
        ----------
        *args, **kwargs:
            Either public ``trace`` positional arguments or the keyword bundle passed
            by ``torchlens.trace``.

        Returns
        -------
        TFCallPlan
            Normalized callable, inputs, call kwargs, and selected execution mode.
        """

        model, input_args, input_kwargs = self._extract_public_call(*args, **kwargs)
        callable_obj = self._select_callable(model)
        concrete_args = self._normalize_input_args(input_args)
        call_kwargs = self._normalize_input_kwargs(input_kwargs)
        mode, reason = self._select_mode(model, callable_obj)
        return TFCallPlan(
            callable_obj=callable_obj,
            args=concrete_args,
            call_kwargs=call_kwargs,
            mode=mode,
            reason=reason,
        )

    def _extract_public_call(self, *args: Any, **kwargs: Any) -> tuple[Any, object, object]:
        """Extract model, positional inputs, and keyword inputs from a backend call.

        Parameters
        ----------
        *args, **kwargs:
            Direct or public keyword-style backend call.

        Returns
        -------
        tuple[Any, object, object]
            Model, input args, and input kwargs.
        """

        if args:
            model = args[0]
            input_args = args[1] if len(args) > 1 else kwargs.get("input_args", ())
            input_kwargs = args[2] if len(args) > 2 else kwargs.get("input_kwargs")
            return model, input_args, input_kwargs
        return kwargs["model"], kwargs.get("input_args", ()), kwargs.get("input_kwargs")

    def _select_callable(self, model: Any) -> Any:
        """Select the single callable used for TensorFlow capture.

        Parameters
        ----------
        model:
            Public model or callable object.

        Returns
        -------
        Any
            Callable forward entry.
        """

        if not callable(model):
            raise BackendUnsupportedError("TensorFlow backend requires a callable capture entry.")
        return model

    def _normalize_input_args(self, input_args: object) -> tuple[Any, ...]:
        """Normalize public positional inputs into a tuple.

        Parameters
        ----------
        input_args:
            Public positional input object.

        Returns
        -------
        tuple[Any, ...]
            Positional call arguments.
        """

        if input_args is None:
            return ()
        if isinstance(input_args, tuple):
            return input_args
        if isinstance(input_args, list):
            return tuple(input_args)
        return (input_args,)

    def _normalize_input_kwargs(self, input_kwargs: object) -> dict[str, Any]:
        """Normalize public keyword inputs into a string-keyed call mapping.

        Parameters
        ----------
        input_kwargs:
            Public keyword input mapping.

        Returns
        -------
        dict[str, Any]
            Keyword call arguments.
        """

        if input_kwargs is None:
            return {}
        if not isinstance(input_kwargs, Mapping):
            raise TypeError("input_kwargs must be a mapping when supplied.")
        return {str(key): value for key, value in input_kwargs.items()}

    def _select_mode(self, model: Any, callable_obj: Any) -> tuple[TFExecutionMode, str]:
        """Select eager or graph-only capture mode for a TensorFlow entry.

        Parameters
        ----------
        model:
            Public model object.
        callable_obj:
            Callable selected for capture.

        Returns
        -------
        tuple[TFExecutionMode, str]
            Selected mode and reason.
        """

        tf = self._import_tensorflow()
        if self._is_predict_entry(callable_obj):
            return "graph_only", "predict entry hides eager interiors"
        if self._is_loaded_saved_model(model):
            return "graph_only", "loaded SavedModel signatures require FuncGraph capture"
        if self._is_tf_function(callable_obj, tf):
            return "graph_only", "callable is a tf.function or ConcreteFunction"
        call_attr = getattr(model, "call", None)
        if call_attr is not None and self._is_tf_function(call_attr, tf):
            return "graph_only", "Model.call is a tf.function or ConcreteFunction"
        return "eager", "callable is eager-executable"

    def _import_tensorflow(self) -> Any:
        """Import TensorFlow lazily.

        Returns
        -------
        Any
            Imported TensorFlow module.
        """

        import tensorflow as tf

        return tf

    def _is_predict_entry(self, callable_obj: Any) -> bool:
        """Return whether ``callable_obj`` is a Keras ``predict`` entry.

        Parameters
        ----------
        callable_obj:
            Candidate callable.

        Returns
        -------
        bool
            True for bound or unbound ``predict`` methods.
        """

        name = getattr(callable_obj, "__name__", "")
        qualname = getattr(callable_obj, "__qualname__", "")
        return name == "predict" or qualname.endswith(".predict")

    def _is_loaded_saved_model(self, model: Any) -> bool:
        """Return whether ``model`` looks like a loaded SavedModel object.

        Parameters
        ----------
        model:
            Candidate model.

        Returns
        -------
        bool
            True when SavedModel signatures are present.
        """

        signatures = getattr(model, "signatures", None)
        return isinstance(signatures, Mapping) and bool(signatures)

    def _is_tf_function(self, value: Any, tf: Any) -> bool:
        """Return whether ``value`` is a TensorFlow graph function object.

        Parameters
        ----------
        value:
            Candidate callable.
        tf:
            Imported TensorFlow module.

        Returns
        -------
        bool
            True for ``tf.function``/``PolymorphicFunction``/``ConcreteFunction`` values.
        """

        if hasattr(value, "get_concrete_function"):
            return True
        concrete_function_type = getattr(tf.types.experimental, "ConcreteFunction", None)
        if concrete_function_type is not None and isinstance(value, concrete_function_type):
            return True
        generic_function_type = getattr(tf.types.experimental, "GenericFunction", None)
        return bool(generic_function_type is not None and isinstance(value, generic_function_type))
