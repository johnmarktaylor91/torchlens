"""Transparent native-framework forward adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ForwardProtocol(Protocol):
    """Runtime protocol required by the execution worker."""

    def forward(self, *args: object, **kwargs: object) -> object:
        """Delegate one native forward call.

        Parameters
        ----------
        *args, **kwargs:
            Native call arguments.

        Returns
        -------
        object
            Unmodified native output.
        """

        ...


@dataclass(frozen=True)
class NativeCallMetadata:
    """Provenance for one transparent native call.

    Parameters
    ----------
    original_framework, run_framework:
        Source and actual execution frameworks.
    native_object_type:
        Qualified native object type.
    native_call_method:
        Method invoked by ``forward``.
    transparent_forward_adapter:
        Always true for this adapter implementation.
    """

    original_framework: str
    run_framework: str
    native_object_type: str
    native_call_method: str
    transparent_forward_adapter: bool = True


class NativeForwardAdapter:
    """Expose ``forward`` as a direct delegation to one native call method.

    Parameters
    ----------
    native_object:
        Framework-native model or bound callable container.
    original_framework:
        Framework in which the architecture is implemented.
    run_framework:
        Framework executing the call. Defaults to the original framework.
    call_method:
        Exact native method delegated to by ``forward``.
    """

    def __init__(
        self,
        native_object: object,
        *,
        original_framework: str,
        run_framework: str | None = None,
        call_method: str = "forward",
    ) -> None:
        """Initialize a transparent native adapter.

        Parameters
        ----------
        native_object:
            Framework-native object.
        original_framework, run_framework, call_method:
            Native call provenance.
        """

        native_call = getattr(native_object, call_method, None)
        if not callable(native_call):
            raise TypeError(
                f"native object {type(native_object).__qualname__} has no callable {call_method!r}"
            )
        self.native_object = native_object
        self._native_call = native_call
        self.training = bool(getattr(native_object, "training", False))
        self.metadata = NativeCallMetadata(
            original_framework=original_framework,
            run_framework=run_framework or original_framework,
            native_object_type=(
                f"{type(native_object).__module__}.{type(native_object).__qualname__}"
            ),
            native_call_method=call_method,
        )

    def forward(self, *args: object, **kwargs: object) -> object:
        """Delegate directly to the recorded native method.

        Parameters
        ----------
        *args, **kwargs:
            Native model inputs.

        Returns
        -------
        object
            Exact native output without tensor transformation.
        """

        return self._native_call(*args, **kwargs)

    def train(self, mode: bool = True) -> "NativeForwardAdapter":
        """Delegate native training-mode selection when supported.

        Parameters
        ----------
        mode:
            Desired training flag.

        Returns
        -------
        NativeForwardAdapter
            This adapter.
        """

        native_train = getattr(self.native_object, "train", None)
        if callable(native_train):
            native_train(mode)
        self.training = mode
        return self

    def eval(self) -> "NativeForwardAdapter":
        """Delegate native evaluation-mode selection when supported.

        Returns
        -------
        NativeForwardAdapter
            This adapter.
        """

        native_eval = getattr(self.native_object, "eval", None)
        if callable(native_eval):
            native_eval()
        else:
            native_train = getattr(self.native_object, "train", None)
            if callable(native_train):
                native_train(False)
        self.training = False
        return self

    def parameters(self, *args: object, **kwargs: object) -> Any:
        """Delegate parameter iteration when the native object exposes it.

        Parameters
        ----------
        *args, **kwargs:
            Native parameter iterator arguments.

        Returns
        -------
        Any
            Native parameter iterator.
        """

        native_parameters = getattr(self.native_object, "parameters", None)
        if not callable(native_parameters):
            return iter(())
        return native_parameters(*args, **kwargs)

    def modules(self) -> Any:
        """Delegate module traversal when supported.

        Returns
        -------
        Any
            Native module iterator, or an iterator containing this adapter.
        """

        native_modules = getattr(self.native_object, "modules", None)
        return native_modules() if callable(native_modules) else iter((self,))


class PyTorchForwardAdapter(NativeForwardAdapter):
    """Transparent adapter for a PyTorch ``forward`` method."""

    def __init__(self, native_object: object) -> None:
        """Wrap a PyTorch object.

        Parameters
        ----------
        native_object:
            Native PyTorch module.
        """

        super().__init__(native_object, original_framework="pytorch", call_method="forward")


class TensorFlowForwardAdapter(NativeForwardAdapter):
    """Transparent adapter for a TensorFlow/Keras ``__call__`` method."""

    def __init__(self, native_object: object) -> None:
        """Wrap a TensorFlow/Keras object.

        Parameters
        ----------
        native_object:
            Native TensorFlow/Keras model.
        """

        super().__init__(native_object, original_framework="tensorflow", call_method="__call__")


class JaxForwardAdapter(NativeForwardAdapter):
    """Transparent adapter for a bound JAX/Flax ``apply`` method."""

    def __init__(self, native_object: object) -> None:
        """Wrap a JAX/Flax object whose parameters are already bound.

        Parameters
        ----------
        native_object:
            Native object exposing ``apply``.
        """

        super().__init__(native_object, original_framework="jax", call_method="apply")


class PaddleForwardAdapter(NativeForwardAdapter):
    """Transparent adapter for a Paddle ``forward`` method."""

    def __init__(self, native_object: object) -> None:
        """Wrap a Paddle layer.

        Parameters
        ----------
        native_object:
            Native Paddle layer.
        """

        super().__init__(native_object, original_framework="paddle", call_method="forward")


def adapt_native_model(native_object: object, framework: str) -> NativeForwardAdapter:
    """Create the closed transparent adapter for a supported framework.

    Parameters
    ----------
    native_object:
        Framework-native model object.
    framework:
        ``pytorch``, ``tensorflow``, ``jax``, or ``paddle`` (including common aliases).

    Returns
    -------
    NativeForwardAdapter
        Framework-specific transparent adapter.
    """

    normalized = framework.lower()
    adapter_types = {
        "torch": PyTorchForwardAdapter,
        "pytorch": PyTorchForwardAdapter,
        "tf": TensorFlowForwardAdapter,
        "tensorflow": TensorFlowForwardAdapter,
        "jax": JaxForwardAdapter,
        "flax": JaxForwardAdapter,
        "paddle": PaddleForwardAdapter,
        "paddlepaddle": PaddleForwardAdapter,
    }
    try:
        adapter_type = adapter_types[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported native framework: {framework!r}") from exc
    return adapter_type(native_object)
