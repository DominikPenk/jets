from __future__ import annotations

from typing import Any, Generic, Optional, ParamSpecKwargs, TypeVar

import numpy as np

from .jet import JetBase
from .typing import Array, ArrayLike, Value

REGISTERED_BACKENDS: dict[str, tuple[type[Backend], dict[str, Any]]] = {}

J = TypeVar("J", bound=JetBase)


def register_backend(
    name: str,
    cls: type[Backend],
    as_default: bool = False,
    **default_args: ParamSpecKwargs,
):
    if not issubclass(cls, Backend):
        raise ValueError("given class is not a subclass of Backend")
    REGISTERED_BACKENDS[name] = (cls, default_args)
    if get_backend._default is None or as_default:
        get_backend._default = cls(**default_args)


def set_default_backend(backend: str | Backend[Any], **backend_args: ParamSpecKwargs):
    backend = (
        backend
        if isinstance(backend, Backend)
        else get_backend(backend, **backend_args)
    )
    get_backend._default = backend


def get_backend(
    backend: Optional[str],
    **kwargs: ParamSpecKwargs,
) -> Backend:
    if backend is None:
        assert get_backend._default is not None, "No default backend is set."
        return get_backend._default
    if backend not in REGISTERED_BACKENDS:
        raise ValueError(f"Backend '{backend}' is not registered.")
    cls, default_args = REGISTERED_BACKENDS[backend]

    backend_args = default_args.copy()
    backend_args.update(kwargs)

    return cls(**backend_args)


get_backend._default = None


class Backend(Generic[J]):
    JetType: type[J]  # To be defined in subclasses

    def __init__(self, dtype: Optional[Any] = None):
        self.dtype = dtype

    def create_jet(self, value: Value, grad: Array) -> JetBase:
        return self.JetType(value, grad, dtype=self.dtype)

    def to_array(self, x: ArrayLike) -> Array:
        raise NotImplementedError("to_array method not implemented for base class.")

    def to_dtype(self, x: Array, dtype) -> Array:
        raise NotImplementedError

    def to_numpy(self, x: Array) -> np.ndarray:
        raise NotImplementedError

    def unstack(self, x: Array, axis: int) -> tuple[Array, ...]:
        raise NotImplementedError("Unstack function not implemented for base class.")

    def where(self, condition: Array, x: Array, y: Array) -> Array:
        raise NotImplementedError("Where function not implemented for base class.")

    def expand_dims(self, x: Array, axis: int | list[int]) -> Array:
        axis = axis if isinstance(axis, list) else [axis]
        axis = [i + len(x.shape) + 1 if i < 0 else i for i in axis]
        target_shape = list(x.shape)
        for ax in sorted(axis):
            target_shape.insert(ax, 1)
        return x.reshape(target_shape)

    def repeat(self, x: Array, repeats: tuple[int, ...]) -> Array:
        raise NotImplementedError("Repeat function not implemented for base class.")

    def einsum(self, equation: str, *operands: Array) -> Array:
        raise NotImplementedError("Einsum function not implemented for base class.")

    def matrix_inverse(self, x: Array) -> Array:
        raise NotImplementedError(
            "Matrix inverse function not implemented for base class."
        )

    def mean(self, x: Array, axis: int | tuple[int, ...] | None = None) -> Array:
        raise NotImplementedError("Mean function not implemented for base class.")

    def moveaxis(
        self, x: Array, source: int | list[int], destination: int | list[int]
    ) -> Array:
        raise NotImplementedError("Moveaxis function not implemented for base class.")

    def eye(
        self, n: int, dtype: Optional[Any] = None, device: Optional[Any] = None
    ) -> Array:
        raise NotImplementedError("Eye function not implemented for base class.")
