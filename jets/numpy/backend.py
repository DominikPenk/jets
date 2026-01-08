from typing import Any, Optional

import numpy as np

from ..backend import Backend
from .jet import NumpyJet


class NumpyBackend(Backend):
    JetType = NumpyJet

    def to_array(self, x: Any) -> np.ndarray:
        return np.asarray(x, dtype=self.dtype)

    def to_numpy(self, x: np.ndarray) -> np.ndarray:
        return x

    def to_dtype(self, x: np.ndarray, dtype) -> np.ndarray:
        return x.astype(dtype)

    def unstack(self, x: np.ndarray, axis: int) -> tuple[np.ndarray, ...]:
        return tuple(np.moveaxis(x, axis, 0))

    repeat = staticmethod(np.repeat)  # type: ignore
    where = staticmethod(np.where)  # type: ignore
    einsum = staticmethod(np.einsum)  # type: ignore
    matrix_inverse = staticmethod(np.linalg.inv)  # type: ignore
    moveaxis = staticmethod(np.moveaxis)  # type: ignore
    mean = staticmethod(np.mean)  # type: ignore
    expand_dims = staticmethod(np.expand_dims)  # type: ignore

    def eye(self, n: int, dtype: Optional[Any] = None, *args, **kwargs) -> np.ndarray:
        dtype = dtype or self.dtype
        return np.eye(n, dtype=dtype)
