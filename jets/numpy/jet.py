from __future__ import annotations

from typing import Self

import numpy as np

from ..jet import JetBase
from ..typing import Numeric, Value


class NumpyJet(JetBase):
    def __init__(self, value: Value, grad: Value, *args, dtype=None, **kwargs):
        dtype = dtype or np.float64
        self.value: np.ndarray = np.asarray(value, dtype=dtype)
        self.grad: np.ndarray = np.asarray(grad, dtype=dtype)

    @classmethod
    def _log(cls, x: np.ndarray) -> np.ndarray:
        return np.log(x)

    def create_similar(self, value: Numeric, grad: Numeric) -> Self:
        return self.__class__(value, grad, dtype=self.dtype)

    def __array_ufunc__(self, func, method, *args, **kwargs):
        from .functions import HANDLE_NUMPY_FUNCTIONS

        if kwargs is None:
            kwargs = {}
        func = HANDLE_NUMPY_FUNCTIONS.get(func, None)
        if func is None:
            print("Not implemented:", func)
            return NotImplemented
        return func(*args, **kwargs)

    @property
    def dtype(self):
        return self.value.dtype
