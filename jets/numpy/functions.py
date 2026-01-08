import functools

import numpy as np

from .jet import NumpyJet

HANDLE_NUMPY_FUNCTIONS = {}


def implements(numpy_function):
    """Register a numpy function override for Jet objects."""

    def decorator(func):
        functools.update_wrapper(func, numpy_function)
        HANDLE_NUMPY_FUNCTIONS[numpy_function] = func
        return func

    return decorator


@implements(np.exp)
def exp(x: NumpyJet) -> NumpyJet:
    value = np.exp(x.value)
    grad = value * x.grad
    return x.createas(value, grad)


@implements(np.sin)
def sin(x: NumpyJet) -> NumpyJet:
    value = np.sin(x.value)
    grad = np.cos(x.value) * x.grad
    return x.createas(value, grad)


@implements(np.cos)
def cos(x: NumpyJet) -> NumpyJet:
    value = np.cos(x.value)
    grad = -np.sin(x.value) * x.grad
    return x.createas(value, grad)


@implements(np.log)
def log(x: NumpyJet) -> NumpyJet:
    value = np.log(x.value)
    grad = x.grad / x.value
    return x.createas(value, grad)
