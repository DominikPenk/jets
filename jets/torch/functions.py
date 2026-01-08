import functools

import torch

from .jet import TorchJet

HANDLE_TORCH_FUNCTIONS = {}


def implements(torch_function):
    """Register a torch function override for Jet objects."""

    def decorator(func):
        functools.update_wrapper(func, torch_function)
        HANDLE_TORCH_FUNCTIONS[torch_function] = func
        return func

    return decorator


@implements(torch.exp)
def exp(x: TorchJet) -> TorchJet:
    value = torch.exp(x.value)
    grad = value * x.grad
    return x.create_similar(value, grad)


@implements(torch.sin)
def sin(x: TorchJet) -> TorchJet:
    value = torch.sin(x.value)
    grad = torch.cos(x.value) * x.grad
    return x.create_similar(value, grad)


@implements(torch.sinh)
def sinh(x: TorchJet) -> TorchJet:
    value = torch.sinh(x.value)
    grad = torch.cosh(x.value) * x.grad
    return x.create_similar(value, grad)


@implements(torch.cos)
def cos(x: TorchJet) -> TorchJet:
    value = torch.cos(x.value)
    grad = -torch.sin(x.value) * x.grad
    return x.create_similar(value, grad)


@implements(torch.cosh)
def cosh(x: TorchJet) -> TorchJet:
    value = torch.cosh(x.value)
    grad = torch.sinh(x.value) * x.grad
    return x.create_similar(value, grad)


@implements(torch.tan)
def tan(x: TorchJet) -> TorchJet:
    value = torch.tan(x.value)
    grad = (1.0 / torch.cos(x.value) ** 2) * x.grad
    return x.create_similar(value, grad)


@implements(torch.tanh)
def tanh(x: TorchJet) -> TorchJet:
    value = torch.tanh(x.value)
    grad = (1 - value**2) * x.grad
    return x.create_similar(value, grad)


@implements(torch.abs)
def abs(x: TorchJet) -> TorchJet:
    value = torch.abs(x.value)
    grad = torch.sign(x.value) * x.grad
    return x.create_similar(value, grad)


@implements(torch.minimum)
def minimum(x: TorchJet, y: TorchJet) -> TorchJet:
    value = torch.minimum(x.value, y.value)
    grad = torch.where(x.value < y.value, x.grad, y.grad)
    return x.create_similar(value, grad)


@implements(torch.maximum)
def maximum(x: TorchJet, y: TorchJet) -> TorchJet:
    value = torch.maximum(x.value, y.value)
    grad = torch.where(x.value > y.value, x.grad, y.grad)
    return x.create_similar(value, grad)


@implements(torch.clamp)
def clamp(x: TorchJet, min=None, max=None) -> TorchJet:
    value = torch.clamp(x.value, min=min, max=max)
    grad = x.grad.clone()
    if min is not None:
        grad = torch.where(x.value < min, torch.zeros_like(grad), grad)
    if max is not None:
        grad = torch.where(x.value > max, torch.zeros_like(grad), grad)
    return x.create_similar(value, grad)


@implements(torch.sqrt)
def sqrt(x: TorchJet) -> TorchJet:
    value = torch.sqrt(x.value)
    grad = 0.5 / value * x.grad
    return x.create_similar(value, grad)


@implements(torch.log)
def log(x: TorchJet) -> TorchJet:
    value = torch.log(x.value)
    grad = 1 / x.value * x.grad
    return x.create_similar(value, grad)


@implements(torch.neg)
def neg(x: TorchJet) -> TorchJet:
    value = -x.value
    grad = -x.grad
    return x.create_similar(value, grad)


@implements(torch.add)
def add(x: TorchJet, y: TorchJet | float | int) -> TorchJet:
    return x + y


@implements(torch.sub)
def sub(x: TorchJet, y: TorchJet | float | int) -> TorchJet:
    return x - y


@implements(torch.mul)
def mul(x: TorchJet, y: TorchJet | float | int) -> TorchJet:
    return x * y


@implements(torch.div)
def div(x: TorchJet, y: TorchJet | float | int) -> TorchJet:
    return x / y


@implements(torch.pow)
def pow(x: TorchJet, y: TorchJet | float | int) -> TorchJet:
    return x**y
