from __future__ import annotations

from typing import Self, TypeAlias

from .typing import Array, ArrayLike, Scalar, Value

Gradient: TypeAlias = Array


class JetBase:
    value: Value
    grad: Gradient

    def __init__(self, value, grad, *args, **kwargs):
        pass

    @classmethod
    def _log(cls, x: Value) -> Value:
        raise NotImplementedError("Log function not implemented for base class.")

    def create_similar(self, value: Value, grad: Gradient | ArrayLike | Scalar) -> Self:
        raise NotImplementedError("create as method must be implemented in subclasses")

    def unwrap_value(self, x: JetBase | Value) -> Value:
        return x.value if isinstance(x, JetBase) else x

    def unwrap_grad(self, x: JetBase | Value) -> Gradient | Value:
        return x.grad if isinstance(x, JetBase) else 0.0  # type: ignore

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(value={self.value}, grad={self.grad})"

    def __repr__(self):
        return f"{self.__class__.__name__}(value={self.value!r}, grad={self.grad!r})"

    def __add__(self, other: JetBase | Value) -> Self:
        value = self.value + self.unwrap_value(other)
        grad = self.grad + self.unwrap_grad(other)
        return self.create_similar(value, grad)

    def __radd__(self, other: Value) -> Self:
        return self.create_similar(self.value + other, self.grad)

    def __sub__(self, other: JetBase | Value) -> Self:
        value = self.value - self.unwrap_value(other)
        grad = self.grad - self.unwrap_grad(other)
        return self.create_similar(value, grad)

    def __rsub__(self, other: Value) -> Self:
        return self.create_similar(other - self.value, -self.grad)

    def __mul__(self, other: JetBase | Value) -> Self:
        value = self.value * self.unwrap_value(other)
        grad = self.grad * self.unwrap_value(other) + self.value * self.unwrap_grad(
            other
        )
        return self.create_similar(value, grad)

    def __rmul__(self, other: Value) -> Self:
        value = self.value * other
        grad = self.grad * other
        return self.create_similar(value, grad)

    def __truediv__(self, other: JetBase | Value) -> Self:
        oval = self.unwrap_value(other)
        value = self.value / oval
        grad = (self.grad * oval - self.value * self.unwrap_grad(other)) / (oval * oval)
        return self.create_similar(value, grad)

    def __rtruediv__(self, other: Value) -> Self:
        value = other / self.value
        grad = -other * self.grad / (self.value * self.value)
        return self.create_similar(value, grad)

    def __neg__(self) -> Self:
        return self.create_similar(-self.value, -self.grad)

    def __pow__(self, other: JetBase | Value) -> Self:
        a = self.value
        ax = self.grad

        if isinstance(other, JetBase):
            b = other.value
            ay = other.grad
            value = a**b
            grad = value * (ay * self._log(a) + (b * ax) / a)
            return self.create_similar(value, grad)

        b = other
        value = a**b
        grad = b * (a ** (b - 1)) * ax
        return self.create_similar(value, grad)

    def __rpow__(self, other: Value) -> Self:
        b = self.value
        ay = self.grad
        value = other**b

        grad = value * ay * self._log(other)
        return self.create_similar(value, grad)
