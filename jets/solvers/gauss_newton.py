from typing import Callable, Optional

from ..backend import Backend, get_backend
from ..typing import ArrayLike
from .utils import create_jets, validate_input_shapes


def gauss_newton(
    fn: Callable,
    x: ArrayLike,
    y_true: ArrayLike,
    initial_guesses: ArrayLike,
    iterations: int,
    weights: Optional[ArrayLike] = None,
    backend: Optional[str | Backend] = None,
    **backend_params,
):
    backend = (
        backend
        if isinstance(backend, Backend)
        else get_backend(backend, **backend_params)
    )

    x = backend.to_array(x)
    y_true = backend.to_array(y_true)
    params = backend.to_array(initial_guesses)

    validate_input_shapes(x, y_true, params)

    for i in range(iterations):
        y_pred = fn(x, *create_jets(params, backend))
        r = (y_pred - y_true) * weights

        JTJ = backend.einsum("j...i,k...i->...jk", r.grad, r.grad)
        JTb = backend.einsum("j...i,...i->...j", r.grad, r.value)

        JTJ_inv = backend.matrix_inverse(JTJ)

        delta = -backend.einsum("...jk,...k->...j", JTJ_inv, JTb)

        params = params + delta
        mse = backend.mean(r.value**2, axis=-1)
        print(f"Iteration {i + 1}: MSE = {backend.mean(mse)}")

    return params
