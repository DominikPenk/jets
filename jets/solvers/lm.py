from typing import Callable, Optional

from ..backend import Backend, get_backend
from ..typing import ArrayLike
from .utils import create_jets, validate_input_shapes


def levenberg_marquardt(
    fn: Callable,
    x: ArrayLike,
    y_true: ArrayLike,
    initial_guesses: ArrayLike,
    iterations: int,
    lambda_init: float = 1e-3,
    lambda_factor: float = 10.0,
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

    num_params, batch_shapes = validate_input_shapes(x, y_true, params)
    num_batch_dims = len(batch_shapes)

    lambda_ = lambda_init * backend.eye(num_params)
    lambda_ = backend.expand_dims(lambda_, axis=list(range(num_batch_dims)))

    for i in range(iterations):
        y_pred = fn(x, *create_jets(params, backend))
        r = y_pred - y_true

        JTJ = backend.einsum("j...i,k...i->...jk", r.grad, r.grad)
        JTb = backend.einsum("j...i,...i->...j", r.grad, r.value)

        JTJ_lm = JTJ + lambda_
        JTJ_lm_inv = backend.matrix_inverse(JTJ_lm)

        delta = -backend.einsum("...jk,...k->...j", JTJ_lm_inv, JTb)

        new_params = params + delta
        new_y_pred = fn(x, *create_jets(new_params, backend))
        new_r = new_y_pred - y_true

        current_mse = backend.mean(r.value**2, axis=-1)
        new_mse = backend.mean(new_r.value**2, axis=-1)

        mask = new_mse < current_mse
        params = backend.where(mask[..., None], new_params, params)
        lambda_ = backend.where(
            mask[..., None, None],
            lambda_ / lambda_factor,
            lambda_ * lambda_factor,
        )

    return params
