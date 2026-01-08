from ..backend import Backend
from ..jet import JetBase
from ..typing import Array, ArrayLike


def create_jets(values: ArrayLike, backend: Backend) -> list[JetBase]:
    values = backend.to_array(values)
    *batch_dims, num_params = values.shape
    n_batch_dims = len(batch_dims)

    jets = []
    grads = backend.eye(num_params)

    for initial_value, grad in zip(backend.unstack(values, axis=-1), grads):
        if n_batch_dims:
            initial_value = backend.expand_dims(initial_value, axis=n_batch_dims)
        grad = backend.expand_dims(grad, axis=list(range(1, n_batch_dims + 2)))
        jets.append(backend.create_jet(initial_value, grad))

    return jets


def validate_input_shapes(x: Array, y: Array, params: Array):
    *x_batch_dims, num_x = x.shape
    *y_batch_dims, num_y = y.shape
    *params_batch_dims, num_params = params.shape

    assert (x_batch_dims == y_batch_dims) & (x_batch_dims == params_batch_dims), (
        "Batch dimensions of x, y, and params must match."
    )
    assert num_x == num_y, "x and y must have the same number of data points."
    return num_params, x_batch_dims
