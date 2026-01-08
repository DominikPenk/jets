from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pytest

from jets.backend import Backend


def _get_shape(x: Any) -> tuple[int, ...]:
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
        try:
            first = next(iter(x))
        except StopIteration:
            return (0,)
        return (len(x),) + _get_shape(first)
    return ()


# ---------- where ----------


def test_where(backend: Backend):
    mask = backend.to_array([True, True, False])
    mask = backend.to_dtype(mask, bool)
    result = backend.where(mask, 1, -1)
    assert result == pytest.approx([1, 1, -1])


# ---------- expand_dims ----------


@pytest.mark.parametrize(
    "input_data, axis, expected_shape",
    [
        # 1D
        ([1, 2, 3], 0, (1, 3)),
        ([1, 2, 3], 1, (3, 1)),
        ([1, 2, 3], -1, (3, 1)),
        # 2D
        ([[1, 2], [3, 4]], 0, (1, 2, 2)),
        ([[1, 2], [3, 4]], 1, (2, 1, 2)),
        ([[1, 2], [3, 4]], 2, (2, 2, 1)),
        ([[1, 2], [3, 4]], -1, (2, 2, 1)),
        # rectangular
        ([[0, 0, 0]], 0, (1, 1, 3)),
        ([[0, 0, 0]], 1, (1, 1, 3)),
        ([[0, 0, 0]], 2, (1, 3, 1)),
        # multiple axes
        ([[1, 2], [3, 4]], [0, 2], (1, 2, 1, 2)),
        ([[1, 2], [3, 4]], [1, 3], (2, 1, 2, 1)),
        ([[1, 2], [3, 4]], [-1, 0], (1, 2, 2, 1)),
    ],
)
def test_expand_dims(backend, input_data, axis, expected_shape):
    arr = backend.to_array(input_data)
    res = backend.expand_dims(arr, axis)

    assert res.shape == expected_shape


@pytest.mark.parametrize("axis", [3, -4, [0, 5]])
def test_expand_dims_invalid_axis(backend, axis):
    arr = backend.to_array([[1, 2]])
    with pytest.raises(Exception):
        backend.expand_dims(arr, axis)


# ---------- moveaxis ----------


@dataclass(frozen=True)
class MoveAxisCase:
    input: Any
    source: int | Sequence[int]
    destination: int | Sequence[int]
    description: str


MOVEAXIS_CASES = [
    MoveAxisCase(
        description="2D: move rows to columns",
        input=[[1, 2], [3, 4]],
        source=0,
        destination=1,
    ),
    MoveAxisCase(
        description="2D: move columns to rows",
        input=[[1, 2], [3, 4]],
        source=1,
        destination=0,
    ),
    MoveAxisCase(
        description="3D: move outer axis to innermost",
        input=[
            [[1], [2], [3]],
            [[4], [5], [6]],
        ],
        source=0,
        destination=2,
    ),
    MoveAxisCase(
        description="negative source axis",
        input=[
            [[1], [2]],
            [[3], [4]],
        ],
        source=-1,
        destination=0,
    ),
    MoveAxisCase(
        description="multiple axes reordered",
        input=[[[[1], [2]]]],
        source=[0, 2],
        destination=[2, 0],
    ),
]


@pytest.mark.parametrize(
    "case",
    MOVEAXIS_CASES,
    ids=lambda c: c.description,
)
def test_moveaxis_contract(backend: Backend, case: MoveAxisCase):
    arr = backend.to_array(case.input)
    result = backend.moveaxis(arr, case.source, case.destination)

    expected = np.moveaxis(np.asarray(case.input), case.source, case.destination)
    result = backend.to_numpy(result)

    # shape contract
    assert result.shape == expected.shape

    # content contract
    assert np.allclose(result, expected)


# ---------- repeat ----------


@dataclass(frozen=True)
class RepeatCase:
    description: str
    input: Any
    repeats: tuple[int, ...] | int
    axis: Optional[int | tuple[int, ...]] = None


REPEAT_CASES = [
    RepeatCase("repeat rows", [[1, 2]], 2, 0),
    RepeatCase(
        "repeat columns",
        [[1, 2]],
        2,
        1,
    ),
    RepeatCase("repeat second", [[[1, 2], [3, 4]]], 3, 1),
]


@pytest.mark.parametrize("case", REPEAT_CASES, ids=lambda c: c.description)
def test_repeat(backend: Backend, case: RepeatCase):
    arr = backend.to_array(case.input)
    result = backend.repeat(arr, case.repeats, case.axis)

    expected = np.asarray(case.input)
    expected = np.repeat(expected, case.repeats, case.axis)

    result = backend.to_numpy(result)

    assert result.shape == expected.shape
    assert np.allclose(result, expected)


# ---------- einsum ----------


@dataclass(frozen=True)
class EinsumCase:
    description: str
    equation: str
    operands: list


EINSUM_CASES = [
    # 1. Matrix-matrix multiplication (2x2 @ 2x2)
    EinsumCase(
        description="matrix-matrix multiplication",
        equation="ij,jk->ik",
        operands=[np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])],
    ),
    # 2. Batched matrix-vector multiplication (batch 2, matrix 2x2, vector 2)
    EinsumCase(
        description="batched matrix-vector multiplication",
        equation="bij,bj->bi",
        operands=[
            np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # shape (2,2,2)
            np.array([[1, 1], [2, 0]]),  # shape (2,2)
        ],
    ),
    # 3. Batched dot product (batch 2, vector 2) -> scalar per batch
    EinsumCase(
        description="batched dot product",
        equation="bi,bi->b",
        operands=[
            np.array([[1, 2], [3, 4]]),  # shape (2,2)
            np.array([[5, 6], [7, 8]]),  # shape (2,2)
        ],
    ),
]


@pytest.mark.parametrize("case", EINSUM_CASES, ids=lambda c: c.description)
def test_einsum(backend, case: EinsumCase):
    # Convert operands to backend arrays
    operands = [backend.to_array(op) for op in case.operands]

    # Compute backend result
    result = backend.einsum(case.equation, *operands)

    # Reference NumPy result
    expected = np.einsum(case.equation, *[np.asarray(op) for op in case.operands])

    # Normalize backend output
    result = backend.to_numpy(result)

    # Shape contract
    assert result.shape == expected.shape

    # Content contract
    assert np.allclose(result, expected)


# ---------- mean ----------


@dataclass(frozen=True)
class MeanCase:
    description: str
    input: Any
    axis: int | tuple[int, ...] | None


MEAN_CASES = [
    MeanCase("full mean", [[1, 2], [3, 4]], None),
    MeanCase("mean axis 0", [[1, 2], [3, 4]], 0),
    MeanCase("mean axis -1", [[1, 2], [3, 4]], 1),
]


@pytest.mark.parametrize("case", MEAN_CASES, ids=lambda c: c.description)
def test_mean_contract(backend, case: MeanCase):
    arr = backend.to_array(case.input)
    result = backend.mean(arr, axis=case.axis)

    expected = np.mean(np.asarray(case.input), axis=case.axis)
    result = backend.to_numpy(result)

    assert result.shape == expected.shape
    assert np.allclose(result, expected)


# ---------- matrix_inverse ----------


@dataclass(frozen=True)
class MatrixInverseCase:
    description: str
    input: Any  # the square matrix or batch of matrices


MATRIX_INVERSE_CASES = [
    # 2x2 invertible matrix
    MatrixInverseCase(description="2x2 matrix", input=np.array([[1, 2], [3, 4]])),
    # 3x3 invertible matrix
    MatrixInverseCase(
        description="3x3 matrix", input=np.array([[2, 0, 1], [1, 1, 0], [3, 2, 1]])
    ),
    # batched 2x2 matrices (batch of 2)
    MatrixInverseCase(
        description="batched 2x2 matrices",
        input=np.array([[[1, 2], [3, 4]], [[2, 1], [1, 2]]]),
    ),
]


@pytest.mark.parametrize("case", MATRIX_INVERSE_CASES, ids=lambda c: c.description)
def test_matrix_inverse(backend, case: MatrixInverseCase):
    arr = backend.to_array(case.input)

    # Backend result
    result = backend.matrix_inverse(arr)

    # Reference NumPy result
    expected = np.linalg.inv(np.asarray(case.input))

    # Normalize backend output
    result = backend.to_numpy(result)

    # Shape contract
    assert result.shape == expected.shape

    # Content contract (allowing float tolerance)
    assert np.allclose(result, expected, rtol=1e-6, atol=1e-6)


# ---------- matrix_inverse ----------


@dataclass(frozen=True)
class EyeCase:
    description: str
    n: int  # size of the square matrix
    dtype: Any = None  # optional dtype


EYE_CASES = [
    EyeCase(description="2x2 identity", n=2),
    EyeCase(description="3x3 identity", n=3),
    EyeCase(description="5x5 identity with float32", n=5, dtype=np.float32),
]


@pytest.mark.parametrize("case", EYE_CASES, ids=lambda c: c.description)
def test_eye(backend, case: EyeCase):
    # Backend result
    result = backend.eye(case.n, dtype=case.dtype)

    # Reference NumPy result
    expected = np.eye(case.n, dtype=case.dtype)

    # Normalize backend output
    result = backend.to_numpy(result)

    # Shape contract
    assert result.shape == expected.shape

    # Content contract
    assert np.allclose(result, expected)


# ---------- matrix_inverse ----------


@dataclass(frozen=True)
class UnstackCase:
    description: str
    input: Any  # input array
    axis: int  # axis to unstack along


UNSTACK_CASES = [
    UnstackCase(
        "1D unstack",
        [1, 2, 3],
        0,
    ),
    UnstackCase("2D unstack axis 0", [[1, 2], [3, 4]], 0),
    UnstackCase("2D unstack axis 1", [[1, 2], [3, 4]], 1),
    UnstackCase(
        "3D unstack last axis",
        np.arange(2 * 2 * 3).reshape(2, 2, 3).tolist(),
        2,
    ),
    UnstackCase(
        "3D unstack negative",
        np.arange(2 * 2 * 3).reshape(2, 2, 3).tolist(),
        -2,
    ),
]


@pytest.mark.parametrize("case", UNSTACK_CASES, ids=lambda c: c.description)
def test_unstack_contract(backend: Backend, case: UnstackCase):
    arr = backend.to_array(case.input)

    # Backend result
    result = backend.unstack(arr, case.axis)  # tuple of arrays

    # Reference NumPy result
    x = np.asarray(case.input)

    for i, slice in enumerate(result):
        slice = backend.to_numpy(slice)
        expected_slice = np.take(x, i, axis=case.axis)

        assert slice.shape == expected_slice.shape
        assert np.allclose(slice, expected_slice)
