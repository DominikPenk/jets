import numpy as np
import pytest

import tests.derivatives as derivatives
from jets.numpy.jet import NumpyJet
from tests.jet_contract import JetContract


class TestNumpyJet(JetContract):
    @pytest.fixture
    def jet(self):
        return NumpyJet

    @pytest.mark.parametrize("dtype", (None, np.int32, np.float64))
    def test_createas(self, jet: type[NumpyJet], dtype):
        x = jet(2.0, 1.0, dtype=dtype)
        y = x.createas(5.0, 3.0)

        self.assert_jet(y, 5.0, 3.0)
        assert x.dtype == y.dtype
        assert type(x) is type(y)

    @pytest.mark.parametrize(
        ("func", "data"),
        [
            (np.sin, derivatives.SINUS),
            (np.cos, derivatives.COSINUS),
            (np.exp, derivatives.EXP),
            (np.log, derivatives.LOG),
        ],
        ids=["sinus", "cosinus", "exp", "log"],
    )
    def test_scalar_unary_numpy_functions(
        self, func, data: list[tuple[float, float, float]]
    ):
        tol = 1e-9
        for x, f_true, df_true in data:
            jx = NumpyJet(x, 1.0)
            jy = func(jx)

            assert np.allclose(jy.value, f_true, atol=tol), f"Value mismatch at x={x}"
            assert np.allclose(jy.grad, df_true, atol=tol), f"Graient mismatch at x={x}"

    @pytest.mark.parametrize(
        ("func", "data"),
        [
            (np.sin, derivatives.SINUS),
            (np.cos, derivatives.COSINUS),
            (np.exp, derivatives.EXP),
            (np.log, derivatives.LOG),
        ],
        ids=["sinus", "cosinus", "exp", "log"],
    )
    def test_vectorized_unary_numpy_functions(
        self, func, data: list[tuple[float, float, float]]
    ):
        tol = 1e-9
        xs, ys, dys = zip(*data)

        j = NumpyJet(xs, np.ones(len(xs)))
        jy: NumpyJet = func(j)

        assert np.allclose(jy.value, ys, atol=tol), "Value mismatch"
        assert np.allclose(jy.grad, dys, atol=tol), "Gradient mismatch"
