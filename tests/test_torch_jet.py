import pytest
import torch

import tests.derivatives as derivatives
from jets.numpy.jet import NumpyJet
from jets.torch.jet import TorchJet
from tests.jet_contract import JetContract


class TestTorchJet(JetContract):
    @pytest.fixture
    def jet(self):
        return TorchJet

    @pytest.mark.parametrize("dtype", (None, torch.int32, torch.float64))
    def test_createas(self, jet: type[TorchJet], dtype):
        x = jet(2.0, 1.0, dtype=dtype)
        y = x.createas(5.0, 3.0)

        self.assert_jet(y, 5.0, 3.0)
        assert x.dtype == y.dtype
        assert x.device == y.device
        assert type(x) is type(y)

    @pytest.mark.parametrize(
        ("func", "data"),
        [
            (torch.sin, derivatives.SINUS),
            (torch.sinh, derivatives.SINH),
            (torch.cos, derivatives.COSINUS),
            (torch.cosh, derivatives.COSH),
            (torch.tan, derivatives.TANGENS),
            (torch.tanh, derivatives.TANH),
            (torch.abs, derivatives.ABS),
            (torch.exp, derivatives.EXP),
            (torch.log, derivatives.LOG),
            (torch.sqrt, derivatives.SQRT),
        ],
    )
    def test_scalar_unary_numpy_functions(
        self, func, data: list[tuple[float, float, float]]
    ):
        tol = 1e-6
        for x, f_true, df_true in data:
            jx = TorchJet(x, 1.0)
            jy = func(jx)

            assert torch.allclose(jy.value, torch.as_tensor(f_true), atol=tol), (
                f"Value mismatch at x={x}"
            )
            assert torch.allclose(jy.grad, torch.as_tensor(df_true), atol=tol), (
                f"Graient mismatch at x={x}"
            )

    @pytest.mark.parametrize(
        ("func", "data"),
        [
            (torch.sin, derivatives.SINUS),
            (torch.sinh, derivatives.SINH),
            (torch.cos, derivatives.COSINUS),
            (torch.cosh, derivatives.COSH),
            (torch.tan, derivatives.TANGENS),
            (torch.tanh, derivatives.TANH),
            (torch.abs, derivatives.ABS),
            (torch.exp, derivatives.EXP),
            (torch.log, derivatives.LOG),
            (torch.sqrt, derivatives.SQRT),
        ],
    )
    def test_vectorized_unary_numpy_functions(
        self, func, data: list[tuple[float, float, float]]
    ):
        tol = 1e-6
        xs, ys, dys = zip(*data)

        j = TorchJet(xs, torch.ones(len(xs)))
        jy: TorchJet = func(j)

        assert torch.allclose(jy.value, torch.as_tensor(ys), atol=tol), "Value mismatch"
        assert torch.allclose(jy.grad, torch.as_tensor(dys), atol=tol), (
            "Gradient mismatch"
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_numpy_conversion(self, device):
        if not torch.cuda.is_available() and device == "cuda":
            pytest.skip("Cuda is not available")
        x_torch = TorchJet(1.0, 1.0, device=device)
        x_numpy = x_torch.numpy()
        assert isinstance(x_numpy, NumpyJet)

    @pytest.mark.parametrize(
        ("src", "dst"),
        [("cpu", "cpu"), ("cpu", "cuda:0"), ("cuda:0", "cpu"), ("cuda:0", "cuda:0")],
    )
    def test_move_to_device(self, src, dst):
        if not torch.cuda.is_available():
            pytest.skip("Cuda is not available")
        x_src = TorchJet([1.0, 0.0], [1.0, 1.0], device=src)
        x_dst = x_src.to(dst)
        assert x_src is x_dst
        assert x_dst.device == torch.device(dst)

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_numpy_conversion(self, device):
        if not torch.cuda.is_available() and device == "cuda":
            pytest.skip("Cuda is not available")
        x_torch = TorchJet(1.0, 1.0, device=device)
        x_numpy = x_torch.numpy()
        assert isinstance(x_numpy, NumpyJet)

    @pytest.mark.parametrize(
        ("orig", "target"),
        [
            (torch.float32, torch.int32),
            (torch.int32, torch.float64),
            (torch.int32, torch.int32),
        ],
    )
    def test_change_dtype(self, orig, target):
        x_src = TorchJet([1.0, 0.0], [1.0, 1.0], dtype=orig)
        x_dst = x_src.to(dtype=target)
        assert x_src is x_dst
        assert x_dst.dtype == target

    def test_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("Cuda is not available")
        x_cpu = TorchJet(1.0, 1.0, device="cpu")
        x_cuda = x_cpu.cuda()
        assert x_cpu is x_cuda
        assert x_cuda.device == torch.device("cuda:0")

    def test_cpu(self):
        if not torch.cuda.is_available():
            pytest.skip("Cuda is not available")
        x_cuda = TorchJet(1.0, 1.0, device="cuda:0")
        x_cpu = x_cuda.cpu()
        assert x_cpu is x_cuda
        assert x_cuda.device == torch.device("cpu")
