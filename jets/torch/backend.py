import torch

from ..backend import Backend
from .jet import TorchJet


class TorchBackend(Backend):
    def __init__(self, dtype=None, device=None):
        self.dtype = dtype or torch.get_default_dtype()
        self.device = device or torch.get_default_device()

    JetType = TorchJet

    def unstack(self, x: torch.Tensor, axis: int) -> tuple[torch.Tensor, ...]:
        return torch.unbind(x, dim=axis)

    def create_jet(self, value, grad) -> TorchJet:
        return TorchJet(value, grad, device=self.device, dtype=self.dtype)

    def to_array(self, x):
        return torch.as_tensor(x, dtype=self.dtype, device=self.device)

    def to_dtype(
        self, x: torch.Tensor, dtype: torch.dtype | type[bool]
    ) -> torch.Tensor:
        return x.bool() if dtype is bool else x.to(dtype=dtype)

    def to_numpy(self, x: torch.Tensor):
        return x.detach().cpu().numpy()

    def expand_dims(self, x: torch.Tensor, axis: int | tuple[int, ...]) -> torch.Tensor:
        if isinstance(axis, int):
            axis = (axis,)

        # Sort axes so insertion does not conflict
        for i, axis in enumerate(axis):
            x = x.unsqueeze(axis)

        return x

    repeat = staticmethod(torch.repeat_interleave)  # type: ignore
    where = staticmethod(torch.where)  # type: ignore
    einsum = staticmethod(torch.einsum)  # type: ignore
    matrix_inverse = staticmethod(torch.linalg.inv)
    moveaxis = staticmethod(torch.moveaxis)  # type: ignore

    def eye(self, n: int, dtype=None) -> torch.Tensor:
        dtype = dtype or self.dtype
        return torch.eye(n, dtype=self.dtype, device=self.device)

    @staticmethod
    def mean(
        x: torch.Tensor, axis: int | tuple[int, ...] | None = None
    ) -> torch.Tensor:
        return torch.mean(x, dim=axis)
