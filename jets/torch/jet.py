import math
from typing import Self

import torch
from torch._prims_common import DeviceLikeType

from ..jet import JetBase
from ..numpy.jet import NumpyJet


class TorchJet(JetBase):
    def __init__(self, value, grad, device=None, dtype: DeviceLikeType | None = None):
        self.value = torch.as_tensor(value, device=device, dtype=dtype)
        self.grad = torch.as_tensor(grad, device=device, dtype=dtype)

    @classmethod
    def _log(cls, x):
        return torch.log(x) if isinstance(x, torch.Tensor) else math.log(x)

    def createas(self, value, grad) -> Self:
        return self.__class__(value, grad, dtype=self.dtype, device=self.device)

    @classmethod
    def __torch_function__(cls, func, types, args, kwargs=None):
        from .functions import HANDLE_TORCH_FUNCTIONS

        if kwargs is None:
            kwargs = {}
        func = HANDLE_TORCH_FUNCTIONS.get(func, None)
        if func is None:
            return NotImplemented
        return func(*args, **kwargs)

    def to(self, device=None, dtype=None) -> Self:
        new_device = device if device is not None else self.device
        new_dtype = dtype if dtype is not None else self.dtype
        self.value = self.value.to(device=new_device, dtype=new_dtype)
        self.grad = self.grad.to(device=new_device, dtype=new_dtype)
        return self

    def cpu(self) -> Self:
        return self.to(device="cpu")

    def cuda(self) -> Self:
        return self.to(device="cuda:0")

    def numpy(self) -> NumpyJet:
        return NumpyJet(
            self.value.detach().cpu().numpy(), self.grad.detach().cpu().numpy()
        )

    @property
    def device(self) -> torch.device:
        return self.grad.device

    @property
    def dtype(self):
        return self.grad.dtype
