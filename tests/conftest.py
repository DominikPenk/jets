import pytest

from jets.numpy.backend import NumpyBackend

try:
    from jets.torch.backend import TorchBackend

    WITH_TORCH = True
except ImportError:
    print("pytorch not supported")
    WITH_TORCH = False


BACKENDS = [NumpyBackend()]
if WITH_TORCH:
    BACKENDS.append(TorchBackend())


@pytest.fixture(params=BACKENDS, scope="function")
def backend(request):
    yield request.param
