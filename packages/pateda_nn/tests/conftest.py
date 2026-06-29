"""Shared pytest fixtures and configuration for the pateda_nn test suite."""
import numpy as np
import pytest

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover - torch should always be installed
    torch = None
    _HAS_CUDA = False


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: mark a test as requiring a CUDA-capable GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests marked ``gpu`` when no CUDA device is available."""
    if _HAS_CUDA:
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA GPU available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def has_cuda():
    """Whether a CUDA GPU is available in this environment."""
    return _HAS_CUDA


@pytest.fixture
def device(has_cuda):
    """The torch device tests should run on (cuda if present, else cpu)."""
    return "cuda" if has_cuda else "cpu"


@pytest.fixture
def rng():
    """A seeded numpy Generator for reproducible test data."""
    return np.random.default_rng(12345)
