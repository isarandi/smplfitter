"""Shared pytest fixtures for smplfitter tests."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Optional

import numpy as np
import pytest


@dataclass
class Backend:
    """Abstraction for testing across different backends."""

    name: str
    module: Any
    to_tensor: Callable[[np.ndarray], Any]
    to_numpy: Callable[[Any], np.ndarray]
    prepare_model: Callable[[Any], Any]
    prepare_fitter: Callable[[Any], Any]
    context: Callable[[], ContextManager]
    num_betas_on_model: bool  # True if num_betas goes on BodyModel, False if on BodyFitter


def _make_pt_backend() -> Backend:
    """Create PyTorch backend."""
    import torch

    import smplfitter.pt as module

    return Backend(
        name='pt',
        module=module,
        to_tensor=lambda x: torch.from_numpy(x).cuda(),
        to_numpy=lambda x: x.cpu().detach().numpy(),
        prepare_model=lambda m: m.cuda(),
        prepare_fitter=lambda f: torch.jit.script(f.cuda()),
        context=torch.inference_mode,
        num_betas_on_model=True,
    )


def _make_nb_backend() -> Backend:
    """Create Numba backend."""
    import smplfitter.nb as module

    return Backend(
        name='nb',
        module=module,
        to_tensor=lambda x: x,
        to_numpy=lambda x: np.asarray(x),
        prepare_model=lambda m: m,
        prepare_fitter=lambda f: f,
        context=nullcontext,
        num_betas_on_model=True,
    )


def _make_np_backend() -> Backend:
    """Create NumPy backend."""
    import smplfitter.np as module

    return Backend(
        name='np',
        module=module,
        to_tensor=lambda x: x,
        to_numpy=lambda x: np.asarray(x),
        prepare_model=lambda m: m,
        prepare_fitter=lambda f: f,
        context=nullcontext,
        num_betas_on_model=True,
    )


def _make_jax_backend() -> Optional[Backend]:
    """Create JAX backend if available."""
    try:
        import jax.numpy as jnp

        import smplfitter.jax as module

        return Backend(
            name='jax',
            module=module,
            to_tensor=lambda x: jnp.array(x),
            to_numpy=lambda x: np.asarray(x),
            prepare_model=lambda m: m,
            prepare_fitter=lambda f: f,
            context=nullcontext,
            num_betas_on_model=True,
        )
    except ImportError:
        return None


def _make_cy_backend() -> Optional[Backend]:
    """Create Cython backend if available."""
    try:
        import smplfitter.cy as module

        return Backend(
            name='cy',
            module=module,
            to_tensor=lambda x: x,
            to_numpy=lambda x: np.asarray(x),
            prepare_model=lambda m: m,
            prepare_fitter=lambda f: f,
            context=nullcontext,
            num_betas_on_model=True,
        )
    except ImportError:
        return None


_BACKEND_FACTORIES = {
    'pt': _make_pt_backend,
    'nb': _make_nb_backend,
    'np': _make_np_backend,
    'jax': _make_jax_backend,
    'cy': _make_cy_backend,
}


def _get_available_backends() -> list[str]:
    """Get list of available backend names."""
    available = []

    # Check PyTorch
    try:
        import torch

        if torch.cuda.is_available():
            available.append('pt')
    except ImportError:
        pass

    # NumPy and Numba are always available (dependencies)
    available.append('nb')
    available.append('np')

    # Check JAX
    try:
        import jax.numpy  # noqa: F401

        available.append('jax')
    except ImportError:
        pass

    # Check Cython
    try:
        import smplfitter.cy  # noqa: F401

        available.append('cy')
    except ImportError:
        pass

    return available


@pytest.fixture(params=_get_available_backends(), ids=lambda x: x)
def backend(request) -> Backend:
    """Fixture providing backend abstraction for cross-backend tests."""
    factory = _BACKEND_FACTORIES[request.param]
    result = factory()
    if result is None:
        pytest.skip(f'{request.param} backend not available')
    return result


@pytest.fixture(params=['pt', 'nb', 'np', 'jax', 'cy'], ids=lambda x: x)
def fitter_backend(request) -> Backend:
    """Fixture for backends that have BodyFitter."""
    if request.param not in _get_available_backends():
        pytest.skip(f'{request.param} backend not available')
    factory = _BACKEND_FACTORIES[request.param]
    result = factory()
    if result is None:
        pytest.skip(f'{request.param} backend not available')
    return result


# Legacy fixtures for backwards compatibility
@pytest.fixture
def random_pose():
    """Generate random pose rotation vectors."""
    return np.random.randn(2, 24 * 3).astype(np.float32) * 0.1


@pytest.fixture
def random_shape():
    """Generate random shape betas."""
    return np.random.randn(2, 10).astype(np.float32)


@pytest.fixture
def random_trans():
    """Generate random translation vectors."""
    return np.random.randn(2, 3).astype(np.float32)
