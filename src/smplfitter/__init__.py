"""SMPLFitter provides forward and inverse kinematics for SMPL-family body models.

Main submodules:
- :mod:`smplfitter.np` - NumPy backend
- :mod:`smplfitter.pt` - PyTorch backend
- :mod:`smplfitter.tf` - TensorFlow backend
- :mod:`smplfitter.jax` - JAX backend
- :mod:`smplfitter.nb` - Numba backend
"""

from __future__ import annotations

from .common import ModelData, initialize, _set_module_for_docs

try:
    from ._version import version as __version__
except ImportError:
    __version__ = '0.0.0'

__all__ = ['ModelData', 'initialize', '__version__']
_set_module_for_docs(__name__, globals(), __all__)
