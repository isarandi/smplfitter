"""SMPLFitter contains three main submodules: :mod:`smplfitter.np`, :mod:`smplfitter.pt`,
:mod:`smplfitter.tf`, which provide the forward and inverse functions for SMPL-like body models
implemented in NumPy, PyTorch, and TensorFlow, respectively.
"""

VERSION = (0, 2, 0)  # PEP 386
__version__ = ".".join([str(x) for x in VERSION])
