"""JAX implementation of body models."""

from __future__ import annotations

import functools

from .bodymodel import BodyModel
from .bodyfitter import BodyFitter
from .bodyconverter import BodyConverter
from . import rotation
from smplfitter.common import _set_module_for_docs

__all__ = ['BodyModel', 'BodyFitter', 'BodyConverter', 'get_cached_body_model', 'rotation']
_set_module_for_docs(__name__, globals(), __all__)


@functools.lru_cache()
def get_cached_body_model(model_name='smpl', gender='neutral', model_root=None):
    return get_body_model(model_name, gender, model_root)


def get_body_model(model_name, gender, model_root=None):
    return BodyModel(model_root=model_root, gender=gender, model_name=model_name)
