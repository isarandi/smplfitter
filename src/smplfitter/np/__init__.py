"""NumPy implementation of body models and the model fitter."""

from __future__ import annotations

from .bodymodel import BodyModel
from .bodyfitter import BodyFitter
from .bodyconverter import BodyConverter
from smplfitter.common import _set_module_for_docs

import functools
import os

__all__ = ['BodyModel', 'BodyFitter', 'BodyConverter', 'get_cached_body_model']
_set_module_for_docs(__name__, globals(), __all__)


@functools.lru_cache()
def get_cached_body_model(model_name='smpl', gender='neutral', model_root=None):
    return get_body_model(model_name, gender, model_root)


def get_body_model(model_name, gender, model_root=None):
    if model_root is None:
        DATA_ROOT = os.getenv('DATA_ROOT', default='.')
        model_root = f'{DATA_ROOT}/body_models/{model_name}'
    return BodyModel(model_root=model_root, gender=gender, model_name=model_name)
