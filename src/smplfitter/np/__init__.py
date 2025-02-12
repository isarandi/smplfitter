"""NumPy implementation of body models and the model fitter."""

from smplfitter.np.bodymodel import BodyModel
from smplfitter.np.bodyfitter import BodyFitter

import functools
import os

__all__ = ['BodyModel', 'BodyFitter', 'get_cached_body_model']

@functools.lru_cache()
def get_cached_body_model(model_name='smpl', gender='neutral', model_root=None):
    return get_body_model(model_name, gender, model_root)


def get_body_model(model_name, gender, model_root=None):
    if model_root is None:
        DATA_ROOT = os.getenv('DATA_ROOT', default='.')
        model_root = f'{DATA_ROOT}/body_models/{model_name}'
    return BodyModel(model_root=model_root, gender=gender, model_name=model_name)
