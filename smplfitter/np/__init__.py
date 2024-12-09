from smplfitter.np.bodymodel import SMPLBodyModel
from smplfitter.np.fitter import SMPLFitter
import functools
import os


@functools.lru_cache()
def get_cached_body_model(model_name='smpl', gender='neutral', model_root=None):
    return get_body_model(model_name, gender, model_root)


def get_body_model(model_name, gender, model_root=None):
    if model_root is None:
        model_root = f'{os.environ["DATA_ROOT"]}/body_models/{model_name}'
    return SMPLBodyModel(model_root=model_root, gender=gender, model_name=model_name)
