import contextlib
import os
import os.path as osp
import pickle
import sys
import warnings

import numpy as np


def initialize(model_name, gender, model_root=None, num_betas=None):
    if model_root is None:
        DATA_ROOT = os.getenv('DATA_ROOT', default='.')
        model_root = f'{DATA_ROOT}/body_models/{model_name}'

    with monkey_patched_for_chumpy():
        if model_name == 'smpl':
            gender_str = dict(f='f', m='m', n='neutral')[gender[0]]
            filename = f'basicmodel_{gender_str}_lbs_10_207_0_v1.1.0.pkl'
            with open(osp.join(model_root, filename), 'rb') as f:
                smpl_data = pickle.load(f, encoding='latin1')
        elif model_name in ('smplx', 'smplxlh', 'smplxmoyo'):
            gender_str = dict(f='FEMALE', m='MALE', n='NEUTRAL')[gender[0]]
            smpl_data = np.load(osp.join(model_root, f'SMPLX_{gender_str}.npz'))
        elif model_name == 'smplh':
            gender_str = dict(f='female', m='male')[gender[0]]
            filename = f'SMPLH_{gender_str}.pkl'
            with open(osp.join(model_root, filename), 'rb') as f:
                smpl_data = pickle.load(f, encoding='latin1')
        elif model_name == 'smplh16':
            gender_str = dict(f='female', m='male', n='neutral')[gender[0]]
            smpl_data = np.load(osp.join(model_root, gender_str, 'model.npz'))
        else:
            raise ValueError(f'Unknown model name: {model_name}')

    res = {}
    res['shapedirs'] = np.array(smpl_data['shapedirs'], dtype=np.float64)
    res['posedirs'] = np.array(smpl_data['posedirs'], dtype=np.float64)
    res['v_template'] = np.array(smpl_data['v_template'], dtype=np.float64)

    if not isinstance(smpl_data['J_regressor'], np.ndarray):
        res['J_regressor'] = np.array(smpl_data['J_regressor'].toarray(), dtype=np.float64)
    else:
        res['J_regressor'] = smpl_data['J_regressor'].astype(np.float64)

    res['weights'] = np.array(smpl_data['weights'])
    res['faces'] = np.array(smpl_data['f'].astype(np.int32))
    res['kintree_parents'] = smpl_data['kintree_table'][0].tolist()
    res['num_joints'] = len(res['kintree_parents'])
    res['num_vertices'] = len(res['v_template'])

    # Kid model has an additional shape parameter which pulls the mesh towards the SMIL mean
    # template
    v_template_smil = np.load(os.path.join(model_root, 'kid_template.npy')).astype(np.float64)
    res['kid_shapedir'] = v_template_smil - np.mean(v_template_smil, axis=0) - res['v_template']
    res['kid_J_shapedir'] = res['J_regressor'] @ res['kid_shapedir']

    if 'J_shapedirs' in smpl_data:
        res['J_shapedirs'] = np.array(smpl_data['J_shapedirs'], dtype=np.float64)
    else:
        res['J_shapedirs'] = np.einsum('jv,vcs->jcs', res['J_regressor'], res['shapedirs'])

    if 'J_template' in smpl_data:
        res['J_template'] = np.array(smpl_data['J_template'], dtype=np.float64)
    else:
        res['J_template'] = res['J_regressor'] @ res['v_template']

    res['v_template'] = res['v_template'] - np.einsum(
        'vcx,x->vc',
        res['posedirs'],
        np.reshape(np.tile(np.eye(3, dtype=np.float64), [res['num_joints'] - 1, 1]), [-1]),
    )

    tensors = {
        'v_template': res['v_template'],
        'shapedirs': res['shapedirs'][:, :, :num_betas],
        'posedirs': res['posedirs'],
        'J_regressor': res['J_regressor'],
        'J_template': res['J_template'],
        'J_shapedirs': res['J_shapedirs'][:, :, :num_betas],
        'kid_shapedir': res['kid_shapedir'],
        'kid_J_shapedir': res['kid_J_shapedir'],
        'weights': res['weights'],
    }

    nontensors = {
        'kintree_parents': res['kintree_parents'],
        'faces': res['faces'],
        'num_joints': res['num_joints'],
        'num_vertices': res['num_vertices'],
    }

    return tensors, nontensors


@contextlib.contextmanager
def monkey_patched_for_chumpy():
    """The pickle file of SMPLH imports chumpy and it tries to import np.bool etc which are
    not available anymore.
    """
    added = []
    for name in ['bool', 'int', 'object', 'str']:
        if name not in dir(np):
            try:
                sys.modules[f'numpy.{name}'] = getattr(np, name + '_')
                added.append(name)
            except:
                pass

    sys.modules[f'numpy.float'] = float
    sys.modules[f'numpy.complex'] = np.complex128
    sys.modules[f'numpy.NINF'] = -np.inf
    np.NINF = -np.inf
    np.complex = np.complex128
    np.float = float

    if 'unicode' not in dir(np):
        sys.modules['numpy.unicode'] = np.str_
        added.append('unicode')

    import inspect

    added_getargspec = False
    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec
        added_getargspec = True

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        yield

    for name in added:
        del sys.modules[f'numpy.{name}']

    if added_getargspec:
        del inspect.getargspec
