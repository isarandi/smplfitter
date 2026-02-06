from __future__ import annotations

import contextlib
import os
import os.path as osp
import pickle
import sys
import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class ModelData:
    """Data loaded from a SMPL-family body model file.

    This dataclass holds all arrays and metadata needed to instantiate a body model
    in any backend (NumPy, PyTorch, TensorFlow, JAX, Numba).
    """

    # Tensor data (numpy arrays, to be converted to framework-specific tensors)
    v_template: np.ndarray
    """Vertex template in T-pose, shape (num_vertices, 3)."""

    shapedirs: np.ndarray
    """Shape blend shapes, shape (num_vertices, 3, num_betas)."""

    posedirs: np.ndarray
    """Pose blend shapes, shape (num_vertices, 3, (num_joints-1)*9)."""

    J_regressor_post_lbs: np.ndarray
    """Joint regressor for post-LBS joint locations, shape (num_joints, num_vertices)."""

    J_template: np.ndarray
    """Joint template positions, shape (num_joints, 3)."""

    J_shapedirs: np.ndarray
    """Joint shape directions, shape (num_joints, 3, num_betas)."""

    kid_shapedir: np.ndarray
    """Kid shape blend shape for vertices, shape (num_vertices, 3)."""

    kid_J_shapedir: np.ndarray
    """Kid shape blend shape for joints, shape (num_joints, 3)."""

    weights: np.ndarray
    """Skinning weights, shape (num_vertices, num_joints)."""

    # Non-tensor data (metadata)
    kintree_parents: list[int]
    """Parent joint indices for kinematic tree."""

    faces: np.ndarray
    """Face indices, shape (num_faces, 3)."""

    num_joints: int
    """Number of joints in the body model."""

    num_vertices: int
    """Number of vertices in the body model mesh."""

    vertex_subset: np.ndarray
    """Indices of vertices used (for partial models)."""


def initialize(
    model_name,
    gender,
    model_root=None,
    num_betas=None,
    vertex_subset_size=None,
    vertex_subset=None,
    faces=None,
    joint_regressor_post_lbs=None,
):
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
        elif model_name == 'mano':
            with open(osp.join(model_root, 'MANO_RIGHT.pkl'), 'rb') as f:
                smpl_data = pickle.load(f, encoding='latin1')
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
    if model_name.lower().startswith('smpl'):
        v_template_smil = np.load(os.path.join(model_root, 'kid_template.npy')).astype(np.float64)
        res['kid_shapedir'] = (
            v_template_smil - np.mean(v_template_smil, axis=0) - res['v_template']
        )
        res['kid_J_shapedir'] = res['J_regressor'] @ res['kid_shapedir']
    else:
        res['kid_shapedir'] = np.zeros_like(res['v_template'])
        res['kid_J_shapedir'] = np.zeros((res['num_joints'], 3))

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

    if vertex_subset_size is not None:
        vertex_subset_dict = np.load(f'{model_root}/vertex_subset_{vertex_subset_size}.npz')
        vertex_subset = vertex_subset_dict['i_verts']
        faces = vertex_subset_dict['faces']
        joint_regressor_post_lbs = np.load(
            f'{model_root}/vertex_subset_joint_regr_post_lbs_{vertex_subset_size}.npy'
        )

    if vertex_subset is None:
        vertex_subset = np.arange(res['num_vertices'], dtype=np.int64)
    else:
        vertex_subset = np.array(vertex_subset, dtype=np.int64)

    if faces is None:
        faces = res['faces']

    if joint_regressor_post_lbs is None:
        joint_regressor_post_lbs = res['J_regressor']

    return ModelData(
        v_template=res['v_template'][vertex_subset],
        shapedirs=res['shapedirs'][vertex_subset, :, :num_betas],
        posedirs=res['posedirs'][vertex_subset],
        J_regressor_post_lbs=joint_regressor_post_lbs,
        J_template=res['J_template'],
        J_shapedirs=res['J_shapedirs'][:, :, :num_betas],
        kid_shapedir=res['kid_shapedir'][vertex_subset],
        kid_J_shapedir=res['kid_J_shapedir'],
        weights=res['weights'][vertex_subset],
        kintree_parents=res['kintree_parents'],
        faces=faces,
        num_joints=res['num_joints'],
        num_vertices=len(vertex_subset),
        vertex_subset=vertex_subset,
    )


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
            except AttributeError:
                pass

    sys.modules['numpy.float'] = float
    sys.modules['numpy.complex'] = np.complex128
    sys.modules['numpy.NINF'] = -np.inf
    np.NINF = -np.inf  # type: ignore[misc]
    np.complex = np.complex128  # type: ignore[misc]
    np.float = float  # type: ignore[misc]

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
