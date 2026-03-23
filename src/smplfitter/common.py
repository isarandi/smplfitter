from __future__ import annotations

import contextlib
import os
import os.path as osp
import pickle
import sys
from dataclasses import dataclass

import numpy as np


# Joint names from the official smplx library (https://github.com/vchoutas/smplx)
SMPL_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hand',
    'right_hand',
]

SMPLH_JOINT_NAMES = SMPL_JOINT_NAMES[:22] + [
    'left_index1',
    'left_index2',
    'left_index3',
    'left_middle1',
    'left_middle2',
    'left_middle3',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
]

SMPLX_JOINT_NAMES = SMPL_JOINT_NAMES[:22] + [
    'jaw',
    'left_eye_smplhf',
    'right_eye_smplhf',
    'left_index1',
    'left_index2',
    'left_index3',
    'left_middle1',
    'left_middle2',
    'left_middle3',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
]

MANO_JOINT_NAMES = [
    'wrist',
    'index1',
    'index2',
    'index3',
    'middle1',
    'middle2',
    'middle3',
    'pinky1',
    'pinky2',
    'pinky3',
    'ring1',
    'ring2',
    'ring3',
    'thumb1',
    'thumb2',
    'thumb3',
]

_JOINT_NAMES_BY_MODEL = {
    'smpl': SMPL_JOINT_NAMES,
    'smplx': SMPLX_JOINT_NAMES,
    'smplxlh': SMPLX_JOINT_NAMES,
    'smplxmoyo': SMPLX_JOINT_NAMES,
    'smplh': SMPLH_JOINT_NAMES,
    'smplh16': SMPLH_JOINT_NAMES,
    'mano': MANO_JOINT_NAMES,
}


def _set_module_for_docs(module_name, module_globals, all_names):
    """Override __module__ on exported objects so Sphinx resolves package-level names.

    sphinx-codeautolink uses __module__ to find the docs page for a name. Without this,
    e.g. ``BodyModel`` imported from ``smplfitter.pt.bodymodel`` would not link to
    ``smplfitter.pt.BodyModel``. The original __module__ is saved as ``_module_original_``
    so that ``inspect.getsourcefile`` can still find the real source file (see
    ``module_restored`` in docs/conf.py).
    """
    for name in all_names:
        obj = module_globals.get(name)
        if obj is not None and callable(obj):
            obj._module_original_ = obj.__module__
            obj.__module__ = module_name


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

    joint_names: list[str]
    """Names of the joints in the body model."""


def _default_body_models_dir():
    """Return the platform-appropriate default body_models directory."""
    import platformdirs

    return osp.join(platformdirs.user_data_dir('smplfitter'), 'body_models')


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
        body_models_dir = os.getenv('SMPLFITTER_BODY_MODELS')
        if body_models_dir is None:
            data_root = os.getenv('DATA_ROOT')
            if data_root is not None:
                body_models_dir = f'{data_root}/body_models'
            elif osp.isdir('body_models'):
                body_models_dir = 'body_models'
            else:
                body_models_dir = _default_body_models_dir()
        model_root = f'{body_models_dir}/{model_name}'

    with _chumpy_stub_modules():
        gender_maps = {
            'smpl': dict(f='f', m='m', n='neutral'),
            'smplx': dict(f='FEMALE', m='MALE', n='NEUTRAL'),
            'smplxlh': dict(f='FEMALE', m='MALE', n='NEUTRAL'),
            'smplxmoyo': dict(f='FEMALE', m='MALE', n='NEUTRAL'),
            'smplh': dict(f='female', m='male'),
            'smplh16': dict(f='female', m='male', n='neutral'),
            'mano': {},
        }

        if model_name not in gender_maps:
            raise ValueError(f'Unknown model name: {model_name}')

        gmap = gender_maps[model_name]
        if model_name != 'mano':
            key = gender[0].lower()
            if key not in gmap:
                available = [{'f': 'female', 'm': 'male', 'n': 'neutral'}[k] for k in gmap]
                raise ValueError(
                    f"Gender '{gender}' is not available for model '{model_name}'. "
                    f"Available: {', '.join(repr(g) for g in available)}."
                )
            gender_str = gmap[key]

        if model_name == 'smpl':
            filename = f'basicmodel_{gender_str}_lbs_10_207_0_v1.1.0.pkl'
        elif model_name in ('smplx', 'smplxlh', 'smplxmoyo'):
            filename = f'SMPLX_{gender_str}.npz'
        elif model_name == 'smplh':
            filename = f'SMPLH_{gender_str}.pkl'
        elif model_name == 'smplh16':
            filename = osp.join(gender_str, 'model.npz')
        elif model_name == 'mano':
            filename = 'MANO_RIGHT.pkl'

        filepath = osp.join(model_root, filename)
        try:
            if filename.endswith('.npz'):
                smpl_data = np.load(filepath)
            else:
                with open(filepath, 'rb') as f, scipy_sparse_forward_compat():
                    smpl_data = pickle.load(f, encoding='latin1')
        except FileNotFoundError:
            raise FileNotFoundError(
                f'Body model file not found: {filepath}\n\n'
                f'Set the body model location using one of:\n'
                f"  1. BodyModel('{model_name}', '{gender}', "
                f"model_root='/your/path/body_models/{model_name}')\n"
                f'  2. export SMPLFITTER_BODY_MODELS=/your/path/body_models\n'
                f'  3. export DATA_ROOT=/your/path   '
                f'(looks for $DATA_ROOT/body_models/)\n\n'
                f'Download models: python -m smplfitter.download\n'
                f'Register first at the relevant site(s):\n'
                f'  https://smpl.is.tue.mpg.de/     (SMPL)\n'
                f'  https://smpl-x.is.tue.mpg.de/   (SMPL-X)\n'
                f'  https://mano.is.tue.mpg.de/     (MANO/SMPL+H)\n'
                f'  https://agora.is.tue.mpg.de/    (kid templates)'
            ) from None

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
    res['kintree_parents'] = np.array(smpl_data['kintree_table'][0], dtype=np.int32).tolist()
    res['num_joints'] = len(res['kintree_parents'])
    res['num_vertices'] = len(res['v_template'])

    # Kid model has an additional shape parameter which pulls the mesh towards the SMIL mean
    # template
    if model_name.lower().startswith('smpl'):
        kid_path = os.path.join(model_root, 'kid_template.npy')
        try:
            v_template_smil = np.load(kid_path).astype(np.float64)
        except FileNotFoundError:
            raise FileNotFoundError(
                f'Kid template not found: {kid_path}\n'
                f'Download it: python -m smplfitter.download'
            ) from None
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
        subset_path = f'{model_root}/vertex_subset_{vertex_subset_size}.npz'
        if not osp.exists(subset_path):
            from .decimation.decimate_body_models import decimate

            i_verts, dec_faces = decimate(res['v_template'], res['faces'], vertex_subset_size)
            np.savez(subset_path, i_verts=i_verts, faces=dec_faces)
        vertex_subset_dict = np.load(subset_path)
        vertex_subset = vertex_subset_dict['i_verts']
        faces = vertex_subset_dict['faces']
        regressor_path = f'{model_root}/vertex_subset_joint_regr_post_lbs_{vertex_subset_size}.npy'
        if osp.exists(regressor_path):
            joint_regressor_post_lbs = np.load(regressor_path)
        else:
            joint_regressor_post_lbs = res['J_regressor'][:, vertex_subset]

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
        joint_names=_JOINT_NAMES_BY_MODEL.get(model_name, []),
    )


@contextlib.contextmanager
def scipy_sparse_forward_compat():
    """Patch sys.modules so pickles saved with old scipy.sparse submodule paths
    (e.g. scipy.sparse.coo.coo_matrix) can still be loaded after SciPy 2.0
    removes those submodules."""
    import scipy.sparse

    saved = {}
    for name in ['coo', 'csr', 'csc']:
        mod_path = f'scipy.sparse.{name}'
        saved[mod_path] = sys.modules.get(mod_path)
        sys.modules[mod_path] = scipy.sparse
    try:
        yield
    finally:
        for mod_path, old_val in saved.items():
            if old_val is None:
                sys.modules.pop(mod_path, None)
            else:
                sys.modules[mod_path] = old_val


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_vertex_converter_csr(vertex_converter_path):
    """Load a vertex converter sparse matrix from a pickle file."""
    with scipy_sparse_forward_compat():
        scipy_csr = load_pickle(vertex_converter_path)['mtx'].tocsr().astype(np.float32)
    return scipy_csr[:, : scipy_csr.shape[1] // 2]


@contextlib.contextmanager
def _chumpy_stub_modules():
    """Register lightweight chumpy stub classes in sys.modules so that pickle files
    containing chumpy objects (Ch, Select) can be loaded without installing chumpy.

    The official SMPL/SMPL+H/MANO .pkl files store shapedirs as chumpy objects.
    These stubs implement just enough to unpickle them and convert to numpy arrays
    via np.array().
    """
    import types

    class _ChStub:
        """Stub for chumpy.ch.Ch — wraps a numpy array in .x"""

        def __array__(self, dtype=None):
            return np.array(self.x, dtype=dtype)

    class _SelectStub:
        """Stub for chumpy.reordering.Select — flat-indexes .a with .idxs"""

        def __array__(self, dtype=None):
            result = np.array(self.a, dtype=dtype).ravel()[self.idxs]
            if hasattr(self, 'preferred_shape') and self.preferred_shape is not None:
                return result.reshape(self.preferred_shape)
            return result

    stub_modules = {
        'chumpy': types.ModuleType('chumpy'),
        'chumpy.ch': types.ModuleType('chumpy.ch'),
        'chumpy.reordering': types.ModuleType('chumpy.reordering'),
    }
    stub_modules['chumpy.ch'].Ch = _ChStub
    stub_modules['chumpy.reordering'].Select = _SelectStub

    saved = {mod_path: sys.modules.get(mod_path) for mod_path in stub_modules}
    sys.modules.update(stub_modules)
    try:
        yield
    finally:
        for mod_path, old_val in saved.items():
            if old_val is None:
                sys.modules.pop(mod_path, None)
            else:
                sys.modules[mod_path] = old_val
