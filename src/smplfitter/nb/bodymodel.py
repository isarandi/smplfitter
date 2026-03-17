from __future__ import annotations

from typing import Optional

import numpy as np
import numba
from numba import prange
from .. import common as smplfitter_common
from .rotation import mat2rotvec, rotvec2mat, rotvec2mat_batch
from .util import matmul_transp_a


class BodyModel:
    """
    Represents a statistical body model of the SMPL family.

    The SMPL (Skinned Multi-Person Linear) model provides a way to represent articulated 3D
    human meshes through a compact shape vector (beta) and pose (body part rotation) parameters.

    Parameters:
        model_name: Name of the model type.
        gender: Gender of the model, which can be 'neutral', 'female' or 'male'.
        model_root: Path to the directory containing model files. By default,
            {DATA_ROOT}/body_models/{model_name} is used, with the DATA_ROOT environment
            variable, or if a DATA_ROOT envvar doesn't exist, ``./body_models/{model_name}``.
        num_betas: Number of shape parameters (betas) to use. By default, all available betas are
            used.
        vertex_subset_size: If specified, loads a pre-computed vertex subset of this size from
            ``{model_root}/vertex_subset_{size}.npz``.
        vertex_subset: Array of vertex indices to use. If specified, only these vertices will be
            computed.
        faces: Custom faces array to use instead of the default.
        joint_regressor_post_lbs: Custom joint regressor for post-LBS joint computation.
    """

    def __init__(
        self,
        model_name='smpl',
        gender='neutral',
        model_root=None,
        num_betas=None,
        vertex_subset_size=None,
        vertex_subset=None,
        faces=None,
        joint_regressor_post_lbs=None,
    ):
        self.gender = gender
        self.model_name = model_name
        data = smplfitter_common.initialize(
            model_name,
            gender,
            model_root,
            num_betas,
            vertex_subset_size,
            vertex_subset,
            faces,
            joint_regressor_post_lbs,
        )
        self.v_template = np.array(data.v_template, np.float32)
        self.shapedirs = np.array(data.shapedirs, np.float32)
        self.posedirs = np.array(data.posedirs, np.float32)
        self.J_regressor_post_lbs = np.array(data.J_regressor_post_lbs, np.float32)
        self.J_template = np.array(data.J_template, np.float32)
        self.J_shapedirs = np.array(data.J_shapedirs, np.float32)
        self.kid_shapedir = np.array(data.kid_shapedir, np.float32)
        self.kid_J_shapedir = np.array(data.kid_J_shapedir, np.float32)
        self.weights = np.array(data.weights, np.float32)
        self.kintree_parents = np.array(data.kintree_parents, np.int64)
        self.faces = data.faces
        self.num_joints = data.num_joints
        self.num_vertices = data.num_vertices
        self.vertex_subset = data.vertex_subset
        self.num_betas = self.shapedirs.shape[2]
        self.joint_names = data.joint_names

    def __call__(
        self,
        pose_rotvecs: Optional[np.ndarray] = None,
        shape_betas: Optional[np.ndarray] = None,
        trans: Optional[np.ndarray] = None,
        kid_factor: Optional[np.ndarray] = None,
        rel_rotmats: Optional[np.ndarray] = None,
        glob_rotmats: Optional[np.ndarray] = None,
        *,
        return_vertices: bool = True,
    ):
        """
        Calculates the body model vertices, joint positions, and orientations for a batch of
        instances given the input pose, shape, and translation parameters. The rotation may be
        specified as one of three options: parent-relative rotation vectors (`pose_rotvecs`),
        parent-relative rotation matrices (`rel_rotmats`), or global rotation matrices
        (`glob_rotmats`).

        Parameters:
            pose_rotvecs: Rotation vectors per joint, shaped as (batch_size, num_joints,
                3) or flattened as (batch_size, num_joints * 3).
            shape_betas: Shape coefficients (betas) for the body shape, shaped as (batch_size,
                num_betas).
            trans: Translation vector to apply after posing, shaped as (batch_size, 3).
            kid_factor: Adjustment factor for kid shapes, shaped as (batch_size, 1).
            rel_rotmats: Parent-relative rotation matrices per joint, shaped as
                (batch_size, num_joints, 3, 3).
            glob_rotmats: Global rotation matrices per joint, shaped as (batch_size, num_joints,
                3, 3).
            return_vertices: Flag indicating whether to compute and return the body model vertices.
                If only joints and orientations are needed, setting this to False is faster.

        Returns:
            A dictionary containing
                - **vertices** -- 3D body model vertices, shaped as (batch_size, num_vertices, 3), \
                    if `return_vertices` is True.
                - **joints** -- 3D joint positions, shaped as (batch_size, num_joints, 3).
                - **orientations** -- Global orientation matrices for each joint, shaped as \
                    (batch_size, num_joints, 3, 3).

        """

        rot_inputs = [
            (name, arg) for name, arg in [
                ('pose_rotvecs', pose_rotvecs),
                ('rel_rotmats', rel_rotmats),
                ('glob_rotmats', glob_rotmats),
            ] if arg is not None
        ]
        if len(rot_inputs) > 1:
            names = [name for name, _ in rot_inputs]
            raise ValueError(
                f'Only one rotation input may be provided. Got: {", ".join(names)}.'
            )

        batch_size = check_batch_size(pose_rotvecs, shape_betas, trans, rel_rotmats, glob_rotmats)

        if batch_size == 0:
            result = dict(
                joints=np.empty((0, self.num_joints, 3), np.float32),
                orientations=np.empty((0, self.num_joints, 3, 3), np.float32),
            )
            if return_vertices:
                result['vertices'] = np.empty((0, self.num_vertices, 3), np.float32)
            return result

        if pose_rotvecs is not None:
            pose_rotvecs = np.asarray(pose_rotvecs, np.float32).reshape(
                batch_size, self.num_joints, 3)
        if rel_rotmats is not None:
            rel_rotmats = np.asarray(rel_rotmats, np.float32)
        if glob_rotmats is not None:
            glob_rotmats = np.asarray(glob_rotmats, np.float32)
        if shape_betas is None:
            shape_betas = np.zeros((batch_size, 0), np.float32)
        else:
            shape_betas = np.asarray(shape_betas, np.float32)
        if trans is None:
            trans = np.zeros((1, 3), np.float32)
        else:
            trans = np.asarray(trans, np.float32)
        if kid_factor is None:
            kid_factor = np.zeros((1,), np.float32)
        else:
            kid_factor = np.asarray(kid_factor, np.float32)

        num_betas = min(shape_betas.shape[1], self.num_betas)
        if num_betas == self.num_betas:
            shapedirs = self.shapedirs
            J_shapedirs = self.J_shapedirs
            shape_betas_slice = shape_betas
        else:
            shapedirs = self.shapedirs[:, :, :num_betas]
            J_shapedirs = self.J_shapedirs[:, :, :num_betas]
            shape_betas_slice = shape_betas[:, :num_betas]

        has_pose_rotvecs = pose_rotvecs is not None
        has_rel_rotmats = rel_rotmats is not None
        has_glob_rotmats = glob_rotmats is not None

        # Use dummy arrays for None inputs (the flags tell the @njit function which to use)
        _dummy3 = np.empty((0, 0, 0), np.float32)
        _dummy4 = np.empty((0, 0, 0, 0), np.float32)

        vertices, joints, orientations = _body_model_forward(
            pose_rotvecs if has_pose_rotvecs else _dummy3,
            rel_rotmats if has_rel_rotmats else _dummy4,
            glob_rotmats if has_glob_rotmats else _dummy4,
            shape_betas_slice,
            trans,
            kid_factor,
            self.v_template,
            shapedirs,
            self.posedirs,
            self.J_template,
            J_shapedirs,
            self.kid_J_shapedir,
            self.kid_shapedir,
            self.weights,
            self.kintree_parents,
            has_pose_rotvecs,
            has_rel_rotmats,
            has_glob_rotmats,
            return_vertices,
        )

        result = dict(joints=joints, orientations=orientations)
        if return_vertices:
            result['vertices'] = vertices
        return result

    def single(self, *args, return_vertices=True, **kwargs):
        """
        Calculates the body model vertices, joint positions, and orientations for a single
        instance given the input pose, shape, and translation parameters. The rotation may be
        specified as one of three options: parent-relative rotation vectors (`pose_rotvecs`),
        parent-relative rotation matrices (`rel_rotmats`), or global rotation matrices (
        `glob_rotmats`). If none of the arguments are given, the default pose and shape are used.

        Parameters:
            pose_rotvecs: Rotation vectors per joint, shaped as (num_joints, 3) or (num_joints *
                3,).
            shape_betas: Shape coefficients (betas) for the body shape, shaped as (num_betas,).
            trans: Translation vector to apply after posing, shaped as (3,).
            kid_factor: Adjustment factor for kid shapes, shaped as (1,). Default is None.
            rel_rotmats: Parent-relative rotation matrices per joint, shaped as (num_joints, 3, 3).
            glob_rotmats: Global rotation matrices per joint, shaped as (num_joints, 3, 3).
            return_vertices: Flag indicating whether to compute and return the body model
                vertices. If only joints and orientations are needed, it is much faster.

        Returns:
            A dictionary containing
                - **vertices** -- 3D body model vertices, shaped as (num_vertices, 3), if \
                    `return_vertices` is True.
                - **joints** -- 3D joint positions, shaped as (num_joints, 3).
                - **orientations** -- Global orientation matrices for each joint, shaped as \
                    (num_joints, 3, 3).

        """
        args = [np.expand_dims(x, axis=0) for x in args]
        kwargs = {k: np.expand_dims(v, axis=0) for k, v in kwargs.items()}
        if len(args) == 0 and len(kwargs) == 0:
            kwargs['shape_betas'] = np.zeros((1, 0), np.float32)
        result = self(*args, return_vertices=return_vertices, **kwargs)
        return {k: np.squeeze(v, axis=0) for k, v in result.items()}

    def rototranslate(
        self,
        R,
        t=None,
        pose_rotvecs=None,
        shape_betas=None,
        trans=None,
        kid_factor=0,
        post_translate=True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Rotates and translates the body in parametric form.

        If `post_translate` is True, the translation is added after rotation by `R`, as:

        `M(new_pose_rotvec, shape, new_trans) = R @ M(pose_rotvecs, shape, trans) + t`,
        where `M` is the body model forward function.

        If `post_translate` is False, the translation is subtracted before rotation by `R`, as:

        `M(new_pose_rotvec, shape, new_trans) = R @ (M(pose_rotvecs, shape, trans) - t)`

        Parameters:
            R: Rotation matrix, shaped as (3, 3).
            t: Translation vector, shaped as (3,). Defaults to zero (pure rotation).
            pose_rotvecs: Initial rotation vectors per joint, shaped as (num_joints * 3,).
            shape_betas: Shape coefficients (betas) for body shape, shaped as (num_betas,).
            trans: Initial translation vector, shaped as (3,).
            kid_factor: Optional in case of kid shapes like in AGORA. Shaped as (1,).
            post_translate: Flag indicating whether to apply the translation after rotation. If
                True, `t` is added after rotation by `R`; if False, `t` is subtracted before
                rotation by `R`.

        Returns:
            A tuple containing
                - **new_pose_rotvec** -- Updated pose rotation vectors, shaped as (num_joints * 3,).
                - **new_trans** -- Updated translation vector, shaped as (3,).


        Notes:
            Rotating a parametric representation is nontrivial because the global orientation
            (first three rotation parameters) performs the rotation around the pelvis joint
            instead of the origin of the canonical coordinate system. This method takes into
            account the offset between the pelvis joint in the shaped T-pose and the origin of
            the canonical coordinate system.
        """
        if t is None:
            t = np.zeros(3, dtype=R.dtype)
        if pose_rotvecs is None or shape_betas is None or trans is None:
            raise ValueError('pose_rotvecs, shape_betas, and trans are required.')
        current_rotmat = rotvec2mat(pose_rotvecs[:3])
        new_rotmat = R @ current_rotmat
        new_pose_rotvec = np.concatenate([mat2rotvec(new_rotmat), pose_rotvecs[3:]], axis=0)

        pelvis = (
            self.J_template[0]
            + self.J_shapedirs[0, :, : shape_betas.shape[0]] @ shape_betas
            + self.kid_J_shapedir[0] * kid_factor
        )

        eye = np.eye(3, dtype=np.float32)
        if post_translate:
            new_trans = pelvis @ (R.T - eye) + trans @ R.T + t
        else:
            new_trans = pelvis @ (R.T - eye) + (trans - t) @ R.T
        return new_pose_rotvec, new_trans


@numba.njit(error_model='numpy', cache=True, parallel=True)
def _forward_kinematics(rel_rotmats, kintree_parents):
    batch_size = rel_rotmats.shape[0]
    num_joints = rel_rotmats.shape[1]
    glob_rotmats = np.empty((batch_size, num_joints, 3, 3), dtype=np.float32)
    for b in prange(batch_size):
        for r in range(3):
            for c in range(3):
                glob_rotmats[b, 0, r, c] = rel_rotmats[b, 0, r, c]
        for i_joint in range(1, num_joints):
            i_parent = kintree_parents[i_joint]
            for r in range(3):
                for c in range(3):
                    val = np.float32(0.0)
                    for k in range(3):
                        val += glob_rotmats[b, i_parent, r, k] * rel_rotmats[b, i_joint, k, c]
                    glob_rotmats[b, i_joint, r, c] = val
    return glob_rotmats


@numba.njit(error_model='numpy', cache=True)
def _accumulate_positions(root_pos, rotated_bones1, kintree_parents):
    batch_size = root_pos.shape[0]
    num_joints = len(kintree_parents)
    glob_positions = np.empty((batch_size, num_joints, 3), dtype=np.float32)
    glob_positions[:, 0] = root_pos
    for i_joint in range(1, num_joints):
        i_parent = kintree_parents[i_joint]
        glob_positions[:, i_joint] = glob_positions[:, i_parent] + rotated_bones1[:, i_joint - 1]
    return glob_positions


@numba.njit(error_model='numpy', cache=True, parallel=True)
def _compute_joints(J_template, J_shapedirs, shape_betas, kid_J_shapedir, kid_factor):
    batch_size = shape_betas.shape[0]
    num_joints = J_template.shape[0]
    j = np.empty((batch_size, num_joints, 3), dtype=np.float32)
    for b in prange(batch_size):
        for i in range(num_joints):
            for c in range(3):
                val = J_template[i, c]
                for s in range(shape_betas.shape[1]):
                    val += J_shapedirs[i, c, s] * shape_betas[b, s]
                if kid_factor.shape[0] == 1:
                    val += kid_J_shapedir[i, c] * kid_factor[0]
                else:
                    val += kid_J_shapedir[i, c] * kid_factor[b]
                j[b, i, c] = val
    return j


@numba.njit(error_model='numpy', cache=True, parallel=True)
def _compute_v_posed(
    v_template, shapedirs, shape_betas, posedirs, pose_feature, kid_shapedir, kid_factor
):
    batch_size = shape_betas.shape[0]
    num_vertices = v_template.shape[0]
    num_betas = shape_betas.shape[1]
    num_pose_features = pose_feature.shape[1]

    v_posed = np.empty((batch_size, num_vertices, 3), dtype=np.float32)
    for v in prange(num_vertices):
        for c in range(3):
            base_val = v_template[v, c]
            for b in range(batch_size):
                val = base_val
                for s in range(num_betas):
                    val += shapedirs[v, c, s] * shape_betas[b, s]
                for p in range(num_pose_features):
                    val += posedirs[v, c, p] * pose_feature[b, p]
                if kid_factor.shape[0] == 1:
                    val += kid_shapedir[v, c] * kid_factor[0]
                else:
                    val += kid_shapedir[v, c] * kid_factor[b]
                v_posed[b, v, c] = val
    return v_posed


@numba.njit(error_model='numpy', cache=True, parallel=True)
def _batched_matvec(mats, vecs):
    batch_size = mats.shape[0]
    n = mats.shape[1]
    result = np.empty((batch_size, n, 3), dtype=np.float32)
    for b in prange(batch_size):
        for i in range(n):
            for r in range(3):
                val = np.float32(0.0)
                for c in range(3):
                    val += mats[b, i, r, c] * vecs[b, i, c]
                result[b, i, r] = val
    return result


@numba.njit(error_model='numpy', cache=True, parallel=True)
def _lbs(glob_rotmats, glob_positions, j, weights, v_posed):
    # glob_rotmats: (batch_size, num_joints, 3, 3)
    # glob_positions: (batch_size, num_joints, 3)
    # j: (batch_size, num_joints, 3)
    # weights: (num_vertices, num_joints)
    # v_posed: (batch_size, num_vertices, 3)
    # Output: (batch_size, num_vertices, 3)
    batch_size = glob_rotmats.shape[0]
    num_joints = glob_rotmats.shape[1]
    num_vertices = weights.shape[0]

    # translations = glob_positions - batched_matvec(glob_rotmats, j)
    translations = np.empty((batch_size, num_joints, 3), dtype=np.float32)
    for b in range(batch_size):
        for i in range(num_joints):
            for r in range(3):
                val = np.float32(0.0)
                for c in range(3):
                    val += glob_rotmats[b, i, r, c] * j[b, i, c]
                translations[b, i, r] = glob_positions[b, i, r] - val

    # vertices = einsum('bjCc,vj,bvc->bvC', glob_rotmats, weights, v_posed) + weights @ translations
    vertices = np.empty((batch_size, num_vertices, 3), dtype=np.float32)
    for v in prange(num_vertices):
        for b in range(batch_size):
            for C in range(3):
                val = np.float32(0.0)
                for i in range(num_joints):
                    w = weights[v, i]
                    if w != 0.0:
                        # Rotation part: sum over c of glob_rotmats[b,j,C,c] * v_posed[b,v,c]
                        rot_val = np.float32(0.0)
                        for c in range(3):
                            rot_val += glob_rotmats[b, i, C, c] * v_posed[b, v, c]
                        val += w * rot_val
                        # Translation part
                        val += w * translations[b, i, C]
                vertices[b, v, C] = val
    return vertices


@numba.njit(error_model='numpy', cache=True, parallel=True)
def _body_model_forward(
    pose_rotvecs, rel_rotmats_in, glob_rotmats_in,
    shape_betas, trans, kid_factor,
    v_template, shapedirs, posedirs, J_template, J_shapedirs,
    kid_J_shapedir, kid_shapedir, weights, kintree_parents,
    has_pose_rotvecs, has_rel_rotmats, has_glob_rotmats, return_vertices,
):
    num_joints = J_template.shape[0]
    num_vertices = v_template.shape[0]

    if has_pose_rotvecs:
        batch_size = pose_rotvecs.shape[0]
    elif has_rel_rotmats:
        batch_size = rel_rotmats_in.shape[0]
    elif has_glob_rotmats:
        batch_size = glob_rotmats_in.shape[0]
    else:
        batch_size = shape_betas.shape[0]

    # Step 1: Get rotation matrices
    if has_glob_rotmats:
        glob_rotmats = glob_rotmats_in.copy()
    else:
        if has_rel_rotmats:
            rel_rotmats = rel_rotmats_in.copy()
        elif has_pose_rotvecs:
            rel_rotmats = rotvec2mat_batch(pose_rotvecs)
        else:
            rel_rotmats = np.empty((batch_size, num_joints, 3, 3), dtype=np.float32)
            for b in range(batch_size):
                for j in range(num_joints):
                    rel_rotmats[b, j, 0, 0] = 1.0
                    rel_rotmats[b, j, 0, 1] = 0.0
                    rel_rotmats[b, j, 0, 2] = 0.0
                    rel_rotmats[b, j, 1, 0] = 0.0
                    rel_rotmats[b, j, 1, 1] = 1.0
                    rel_rotmats[b, j, 1, 2] = 0.0
                    rel_rotmats[b, j, 2, 0] = 0.0
                    rel_rotmats[b, j, 2, 1] = 0.0
                    rel_rotmats[b, j, 2, 2] = 1.0
        glob_rotmats = _forward_kinematics(rel_rotmats, kintree_parents)

    # Step 2: Compute joints
    j = _compute_joints(J_template, J_shapedirs, shape_betas, kid_J_shapedir, kid_factor)

    # Step 3: Compute bones, rotate, accumulate positions
    bones1 = np.empty((batch_size, num_joints - 1, 3), dtype=np.float32)
    parent_glob_rotmats1 = np.empty((batch_size, num_joints - 1, 3, 3), dtype=np.float32)
    for b in range(batch_size):
        for ji in range(1, num_joints):
            p = kintree_parents[ji]
            for c in range(3):
                bones1[b, ji - 1, c] = j[b, ji, c] - j[b, p, c]
            for r in range(3):
                for c in range(3):
                    parent_glob_rotmats1[b, ji - 1, r, c] = glob_rotmats[b, p, r, c]

    rotated_bones1 = _batched_matvec(parent_glob_rotmats1, bones1)
    glob_positions = _accumulate_positions(j[:, 0].copy(), rotated_bones1, kintree_parents)

    # Step 4: Add translation to joint positions
    joints_out = np.empty((batch_size, num_joints, 3), dtype=np.float32)
    for b in range(batch_size):
        tb = 0 if trans.shape[0] == 1 else b
        for ji in range(num_joints):
            for c in range(3):
                joints_out[b, ji, c] = glob_positions[b, ji, c] + trans[tb, c]

    if not return_vertices:
        return np.empty((0, 0, 0), dtype=np.float32), joints_out, glob_rotmats

    # Step 5: Compute rel_rotmats1 for pose feature
    if has_glob_rotmats and not has_rel_rotmats and not has_pose_rotvecs:
        rel_rotmats1 = matmul_transp_a(parent_glob_rotmats1, glob_rotmats[:, 1:])
    else:
        rel_rotmats1 = rel_rotmats[:, 1:].copy()

    pose_feature = rel_rotmats1.reshape(batch_size, (num_joints - 1) * 9)

    # Step 6: Compute v_posed
    v_posed = _compute_v_posed(
        v_template, shapedirs, shape_betas, posedirs, pose_feature,
        kid_shapedir, kid_factor,
    )

    # Step 7: LBS
    vertices = _lbs(glob_rotmats, glob_positions, j, weights, v_posed)

    # Step 8: Add translation to vertices
    vertices_out = np.empty((batch_size, num_vertices, 3), dtype=np.float32)
    for b in prange(num_vertices):
        for bi in range(batch_size):
            tb = 0 if trans.shape[0] == 1 else bi
            for c in range(3):
                vertices_out[bi, b, c] = vertices[bi, b, c] + trans[tb, c]

    return vertices_out, joints_out, glob_rotmats


def check_batch_size(pose_rotvecs, shape_betas, trans, rel_rotmats, glob_rotmats=None):
    batch_sizes = [
        np.asarray(x).shape[0]
        for x in [pose_rotvecs, shape_betas, trans, rel_rotmats, glob_rotmats]
        if x is not None
    ]

    if len(batch_sizes) == 0:
        return 0

    if not all(b == batch_sizes[0] for b in batch_sizes[1:]):
        raise RuntimeError('The batch sizes must be equal.')

    return batch_sizes[0]
