from __future__ import annotations

from typing import Optional

import numpy as np
from .. import common as smplfitter_common
from .rotation import mat2rotvec, rotvec2mat
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
        self.kintree_parents = data.kintree_parents
        self.faces = data.faces
        self.num_joints = data.num_joints
        self.num_vertices = data.num_vertices
        self.vertex_subset = data.vertex_subset
        self.num_betas = self.shapedirs.shape[2]

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

        batch_size = check_batch_size(pose_rotvecs, shape_betas, trans, rel_rotmats, glob_rotmats)

        if rel_rotmats is not None:
            rel_rotmats = np.asarray(rel_rotmats, np.float32)
        elif pose_rotvecs is not None:
            pose_rotvecs = np.asarray(pose_rotvecs, np.float32)
            rel_rotmats = rotvec2mat(np.reshape(pose_rotvecs, (batch_size, self.num_joints, 3)))
        elif glob_rotmats is None:
            rel_rotmats = np.tile(np.eye(3, dtype=np.float32), [batch_size, self.num_joints, 1, 1])

        if glob_rotmats is None:
            assert rel_rotmats is not None
            glob_rotmats_list = [rel_rotmats[:, 0]]
            for i_joint in range(1, self.num_joints):
                i_parent = self.kintree_parents[i_joint]
                glob_rotmats_list.append(glob_rotmats_list[i_parent] @ rel_rotmats[:, i_joint])
            glob_rotmats = np.stack(glob_rotmats_list, axis=1)

        parent_indices1 = self.kintree_parents[1:]
        parent_glob_rotmats1 = glob_rotmats[:, parent_indices1]

        if rel_rotmats is None:
            rel_rotmats1 = matmul_transp_a(parent_glob_rotmats1, glob_rotmats[:, 1:])
        else:
            rel_rotmats1 = rel_rotmats[:, 1:]

        if shape_betas is None:
            shape_betas = np.zeros((batch_size, 0), np.float32)
        else:
            shape_betas = np.asarray(shape_betas, np.float32)
        num_betas = np.minimum(shape_betas.shape[1], self.shapedirs.shape[2])

        if kid_factor is None:
            kid_factor = np.zeros((1,), np.float32)
        else:
            kid_factor = np.asarray(kid_factor, np.float32)

        j = (
            self.J_template
            + np.einsum(
                'jcs,bs->bjc', self.J_shapedirs[:, :, :num_betas], shape_betas[:, :num_betas]
            )
            + np.einsum('jc,b->bjc', self.kid_J_shapedir, kid_factor)
        )

        bones1 = j[:, 1:] - j[:, parent_indices1]
        rotated_bones1 = np.einsum('bjCc,bjc->bjC', parent_glob_rotmats1, bones1)

        glob_positions = [j[:, 0]]
        for i_joint in range(1, self.num_joints):
            i_parent = self.kintree_parents[i_joint]
            glob_positions.append(glob_positions[i_parent] + rotated_bones1[:, i_joint - 1])
        glob_positions = np.stack(glob_positions, axis=1)

        if trans is None:
            trans = np.zeros((1, 3), np.float32)
        else:
            trans = trans.astype(np.float32)

        if not return_vertices:
            return dict(joints=(glob_positions + trans[:, np.newaxis]), orientations=glob_rotmats)

        pose_feature = np.reshape(rel_rotmats1, [-1, (self.num_joints - 1) * 3 * 3])
        v_posed = (
            self.v_template
            + np.einsum(
                'vcp,bp->bvc', self.shapedirs[:, :, :num_betas], shape_betas[:, :num_betas]
            )
            + np.einsum('vcp,bp->bvc', self.posedirs, pose_feature)
            + np.einsum('vc,b->bvc', self.kid_shapedir, kid_factor)
        )

        translations = glob_positions - np.einsum('bjCc,bjc->bjC', glob_rotmats, j)
        vertices = (
            np.einsum('bjCc,vj,bvc->bvC', glob_rotmats, self.weights, v_posed)
            + self.weights @ translations
        )

        return dict(
            vertices=vertices + trans[:, np.newaxis],
            joints=glob_positions + trans[:, np.newaxis],
            orientations=glob_rotmats,
        )

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
        self, R, t, pose_rotvecs, shape_betas, trans, kid_factor=0, post_translate=True
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
            t: Translation vector, shaped as (3,).
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


def check_batch_size(pose_rotvecs, shape_betas, trans, rel_rotmats, glob_rotmats=None):
    batch_sizes = [
        np.asarray(x).shape[0]
        for x in [pose_rotvecs, shape_betas, trans, rel_rotmats, glob_rotmats]
        if x is not None
    ]

    if len(batch_sizes) == 0:
        raise RuntimeError(
            'At least one argument must be given among pose_rotvecs, shape_betas, trans, '
            'rel_rotmats.'
        )

    if not all(b == batch_sizes[0] for b in batch_sizes[1:]):
        raise RuntimeError('The batch sizes must be equal.')

    return batch_sizes[0]
