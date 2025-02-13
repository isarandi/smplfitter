from typing import Optional

import numpy as np
import smplfitter.common
from smplfitter.np.rotation import mat2rotvec, rotvec2mat
from smplfitter.np.util import matmul_transp_a


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
    """

    def __init__(self, model_name='smpl', gender='neutral', model_root=None, num_betas=None):
        self.gender = gender
        self.model_name = model_name
        tensors, nontensors = smplfitter.common.initialize(
            model_name, gender, model_root, num_betas
        )
        self.v_template = np.array(tensors['v_template'], np.float32)
        self.shapedirs = np.array(tensors['shapedirs'], np.float32)
        self.posedirs = np.array(tensors['posedirs'], np.float32)
        self.J_regressor = np.array(tensors['J_regressor'], np.float32)
        self.J_template = np.array(tensors['J_template'], np.float32)
        self.J_shapedirs = np.array(tensors['J_shapedirs'], np.float32)
        self.kid_shapedir = np.array(tensors['kid_shapedir'], np.float32)
        self.kid_J_shapedir = np.array(tensors['kid_J_shapedir'], np.float32)
        self.weights = np.array(tensors['weights'], np.float32)
        self.kintree_parents = nontensors['kintree_parents']
        self.faces = nontensors['faces']
        self.num_joints = nontensors['num_joints']
        self.num_vertices = nontensors['num_vertices']

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

        batch_size = check_batch_size(pose_rotvecs, shape_betas, trans, rel_rotmats)

        if rel_rotmats is not None:
            rel_rotmats = np.asarray(rel_rotmats, np.float32)
        elif pose_rotvecs is not None:
            pose_rotvecs = np.asarray(pose_rotvecs, np.float32)
            rel_rotmats = rotvec2mat(np.reshape(pose_rotvecs, (batch_size, self.num_joints, 3)))
        elif glob_rotmats is None:
            rel_rotmats = np.tile(np.eye(3, dtype=np.float32), [batch_size, self.num_joints, 1, 1])

        if glob_rotmats is None:
            glob_rotmats = [rel_rotmats[:, 0]]
            for i_joint in range(1, self.num_joints):
                i_parent = self.kintree_parents[i_joint]
                glob_rotmats.append(glob_rotmats[i_parent] @ rel_rotmats[:, i_joint])
            glob_rotmats = np.stack(glob_rotmats, axis=1)

        parent_indices = self.kintree_parents[1:]
        parent_glob_rotmats = np.concatenate(
            [
                np.tile(np.eye(3), [glob_rotmats.shape[0], 1, 1, 1]),
                glob_rotmats[:, parent_indices],
            ],
            axis=1,
        )

        if rel_rotmats is None:
            rel_rotmats = matmul_transp_a(parent_glob_rotmats, glob_rotmats)

        if shape_betas is None:
            shape_betas = np.zeros((batch_size, 0), np.float32)
        else:
            shape_betas = np.asarray(shape_betas, np.float32)
        num_betas = np.minimum(shape_betas.shape[1], self.shapedirs.shape[2])

        if kid_factor is None:
            kid_factor = np.zeros((1,), np.float32)
        else:
            kid_factor = np.float32(kid_factor)

        j = (
            self.J_template
            + np.einsum(
                'jcs,bs->bjc', self.J_shapedirs[:, :, :num_betas], shape_betas[:, :num_betas]
            )
            + np.einsum('jc,b->bjc', self.kid_J_shapedir, kid_factor)
        )

        glob_rotmats = [rel_rotmats[:, 0]]
        glob_positions = [j[:, 0]]

        for i_joint in range(1, self.num_joints):
            i_parent = self.kintree_parents[i_joint]
            glob_rotmats.append(glob_rotmats[i_parent] @ rel_rotmats[:, i_joint])
            glob_positions.append(
                glob_positions[i_parent]
                + np.einsum('bCc,bc->bC', glob_rotmats[i_parent], j[:, i_joint] - j[:, i_parent])
            )

        glob_rotmats = np.stack(glob_rotmats, axis=1)
        glob_positions = np.stack(glob_positions, axis=1)

        if trans is None:
            trans = np.zeros((1, 3), np.float32)
        else:
            trans = trans.astype(np.float32)

        if not return_vertices:
            return dict(joints=(glob_positions + trans[:, np.newaxis]), orientations=glob_rotmats)

        pose_feature = np.reshape(rel_rotmats[:, 1:], [-1, (self.num_joints - 1) * 3 * 3])
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

        if post_translate:
            new_trans = pelvis @ (R.T - np.eye(3)) + trans @ R.T + t
        else:
            new_trans = pelvis @ (R.T - np.eye(3)) + (trans - t) @ R.T
        return new_pose_rotvec, new_trans


def check_batch_size(pose_rotvecs, shape_betas, trans, rel_rotmats):
    batch_sizes = [
        np.asarray(x).shape[0]
        for x in [pose_rotvecs, shape_betas, trans, rel_rotmats]
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
