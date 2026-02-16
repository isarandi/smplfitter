from __future__ import annotations

from typing import Optional
import tensorflow as tf

from .. import common as smplfitter_common
from .rotation import mat2rotvec, rotvec2mat


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
            variable.
        num_betas: Number of shape parameters (betas) to use. By default, all available betas
            are used.
        vertex_subset_size: Size of a pre-computed vertex subset to use for faster fitting.
        vertex_subset: Custom array of vertex indices to use.
        joint_regressor_post_lbs: Custom joint regressor matrix for post-LBS joint locations.
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
        self.v_template = tf.constant(data.v_template, tf.float32)
        self.shapedirs = tf.constant(data.shapedirs, tf.float32)
        self.posedirs = tf.constant(data.posedirs, tf.float32)
        self.J_regressor_post_lbs = tf.constant(data.J_regressor_post_lbs, tf.float32)
        self.J_template = tf.constant(data.J_template, tf.float32)
        self.J_shapedirs = tf.constant(data.J_shapedirs, tf.float32)
        self.kid_shapedir = tf.constant(data.kid_shapedir, tf.float32)
        self.kid_J_shapedir = tf.constant(data.kid_J_shapedir, tf.float32)
        self.weights = tf.constant(data.weights, tf.float32)
        self.kintree_parents = data.kintree_parents
        self.faces = data.faces
        self.num_joints = data.num_joints
        self.num_vertices = data.num_vertices
        self.num_betas = self.shapedirs.shape[2]
        self.vertex_subset = data.vertex_subset

    def __call__(
        self,
        pose_rotvecs: Optional[tf.Tensor] = None,
        shape_betas: Optional[tf.Tensor] = None,
        trans: Optional[tf.Tensor] = None,
        kid_factor: Optional[tf.Tensor] = None,
        rel_rotmats: Optional[tf.Tensor] = None,
        glob_rotmats: Optional[tf.Tensor] = None,
        *,
        return_vertices: bool = True,
    ):
        """
        Calculates the body model vertices, joint positions, and orientations for a batch of
        instances given the input pose, shape, and translation parameters. The rotation may be
        specified as one of three options: parent-relative rotation vectors (`pose_rotvecs`),
        parent-relative rotation matrices (`rel_rotmats`), or global rotation matrices (
        `glob_rotmats`).

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
                - 'vertices': 3D body model vertices, shaped as (batch_size, num_vertices, 3), if \
                    `return_vertices` is True.
                - 'joints': 3D joint positions, shaped as (batch_size, num_joints, 3).
                - 'orientations': Global orientation matrices for each joint, shaped as \
                    (batch_size, num_joints, 3, 3).
        """
        if isinstance(shape_betas, tf.RaggedTensor):
            assert pose_rotvecs is not None
            assert trans is not None
            res = self(
                pose_rotvecs=pose_rotvecs.flat_values,
                shape_betas=shape_betas.flat_values,
                trans=trans.flat_values,
                return_vertices=return_vertices,
            )
            return tf.nest.map_structure(
                lambda x: tf.RaggedTensor.from_row_splits(x, shape_betas.row_splits), res
            )

        batch_size = check_batch_size(pose_rotvecs, shape_betas, trans, rel_rotmats, glob_rotmats)
        if rel_rotmats is not None:
            rel_rotmats = tf.cast(rel_rotmats, tf.float32)
        elif pose_rotvecs is not None:
            pose_rotvecs = tf.cast(pose_rotvecs, tf.float32)
            rel_rotmats = rotvec2mat(tf.reshape(pose_rotvecs, (batch_size, self.num_joints, 3)))
        elif glob_rotmats is None:
            rel_rotmats = tf.eye(3, batch_shape=[batch_size, self.num_joints])

        if glob_rotmats is None:
            assert rel_rotmats is not None
            glob_rotmats_list = [rel_rotmats[:, 0]]
            for i_joint in range(1, self.num_joints):
                i_parent = self.kintree_parents[i_joint]
                glob_rotmats_list.append(glob_rotmats_list[i_parent] @ rel_rotmats[:, i_joint])
            glob_rotmats = tf.stack(glob_rotmats_list, axis=1)

        parent_glob_rotmats = tf.concat(
            [
                tf.broadcast_to(tf.eye(3), tf.shape(glob_rotmats[:, :1])),
                tf.gather(glob_rotmats, self.kintree_parents[1:], axis=1),
            ],
            axis=1,
        )

        if rel_rotmats is None:
            rel_rotmats = tf.linalg.matmul(parent_glob_rotmats, glob_rotmats, transpose_a=True)

        shape_betas = (
            tf.cast(shape_betas, tf.float32)
            if shape_betas is not None
            else tf.zeros((batch_size, 0), tf.float32)
        )
        num_betas = tf.minimum(tf.shape(shape_betas)[1], self.shapedirs.shape[2])

        if kid_factor is None:
            kid_factor = tf.zeros((1,), tf.float32)
        else:
            kid_factor = tf.cast(kid_factor, tf.float32)
        j = (
            self.J_template
            + tf.einsum(
                'jcs,bs->bjc', self.J_shapedirs[:, :, :num_betas], shape_betas[:, :num_betas]
            )
            + tf.einsum('jc,b->bjc', self.kid_J_shapedir, kid_factor)
        )

        j_parent = tf.concat(
            [
                tf.broadcast_to(tf.zeros(3), tf.shape(j[:, :1])),
                tf.gather(j, self.kintree_parents[1:], axis=1),
            ],
            axis=1,
        )
        bones = j - j_parent
        rotated_bones = tf.einsum('bjCc,bjc->bjC', parent_glob_rotmats, bones)

        glob_positions = [j[:, 0]]
        for i_joint in range(1, self.num_joints):
            i_parent = self.kintree_parents[i_joint]
            glob_positions.append(glob_positions[i_parent] + rotated_bones[:, i_joint])
        glob_positions = tf.stack(glob_positions, axis=1)

        if trans is None:
            trans = tf.zeros((1, 3), tf.float32)
        else:
            trans = tf.cast(trans, tf.float32)

        if not return_vertices:
            return dict(joints=glob_positions + trans[:, tf.newaxis], orientations=glob_rotmats)

        pose_feature = tf.reshape(rel_rotmats[:, 1:], [-1, (self.num_joints - 1) * 3 * 3])
        v_posed = (
            self.v_template
            + tf.einsum(
                'vcp,bp->bvc', self.shapedirs[:, :, :num_betas], shape_betas[:, :num_betas]
            )
            + tf.einsum('vcp,bp->bvc', self.posedirs, pose_feature)
            + tf.einsum('vc,b->bvc', self.kid_shapedir, kid_factor)
        )

        translations = glob_positions - tf.einsum('bjCc,bjc->bjC', glob_rotmats, j)
        vertices = (
            tf.einsum('bjCc,vj,bvc->bvC', glob_rotmats, self.weights, v_posed)
            + self.weights @ translations
        )

        return dict(
            joints=(glob_positions + trans[:, tf.newaxis]),
            vertices=(vertices + trans[:, tf.newaxis]),
            orientations=glob_rotmats,
        )

    def single(self, *args, return_vertices=True, **kwargs):
        """
        Calculates the body model vertices, joint positions, and orientations for a single
        instance given the input pose, shape, and translation parameters. The rotation may be
        specified as one of three options: parent-relative rotation vectors (`pose_rotvecs`),
        parent-relative rotation matrices (`rel_rotmats`), or global rotation matrices (
        `glob_rotmats`).

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
                    `return_vertices` is True
                - **joints** -- 3D joint positions, shaped as (num_joints, 3)
                - **orientations** -- Global orientation matrices for each joint, shaped as \
                    (num_joints, 3, 3)
        """
        args = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), args)
        kwargs = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), kwargs)
        if len(args) == 0 and len(kwargs) == 0:
            kwargs['shape_betas'] = tf.zeros((1, 0), tf.float32)
        result = self(*args, return_vertices=return_vertices, **kwargs)
        return tf.nest.map_structure(lambda x: tf.squeeze(x, 0), result)

    def rototranslate(
        self,
        R: tf.Tensor,
        t=None,
        pose_rotvecs=None,
        shape_betas=None,
        trans=None,
        kid_factor=0,
        post_translate: bool = True,
    ):
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
                - **new_pose_rotvec** -- Updated pose rotation vectors, shaped as (num_joints * 3,)
                - **new_trans** -- Updated translation vector, shaped as (3,)

        Notes:
            Rotating a parametric representation is nontrivial because the global orientation
            (first three rotation parameters) performs the rotation around the pelvis joint
            instead of the origin of the canonical coordinate system. This method takes into
            account the offset between the pelvis joint in the shaped T-pose and the origin of
            the canonical coordinate system.
        """
        if t is None:
            t = tf.zeros(3, dtype=R.dtype)
        if pose_rotvecs is None or shape_betas is None or trans is None:
            raise ValueError('pose_rotvecs, shape_betas, and trans are required.')
        current_rotmat = rotvec2mat(pose_rotvecs[..., :3])
        new_rotmat = R @ current_rotmat
        new_pose_rotvec = tf.concat([mat2rotvec(new_rotmat), pose_rotvecs[3:]], axis=0)

        pelvis = (
            self.J_template[0]
            + self.J_shapedirs[0, :, : shape_betas.shape[0]] @ shape_betas
            + self.kid_J_shapedir[0] * kid_factor
        )
        if post_translate:
            new_trans = (
                tf.matmul(trans, R, transpose_b=True)
                + t
                + tf.matmul(pelvis, R - tf.eye(3), transpose_b=True)
            )
        else:
            new_trans = tf.matmul(trans - t, R, transpose_b=True) + tf.matmul(
                pelvis, R - tf.eye(3), transpose_b=True
            )
        return new_pose_rotvec, new_trans


def check_batch_size(pose_rotvecs, shape_betas, trans, rel_rotmats, glob_rotmats):
    batch_sizes = [
        tf.shape(x)[0]
        for x in [pose_rotvecs, shape_betas, trans, rel_rotmats, glob_rotmats]
        if x is not None
    ]

    if len(batch_sizes) == 0:
        raise RuntimeError(
            'At least one argument must be given among pose_rotvecs, shape_betas, trans, '
            'rel_rotmats.'
        )

    return batch_sizes[0]
