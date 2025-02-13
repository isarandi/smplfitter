from typing import Optional

import tensorflow as tf
from smplfitter.tf.bodymodel import BodyModel
from smplfitter.tf.lstsq import lstsq, lstsq_partial_share
from smplfitter.tf.rotation import kabsch, mat2rotvec
from smplfitter.tf.util import safe_nan_to_zero


class BodyFitter:
    """
    Fits body model (SMPL/SMPL-X/SMPL+H) parameters to target vertices and joints.

    Parameters:
        body_model: The SMPL model instance we wish to fit, of a certain model variant and gender.
        num_betas: Number of shape parameters (betas) to use when fitting.
        enable_kid: Enables the use of a kid blendshape, allowing for fitting kid shapes as in
            AGORA.
        vertex_subset: A tensor or list specifying a subset of vertices to use in the fitting
            process, allowing faster partial fitting. The subset of vertices should cover all body
            parts to provide enough constraints.
        joint_regressor: A regression matrix of shape (num_joints, num_vertices) for obtaining
            joint locations, in case the target joints are not specified when fitting. A custom one
            must be supplied if `vertex_subset` is partial and target joint locations will not be
            provided.
    """

    def __init__(
        self,
        body_model: 'smplfitter.tf.BodyModel',
        num_betas: int = 10,
        enable_kid: bool = False,
        vertex_subset=None,
        joint_regressor=None,
    ):

        self.body_model = body_model
        self.n_betas = num_betas
        self.enable_kid = enable_kid

        if vertex_subset is None:
            vertex_subset = tf.range(body_model.num_vertices)

        self.vertex_subset = vertex_subset
        self.default_mesh_tf = tf.gather(
            body_model.single()['vertices'], self.vertex_subset, axis=0
        )

        self.J_template_ext = tf.concat(
            [
                tf.reshape(body_model.J_template, [-1, 3, 1]),
                body_model.J_shapedirs[:, :, : self.n_betas],
            ]
            + ([tf.reshape(body_model.kid_J_shapedir, [-1, 3, 1])] if enable_kid else []),
            axis=2,
        )

        self.children_and_self = [[i_joint] for i_joint in range(body_model.num_joints)]
        for i_joint in range(1, body_model.num_joints):
            i_parent = body_model.kintree_parents[i_joint]
            self.children_and_self[i_parent].append(i_joint)

        self.descendants_and_self = [[i_joint] for i_joint in range(body_model.num_joints)]
        for i_joint in range(body_model.num_joints - 1, 0, -1):
            i_parent = body_model.kintree_parents[i_joint]
            self.descendants_and_self[i_parent].extend(self.descendants_and_self[i_joint])

        self.shapedirs = tf.gather(body_model.shapedirs, self.vertex_subset, axis=0)
        self.kid_shapedir = tf.gather(body_model.kid_shapedir, self.vertex_subset, axis=0)
        self.v_template = tf.gather(body_model.v_template, self.vertex_subset, axis=0)
        self.weights = tf.gather(body_model.weights, self.vertex_subset, axis=0)
        self.posedirs = tf.gather(body_model.posedirs, self.vertex_subset, axis=0)
        self.num_vertices = tf.shape(self.v_template)[0]
        if joint_regressor is not None:
            self.J_regressor = joint_regressor
        else:
            self.J_regressor = body_model.J_regressor

    def fit(
        self,
        target_vertices: tf.Tensor,
        target_joints: Optional[tf.Tensor] = None,
        vertex_weights: Optional[tf.Tensor] = None,
        joint_weights: Optional[tf.Tensor] = None,
        num_iter: int = 1,
        beta_regularizer: float = 1,
        beta_regularizer2: float = 0,
        scale_regularizer: float = 0,
        kid_regularizer: Optional[float] = None,
        share_beta: bool = False,
        final_adjust_rots: bool = True,
        scale_target: bool = False,
        scale_fit: bool = False,
        allow_nan: bool = False,
        requested_keys=(),
    ):
        """
        Fits the body model to target vertices and optionally joints by optimizing for shape and
        pose, and optionally others.

        Parameters:
            target_vertices: Target mesh vertices, shaped as (batch_size, num_vertices, 3).
            target_joints: Target joint locations, shaped as (batch_size, num_joints, 3).
            vertex_weights: Importance weights for each vertex during the fitting process.
            joint_weights: Importance weights for each joint during the fitting process.
            num_iter: Number of iterations for the optimization process. Reasonable values are in
                the range of 1-4.
            beta_regularizer: L2 regularization weight for shape parameters (betas).
                Set small for easy poses and extreme body shapes, set high for harder poses and
                non-extreme body shape. (Good choices can be 0, 0.1, 1, 10.)
            beta_regularizer2: Secondary regularization for betas, affecting the first two
                parameters. Often zero works well.
            scale_regularizer: Regularization term to penalize the scale factor deviating from 1.
                Has no effect unless `scale_target` or `scale_fit` is True.
            kid_regularizer: Regularization weight for the kid blendshape factor. Has no effect
                unless `enable_kid` on the object is True.
            share_beta: If True, shares the shape parameters (betas) across instances in the
                batch.
            final_adjust_rots: Whether to perform a final refinement of the body part
                orientations to improve alignment.
            scale_target: If True, estimates a scale factor to apply to the target vertices for
                alignment.
            scale_fit: If True, estimates a scale factor to apply to the fitted mesh for
                alignment.
            allow_nan: If True, allows NaN values in the output. If False, replaces NaN values
                with zeros.
            requested_keys: List of keys specifying which results to return.

        Returns:
            A dictionary containing the following items, based on requested keys
                - **pose_rotvecs** -- Estimated pose in concatenated rotation vector format
                - **shape_betas** -- Estimated shape parameters (betas)
                - **trans** -- Estimated translation parameters
                - **joints** -- Estimated joint positions, if requested
                - **vertices** -- Fitted mesh vertices, if requested
                - **orientations** -- Global body part orientations as rotation matrices
                - **relative_orientations** -- Parent-relative body part orientations as rotation \
                    matrices
                - **kid_factor** -- Estimated kid blendshape factor, if `enable_kid` is True
                - **scale_corr** -- Estimated scale correction factor, if `scale_target` or \
                    `scale_fit` is True

        """

        # Subtract mean first for better numerical stability (and add it back later)
        if target_joints is None:
            target_mean = tf.reduce_mean(target_vertices, axis=1)
            target_vertices = target_vertices - target_mean[:, tf.newaxis]
        else:
            target_mean = tf.reduce_mean(
                tf.concat([target_vertices, target_joints], axis=1), axis=1
            )
            target_vertices = target_vertices - target_mean[:, tf.newaxis]
            target_joints = target_joints - target_mean[:, tf.newaxis]

        initial_joints = self.body_model.J_template[tf.newaxis]
        initial_vertices = self.default_mesh_tf[tf.newaxis]

        glob_rotmats = self.fit_global_rotations(
            target_vertices,
            target_joints,
            initial_vertices,
            initial_joints,
            vertex_weights,
            joint_weights,
        )

        for i in range(num_iter - 1):

            result = self.fit_shape(
                glob_rotmats,
                target_vertices,
                target_joints,
                vertex_weights,
                joint_weights,
                beta_regularizer,
                beta_regularizer2,
                scale_regularizer=0,
                kid_regularizer=kid_regularizer,
                share_beta=share_beta,
                scale_target=False,
                scale_fit=False,
                requested_keys=['vertices'] + (['joints'] if target_joints is not None else []),
            )
            glob_rotmats = (
                self.fit_global_rotations(
                    target_vertices,
                    target_joints,
                    result['vertices'],
                    result['joints'],
                    vertex_weights,
                    joint_weights,
                )
                @ glob_rotmats
            )

        result = self.fit_shape(
            glob_rotmats,
            target_vertices,
            target_joints,
            vertex_weights,
            joint_weights,
            beta_regularizer,
            beta_regularizer2,
            scale_regularizer,
            kid_regularizer,
            share_beta,
            scale_target,
            scale_fit,
            requested_keys=['vertices']
            + (['joints'] if target_joints is not None or final_adjust_rots else []),
        )

        if final_adjust_rots:
            if scale_target:
                factor = result['scale_corr'][:, tf.newaxis, tf.newaxis]
                glob_rotmats = self.fit_global_rotations_dependent(
                    target_vertices * factor,
                    target_joints * factor,
                    result['vertices'],
                    result['joints'],
                    vertex_weights,
                    joint_weights,
                    glob_rotmats,
                    result['shape_betas'],
                    None,
                    result['kid_factor'],
                    result['trans'],
                )
            elif scale_fit:
                factor = result['scale_corr'][:, tf.newaxis, tf.newaxis]

                def scale_adjust(x):
                    return factor * x + (1 - factor) * tf.expand_dims(result['trans'], -2)

                glob_rotmats = self.fit_global_rotations_dependent(
                    target_vertices,
                    target_joints,
                    scale_adjust(result['vertices']),
                    scale_adjust(result['joints']),
                    vertex_weights,
                    joint_weights,
                    glob_rotmats,
                    result['shape_betas'],
                    result['scale_corr'],
                    result['kid_factor'],
                    result['trans'],
                )
            else:
                glob_rotmats = self.fit_global_rotations_dependent(
                    target_vertices,
                    target_joints,
                    result['vertices'],
                    result['joints'],
                    vertex_weights,
                    joint_weights,
                    glob_rotmats,
                    result['shape_betas'],
                    None,
                    result['kid_factor'],
                    result['trans'],
                )

        if 'joints' in requested_keys or 'vertices' in requested_keys:
            forw = self.body_model(
                glob_rotmats=glob_rotmats,
                shape_betas=result['shape_betas'],
                trans=result['trans'],
                kid_factor=result['kid_factor'],
            )

        # Add the mean back
        result['trans'] = result['trans'] + target_mean
        if 'joints' in requested_keys:
            result['joints'] = forw['joints'] + target_mean[:, tf.newaxis]
        if 'vertices' in requested_keys:
            result['vertices'] = forw['vertices'] + target_mean[:, tf.newaxis]

        result['orientations'] = glob_rotmats

        # Provide other requested rotation formats
        if 'relative_orientations' in requested_keys or 'pose_rotvecs' in requested_keys:
            parent_glob_rotmats = tf.concat(
                [
                    tf.broadcast_to(tf.eye(3), tf.shape(glob_rotmats[:, :1])),
                    tf.gather(glob_rotmats, self.body_model.kintree_parents[1:], axis=1),
                ],
                axis=1,
            )
            result['relative_orientations'] = tf.linalg.matmul(
                parent_glob_rotmats, glob_rotmats, transpose_a=True
            )

        if 'pose_rotvecs' in requested_keys:
            rotvecs = mat2rotvec(result['relative_orientations'])
            result['pose_rotvecs'] = tf.reshape(rotvecs, [tf.shape(rotvecs)[0], -1])

        if not allow_nan:
            return {k: safe_nan_to_zero(v) if v is not None else None for k, v in result.items()}
        return result

    def fit_shape(
        self,
        glob_rotmats,
        target_vertices,
        target_joints=None,
        vertex_weights=None,
        joint_weights=None,
        beta_regularizer=1,
        beta_regularizer2=0,
        scale_regularizer=0,
        kid_regularizer=None,
        share_beta=False,
        scale_target=False,
        scale_fit=False,
        requested_keys=(),
    ):

        if scale_target and scale_fit:
            raise ValueError("Only one of estim_scale_target and estim_scale_fit can be True")

        glob_rotmats = tf.cast(glob_rotmats, tf.float32)
        batch_size = tf.shape(target_vertices)[0]

        parent_glob_rot_mats = tf.concat(
            [
                tf.broadcast_to(tf.eye(3), tf.shape(glob_rotmats[:, :1])),
                tf.gather(glob_rotmats, self.body_model.kintree_parents[1:], axis=1),
            ],
            axis=1,
        )
        rel_rotmats = tf.linalg.matmul(parent_glob_rot_mats, glob_rotmats, transpose_a=True)

        glob_positions_ext = [tf.repeat(self.J_template_ext[tf.newaxis, 0], batch_size, axis=0)]
        for i_joint, i_parent in enumerate(self.body_model.kintree_parents[1:], start=1):
            glob_positions_ext.append(
                glob_positions_ext[i_parent]
                + tf.einsum(
                    'bCc,cs->bCs',
                    glob_rotmats[:, i_parent],
                    self.J_template_ext[i_joint] - self.J_template_ext[i_parent],
                )
            )
        glob_positions_ext = tf.stack(glob_positions_ext, axis=1)
        translations_ext = glob_positions_ext - tf.einsum(
            'bjCc,jcs->bjCs', glob_rotmats, self.J_template_ext
        )

        rot_params = tf.reshape(rel_rotmats[:, 1:], [-1, (self.body_model.num_joints - 1) * 3 * 3])
        v_posed = self.v_template + tf.einsum('vcp,bp->bvc', self.posedirs, rot_params)
        v_rotated = tf.einsum('bjCc,vj,bvc->bvC', glob_rotmats, self.weights, v_posed)

        shapedirs = (
            tf.concat(
                [self.shapedirs[:, :, : self.n_betas], self.kid_shapedir[:, :, tf.newaxis]], axis=2
            )
            if self.enable_kid
            else self.shapedirs[:, :, : self.n_betas]
        )
        v_grad_rotated = tf.einsum('bjCc,lj,lcs->blCs', glob_rotmats, self.weights, shapedirs)

        v_rotated_ext = tf.concat([v_rotated[:, :, :, tf.newaxis], v_grad_rotated], axis=3)
        v_translations_ext = tf.einsum('vj,bjcs->bvcs', self.weights, translations_ext)
        v_posed_posed_ext = v_translations_ext + v_rotated_ext

        if target_joints is None:
            target_both = target_vertices
            pos_both = v_posed_posed_ext[..., 0]
            jac_pos_both = v_posed_posed_ext[..., 1:]
        else:
            target_both = tf.concat([target_vertices, target_joints], axis=1)
            pos_both = tf.concat([v_posed_posed_ext[..., 0], glob_positions_ext[..., 0]], axis=1)
            jac_pos_both = tf.concat(
                [v_posed_posed_ext[..., 1:], glob_positions_ext[..., 1:]], axis=1
            )

        if scale_target:
            A = tf.concat([jac_pos_both, -target_both[..., tf.newaxis]], axis=3)
        elif scale_fit:
            A = tf.concat([jac_pos_both, pos_both[..., tf.newaxis]], axis=3)
        else:
            A = jac_pos_both

        b = target_both - pos_both
        mean_A = tf.reduce_mean(A, axis=1, keepdims=True)
        mean_b = tf.reduce_mean(b, axis=1, keepdims=True)
        A = A - mean_A
        b = b - mean_b

        if target_joints is not None and vertex_weights is not None and joint_weights is not None:
            weights = tf.concat([vertex_weights, joint_weights], axis=1)
        elif target_joints is None and vertex_weights is not None:
            weights = vertex_weights
        else:
            weights = tf.ones(tf.shape(A)[:2], tf.float32)

        n_params = (
            self.n_betas + (1 if self.enable_kid else 0) + (1 if scale_target or scale_fit else 0)
        )
        A = tf.reshape(A, [batch_size, -1, n_params])
        b = tf.reshape(b, [batch_size, -1, 1])
        w = tf.repeat(tf.reshape(weights, [batch_size, -1]), 3, axis=1)
        beta_regularizer = tf.convert_to_tensor(beta_regularizer, tf.float32)
        beta_regularizer2 = tf.convert_to_tensor(
            beta_regularizer2, tf.float32
        )  # regul of first two

        l2_regularizer_all = tf.concat(
            [
                tf.fill([2], beta_regularizer2),
                tf.fill([self.n_betas - 2], beta_regularizer),
            ],
            axis=0,
        )

        if self.enable_kid:
            if kid_regularizer is None:
                kid_regularizer = beta_regularizer
            else:
                kid_regularizer = tf.convert_to_tensor(kid_regularizer, tf.float32)
            l2_regularizer_all = tf.concat(
                [l2_regularizer_all, kid_regularizer[tf.newaxis]], axis=0
            )

        if scale_target or scale_fit:
            scale_regularizer = tf.convert_to_tensor(scale_regularizer, tf.float32)
            l2_regularizer_all = tf.concat(
                [l2_regularizer_all, scale_regularizer[tf.newaxis]], axis=0
            )


        if share_beta:
            x = lstsq_partial_share(
                A, b, w, l2_regularizer_all, n_shared=self.n_betas + (1 if self.enable_kid else 0)
            )
        else:
            x = lstsq(A, b, w, l2_regularizer_all)

        x = tf.squeeze(x, -1)
        new_trans = tf.squeeze(mean_b, 1) - tf.linalg.matvec(tf.squeeze(mean_A, 1), x)
        new_shape = x[:, : self.n_betas]
        new_kid_factor = None
        new_scale_corr = None

        if self.enable_kid:
            new_kid_factor = x[:, self.n_betas]
        if scale_target or scale_fit:
            new_scale_corr = x[:, -1] + 1
            if scale_fit:
                new_shape /= new_scale_corr[..., tf.newaxis]

        result = dict(
            shape_betas=new_shape,
            kid_factor=new_kid_factor,
            trans=new_trans,
            relative_orientations=rel_rotmats,
            joints=None,
            vertices=None,
            scale_corr=new_scale_corr,
        )

        if self.enable_kid:
            new_shape = tf.concat([new_shape, new_kid_factor[:, tf.newaxis]], axis=1)

        if 'joints' in requested_keys:
            result['joints'] = (
                glob_positions_ext[..., 0]
                + tf.einsum('bvcs,bs->bvc', glob_positions_ext[..., 1:], new_shape)
                + new_trans[:, tf.newaxis]
            )

        if 'vertices' in requested_keys:
            result['vertices'] = (
                v_posed_posed_ext[..., 0]
                + tf.einsum('bvcs,bs->bvc', v_posed_posed_ext[..., 1:], new_shape)
                + new_trans[:, tf.newaxis]
            )
        return result

    def fit_global_rotations(
        self,
        target_vertices,
        target_joints,
        reference_vertices,
        reference_joints,
        vertex_weights,
        joint_weights,
    ):
        glob_rots = []
        mesh_weight = 1e-6
        joint_weight = 1 - mesh_weight

        if target_joints is None or reference_joints is None:
            target_joints = self.J_regressor @ target_vertices
            reference_joints = self.J_regressor @ reference_vertices

        part_assignment = tf.argmax(self.weights, axis=1)
        # Disable the rotation of toes separately from the feet
        part_assignment = tf.where(part_assignment == 10, tf.cast(7, tf.int64), part_assignment)
        part_assignment = tf.where(part_assignment == 11, tf.cast(8, tf.int64), part_assignment)

        for i in range(self.body_model.num_joints):
            # Disable the rotation of toes separately from the feet
            if i == 10:
                glob_rots.append(glob_rots[7])
                continue
            elif i == 11:
                glob_rots.append(glob_rots[8])
                continue

            selector = tf.where(part_assignment == i)[:, 0]
            default_body_part = tf.gather(reference_vertices, selector, axis=1)
            estim_body_part = tf.gather(target_vertices, selector, axis=1)
            weights_body_part = (
                tf.gather(vertex_weights, selector, axis=1)[..., tf.newaxis] * mesh_weight
                if vertex_weights is not None
                else mesh_weight
            )

            default_joints = tf.gather(reference_joints, self.children_and_self[i], axis=1)
            estim_joints = tf.gather(target_joints, self.children_and_self[i], axis=1)
            weights_joints = (
                tf.gather(joint_weights, self.children_and_self[i], axis=1)[..., tf.newaxis]
                * joint_weight
                if joint_weights is not None
                else joint_weight
            )

            body_part_mean_reference = tf.reduce_mean(default_joints, axis=1, keepdims=True)
            default_points = tf.concat(
                [
                    (default_body_part - body_part_mean_reference) * weights_body_part,
                    (default_joints - body_part_mean_reference) * weights_joints,
                ],
                axis=1,
            )

            body_part_mean_target = tf.reduce_mean(estim_joints, axis=1, keepdims=True)

            estim_points = tf.concat(
                [
                    (estim_body_part - body_part_mean_target),
                    (estim_joints - body_part_mean_target),
                ],
                axis=1,
            )

            glob_rot = kabsch(estim_points, default_points)
            glob_rots.append(glob_rot)

        return tf.stack(glob_rots, axis=1)

    def fit_global_rotations_dependent(
        self,
        target_vertices,
        target_joints,
        reference_vertices,
        reference_joints,
        vertex_weights,
        joint_weights,
        glob_rots_prev,
        shape_betas,
        scale_corr,
        kid_factor,
        trans,
    ):
        glob_rots = []

        true_reference_joints = reference_joints
        if target_joints is None or reference_joints is None:
            target_joints = self.J_regressor @ target_vertices
            reference_joints = self.J_regressor @ reference_vertices

        part_assignment = tf.argmax(self.weights, axis=1)
        part_assignment = tf.where(part_assignment == 10, tf.cast(7, tf.int64), part_assignment)
        part_assignment = tf.where(part_assignment == 11, tf.cast(8, tf.int64), part_assignment)

        j = self.body_model.J_template + tf.einsum(
            'jcs,...s->...jc', self.body_model.J_shapedirs[:, :, : self.n_betas], shape_betas
        )
        if kid_factor is not None:
            j += tf.einsum('jc,...->...jc', self.body_model.kid_J_shapedir, kid_factor)

        if scale_corr is not None:
            j *= scale_corr[..., tf.newaxis, tf.newaxis]

        j_parent = tf.concat(
            [
                tf.broadcast_to(tf.zeros(3), tf.shape(j[:, :1])),
                tf.gather(j, self.body_model.kintree_parents[1:], axis=1),
            ],
            axis=1,
        )
        bones = j - j_parent

        glob_positions = []

        for i in range(self.body_model.num_joints):
            if i == 0:
                glob_position = j[:, i] + trans
            else:
                i_parent = self.body_model.kintree_parents[i]
                glob_position = glob_positions[i_parent] + tf.linalg.matvec(
                    glob_rots[i_parent], bones[:, i]
                )
            glob_positions.append(glob_position)

            if i == 10:
                glob_rots.append(glob_rots[7])
                continue
            elif i == 11:
                glob_rots.append(glob_rots[8])
                continue
            elif i not in [1, 2, 4, 5, 7, 8, 16, 17, 18, 19]:
                glob_rots.append(glob_rots_prev[:, i])
                continue

            vertex_selector = tf.where(part_assignment == i)[:, 0]
            joint_selector = self.children_and_self[i]

            default_body_part = tf.gather(reference_vertices, vertex_selector, axis=1)
            estim_body_part = tf.gather(target_vertices, vertex_selector, axis=1)
            weights_body_part = (
                tf.gather(vertex_weights, vertex_selector, axis=1)[..., tf.newaxis]
                if vertex_weights is not None
                else tf.constant(1, tf.float32)
            )

            default_joints = tf.gather(reference_joints, joint_selector, axis=1)
            estim_joints = tf.gather(target_joints, joint_selector, axis=1)
            weights_joints = (
                tf.gather(joint_weights, joint_selector, axis=1)[..., tf.newaxis]
                if joint_weights is not None
                else tf.constant(1, tf.float32)
            )

            reference_point = glob_position[:, tf.newaxis]
            default_reference_point = true_reference_joints[:, i : i + 1]
            default_points = tf.concat(
                [
                    (default_body_part - default_reference_point) * weights_body_part,
                    (default_joints - default_reference_point) * weights_joints,
                ],
                axis=1,
            )
            estim_points = tf.concat(
                [(estim_body_part - reference_point), (estim_joints - reference_point)], axis=1
            )
            glob_rot = kabsch(estim_points, default_points) @ glob_rots_prev[:, i]
            glob_rots.append(glob_rot)

        return tf.stack(glob_rots, axis=1)
