"""TensorFlow body fitter with feature parity to PyTorch version.

This version adds:
- fit_with_known_pose() and fit_with_known_shape() methods
- initial_pose_rotvecs, initial_shape_betas, initial_kid_factor parameters
- Weighted mean centering for numerical stability
- Regularizer reference support
- Cached part_assignment
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import tensorflow as tf

from .lstsq import lstsq, lstsq_partial_share
from .rotation import kabsch, mat2rotvec, rotvec2mat
from .util import safe_nan_to_zero

if TYPE_CHECKING:
    import smplfitter.tf


class BodyFitter:
    """Fits body model parameters to target vertices and joints.

    Parameters:
        body_model: The SMPL model instance to fit.
        enable_kid: Enable kid blendshape fitting.
    """

    def __init__(
        self,
        body_model: 'smplfitter.tf.BodyModel',
        enable_kid: bool = False,
    ):
        self.body_model = body_model
        self.n_betas = body_model.num_betas
        self.enable_kid = enable_kid
        self.is_smpl_family = body_model.model_name.startswith('smpl')

        self.default_mesh_tf = body_model.single()['vertices']

        self.J_template_ext = tf.concat(
            [
                tf.reshape(body_model.J_template, [-1, 3, 1]),
                body_model.J_shapedirs,
            ]
            + ([tf.reshape(body_model.kid_J_shapedir, [-1, 3, 1])] if enable_kid else []),
            axis=2,
        )

        # Build joint hierarchy
        self.children_and_self = [[i] for i in range(body_model.num_joints)]
        for i_joint in range(1, body_model.num_joints):
            i_parent = body_model.kintree_parents[i_joint]
            self.children_and_self[i_parent].append(i_joint)

        self.descendants_and_self = [[i] for i in range(body_model.num_joints)]
        for i_joint in range(body_model.num_joints - 1, 0, -1):
            i_parent = body_model.kintree_parents[i_joint]
            self.descendants_and_self[i_parent].extend(self.descendants_and_self[i_joint])

        # Use model's data directly (already subsetted if vertex_subset was provided to model)
        self.shapedirs = body_model.shapedirs
        self.kid_shapedir = body_model.kid_shapedir
        self.v_template = body_model.v_template
        self.weights = body_model.weights
        self.posedirs = body_model.posedirs
        self.num_vertices = body_model.num_vertices

        self.J_regressor = body_model.J_regressor_post_lbs

        # Cache part assignment (with toes merged into feet for SMPL-family models)
        part_assignment = tf.argmax(self.weights, axis=1)
        if self.is_smpl_family:
            part_assignment = tf.where(part_assignment == 10, tf.cast(7, tf.int64), part_assignment)
            part_assignment = tf.where(part_assignment == 11, tf.cast(8, tf.int64), part_assignment)
        self.part_assignment = part_assignment
        self.part_vertex_selectors = [
            tf.where(part_assignment == i)[:, 0] for i in range(body_model.num_joints)
        ]

    def fit(
        self,
        target_vertices: tf.Tensor,
        target_joints: Optional[tf.Tensor] = None,
        vertex_weights: Optional[tf.Tensor] = None,
        joint_weights: Optional[tf.Tensor] = None,
        num_iter: int = 1,
        beta_regularizer: float = 1.0,
        beta_regularizer2: float = 0.0,
        scale_regularizer: float = 0.0,
        kid_regularizer: Optional[float] = None,
        share_beta: bool = False,
        final_adjust_rots: bool = True,
        scale_target: bool = False,
        scale_fit: bool = False,
        initial_pose_rotvecs: Optional[tf.Tensor] = None,
        initial_shape_betas: Optional[tf.Tensor] = None,
        initial_kid_factor: Optional[tf.Tensor] = None,
        allow_nan: bool = False,
        requested_keys: Optional[list[str]] = None,
    ) -> dict[str, tf.Tensor]:
        """Fit body model to target vertices and optionally joints."""
        if requested_keys is None:
            requested_keys = ['pose_rotvecs']

        # Center targets for numerical stability
        target_vertices, target_joints, target_mean = self._center_targets(
            target_vertices, target_joints, vertex_weights, joint_weights
        )

        # Initialize global rotations
        glob_rotmats = self._initialize_rotations(
            target_vertices,
            target_joints,
            vertex_weights,
            joint_weights,
            initial_pose_rotvecs,
            initial_shape_betas,
            initial_kid_factor,
        )

        # Alternating optimization iterations
        for _ in range(num_iter - 1):
            result = self._fit_shape(
                glob_rotmats,
                target_vertices,
                target_joints,
                vertex_weights,
                joint_weights,
                beta_regularizer,
                beta_regularizer2,
                scale_regularizer=0.0,
                kid_regularizer=kid_regularizer,
                share_beta=share_beta,
                scale_target=False,
                scale_fit=False,
                beta_regularizer_reference=initial_shape_betas,
                kid_regularizer_reference=initial_kid_factor,
                requested_keys=['vertices', 'joints'] if target_joints is not None else ['vertices'],
            )
            glob_rotmats = (
                self._fit_global_rotations(
                    target_vertices,
                    target_joints,
                    result['vertices'],
                    result['joints'] if target_joints is not None else None,
                    vertex_weights,
                    joint_weights,
                )
                @ glob_rotmats
            )

        # Final shape fitting
        result = self._fit_shape(
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
            beta_regularizer_reference=initial_shape_betas,
            kid_regularizer_reference=initial_kid_factor,
            requested_keys=['vertices', 'joints'] if target_joints is not None or final_adjust_rots else ['vertices'],
        )

        # Final rotation refinement
        if final_adjust_rots:
            glob_rotmats = self._final_rotation_adjustment(
                target_vertices,
                target_joints,
                result,
                vertex_weights,
                joint_weights,
                glob_rotmats,
                scale_target,
                scale_fit,
            )

        # Build output
        output = self._build_result(
            glob_rotmats,
            result['shape_betas'],
            result['trans'],
            target_mean,
            result.get('kid_factor'),
            result.get('scale_corr'),
            scale_target,
            scale_fit,
            requested_keys,
        )

        if not allow_nan:
            return {k: safe_nan_to_zero(v) if v is not None else None for k, v in output.items()}
        return output

    def fit_with_known_pose(
        self,
        pose_rotvecs: tf.Tensor,
        target_vertices: tf.Tensor,
        target_joints: Optional[tf.Tensor] = None,
        vertex_weights: Optional[tf.Tensor] = None,
        joint_weights: Optional[tf.Tensor] = None,
        beta_regularizer: float = 1.0,
        beta_regularizer2: float = 0.0,
        scale_regularizer: float = 0.0,
        kid_regularizer: Optional[float] = None,
        share_beta: bool = False,
        scale_target: bool = False,
        scale_fit: bool = False,
        beta_regularizer_reference: Optional[tf.Tensor] = None,
        kid_regularizer_reference: Optional[tf.Tensor] = None,
        allow_nan: bool = False,
        requested_keys: Optional[list[str]] = None,
    ) -> dict[str, tf.Tensor]:
        """Fit shape and translation with known pose."""
        if requested_keys is None:
            requested_keys = []

        target_vertices, target_joints, target_mean = self._center_targets(
            target_vertices, target_joints, vertex_weights, joint_weights
        )

        glob_rotmats = self._rotvecs_to_global_rotmats(pose_rotvecs)

        result = self._fit_shape(
            glob_rotmats,
            target_vertices,
            target_joints,
            vertex_weights,
            joint_weights,
            beta_regularizer,
            beta_regularizer2,
            beta_regularizer2,
            scale_regularizer,
            kid_regularizer,
            share_beta,
            scale_target,
            scale_fit,
            beta_regularizer_reference=beta_regularizer_reference,
            kid_regularizer_reference=kid_regularizer_reference,
        )

        output = self._build_result(
            glob_rotmats,
            result['shape_betas'],
            result['trans'] + target_mean,
            tf.zeros_like(target_mean),
            result.get('kid_factor'),
            result.get('scale_corr'),
            scale_target,
            scale_fit,
            requested_keys,
        )

        if not allow_nan:
            return {k: safe_nan_to_zero(v) if v is not None else None for k, v in output.items()}
        return output

    def fit_with_known_shape(
        self,
        shape_betas: tf.Tensor,
        target_vertices: tf.Tensor,
        target_joints: Optional[tf.Tensor] = None,
        vertex_weights: Optional[tf.Tensor] = None,
        joint_weights: Optional[tf.Tensor] = None,
        kid_factor: Optional[tf.Tensor] = None,
        num_iter: int = 1,
        final_adjust_rots: bool = True,
        initial_pose_rotvecs: Optional[tf.Tensor] = None,
        scale_fit: bool = False,
        allow_nan: bool = False,
        requested_keys: Optional[list[str]] = None,
    ) -> dict[str, tf.Tensor]:
        """Fit pose and translation with known shape."""
        if requested_keys is None:
            requested_keys = ['pose_rotvecs']

        target_vertices, target_joints, target_mean = self._center_targets(
            target_vertices, target_joints, vertex_weights, joint_weights
        )

        # Initialize from shape
        initial_forw = self.body_model(
            shape_betas=shape_betas, kid_factor=kid_factor, pose_rotvecs=initial_pose_rotvecs
        )
        glob_rotmats = (
            self._fit_global_rotations(
                target_vertices,
                target_joints,
                tf.gather(initial_forw['vertices'], self.vertex_subset, axis=1),
                initial_forw['joints'],
                vertex_weights,
                joint_weights,
            )
            @ initial_forw['orientations']
        )

        # Iterative refinement
        for _ in range(num_iter - 1):
            result = self.body_model(
                glob_rotmats=glob_rotmats, shape_betas=shape_betas, kid_factor=kid_factor
            )
            glob_rotmats = (
                self._fit_global_rotations(
                    target_vertices,
                    target_joints,
                    tf.gather(result['vertices'], self.vertex_subset, axis=1),
                    result['joints'] if target_joints is not None else None,
                    vertex_weights,
                    joint_weights,
                )
                @ glob_rotmats
            )

        # Final forward pass
        result = self.body_model(
            glob_rotmats=glob_rotmats, shape_betas=shape_betas, kid_factor=kid_factor
        )
        scale_corr, trans = self._fit_scale_and_translation(
            target_vertices,
            tf.gather(result['vertices'], self.vertex_subset, axis=1),
            target_joints,
            result['joints'],
            vertex_weights,
            joint_weights,
            scale=scale_fit,
        )

        # Final rotation adjustment
        if final_adjust_rots:
            if scale_fit and scale_corr is not None:
                factor = scale_corr[:, tf.newaxis, tf.newaxis]
                ref_verts = factor * tf.gather(result['vertices'], self.vertex_subset, axis=1) + trans[:, tf.newaxis]
                ref_joints = factor * result['joints'] + trans[:, tf.newaxis]
            else:
                ref_verts = tf.gather(result['vertices'], self.vertex_subset, axis=1) + trans[:, tf.newaxis]
                ref_joints = result['joints'] + trans[:, tf.newaxis]

            glob_rotmats = self._fit_global_rotations_dependent(
                target_vertices,
                target_joints,
                ref_verts,
                ref_joints,
                vertex_weights,
                joint_weights,
                glob_rotmats,
                shape_betas,
                scale_corr[:, tf.newaxis, tf.newaxis] if scale_fit and scale_corr is not None else None,
                trans,
                kid_factor,
            )

        output = self._build_result(
            glob_rotmats,
            shape_betas,
            trans + target_mean,
            tf.zeros_like(target_mean),
            kid_factor,
            scale_corr,
            False,
            scale_fit,
            requested_keys,
        )

        if not allow_nan:
            return {k: safe_nan_to_zero(v) if v is not None else None for k, v in output.items()}
        return output

    # ==================== Helper Methods ====================

    def _center_targets(
        self,
        target_vertices: tf.Tensor,
        target_joints: Optional[tf.Tensor],
        vertex_weights: Optional[tf.Tensor],
        joint_weights: Optional[tf.Tensor],
    ) -> tuple[tf.Tensor, Optional[tf.Tensor], tf.Tensor]:
        """Subtract weighted mean for numerical stability."""
        if target_joints is None:
            if vertex_weights is not None:
                w = vertex_weights[:, :, tf.newaxis]
                target_mean = tf.reduce_sum(target_vertices * w, axis=1) / tf.reduce_sum(w, axis=1)
            else:
                target_mean = tf.reduce_mean(target_vertices, axis=1)
            return target_vertices - target_mean[:, tf.newaxis], None, target_mean
        else:
            combined = tf.concat([target_vertices, target_joints], axis=1)
            if vertex_weights is not None and joint_weights is not None:
                w = tf.concat([vertex_weights, joint_weights], axis=1)[:, :, tf.newaxis]
                target_mean = tf.reduce_sum(combined * w, axis=1) / tf.reduce_sum(w, axis=1)
            else:
                target_mean = tf.reduce_mean(combined, axis=1)
            return (
                target_vertices - target_mean[:, tf.newaxis],
                target_joints - target_mean[:, tf.newaxis],
                target_mean,
            )

    def _rotvecs_to_global_rotmats(self, pose_rotvecs: tf.Tensor) -> tf.Tensor:
        """Convert rotation vectors to global rotation matrices."""
        rel_rotmats = rotvec2mat(tf.reshape(pose_rotvecs, [-1, self.body_model.num_joints, 3]))
        glob_rotmats_list = [rel_rotmats[:, 0]]
        for i_joint in range(1, self.body_model.num_joints):
            i_parent = self.body_model.kintree_parents[i_joint]
            glob_rotmats_list.append(glob_rotmats_list[i_parent] @ rel_rotmats[:, i_joint])
        return tf.stack(glob_rotmats_list, axis=1)

    def _global_to_relative_rotmats(self, glob_rotmats: tf.Tensor) -> tf.Tensor:
        """Convert global rotation matrices to parent-relative."""
        parent_glob_rotmats = tf.concat(
            [
                tf.broadcast_to(tf.eye(3), tf.shape(glob_rotmats[:, :1])),
                tf.gather(glob_rotmats, self.body_model.kintree_parents[1:], axis=1),
            ],
            axis=1,
        )
        return tf.linalg.matmul(parent_glob_rotmats, glob_rotmats, transpose_a=True)

    def _initialize_rotations(
        self,
        target_vertices: tf.Tensor,
        target_joints: Optional[tf.Tensor],
        vertex_weights: Optional[tf.Tensor],
        joint_weights: Optional[tf.Tensor],
        initial_pose_rotvecs: Optional[tf.Tensor],
        initial_shape_betas: Optional[tf.Tensor],
        initial_kid_factor: Optional[tf.Tensor],
    ) -> tf.Tensor:
        """Initialize global rotations from targets."""
        if initial_pose_rotvecs is not None or initial_shape_betas is not None:
            initial_forw = self.body_model(
                shape_betas=initial_shape_betas,
                kid_factor=initial_kid_factor,
                pose_rotvecs=initial_pose_rotvecs,
            )
            return (
                self._fit_global_rotations(
                    target_vertices,
                    target_joints,
                    tf.gather(initial_forw['vertices'], self.vertex_subset, axis=1),
                    initial_forw['joints'],
                    vertex_weights,
                    joint_weights,
                )
                @ initial_forw['orientations']
            )
        else:
            return self._fit_global_rotations(
                target_vertices,
                target_joints,
                self.default_mesh_tf[tf.newaxis],
                self.body_model.J_template[tf.newaxis],
                vertex_weights,
                joint_weights,
            )

    def _final_rotation_adjustment(
        self,
        target_vertices: tf.Tensor,
        target_joints: Optional[tf.Tensor],
        shape_result: dict[str, tf.Tensor],
        vertex_weights: Optional[tf.Tensor],
        joint_weights: Optional[tf.Tensor],
        glob_rotmats: tf.Tensor,
        scale_target: bool,
        scale_fit: bool,
    ) -> tf.Tensor:
        """Final sequential rotation refinement."""
        ref_verts = shape_result['vertices']
        ref_joints = shape_result['joints']
        ref_shape = shape_result['shape_betas']
        ref_trans = shape_result['trans']
        ref_kid_factor = shape_result.get('kid_factor')
        ref_scale_corr = shape_result.get('scale_corr')

        if scale_target and ref_scale_corr is not None:
            factor = ref_scale_corr[:, tf.newaxis, tf.newaxis]
            return self._fit_global_rotations_dependent(
                target_vertices * factor,
                target_joints * factor if target_joints is not None else None,
                ref_verts,
                ref_joints,
                vertex_weights,
                joint_weights,
                glob_rotmats,
                ref_shape,
                None,
                ref_trans,
                ref_kid_factor,
            )
        elif scale_fit and ref_scale_corr is not None:
            factor = ref_scale_corr[:, tf.newaxis, tf.newaxis]
            return self._fit_global_rotations_dependent(
                target_vertices,
                target_joints,
                factor * ref_verts + (1 - factor) * ref_trans[:, tf.newaxis],
                factor * ref_joints + (1 - factor) * ref_trans[:, tf.newaxis],
                vertex_weights,
                joint_weights,
                glob_rotmats,
                ref_shape,
                factor,
                ref_trans,
                ref_kid_factor,
            )
        else:
            return self._fit_global_rotations_dependent(
                target_vertices,
                target_joints,
                ref_verts,
                ref_joints,
                vertex_weights,
                joint_weights,
                glob_rotmats,
                ref_shape,
                None,
                ref_trans,
                ref_kid_factor,
            )

    def _build_result(
        self,
        glob_rotmats: tf.Tensor,
        shape_betas: tf.Tensor,
        trans: tf.Tensor,
        target_mean: tf.Tensor,
        kid_factor: Optional[tf.Tensor],
        scale_corr: Optional[tf.Tensor],
        scale_target: bool,
        scale_fit: bool,
        requested_keys: list[str],
    ) -> dict[str, tf.Tensor]:
        """Build the output dictionary."""
        result: dict[str, tf.Tensor] = {}

        # Compute final translation with mean restored
        if scale_target and scale_corr is not None:
            result['trans'] = trans + target_mean * scale_corr[:, tf.newaxis]
        elif scale_fit and scale_corr is not None:
            result['trans'] = trans + target_mean / scale_corr[:, tf.newaxis]
        else:
            result['trans'] = trans + target_mean

        result['shape_betas'] = shape_betas
        result['orientations'] = glob_rotmats

        if kid_factor is not None:
            result['kid_factor'] = kid_factor

        if scale_corr is not None:
            result['scale_corr'] = scale_corr

        # Compute rotation formats if requested
        if 'relative_orientations' in requested_keys or 'pose_rotvecs' in requested_keys:
            result['relative_orientations'] = self._global_to_relative_rotmats(glob_rotmats)

        if 'pose_rotvecs' in requested_keys:
            rel_ori = result['relative_orientations']
            rotvecs = mat2rotvec(rel_ori)
            result['pose_rotvecs'] = tf.reshape(rotvecs, [tf.shape(rotvecs)[0], -1])

        return result

    # ==================== Core Fitting Methods ====================

    def _fit_shape(
        self,
        glob_rotmats: tf.Tensor,
        target_vertices: tf.Tensor,
        target_joints: Optional[tf.Tensor],
        vertex_weights: Optional[tf.Tensor],
        joint_weights: Optional[tf.Tensor],
        beta_regularizer: float,
        beta_regularizer2: float,
        scale_regularizer: float,
        kid_regularizer: Optional[float],
        share_beta: bool,
        scale_target: bool,
        scale_fit: bool,
        beta_regularizer_reference: Optional[tf.Tensor] = None,
        kid_regularizer_reference: Optional[tf.Tensor] = None,
        requested_keys: Optional[list[str]] = None,
    ) -> dict[str, tf.Tensor]:
        """Fit shape parameters given global rotations."""
        if scale_target and scale_fit:
            raise ValueError('Only one of scale_target and scale_fit can be True')
        if requested_keys is None:
            requested_keys = []

        glob_rotmats = tf.cast(glob_rotmats, tf.float32)
        batch_size = tf.shape(target_vertices)[0]

        # Build shape Jacobian
        v_posed_ext, glob_positions_ext, rel_rotmats = self._build_shape_jacobian(
            glob_rotmats, batch_size
        )

        # Setup target and design matrix
        if target_joints is None:
            target_both = target_vertices
            pos_both = v_posed_ext[..., 0]
            jac_pos_both = v_posed_ext[..., 1:]
        else:
            target_both = tf.concat([target_vertices, target_joints], axis=1)
            pos_both = tf.concat([v_posed_ext[..., 0], glob_positions_ext[..., 0]], axis=1)
            jac_pos_both = tf.concat([v_posed_ext[..., 1:], glob_positions_ext[..., 1:]], axis=1)

        # Design matrix
        if scale_target:
            A = tf.concat([jac_pos_both, -target_both[..., tf.newaxis]], axis=3)
        elif scale_fit:
            A = tf.concat([jac_pos_both, pos_both[..., tf.newaxis]], axis=3)
        else:
            A = jac_pos_both

        b = target_both - pos_both

        # Weights
        if target_joints is not None and vertex_weights is not None and joint_weights is not None:
            weights = tf.concat([vertex_weights, joint_weights], axis=1)
        elif target_joints is None and vertex_weights is not None:
            weights = vertex_weights
        else:
            weights = tf.ones(tf.shape(A)[:2], tf.float32)

        # Weighted centering for numerical stability
        w_expanded_A = weights[:, :, tf.newaxis, tf.newaxis]
        w_expanded_b = weights[:, :, tf.newaxis]
        w_sum_A = tf.reduce_sum(w_expanded_A, axis=1, keepdims=True)
        w_sum_b = tf.reduce_sum(w_expanded_b, axis=1, keepdims=True)

        mean_A = tf.where(
            w_sum_A > 0,
            tf.reduce_sum(w_expanded_A * A, axis=1, keepdims=True) / w_sum_A,
            tf.zeros_like(A[:, :1])
        )
        mean_b = tf.where(
            w_sum_b > 0,
            tf.reduce_sum(w_expanded_b * b, axis=1, keepdims=True) / w_sum_b,
            tf.zeros_like(b[:, :1])
        )
        A = A - mean_A
        b = b - mean_b

        # Flatten spatial dimensions
        n_params = self.n_betas + (1 if self.enable_kid else 0) + (1 if scale_target or scale_fit else 0)
        A = tf.reshape(A, [batch_size, -1, n_params])
        b = tf.reshape(b, [batch_size, -1, 1])
        w = tf.repeat(tf.reshape(weights, [batch_size, -1]), 3, axis=1)

        # Setup regularization
        l2_reg, l2_reg_rhs = self._setup_regularization(
            batch_size,
            beta_regularizer,
            beta_regularizer2,
            scale_regularizer,
            kid_regularizer,
            beta_regularizer_reference,
            kid_regularizer_reference,
            scale_target or scale_fit,
        )

        # Solve least squares
        n_shared = self.n_betas + (1 if self.enable_kid else 0)
        if share_beta:
            x = lstsq_partial_share(A, b, w, l2_reg, l2_reg_rhs, n_shared)
        else:
            x = lstsq(A, b, w, l2_reg, l2_reg_rhs)
        x = tf.squeeze(x, -1)

        # Compute translation
        new_trans = tf.squeeze(mean_b, 1) - tf.linalg.matvec(tf.squeeze(mean_A, 1), x)
        new_shape = x[:, :self.n_betas]

        result = {
            'shape_betas': new_shape,
            'trans': new_trans,
            'relative_orientations': rel_rotmats,
            'kid_factor': None,
            'scale_corr': None,
            'joints': None,
            'vertices': None,
        }

        if self.enable_kid:
            new_kid_factor = x[:, self.n_betas]
            result['kid_factor'] = new_kid_factor
        else:
            new_kid_factor = None

        if scale_target or scale_fit:
            new_scale_corr = x[:, -1] + 1
            if scale_fit:
                result['shape_betas'] = result['shape_betas'] / new_scale_corr[:, tf.newaxis]
                if new_kid_factor is not None:
                    result['kid_factor'] = new_kid_factor / new_scale_corr
            result['scale_corr'] = new_scale_corr

        # Compute vertices/joints if requested
        shape_for_output = result['shape_betas']
        if self.enable_kid and result['kid_factor'] is not None:
            shape_for_output = tf.concat([shape_for_output, result['kid_factor'][:, tf.newaxis]], axis=1)

        if 'joints' in requested_keys:
            result['joints'] = (
                glob_positions_ext[..., 0]
                + tf.einsum('bvcs,bs->bvc', glob_positions_ext[..., 1:], shape_for_output)
                + new_trans[:, tf.newaxis]
            )

        if 'vertices' in requested_keys:
            result['vertices'] = (
                v_posed_ext[..., 0]
                + tf.einsum('bvcs,bs->bvc', v_posed_ext[..., 1:], shape_for_output)
                + new_trans[:, tf.newaxis]
            )

        return result

    def _build_shape_jacobian(
        self,
        glob_rotmats: tf.Tensor,
        batch_size: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Build posed vertices and joints with shape Jacobians."""
        parent_glob_rotmats = tf.concat(
            [
                tf.broadcast_to(tf.eye(3), tf.shape(glob_rotmats[:, :1])),
                tf.gather(glob_rotmats, self.body_model.kintree_parents[1:], axis=1),
            ],
            axis=1,
        )
        rel_rotmats = tf.linalg.matmul(parent_glob_rotmats, glob_rotmats, transpose_a=True)

        # Forward kinematics with shape derivatives
        glob_positions_ext_list = [tf.repeat(self.J_template_ext[tf.newaxis, 0], batch_size, axis=0)]
        for i_joint, i_parent in enumerate(self.body_model.kintree_parents[1:], start=1):
            glob_positions_ext_list.append(
                glob_positions_ext_list[i_parent]
                + tf.einsum(
                    'bCc,cs->bCs',
                    glob_rotmats[:, i_parent],
                    self.J_template_ext[i_joint] - self.J_template_ext[i_parent],
                )
            )
        glob_positions_ext = tf.stack(glob_positions_ext_list, axis=1)
        translations_ext = glob_positions_ext - tf.einsum(
            'bjCc,jcs->bjCs', glob_rotmats, self.J_template_ext
        )

        # Pose blend shapes
        rot_params = tf.reshape(rel_rotmats[:, 1:], [-1, (self.body_model.num_joints - 1) * 9])
        v_posed = self.v_template + tf.einsum('vcp,bp->bvc', self.posedirs, rot_params)
        v_rotated = tf.einsum('bjCc,vj,bvc->bvC', glob_rotmats, self.weights, v_posed)

        # Shape derivatives
        shapedirs = (
            tf.concat([self.shapedirs[:, :, :self.n_betas], self.kid_shapedir[:, :, tf.newaxis]], axis=2)
            if self.enable_kid
            else self.shapedirs[:, :, :self.n_betas]
        )
        v_grad_rotated = tf.einsum('bjCc,lj,lcs->blCs', glob_rotmats, self.weights, shapedirs)

        # Combine position and Jacobian
        v_rotated_ext = tf.concat([v_rotated[:, :, :, tf.newaxis], v_grad_rotated], axis=3)
        v_translations_ext = tf.einsum('vj,bjcs->bvcs', self.weights, translations_ext)
        v_posed_ext = v_translations_ext + v_rotated_ext

        return v_posed_ext, glob_positions_ext, rel_rotmats

    def _setup_regularization(
        self,
        batch_size: tf.Tensor,
        beta_regularizer: float,
        beta_regularizer2: float,
        scale_regularizer: float,
        kid_regularizer: Optional[float],
        beta_regularizer_reference: Optional[tf.Tensor],
        kid_regularizer_reference: Optional[tf.Tensor],
        has_scale: bool,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Setup L2 regularization."""
        # Beta regularization weights
        l2_reg = tf.concat([
            tf.fill([2], tf.cast(beta_regularizer2, tf.float32)),
            tf.fill([self.n_betas - 2], tf.cast(beta_regularizer, tf.float32)),
        ], axis=0)

        # Beta regularization reference
        if beta_regularizer_reference is None:
            l2_reg_ref = tf.zeros([batch_size, self.n_betas], tf.float32)
        else:
            l2_reg_ref = beta_regularizer_reference

        # Kid factor regularization
        if self.enable_kid:
            kid_reg = kid_regularizer if kid_regularizer is not None else beta_regularizer
            l2_reg = tf.concat([l2_reg, tf.constant([kid_reg], tf.float32)], axis=0)
            kid_ref = (
                kid_regularizer_reference
                if kid_regularizer_reference is not None
                else tf.zeros([batch_size], tf.float32)
            )
            l2_reg_ref = tf.concat([l2_reg_ref, kid_ref[:, tf.newaxis]], axis=1)

        # Scale regularization
        if has_scale:
            l2_reg = tf.concat([l2_reg, tf.constant([scale_regularizer], tf.float32)], axis=0)
            l2_reg_ref = tf.concat([l2_reg_ref, tf.zeros([batch_size, 1], tf.float32)], axis=1)

        l2_reg_rhs = (l2_reg * l2_reg_ref)[:, :, tf.newaxis]
        return l2_reg, l2_reg_rhs

    def _fit_global_rotations(
        self,
        target_vertices: tf.Tensor,
        target_joints: Optional[tf.Tensor],
        reference_vertices: tf.Tensor,
        reference_joints: Optional[tf.Tensor],
        vertex_weights: Optional[tf.Tensor],
        joint_weights: Optional[tf.Tensor],
    ) -> tf.Tensor:
        """Fit global rotations via Kabsch algorithm per body part."""
        mesh_weight = 1e-6
        joint_weight = 1 - mesh_weight

        if target_joints is None or reference_joints is None:
            target_joints = self.J_regressor @ target_vertices
            reference_joints = self.J_regressor @ reference_vertices

        glob_rots = []

        for i in range(self.body_model.num_joints):
            if self.is_smpl_family:
                # Toes use foot rotation
                if i == 10:
                    glob_rots.append(glob_rots[7])
                    continue
                elif i == 11:
                    glob_rots.append(glob_rots[8])
                    continue

            selector = self.part_vertex_selectors[i]
            default_body_part = tf.gather(reference_vertices, selector, axis=1)
            estim_body_part = tf.gather(target_vertices, selector, axis=1)
            weights_body_part = (
                tf.gather(vertex_weights, selector, axis=1)[:, :, tf.newaxis] * mesh_weight
                if vertex_weights is not None
                else mesh_weight
            )

            default_joints = tf.gather(reference_joints, self.children_and_self[i], axis=1)
            estim_joints = tf.gather(target_joints, self.children_and_self[i], axis=1)
            weights_joints = (
                tf.gather(joint_weights, self.children_and_self[i], axis=1)[:, :, tf.newaxis] * joint_weight
                if joint_weights is not None
                else joint_weight
            )

            # Center on joint mean
            body_part_mean_ref = tf.reduce_mean(default_joints, axis=1, keepdims=True)
            body_part_mean_tgt = tf.reduce_mean(estim_joints, axis=1, keepdims=True)

            default_points = tf.concat([
                (default_body_part - body_part_mean_ref) * weights_body_part,
                (default_joints - body_part_mean_ref) * weights_joints,
            ], axis=1)

            estim_points = tf.concat([
                estim_body_part - body_part_mean_tgt,
                estim_joints - body_part_mean_tgt,
            ], axis=1)

            glob_rots.append(kabsch(estim_points, default_points))

        return tf.stack(glob_rots, axis=1)

    def _fit_global_rotations_dependent(
        self,
        target_vertices: tf.Tensor,
        target_joints: Optional[tf.Tensor],
        reference_vertices: tf.Tensor,
        reference_joints: tf.Tensor,
        vertex_weights: Optional[tf.Tensor],
        joint_weights: Optional[tf.Tensor],
        glob_rots_prev: tf.Tensor,
        shape_betas: tf.Tensor,
        scale_corr: Optional[tf.Tensor],
        trans: tf.Tensor,
        kid_factor: Optional[tf.Tensor],
    ) -> tf.Tensor:
        """Sequential rotation refinement respecting kinematic chain."""
        true_reference_joints = reference_joints

        if target_joints is None:
            target_joints = self.J_regressor @ target_vertices
            reference_joints = self.J_regressor @ reference_vertices
        if true_reference_joints is None:
            true_reference_joints = reference_joints

        # Compute T-pose joint positions
        j = self.body_model.J_template + tf.einsum(
            'jcs,...s->...jc',
            self.body_model.J_shapedirs[:, :, :self.n_betas],
            shape_betas,
        )
        if kid_factor is not None:
            j = j + tf.einsum('jc,...->...jc', self.body_model.kid_J_shapedir, kid_factor)
        if scale_corr is not None:
            j = j * scale_corr

        # Bone vectors
        j_parent = tf.concat([
            tf.broadcast_to(tf.zeros(3), tf.shape(j[:, :1])),
            tf.gather(j, self.body_model.kintree_parents[1:], axis=1),
        ], axis=1)
        bones = j - j_parent

        glob_rots = []
        glob_positions = []

        for i in range(self.body_model.num_joints):
            # Compute position via FK
            if i == 0:
                glob_position = j[:, i] + trans
            else:
                i_parent = self.body_model.kintree_parents[i]
                glob_position = glob_positions[i_parent] + tf.linalg.matvec(
                    glob_rots[i_parent], bones[:, i]
                )
            glob_positions.append(glob_position)

            if self.is_smpl_family:
                # Toes use foot rotation
                if i == 10:
                    glob_rots.append(glob_rots[7])
                    continue
                elif i == 11:
                    glob_rots.append(glob_rots[8])
                    continue
                elif i not in (1, 2, 4, 5, 7, 8, 16, 17, 18, 19):
                    glob_rots.append(glob_rots_prev[:, i])
                    continue

            # Kabsch alignment for this joint
            vertex_selector = self.part_vertex_selectors[i]
            joint_selector = self.children_and_self[i]

            default_body_part = tf.gather(reference_vertices, vertex_selector, axis=1)
            estim_body_part = tf.gather(target_vertices, vertex_selector, axis=1)
            weights_body_part = (
                tf.gather(vertex_weights, vertex_selector, axis=1)[:, :, tf.newaxis]
                if vertex_weights is not None
                else tf.constant(1.0, tf.float32)
            )

            default_joints_sel = tf.gather(reference_joints, joint_selector, axis=1)
            estim_joints_sel = tf.gather(target_joints, joint_selector, axis=1)
            weights_joints = (
                tf.gather(joint_weights, joint_selector, axis=1)[:, :, tf.newaxis]
                if joint_weights is not None
                else tf.constant(1.0, tf.float32)
            )

            reference_point = glob_position[:, tf.newaxis]
            default_reference_point = true_reference_joints[:, i:i+1]

            default_points = tf.concat([
                (default_body_part - default_reference_point) * weights_body_part,
                (default_joints_sel - default_reference_point) * weights_joints,
            ], axis=1)

            estim_points = tf.concat([
                estim_body_part - reference_point,
                estim_joints_sel - reference_point,
            ], axis=1)

            glob_rots.append(kabsch(estim_points, default_points) @ glob_rots_prev[:, i])

        return tf.stack(glob_rots, axis=1)

    def _fit_scale_and_translation(
        self,
        target_vertices: tf.Tensor,
        reference_vertices: tf.Tensor,
        target_joints: Optional[tf.Tensor],
        reference_joints: tf.Tensor,
        vertex_weights: Optional[tf.Tensor] = None,
        joint_weights: Optional[tf.Tensor] = None,
        scale: bool = False,
    ) -> tuple[Optional[tf.Tensor], tf.Tensor]:
        """Fit scale and translation to align reference to target."""
        if target_joints is None:
            target_both = target_vertices
            reference_both = reference_vertices
            weights_both = (
                vertex_weights
                if vertex_weights is not None
                else tf.ones(tf.shape(target_vertices)[:2], tf.float32)
            )
        else:
            target_both = tf.concat([target_vertices, target_joints], axis=1)
            reference_both = tf.concat([reference_vertices, reference_joints], axis=1)
            if vertex_weights is not None and joint_weights is not None:
                weights_both = tf.concat([vertex_weights, joint_weights], axis=1)
            else:
                weights_both = tf.ones(tf.shape(target_both)[:2], tf.float32)

        weights_both = weights_both / tf.reduce_sum(weights_both, axis=1, keepdims=True)

        weighted_mean_target = tf.reduce_sum(target_both * weights_both[:, :, tf.newaxis], axis=1)
        weighted_mean_reference = tf.reduce_sum(reference_both * weights_both[:, :, tf.newaxis], axis=1)

        if scale:
            target_centered = target_both - weighted_mean_target[:, tf.newaxis]
            reference_centered = reference_both - weighted_mean_reference[:, tf.newaxis]

            ssq_reference = tf.reduce_sum(reference_centered**2 * weights_both[:, :, tf.newaxis], axis=[1, 2])
            ssq_target = tf.reduce_sum(target_centered**2 * weights_both[:, :, tf.newaxis], axis=[1, 2])

            scale_factor = tf.sqrt(ssq_target / ssq_reference)
            trans = weighted_mean_target - scale_factor[:, tf.newaxis] * weighted_mean_reference
            return scale_factor, trans
        else:
            return None, weighted_mean_target - weighted_mean_reference

