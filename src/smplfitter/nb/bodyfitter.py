from __future__ import annotations

from typing import Optional, TYPE_CHECKING
import numpy as np
import numba

from .lstsq import lstsq, lstsq_partial_share
from .rotation import mat2rotvec, rotvec2mat

if TYPE_CHECKING:
    import smplfitter.nb


class BodyFitter:
    """
    Fits body model (SMPL/SMPL-X/SMPL+H) parameters to target vertices and joints.

    Parameters:
        body_model: The body model instance to fit.
        enable_kid: Enables the use of a kid blendshape.
    """

    def __init__(
        self,
        body_model: 'smplfitter.nb.BodyModel',
        enable_kid: bool = False,
    ):
        self.body_model = body_model
        self.n_betas = body_model.shapedirs.shape[2]
        self.enable_kid = enable_kid
        self.is_smpl_family = body_model.model_name.startswith('smpl')

        self.default_mesh_tf = body_model.single()['vertices']

        # Template for joints with shape adjustments
        # Shape: (num_joints, 3, 1 + num_betas [+ 1 if kid])
        J_template_ext_parts = [
            body_model.J_template[:, :, np.newaxis],
            body_model.J_shapedirs,
        ]
        if enable_kid:
            J_template_ext_parts.append(body_model.kid_J_shapedir[:, :, np.newaxis])
        self.J_template_ext = np.concatenate(J_template_ext_parts, axis=2).astype(np.float32)

        # Pre-compute part assignment (which vertices belong to which joint)
        self.part_assignment = np.argmax(body_model.weights, axis=1).astype(np.int64)
        if self.is_smpl_family:
            # Disable rotation of toes separately from feet
            self.part_assignment = np.where(self.part_assignment == 10, 7, self.part_assignment)
            self.part_assignment = np.where(self.part_assignment == 11, 8, self.part_assignment)

        # Toe-copy pairs and refine joints for SMPL-family models
        if self.is_smpl_family:
            self.toe_copy_pairs = np.array([[10, 7], [11, 8]], dtype=np.int64)
            self.refine_joints = np.array([1, 2, 4, 5, 7, 8, 16, 17, 18, 19], dtype=np.int64)
        else:
            self.toe_copy_pairs = np.empty((0, 2), dtype=np.int64)
            self.refine_joints = np.arange(body_model.num_joints, dtype=np.int64)

        # Build children_and_self as a padded array for Numba
        # children_and_self[i] contains joint indices that are children of i (including i)
        children_and_self_list = [[i] for i in range(body_model.num_joints)]
        for i_joint in range(1, body_model.num_joints):
            i_parent = body_model.kintree_parents[i_joint]
            children_and_self_list[i_parent].append(i_joint)

        # Pad to max length and create mask
        max_children = max(len(c) for c in children_and_self_list)
        self.children_and_self = np.zeros((body_model.num_joints, max_children), dtype=np.int64)
        self.children_and_self_count = np.zeros(body_model.num_joints, dtype=np.int64)
        for i, children in enumerate(children_and_self_list):
            self.children_and_self[i, : len(children)] = children
            self.children_and_self_count[i] = len(children)

    def fit(
        self,
        target_vertices: np.ndarray,
        target_joints: Optional[np.ndarray] = None,
        vertex_weights: Optional[np.ndarray] = None,
        joint_weights: Optional[np.ndarray] = None,
        num_iter: int = 1,
        beta_regularizer: float = 1.0,
        beta_regularizer2: float = 0.0,
        scale_regularizer: float = 0.0,
        kid_regularizer: Optional[float] = None,
        share_beta: bool = False,
        final_adjust_rots: bool = True,
        scale_target: bool = False,
        scale_fit: bool = False,
        initial_pose_rotvecs: Optional[np.ndarray] = None,
        initial_shape_betas: Optional[np.ndarray] = None,
        initial_kid_factor: Optional[np.ndarray] = None,
        requested_keys: Optional[tuple] = None,
    ) -> dict:
        """Fits the body model to target vertices and optionally joints."""
        if requested_keys is None:
            requested_keys = ('pose_rotvecs',)

        target_vertices = np.asarray(target_vertices, dtype=np.float32)
        if target_joints is not None:
            target_joints = np.asarray(target_joints, dtype=np.float32)

        # Subtract mean for numerical stability
        if target_joints is None:
            target_mean = np.mean(target_vertices, axis=1)
            target_vertices = target_vertices - target_mean[:, np.newaxis]
        else:
            target_mean = np.mean(np.concatenate([target_vertices, target_joints], axis=1), axis=1)
            target_vertices = target_vertices - target_mean[:, np.newaxis]
            target_joints = target_joints - target_mean[:, np.newaxis]

        if initial_pose_rotvecs is not None or initial_shape_betas is not None:
            initial_forw = self.body_model(
                shape_betas=initial_shape_betas,
                kid_factor=initial_kid_factor,
                pose_rotvecs=initial_pose_rotvecs,
            )
            initial_joints = initial_forw['joints']
            initial_vertices = initial_forw['vertices']
            initial_orientations = initial_forw['orientations']
        else:
            initial_vertices = self.default_mesh_tf[np.newaxis]
            initial_joints = None
            initial_orientations = None

        # Compute regressed joints if needed (match PT: if either is None, recompute both)
        if target_joints is None or initial_joints is None:
            target_joints_for_rot = self.body_model.J_regressor_post_lbs @ target_vertices
            reference_joints_for_rot = self.body_model.J_regressor_post_lbs @ initial_vertices
        else:
            target_joints_for_rot = target_joints
            reference_joints_for_rot = initial_joints

        glob_rotmats = _fit_global_rotations(
            target_vertices,
            target_joints_for_rot,
            initial_vertices,
            reference_joints_for_rot,
            vertex_weights,
            joint_weights,
            self.part_assignment,
            self.children_and_self,
            self.children_and_self_count,
            self.body_model.num_joints,
            self.toe_copy_pairs,
        )

        if initial_orientations is not None:
            glob_rotmats = _batched_matmul_4d(glob_rotmats, initial_orientations)

        parent_indices = self.body_model.kintree_parents[1:]

        for i in range(num_iter - 1):
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
                requested_keys=('vertices', 'joints')
                if target_joints is not None
                else ('vertices',),
            )
            ref_verts = result['vertices']
            ref_joints = result.get('joints')

            if target_joints is None:
                target_joints_for_rot = self.body_model.J_regressor_post_lbs @ target_vertices
            else:
                target_joints_for_rot = target_joints

            if ref_joints is None:
                ref_joints_for_rot = self.body_model.J_regressor_post_lbs @ ref_verts
            else:
                ref_joints_for_rot = ref_joints

            delta_rot = _fit_global_rotations(
                target_vertices,
                target_joints_for_rot,
                ref_verts,
                ref_joints_for_rot,
                vertex_weights,
                joint_weights,
                self.part_assignment,
                self.children_and_self,
                self.children_and_self_count,
                self.body_model.num_joints,
                self.toe_copy_pairs,
            )
            glob_rotmats = _batched_matmul_4d(delta_rot, glob_rotmats)

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
            requested_keys=('vertices', 'joints')
            if target_joints is not None or final_adjust_rots
            else ('vertices',),
        )

        ref_verts = result['vertices']
        ref_joints = result.get('joints')
        ref_shape = result['shape_betas']
        ref_trans = result['trans']
        ref_kid_factor = result.get('kid_factor')
        ref_scale_corr = result.get('scale_corr')

        if final_adjust_rots and ref_joints is not None:
            if scale_target and ref_scale_corr is not None:
                factor = ref_scale_corr[:, np.newaxis, np.newaxis]
                glob_rotmats = self._fit_global_rotations_dependent(
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
                factor = ref_scale_corr[:, np.newaxis, np.newaxis]
                glob_rotmats = self._fit_global_rotations_dependent(
                    target_vertices,
                    target_joints,
                    factor * ref_verts + (1 - factor) * ref_trans[:, np.newaxis],
                    factor * ref_joints + (1 - factor) * ref_trans[:, np.newaxis],
                    vertex_weights,
                    joint_weights,
                    glob_rotmats,
                    ref_shape,
                    ref_scale_corr,
                    ref_trans,
                    ref_kid_factor,
                )
            else:
                glob_rotmats = self._fit_global_rotations_dependent(
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

        # Add mean back (scaled appropriately if using scale correction)
        if scale_target and ref_scale_corr is not None:
            result['trans'] = ref_trans + target_mean * ref_scale_corr[:, np.newaxis]
        elif scale_fit and ref_scale_corr is not None:
            result['trans'] = ref_trans + target_mean / ref_scale_corr[:, np.newaxis]
        else:
            result['trans'] = ref_trans + target_mean

        result['orientations'] = glob_rotmats

        # Provide requested rotation formats
        if 'relative_orientations' in requested_keys or 'pose_rotvecs' in requested_keys:
            result['relative_orientations'] = _compute_relative_orientations(
                glob_rotmats, parent_indices
            )

        if 'pose_rotvecs' in requested_keys:
            rel_ori = result['relative_orientations']
            rotvecs = mat2rotvec(rel_ori)
            result['pose_rotvecs'] = rotvecs.reshape(rotvecs.shape[0], -1)

        # Remove internal keys
        if 'vertices' in result:
            del result['vertices']
        if 'joints' in result:
            del result['joints']

        return result

    def fit_with_known_pose(
        self,
        pose_rotvecs: np.ndarray,
        target_vertices: np.ndarray,
        target_joints: Optional[np.ndarray] = None,
        vertex_weights: Optional[np.ndarray] = None,
        joint_weights: Optional[np.ndarray] = None,
        beta_regularizer: float = 1.0,
        beta_regularizer2: float = 0.0,
        scale_regularizer: float = 0.0,
        kid_regularizer: Optional[float] = None,
        share_beta: bool = False,
        scale_target: bool = False,
        scale_fit: bool = False,
        beta_regularizer_reference: Optional[np.ndarray] = None,
        kid_regularizer_reference: Optional[np.ndarray] = None,
        requested_keys: tuple = (),
    ) -> dict[str, np.ndarray]:
        """Fits the body shape and translation with known output pose."""
        # Subtract mean for numerical stability
        if target_joints is None:
            target_mean = np.mean(target_vertices, axis=1)
            target_vertices = target_vertices - target_mean[:, np.newaxis]
        else:
            target_mean = np.mean(np.concatenate([target_vertices, target_joints], axis=1), axis=1)
            target_vertices = target_vertices - target_mean[:, np.newaxis]
            target_joints = target_joints - target_mean[:, np.newaxis]

        # Convert rotation vectors to global rotation matrices
        rel_rotmats = rotvec2mat(pose_rotvecs.reshape(-1, self.body_model.num_joints, 3))
        glob_rotmats = _rel_to_glob_rotmats(rel_rotmats, self.body_model.kintree_parents)

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
            beta_regularizer_reference=beta_regularizer_reference,
            kid_regularizer_reference=kid_regularizer_reference,
        )

        result['trans'] = result['trans'] + target_mean
        result.pop('vertices', None)
        result.pop('joints', None)
        result.pop('relative_orientations', None)
        return {k: v for k, v in result.items() if v is not None}

    def fit_with_known_shape(
        self,
        shape_betas: np.ndarray,
        target_vertices: np.ndarray,
        target_joints: Optional[np.ndarray] = None,
        vertex_weights: Optional[np.ndarray] = None,
        joint_weights: Optional[np.ndarray] = None,
        kid_factor: Optional[np.ndarray] = None,
        num_iter: int = 1,
        final_adjust_rots: bool = True,
        initial_pose_rotvecs: Optional[np.ndarray] = None,
        scale_fit: bool = False,
        requested_keys: tuple = (),
    ) -> dict[str, np.ndarray]:
        """Fits the body pose and translation with known shape parameters."""
        if not requested_keys:
            requested_keys = ('pose_rotvecs',)

        # Slice shape_betas to n_betas
        shape_betas = shape_betas[:, : self.n_betas]
        batch_size = shape_betas.shape[0]

        # Subtract mean for numerical stability
        if target_joints is None:
            target_mean = np.mean(target_vertices, axis=1)
            target_vertices = target_vertices - target_mean[:, np.newaxis]
        else:
            target_mean = np.mean(np.concatenate([target_vertices, target_joints], axis=1), axis=1)
            target_vertices = target_vertices - target_mean[:, np.newaxis]
            target_joints = target_joints - target_mean[:, np.newaxis]

        # Initialize from shape
        initial_forw = self.body_model(
            shape_betas=shape_betas, kid_factor=kid_factor, pose_rotvecs=initial_pose_rotvecs
        )
        initial_joints = initial_forw['joints']
        initial_vertices = initial_forw['vertices']

        # Compute regressed joints if needed
        if target_joints is None:
            target_joints_for_rot = self.body_model.J_regressor_post_lbs @ target_vertices
            reference_joints_for_rot = self.body_model.J_regressor_post_lbs @ initial_vertices
        else:
            target_joints_for_rot = target_joints
            reference_joints_for_rot = initial_joints

        glob_rotmats = (
            _fit_global_rotations(
                target_vertices,
                target_joints_for_rot,
                initial_vertices,
                reference_joints_for_rot,
                vertex_weights,
                joint_weights,
                self.part_assignment,
                self.children_and_self,
                self.children_and_self_count,
                self.body_model.num_joints,
                self.toe_copy_pairs,
            )
            @ initial_forw['orientations']
        )

        # Iterative refinement
        for _ in range(num_iter - 1):
            forw = self.body_model(
                glob_rotmats=glob_rotmats, shape_betas=shape_betas, kid_factor=kid_factor
            )
            ref_verts = forw['vertices']
            if target_joints is None:
                ref_joints_for_rot = self.body_model.J_regressor_post_lbs @ ref_verts
            else:
                ref_joints_for_rot = forw['joints']
            glob_rotmats = (
                _fit_global_rotations(
                    target_vertices,
                    target_joints_for_rot,
                    ref_verts,
                    ref_joints_for_rot,
                    vertex_weights,
                    joint_weights,
                    self.part_assignment,
                    self.children_and_self,
                    self.children_and_self_count,
                    self.body_model.num_joints,
                    self.toe_copy_pairs,
                )
                @ glob_rotmats
            )

        # Final forward pass
        forw = self.body_model(
            glob_rotmats=glob_rotmats, shape_betas=shape_betas, kid_factor=kid_factor
        )
        ref_verts = forw['vertices']
        ref_joints = forw['joints']

        # Compute translation (and optionally scale)
        ref_scale_corr, trans = fit_scale_and_translation(
            target_vertices,
            ref_verts,
            target_joints,
            ref_joints,
            vertex_weights,
            joint_weights,
            scale=scale_fit,
        )

        # Optional final rotation adjustment
        if final_adjust_rots:
            if scale_fit and ref_scale_corr is not None:
                glob_rotmats = self._fit_global_rotations_dependent(
                    target_vertices,
                    target_joints,
                    ref_scale_corr[:, np.newaxis, np.newaxis] * ref_verts + trans[:, np.newaxis],
                    ref_scale_corr[:, np.newaxis, np.newaxis] * ref_joints + trans[:, np.newaxis],
                    vertex_weights,
                    joint_weights,
                    glob_rotmats,
                    shape_betas,
                    ref_scale_corr,
                    trans,
                    kid_factor,
                )
            else:
                glob_rotmats = self._fit_global_rotations_dependent(
                    target_vertices,
                    target_joints,
                    ref_verts + trans[:, np.newaxis],
                    ref_joints + trans[:, np.newaxis],
                    vertex_weights,
                    joint_weights,
                    glob_rotmats,
                    shape_betas,
                    None,  # scale_corr
                    trans,
                    kid_factor,
                )

        # Build result
        result: dict[str, np.ndarray] = {
            'shape_betas': shape_betas,
            'trans': trans + target_mean,
            'orientations': glob_rotmats,
        }
        if kid_factor is not None:
            result['kid_factor'] = kid_factor
        if scale_fit and ref_scale_corr is not None:
            result['scale_corr'] = ref_scale_corr

        # Compute relative orientations and pose_rotvecs if requested
        parent_indices = self.body_model.kintree_parents[1:]
        if 'relative_orientations' in requested_keys or 'pose_rotvecs' in requested_keys:
            result['relative_orientations'] = _compute_relative_orientations(
                glob_rotmats, parent_indices
            )

        if 'pose_rotvecs' in requested_keys:
            rotvecs = mat2rotvec(result['relative_orientations'])
            result['pose_rotvecs'] = rotvecs.reshape(batch_size, -1)

        return result

    def _fit_shape(
        self,
        glob_rotmats: np.ndarray,
        target_vertices: np.ndarray,
        target_joints: Optional[np.ndarray] = None,
        vertex_weights: Optional[np.ndarray] = None,
        joint_weights: Optional[np.ndarray] = None,
        beta_regularizer: float = 1.0,
        beta_regularizer2: float = 0.0,
        scale_regularizer: float = 0.0,
        kid_regularizer: Optional[float] = None,
        share_beta: bool = False,
        scale_target: bool = False,
        scale_fit: bool = False,
        beta_regularizer_reference: Optional[np.ndarray] = None,
        kid_regularizer_reference: Optional[np.ndarray] = None,
        requested_keys: tuple = (),
    ) -> dict:
        """Internal method to fit shape given global rotations."""
        if scale_target and scale_fit:
            raise ValueError('Only one of scale_target and scale_fit can be True')

        batch_size = target_vertices.shape[0]
        parent_indices = self.body_model.kintree_parents[1:]

        # Compute parent relative rotations
        rel_rotmats = _compute_relative_orientations(glob_rotmats, parent_indices)

        # Prepare shapedirs
        if self.enable_kid:
            shapedirs = np.concatenate(
                [self.body_model.shapedirs, self.body_model.kid_shapedir[:, :, np.newaxis]],
                axis=2,
            ).astype(np.float32)
        else:
            shapedirs = self.body_model.shapedirs

        # Compute all the heavy stuff in JIT
        glob_positions_ext, v_posed_posed_ext = _fit_shape_core(
            glob_rotmats,
            rel_rotmats,
            self.J_template_ext,
            self.body_model.kintree_parents,
            self.body_model.v_template,
            self.body_model.posedirs,
            self.body_model.weights,
            shapedirs,
            batch_size,
        )

        # Combine vertices and joints
        if target_joints is None:
            target_both = target_vertices
            pos_both = v_posed_posed_ext[..., 0]
            jac_pos_both = v_posed_posed_ext[..., 1:]
        else:
            target_both = np.concatenate([target_vertices, target_joints], axis=1)
            pos_both = np.concatenate(
                [v_posed_posed_ext[..., 0], glob_positions_ext[..., 0]], axis=1
            )
            jac_pos_both = np.concatenate(
                [v_posed_posed_ext[..., 1:], glob_positions_ext[..., 1:]], axis=1
            )

        # Build design matrix
        if scale_target:
            A = np.concatenate([jac_pos_both, -target_both[:, :, :, np.newaxis]], axis=3)
        elif scale_fit:
            A = np.concatenate([jac_pos_both, pos_both[:, :, :, np.newaxis]], axis=3)
        else:
            A = jac_pos_both

        b = target_both - pos_both

        # Weights
        if target_joints is not None and vertex_weights is not None and joint_weights is not None:
            weights = np.concatenate([vertex_weights, joint_weights], axis=1)
        elif target_joints is None and vertex_weights is not None:
            weights = vertex_weights
        else:
            weights = np.ones(A.shape[:2], dtype=np.float32)

        n_params = (
            self.n_betas + (1 if self.enable_kid else 0) + (1 if scale_target or scale_fit else 0)
        )

        # Weighted mean centering
        weights_sum = np.sum(weights, axis=1, keepdims=True)
        weights_sum = np.where(weights_sum == 0, 1.0, weights_sum)
        mean_A = (
            np.sum(weights[:, :, np.newaxis, np.newaxis] * A, axis=1, keepdims=True)
            / weights_sum[:, :, np.newaxis, np.newaxis]
        )
        mean_b = (
            np.sum(weights[:, :, np.newaxis] * b, axis=1, keepdims=True)
            / weights_sum[:, :, np.newaxis]
        )
        mean_A = np.nan_to_num(mean_A)
        mean_b = np.nan_to_num(mean_b)
        A = A - mean_A
        b = b - mean_b

        A = A.reshape(batch_size, -1, n_params)
        b = b.reshape(batch_size, -1, 1)
        w = np.repeat(weights.reshape(batch_size, -1), 3, axis=1)

        # Regularization
        l2_regularizer_all = np.concatenate(
            [
                np.full((2,), beta_regularizer2, dtype=np.float32),
                np.full((self.n_betas - 2,), beta_regularizer, dtype=np.float32),
            ]
        )

        if beta_regularizer_reference is None:
            l2_regularizer_reference_all = np.zeros((batch_size, self.n_betas), dtype=np.float32)
        else:
            l2_regularizer_reference_all = beta_regularizer_reference.astype(np.float32)

        if self.enable_kid:
            if kid_regularizer is None:
                kid_regularizer = beta_regularizer
            l2_regularizer_all = np.concatenate(
                [l2_regularizer_all, np.array([kid_regularizer], dtype=np.float32)]
            )
            if kid_regularizer_reference is None:
                kid_ref = np.zeros((batch_size, 1), dtype=np.float32)
            else:
                kid_ref = kid_regularizer_reference[:, np.newaxis].astype(np.float32)
            l2_regularizer_reference_all = np.concatenate(
                [l2_regularizer_reference_all, kid_ref], axis=1
            )

        if scale_target or scale_fit:
            l2_regularizer_all = np.concatenate(
                [l2_regularizer_all, np.array([scale_regularizer], dtype=np.float32)]
            )
            l2_regularizer_reference_all = np.concatenate(
                [l2_regularizer_reference_all, np.zeros((batch_size, 1), dtype=np.float32)], axis=1
            )

        l2_regularizer_rhs = (l2_regularizer_all * l2_regularizer_reference_all)[:, :, np.newaxis]

        # Solve
        if share_beta:
            x = lstsq_partial_share(
                A,
                b,
                w,
                l2_regularizer_all,
                l2_regularizer_rhs,
                n_shared=self.n_betas + (1 if self.enable_kid else 0),
            )
        else:
            x = lstsq(A, b, w, l2_regularizer_all, l2_regularizer_rhs)

        x = x.squeeze(-1)
        new_trans = mean_b.squeeze(1) - np.matmul(mean_A.squeeze(1), x[:, :, np.newaxis]).squeeze(
            -1
        )
        new_shape = x[:, : self.n_betas]

        result = dict(
            shape_betas=new_shape,
            trans=new_trans,
            relative_orientations=rel_rotmats,
        )

        new_kid_factor = None
        if self.enable_kid:
            new_kid_factor = x[:, self.n_betas]
            result['kid_factor'] = new_kid_factor

        new_scale_corr = None
        if scale_target or scale_fit:
            new_scale_corr = x[:, -1] + 1
            if scale_fit and new_scale_corr is not None:
                new_shape = new_shape / new_scale_corr[:, np.newaxis]
                result['shape_betas'] = new_shape
                if new_kid_factor is not None:
                    new_kid_factor = new_kid_factor / new_scale_corr
                    result['kid_factor'] = new_kid_factor
            result['scale_corr'] = new_scale_corr

        # Compute output vertices/joints if requested
        if self.enable_kid and new_kid_factor is not None:
            new_shape_full = np.concatenate([new_shape, new_kid_factor[:, np.newaxis]], axis=1)
        else:
            new_shape_full = new_shape

        if 'joints' in requested_keys:
            result['joints'] = _apply_shape_coeffs(glob_positions_ext, new_shape_full, new_trans)

        if 'vertices' in requested_keys:
            result['vertices'] = _apply_shape_coeffs(v_posed_posed_ext, new_shape_full, new_trans)

        return result

    def _fit_global_rotations_dependent(
        self,
        target_vertices: np.ndarray,
        target_joints: Optional[np.ndarray],
        reference_vertices: np.ndarray,
        reference_joints: np.ndarray,
        vertex_weights: Optional[np.ndarray],
        joint_weights: Optional[np.ndarray],
        glob_rots_prev: np.ndarray,
        shape_betas: np.ndarray,
        scale_corr: Optional[np.ndarray],
        trans: np.ndarray,
        kid_factor: Optional[np.ndarray],
    ) -> np.ndarray:
        """Fit global rotations dependent on skeleton structure."""
        true_reference_joints = reference_joints
        if target_joints is None:
            target_joints = self.body_model.J_regressor_post_lbs @ target_vertices
        if reference_joints is None:
            reference_joints = self.body_model.J_regressor_post_lbs @ reference_vertices
        if true_reference_joints is None:
            true_reference_joints = reference_joints

        # Compute joint positions from shape
        j = self.body_model.J_template + np.einsum(
            'jcs,...s->...jc',
            self.body_model.J_shapedirs[:, :, : self.n_betas],
            shape_betas[:, : self.n_betas],
        )
        if kid_factor is not None:
            j = j + np.einsum('jc,...->...jc', self.body_model.kid_J_shapedir, kid_factor)

        if scale_corr is not None:
            j = j * scale_corr[:, np.newaxis, np.newaxis]

        return _fit_global_rotations_dependent_core(
            target_vertices,
            target_joints,
            reference_vertices,
            reference_joints,
            true_reference_joints,
            vertex_weights,
            joint_weights,
            glob_rots_prev,
            j,
            trans,
            self.body_model.kintree_parents,
            self.part_assignment,
            self.children_and_self,
            self.children_and_self_count,
            self.body_model.num_joints,
            self.toe_copy_pairs,
            self.refine_joints,
        )


def fit_scale_and_translation(
    target_vertices,
    reference_vertices,
    target_joints,
    reference_joints,
    vertex_weights=None,
    joint_weights=None,
    scale=False,
):
    if target_joints is None or reference_joints is None:
        target_both = target_vertices
        reference_both = reference_vertices
        if vertex_weights is not None:
            weights_both = vertex_weights
        else:
            weights_both = np.ones(target_vertices.shape[:2], dtype=np.float32)
    else:
        target_both = np.concatenate([target_vertices, target_joints], axis=1)
        reference_both = np.concatenate([reference_vertices, reference_joints], axis=1)
        if vertex_weights is not None and joint_weights is not None:
            weights_both = np.concatenate([vertex_weights, joint_weights], axis=1)
        else:
            weights_both = np.ones(
                (target_vertices.shape[0], target_vertices.shape[1] + target_joints.shape[1]),
                dtype=np.float32,
            )

    weights_both = weights_both / np.sum(weights_both, axis=1, keepdims=True)

    weighted_mean_target = np.sum(target_both * weights_both[..., np.newaxis], axis=1)
    weighted_mean_reference = np.sum(reference_both * weights_both[..., np.newaxis], axis=1)

    if scale:
        target_centered = target_both - weighted_mean_target[:, np.newaxis]
        reference_centered = reference_both - weighted_mean_reference[:, np.newaxis]
        ssq_reference = np.sum(reference_centered**2 * weights_both[..., np.newaxis], axis=(1, 2))
        ssq_target = np.sum(target_centered**2 * weights_both[..., np.newaxis], axis=(1, 2))
        scale_factor = np.sqrt(ssq_target / ssq_reference)
        trans = weighted_mean_target - scale_factor[:, np.newaxis] * weighted_mean_reference
    else:
        scale_factor = None
        trans = weighted_mean_target - weighted_mean_reference

    return scale_factor, trans


# ============== Global Numba-compiled functions ==============


@numba.njit(error_model='numpy', cache=True)
def _compute_relative_orientations(glob_rotmats, parent_indices):
    """Compute parent-relative rotations from global rotations."""
    batch_size = glob_rotmats.shape[0]
    num_joints = glob_rotmats.shape[1]

    # Build parent global rotations (identity for root)
    parent_glob = np.empty((batch_size, num_joints, 3, 3), dtype=np.float32)
    for b in range(batch_size):
        # Root's parent is identity
        parent_glob[b, 0, 0, 0] = 1.0
        parent_glob[b, 0, 0, 1] = 0.0
        parent_glob[b, 0, 0, 2] = 0.0
        parent_glob[b, 0, 1, 0] = 0.0
        parent_glob[b, 0, 1, 1] = 1.0
        parent_glob[b, 0, 1, 2] = 0.0
        parent_glob[b, 0, 2, 0] = 0.0
        parent_glob[b, 0, 2, 1] = 0.0
        parent_glob[b, 0, 2, 2] = 1.0
        # Other joints
        for j in range(len(parent_indices)):
            pi = parent_indices[j]
            for r in range(3):
                for c in range(3):
                    parent_glob[b, j + 1, r, c] = glob_rotmats[b, pi, r, c]

    # Compute relative: parent.T @ global
    rel_rotmats = np.empty((batch_size, num_joints, 3, 3), dtype=np.float32)
    for b in range(batch_size):
        for j in range(num_joints):
            for i in range(3):
                for k in range(3):
                    val = np.float32(0.0)
                    for c in range(3):
                        val += parent_glob[b, j, c, i] * glob_rotmats[b, j, c, k]
                    rel_rotmats[b, j, i, k] = val
    return rel_rotmats


@numba.njit(error_model='numpy', cache=True)
def _rel_to_glob_rotmats(rel_rotmats, kintree_parents):
    """Convert parent-relative rotations to global rotations."""
    batch_size = rel_rotmats.shape[0]
    num_joints = rel_rotmats.shape[1]
    glob_rotmats = np.empty((batch_size, num_joints, 3, 3), dtype=np.float32)

    # Root is same as relative
    for b in range(batch_size):
        for r in range(3):
            for c in range(3):
                glob_rotmats[b, 0, r, c] = rel_rotmats[b, 0, r, c]

    # Forward kinematics: glob[i] = glob[parent[i]] @ rel[i]
    for i_joint in range(1, num_joints):
        i_parent = kintree_parents[i_joint]
        for b in range(batch_size):
            for r in range(3):
                for c in range(3):
                    val = np.float32(0.0)
                    for k in range(3):
                        val += glob_rotmats[b, i_parent, r, k] * rel_rotmats[b, i_joint, k, c]
                    glob_rotmats[b, i_joint, r, c] = val

    return glob_rotmats


@numba.njit(error_model='numpy', cache=True)
def _compute_glob_positions_ext(glob_rotmats, J_template_ext, kintree_parents, batch_size):
    """Compute global positions with shape gradients."""
    num_joints = J_template_ext.shape[0]
    n_shape_params = J_template_ext.shape[2]

    glob_positions_ext = np.empty((batch_size, num_joints, 3, n_shape_params), dtype=np.float32)

    # Initialize root joint
    for b in range(batch_size):
        for c in range(3):
            for s in range(n_shape_params):
                glob_positions_ext[b, 0, c, s] = J_template_ext[0, c, s]

    # Forward kinematics for other joints
    for i_joint in range(1, num_joints):
        i_parent = kintree_parents[i_joint]
        for b in range(batch_size):
            for C in range(3):
                for s in range(n_shape_params):
                    # Rotate bone offset
                    val = np.float32(0.0)
                    for c in range(3):
                        bone = J_template_ext[i_joint, c, s] - J_template_ext[i_parent, c, s]
                        val += glob_rotmats[b, i_parent, C, c] * bone
                    glob_positions_ext[b, i_joint, C, s] = (
                        glob_positions_ext[b, i_parent, C, s] + val
                    )

    return glob_positions_ext


@numba.njit(error_model='numpy', cache=True)
def _batched_matmul_4d(a, b):
    """Batched matrix multiply for (batch, n, 3, 3) arrays."""
    batch_size = a.shape[0]
    n = a.shape[1]
    result = np.empty((batch_size, n, 3, 3), dtype=np.float32)
    for bi in range(batch_size):
        for ni in range(n):
            for i in range(3):
                for j in range(3):
                    val = np.float32(0.0)
                    for k in range(3):
                        val += a[bi, ni, i, k] * b[bi, ni, k, j]
                    result[bi, ni, i, j] = val
    return result


@numba.njit(error_model='numpy', cache=True)
def _fit_global_rotations(
    target_vertices,
    target_joints,
    reference_vertices,
    reference_joints,
    vertex_weights,
    joint_weights,
    part_assignment,
    children_and_self,
    children_and_self_count,
    num_joints,
    toe_copy_pairs,
):
    """Fit global rotations using Kabsch algorithm."""
    batch_size = target_vertices.shape[0]
    ref_batch = reference_vertices.shape[0]

    mesh_weight = np.float32(1e-6)
    joint_weight = np.float32(1.0) - mesh_weight

    glob_rots = np.empty((batch_size, num_joints, 3, 3), dtype=np.float32)

    for i in range(num_joints):
        # Check if this joint should copy rotation from another (e.g. toes from feet)
        copy_src = -1
        for p in range(toe_copy_pairs.shape[0]):
            if toe_copy_pairs[p, 0] == i:
                copy_src = toe_copy_pairs[p, 1]
                break
        if copy_src >= 0:
            for b in range(batch_size):
                for r in range(3):
                    for c in range(3):
                        glob_rots[b, i, r, c] = glob_rots[b, copy_src, r, c]
            continue

        # Get children for this joint
        n_children = children_and_self_count[i]
        children = children_and_self[i, :n_children]

        # Compute reference joint mean (for centering)
        ref_joint_mean = np.zeros((ref_batch, 3), dtype=np.float32)
        for b in range(ref_batch):
            for ci in range(n_children):
                child_idx = children[ci]
                for c in range(3):
                    ref_joint_mean[b, c] += reference_joints[b, child_idx, c]
            for c in range(3):
                ref_joint_mean[b, c] /= n_children

        # Compute target joint mean
        target_joint_mean = np.zeros((batch_size, 3), dtype=np.float32)
        for b in range(batch_size):
            for ci in range(n_children):
                child_idx = children[ci]
                for c in range(3):
                    target_joint_mean[b, c] += target_joints[b, child_idx, c]
            for c in range(3):
                target_joint_mean[b, c] /= n_children

        # Run Kabsch for each batch element
        for b in range(batch_size):
            rb = b if ref_batch > 1 else 0

            # Compute A = X.T @ Y where X is target, Y is reference (weighted)
            A = np.zeros((3, 3), dtype=np.float32)

            # Add vertex contributions
            n_verts = part_assignment.shape[0]
            for v in range(n_verts):
                if part_assignment[v] == i:
                    for r in range(3):
                        x_val = target_vertices[b, v, r] - target_joint_mean[b, r]
                        for c in range(3):
                            y_val = (
                                reference_vertices[rb, v, c] - ref_joint_mean[rb, c]
                            ) * mesh_weight
                            A[r, c] += x_val * y_val

            # Add joint contributions
            for ci in range(n_children):
                child_idx = children[ci]
                for r in range(3):
                    x_val = target_joints[b, child_idx, r] - target_joint_mean[b, r]
                    for c in range(3):
                        y_val = (
                            reference_joints[rb, child_idx, c] - ref_joint_mean[rb, c]
                        ) * joint_weight
                        A[r, c] += x_val * y_val

            # SVD and rotation
            U, _, Vh = np.linalg.svd(A)
            T = U @ Vh
            if np.linalg.det(T) < 0:
                T = T - np.float32(2.0) * U[:, -1:] @ Vh[-1:, :]

            for r in range(3):
                for c in range(3):
                    glob_rots[b, i, r, c] = T[r, c]

    return glob_rots


@numba.njit(error_model='numpy', cache=True)
def _fit_global_rotations_dependent_core(
    target_vertices,
    target_joints,
    reference_vertices,
    reference_joints,
    true_reference_joints,
    vertex_weights,
    joint_weights,
    glob_rots_prev,
    j,  # joint positions from shape
    trans,
    kintree_parents,
    part_assignment,
    children_and_self,
    children_and_self_count,
    num_joints,
    toe_copy_pairs,
    refine_joints,
):
    """Core computation for dependent global rotations."""
    batch_size = target_vertices.shape[0]

    # Compute bones
    bones = np.empty_like(j)
    for b in range(batch_size):
        for ji in range(num_joints):
            if ji == 0:
                for c in range(3):
                    bones[b, ji, c] = j[b, ji, c]
            else:
                pi = kintree_parents[ji]
                for c in range(3):
                    bones[b, ji, c] = j[b, ji, c] - j[b, pi, c]

    glob_rots = np.empty((batch_size, num_joints, 3, 3), dtype=np.float32)
    glob_positions = np.empty((batch_size, num_joints, 3), dtype=np.float32)

    for i in range(num_joints):
        # Compute global position
        if i == 0:
            for b in range(batch_size):
                for c in range(3):
                    glob_positions[b, i, c] = j[b, i, c] + trans[b, c]
        else:
            i_parent = kintree_parents[i]
            for b in range(batch_size):
                for c in range(3):
                    val = np.float32(0.0)
                    for k in range(3):
                        val += glob_rots[b, i_parent, c, k] * bones[b, i, k]
                    glob_positions[b, i, c] = glob_positions[b, i_parent, c] + val

        # Check if this joint should copy rotation from another (e.g. toes from feet)
        copy_src = -1
        for p in range(toe_copy_pairs.shape[0]):
            if toe_copy_pairs[p, 0] == i:
                copy_src = toe_copy_pairs[p, 1]
                break
        if copy_src >= 0:
            for b in range(batch_size):
                for r in range(3):
                    for c in range(3):
                        glob_rots[b, i, r, c] = glob_rots[b, copy_src, r, c]
            continue

        # Check if we should refine this joint
        should_refine = False
        for ri in range(len(refine_joints)):
            if refine_joints[ri] == i:
                should_refine = True
                break

        if not should_refine:
            for b in range(batch_size):
                for r in range(3):
                    for c in range(3):
                        glob_rots[b, i, r, c] = glob_rots_prev[b, i, r, c]
            continue

        # Get children
        n_children = children_and_self_count[i]
        children = children_and_self[i, :n_children]

        # Compute Kabsch for each batch element
        for b in range(batch_size):
            A = np.zeros((3, 3), dtype=np.float32)

            # Add vertex contributions
            n_verts = part_assignment.shape[0]
            for v in range(n_verts):
                if part_assignment[v] == i:
                    w = np.float32(1.0)
                    for r in range(3):
                        x_val = target_vertices[b, v, r] - glob_positions[b, i, r]
                        for c in range(3):
                            y_val = (
                                reference_vertices[b, v, c] - true_reference_joints[b, i, c]
                            ) * w
                            A[r, c] += x_val * y_val

            # Add joint contributions
            for ci in range(n_children):
                child_idx = children[ci]
                w = np.float32(1.0)
                for r in range(3):
                    x_val = target_joints[b, child_idx, r] - glob_positions[b, i, r]
                    for c in range(3):
                        y_val = (
                            reference_joints[b, child_idx, c] - true_reference_joints[b, i, c]
                        ) * w
                        A[r, c] += x_val * y_val

            # SVD
            U, _, Vh = np.linalg.svd(A)
            T = U @ Vh
            if np.linalg.det(T) < 0:
                T = T - np.float32(2.0) * U[:, -1:] @ Vh[-1:, :]

            # Multiply by previous rotation
            for r in range(3):
                for c in range(3):
                    val = np.float32(0.0)
                    for k in range(3):
                        val += T[r, k] * glob_rots_prev[b, i, k, c]
                    glob_rots[b, i, r, c] = val

    return glob_rots


@numba.njit(error_model='numpy', cache=True, parallel=True)
def _fit_shape_core(
    glob_rotmats,
    rel_rotmats,
    J_template_ext,
    kintree_parents,
    v_template,
    posedirs,
    weights,
    shapedirs,
    batch_size,
):
    """
    Core computation for shape fitting.
    Replaces multiple einsum operations with explicit loops.
    """
    num_joints = J_template_ext.shape[0]
    n_shape_params = J_template_ext.shape[2]
    num_vertices = v_template.shape[0]
    num_pose_features = (num_joints - 1) * 9

    # 1. Compute glob_positions_ext (forward kinematics with shape gradients)
    glob_positions_ext = np.empty((batch_size, num_joints, 3, n_shape_params), dtype=np.float32)
    for b in range(batch_size):
        for c in range(3):
            for s in range(n_shape_params):
                glob_positions_ext[b, 0, c, s] = J_template_ext[0, c, s]

    for i_joint in range(1, num_joints):
        i_parent = kintree_parents[i_joint]
        for b in range(batch_size):
            for C in range(3):
                for s in range(n_shape_params):
                    val = np.float32(0.0)
                    for c in range(3):
                        bone = J_template_ext[i_joint, c, s] - J_template_ext[i_parent, c, s]
                        val += glob_rotmats[b, i_parent, C, c] * bone
                    glob_positions_ext[b, i_joint, C, s] = (
                        glob_positions_ext[b, i_parent, C, s] + val
                    )

    # 2. Compute translations_ext = glob_positions_ext - einsum('bjCc,jcs->bjCs', glob_rotmats, J_template_ext)
    translations_ext = np.empty((batch_size, num_joints, 3, n_shape_params), dtype=np.float32)
    for b in range(batch_size):
        for j in range(num_joints):
            for C in range(3):
                for s in range(n_shape_params):
                    val = np.float32(0.0)
                    for c in range(3):
                        val += glob_rotmats[b, j, C, c] * J_template_ext[j, c, s]
                    translations_ext[b, j, C, s] = glob_positions_ext[b, j, C, s] - val

    # 3. Compute rot_params from rel_rotmats[:, 1:] reshaped
    rot_params = np.empty((batch_size, num_pose_features), dtype=np.float32)
    for b in range(batch_size):
        idx = 0
        for j in range(1, num_joints):
            for r in range(3):
                for c in range(3):
                    rot_params[b, idx] = rel_rotmats[b, j, r, c]
                    idx += 1

    # 4. Compute v_posed = v_template + einsum('vcp,bp->bvc', posedirs, rot_params)
    v_posed = np.empty((batch_size, num_vertices, 3), dtype=np.float32)
    for v in numba.prange(num_vertices):
        for b in range(batch_size):
            for c in range(3):
                val = v_template[v, c]
                for p in range(num_pose_features):
                    val += posedirs[v, c, p] * rot_params[b, p]
                v_posed[b, v, c] = val

    # 5. Compute v_rotated = einsum('bjCc,vj,bvc->bvC', glob_rotmats, weights, v_posed)
    v_rotated = np.empty((batch_size, num_vertices, 3), dtype=np.float32)
    for v in numba.prange(num_vertices):
        for b in range(batch_size):
            for C in range(3):
                val = np.float32(0.0)
                for j in range(num_joints):
                    w = weights[v, j]
                    if w != 0.0:
                        for c in range(3):
                            val += glob_rotmats[b, j, C, c] * w * v_posed[b, v, c]
                v_rotated[b, v, C] = val

    # 6. Compute v_grad_rotated = einsum('bjCc,lj,lcs->blCs', glob_rotmats, weights, shapedirs)
    n_shape_coeffs = shapedirs.shape[2]
    v_grad_rotated = np.empty((batch_size, num_vertices, 3, n_shape_coeffs), dtype=np.float32)
    for v in numba.prange(num_vertices):
        for b in range(batch_size):
            for C in range(3):
                for s in range(n_shape_coeffs):
                    val = np.float32(0.0)
                    for j in range(num_joints):
                        w = weights[v, j]
                        if w != 0.0:
                            for c in range(3):
                                val += glob_rotmats[b, j, C, c] * w * shapedirs[v, c, s]
                    v_grad_rotated[b, v, C, s] = val

    # 7. Compute v_translations_ext = einsum('vj,bjcs->bvcs', weights, translations_ext)
    v_translations_ext = np.empty((batch_size, num_vertices, 3, n_shape_params), dtype=np.float32)
    for v in numba.prange(num_vertices):
        for b in range(batch_size):
            for c in range(3):
                for s in range(n_shape_params):
                    val = np.float32(0.0)
                    for j in range(num_joints):
                        w = weights[v, j]
                        if w != 0.0:
                            val += w * translations_ext[b, j, c, s]
                    v_translations_ext[b, v, c, s] = val

    # 8. Combine: v_posed_posed_ext = v_translations_ext + [v_rotated[..., newaxis], v_grad_rotated]
    # Shape: (batch, num_vertices, 3, 1 + n_shape_coeffs)
    total_params = 1 + n_shape_coeffs
    v_posed_posed_ext = np.empty((batch_size, num_vertices, 3, total_params), dtype=np.float32)
    for v in numba.prange(num_vertices):
        for b in range(batch_size):
            for c in range(3):
                # First slot is v_rotated + v_translations_ext[..., 0]
                v_posed_posed_ext[b, v, c, 0] = v_translations_ext[b, v, c, 0] + v_rotated[b, v, c]
                # Rest is v_grad_rotated + v_translations_ext[..., 1:]
                for s in range(n_shape_coeffs):
                    v_posed_posed_ext[b, v, c, 1 + s] = (
                        v_translations_ext[b, v, c, 1 + s] + v_grad_rotated[b, v, c, s]
                    )

    return glob_positions_ext, v_posed_posed_ext


@numba.njit(error_model='numpy', cache=True, parallel=True)
def _apply_shape_coeffs(pos_ext, shape_coeffs, trans):
    """
    Apply shape coefficients: pos_ext[..., 0] + einsum('bvcs,bs->bvc', pos_ext[..., 1:], shape_coeffs) + trans
    """
    batch_size = pos_ext.shape[0]
    num_points = pos_ext.shape[1]
    n_coeffs = shape_coeffs.shape[1]

    result = np.empty((batch_size, num_points, 3), dtype=np.float32)
    for v in numba.prange(num_points):
        for b in range(batch_size):
            for c in range(3):
                val = pos_ext[b, v, c, 0] + trans[b, c]
                for s in range(n_coeffs):
                    val += pos_ext[b, v, c, 1 + s] * shape_coeffs[b, s]
                result[b, v, c] = val
    return result
