from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .lstsq import lstsq, lstsq_partial_share
from .rotation import align_unit_vectors, divide_no_nan, mat2rotvec, proj_SO3, rotvec2mat
from .util import matmul_transp_a

if TYPE_CHECKING:
    import smplfitter.np


class BodyFitter:
    """
    Fits body model (SMPL/SMPL-X/SMPL+H) parameters to lists of target vertices and joints.

    Parameters:
        body_model: The SMPL model instance we wish to fit, of a certain model variant and gender.
        enable_kid: Enables the use of a kid blendshape, allowing for fitting kid shapes as in
            AGORA.
    """

    def __init__(
        self,
        body_model: 'smplfitter.np.BodyModel',
        enable_kid: bool = False,
    ):
        self.body_model = body_model
        self.n_betas = body_model.num_betas
        self.enable_kid = enable_kid
        self.is_smpl_family = body_model.model_name.startswith('smpl')

        part_assignment = np.argmax(body_model.weights, axis=1)
        if self.is_smpl_family:
            part_assignment = np.where(
                part_assignment == 10, np.array(7, dtype=np.int64), part_assignment
            )
            part_assignment = np.where(
                part_assignment == 11, np.array(8, dtype=np.int64), part_assignment
            )
        self.part_assignment = part_assignment
        self.part_vertex_selectors = [
            np.where(part_assignment == i)[0] for i in range(body_model.num_joints)
        ]

        self.default_mesh_tf = body_model.single()['vertices']

        self.J_template_ext = np.concatenate(
            [body_model.J_template.reshape(-1, 3, 1), body_model.J_shapedirs]
            + ([body_model.kid_J_shapedir.reshape(-1, 3, 1)] if enable_kid else []),
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

        # Use model's data directly (already subsetted if vertex_subset was provided to model)
        self.shapedirs = body_model.shapedirs
        self.kid_shapedir = body_model.kid_shapedir
        self.v_template = body_model.v_template
        self.weights = body_model.weights
        self.posedirs = body_model.posedirs
        self.num_vertices = body_model.num_vertices
        self.J_regressor = body_model.J_regressor_post_lbs

        # --- Swing-twist rotation solver precompute (static dispatch and part statistics) ---
        # We estimate the global orientation of each BODY PART (the partition of the mesh
        # by dominant skinning weight; part i belongs to joint i). Each part contains its
        # own joint plus its child joints; parts are bucketed by how many joints they
        # contain, since the joints pin the part's orientation to different degrees:
        #   n >= 3 joints: the joints alone determine the orientation (Kabsch on joints)
        #   n == 2 joints (a bone): the joints pin the bone direction (swing);
        #       the part's vertices resolve the remaining twist about the bone
        #   n == 1 joint (leaf part): the vertices alone determine the orientation
        # SMPL-family toe parts (10, 11) copy the feet (7, 8) and are excluded.
        J = body_model.num_joints
        multi_joint_parts = []
        bone_parts = []
        leaf_parts = []
        for i in range(J):
            if self.is_smpl_family and (i == 10 or i == 11):
                continue
            n = len(self.children_and_self[i])
            if n >= 3:
                multi_joint_parts.append(i)
            elif n == 2:
                bone_parts.append(i)
            else:
                leaf_parts.append(i)
        self.multi_joint_parts = multi_joint_parts
        self.bone_parts = bone_parts
        self.leaf_parts = leaf_parts

        # Joints whose rotation the final adjustment pass refines (must have a bone to
        # re-anchor at the recomputed joint position; others keep their initial rotation).
        if self.is_smpl_family:
            self.adjustable_parts = [1, 2, 4, 5, 7, 8, 16, 17, 18, 19]
        else:
            self.adjustable_parts = [i for i in range(J)]

        # Vertices that participate in some vertex-based statistic (bone twist, leaf Kabsch,
        # or the final adjustment). Parts whose orientation comes from joints alone are
        # skipped where possible.
        stat_parts = sorted(set(bone_parts + leaf_parts + self.adjustable_parts))
        used_mask = np.zeros(body_model.num_vertices, dtype=bool)
        for i in stat_parts:
            used_mask[self.part_vertex_selectors[i]] = True
        used_vertex_indices = np.where(used_mask)[0]
        self.used_vertex_indices = used_vertex_indices

        # One-hot part membership over the used vertices: row j sums vertices of part j.
        part_matrix = np.zeros((J, len(used_vertex_indices)), dtype=np.float32)
        part_matrix[part_assignment[used_vertex_indices], np.arange(len(used_vertex_indices))] = (
            1.0
        )
        self.part_matrix = part_matrix
        self.part_counts = part_matrix.sum(axis=1).reshape(1, J, 1)

        # Children-mean centering matrix: row i averages children_and_self[i] joint positions.
        center_matrix = np.zeros((J, J), dtype=np.float32)
        for i in range(J):
            js = self.children_and_self[i]
            center_matrix[i, js] = 1.0 / len(js)
        self.center_matrix = center_matrix

        # Joint membership per multi-joint part (their orientation is Kabsch-fit from joints).
        mjp_joint_membership = np.zeros((len(multi_joint_parts), J), dtype=np.float32)
        for k, i in enumerate(multi_joint_parts):
            mjp_joint_membership[k, self.children_and_self[i]] = 1.0
        self.mjp_joint_membership = mjp_joint_membership
        self.mjp_joint_counts = mjp_joint_membership.sum(axis=1).reshape(1, -1, 1)
        self.mjp_center_matrix = center_matrix[multi_joint_parts]

        # Bone endpoints (start joint, end joint) per bone part.
        self.bone_pairs = np.array(
            [[self.children_and_self[i][0], self.children_and_self[i][1]] for i in bone_parts],
            dtype=np.int64,
        ).reshape(len(bone_parts), 2)

        # Assembly permutation: R_concat = cat([R_multi, R_leaf, R_bone]) is scattered back
        # to per-part order; SMPL toe parts take the feet slots directly (10 <- 7, 11 <- 8).
        concat_order = multi_joint_parts + leaf_parts + bone_parts
        inverse_perm = [0] * J
        for pos, jj in enumerate(concat_order):
            inverse_perm[jj] = pos
        if self.is_smpl_family:
            inverse_perm[10] = inverse_perm[7]
            inverse_perm[11] = inverse_perm[8]
        self.assemble_indices = np.array(inverse_perm, dtype=np.int64)

    def _part_sums(self, target_vertices, reference_vertices, vertex_weights):
        """Per-part sufficient statistics for cross-covariances, computed loop-free.

        Returns per-part weighted sums over each part's vertices:
        ``raw = sum w t a^T`` (B, J, 3, 3), ``s_t = sum w t`` (B, J, 3),
        ``s_a = sum w a`` (B_ref, J, 3), ``s_w = sum w`` (B or 1, J, 1).
        The centered cross-covariance about any centers (c_t, c_a) then follows as
        ``raw - s_t c_a^T - c_t s_a^T + s_w c_t c_a^T``.
        """
        t = target_vertices[:, self.used_vertex_indices]
        a = reference_vertices[:, self.used_vertex_indices]
        if vertex_weights is not None:
            w = vertex_weights[:, self.used_vertex_indices]
            a = a * w[..., np.newaxis]
            t_sum_side = t * w[..., np.newaxis]
            s_w = self.part_matrix @ w[..., np.newaxis]
        else:
            t_sum_side = t
            s_w = self.part_counts
        B = t.shape[0] if t.shape[0] >= a.shape[0] else a.shape[0]
        outer = (t[..., np.newaxis] * a[..., np.newaxis, :]).reshape(B, t.shape[1], 9)
        raw = (self.part_matrix @ outer).reshape(B, -1, 3, 3)
        s_t = self.part_matrix @ t_sum_side
        s_a = self.part_matrix @ a
        return raw, s_t, s_a, s_w

    def fit(
        self,
        target_vertices,
        target_joints=None,
        vertex_weights=None,
        joint_weights=None,
        num_iter=1,
        beta_regularizer=1,
        beta_regularizer2=0,
        scale_regularizer=0,
        kid_regularizer=None,
        share_beta=False,
        final_adjust_rots=True,
        scale_target=False,
        scale_fit=False,
        initial_pose_rotvecs=None,
        initial_shape_betas=None,
        initial_kid_factor=None,
        allow_nan=True,
        requested_keys=('pose_rotvecs',),
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
            initial_kid_factor: Same as above, but for the kid blendshape factor.
            requested_keys: List of keys specifying which results to return.

        Returns:
            A dictionary containing the following items, based on requested keys
                - **pose_rotvecs** -- Estimated pose in concatenated rotation vector format.
                - **shape_betas** -- Estimated shape parameters (betas).
                - **trans** -- Estimated translation parameters.
                - **joints** -- Estimated joint positions, if requested.
                - **vertices** -- Fitted mesh vertices, if requested.
                - **orientations** -- Global body part orientations as rotation matrices.
                - **relative_orientations** -- Parent-relative body part orientations as rotation \
                    matrices.
                - **kid_factor** -- Estimated kid blendshape factor, if `enable_kid` is True.
                - **scale_corr** -- Estimated scale correction factor, if `scale_target` or \
                    `scale_fit` is True.

        """

        # Subtract mean first for better numerical stability (and add it back later)
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
            initial_vertices = initial_forw['vertices'][:, self.vertex_subset]
            initial_orientations = initial_forw['orientations']
        else:
            initial_joints = self.body_model.J_template[np.newaxis]
            initial_vertices = self.default_mesh_tf[np.newaxis]
            initial_orientations = None

        glob_rotmats = self._fit_global_rotations(
            target_vertices,
            target_joints,
            initial_vertices,
            initial_joints if target_joints is not None else None,
            vertex_weights,
            joint_weights,
        )

        if initial_orientations is not None:
            glob_rotmats = glob_rotmats @ initial_orientations
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
                scale_regularizer=0,
                kid_regularizer=kid_regularizer,
                share_beta=share_beta,
                scale_target=False,
                scale_fit=False,
                beta_regularizer_reference=initial_shape_betas,
                kid_regularizer_reference=initial_kid_factor,
                requested_keys=['vertices'] + (['joints'] if target_joints is not None else []),
            )
            glob_rotmats = (
                self._fit_global_rotations(
                    target_vertices,
                    target_joints,
                    result['vertices'],
                    result['joints'],
                    vertex_weights,
                    joint_weights,
                )
                @ glob_rotmats
            )

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
            requested_keys=['vertices']
            + (['joints'] if target_joints is not None or final_adjust_rots else []),
        )
        if final_adjust_rots:
            ref_verts = result['vertices']
            ref_joints = result['joints']
            ref_trans = result['trans']
            if scale_target:
                factor = result['scale_corr'][:, np.newaxis, np.newaxis]
                glob_rotmats = self._fit_global_rotations_dependent(
                    target_vertices * factor,
                    target_joints * factor if target_joints is not None else None,
                    ref_verts,
                    ref_joints,
                    vertex_weights,
                    joint_weights,
                    glob_rotmats,
                    result['shape_betas'],
                    None,
                    ref_trans,
                    result['kid_factor'],
                )
            elif scale_fit:
                factor = result['scale_corr'][:, np.newaxis, np.newaxis]
                glob_rotmats = self._fit_global_rotations_dependent(
                    target_vertices,
                    target_joints,
                    factor * ref_verts + (1 - factor) * ref_trans[:, np.newaxis],
                    factor * ref_joints + (1 - factor) * ref_trans[:, np.newaxis],
                    vertex_weights,
                    joint_weights,
                    glob_rotmats,
                    result['shape_betas'],
                    result['scale_corr'],
                    ref_trans,
                    result['kid_factor'],
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
                    result['shape_betas'],
                    None,
                    ref_trans,
                    result['kid_factor'],
                )

        if 'joints' in requested_keys or 'vertices' in requested_keys:
            forw = self.body_model(
                glob_rotmats=glob_rotmats,
                shape_betas=result['shape_betas'],
                trans=result['trans'],
                kid_factor=result['kid_factor'],
            )

        # Add the mean back (scaled appropriately if using scale correction)
        if scale_target:
            result['trans'] = result['trans'] + target_mean * result['scale_corr'][:, np.newaxis]
        elif scale_fit:
            result['trans'] = result['trans'] + target_mean / result['scale_corr'][:, np.newaxis]
        else:
            result['trans'] = result['trans'] + target_mean
        if 'joints' in requested_keys:
            result['joints'] = forw['joints'] + target_mean[:, np.newaxis]
        if 'vertices' in requested_keys:
            result['vertices'] = forw['vertices'] + target_mean[:, np.newaxis]

        result['orientations'] = glob_rotmats

        # Provide other requested rotation formats
        if 'relative_orientations' in requested_keys or 'pose_rotvecs' in requested_keys:
            parent_glob_rotmats = np.concatenate(
                [
                    np.broadcast_to(np.eye(3, dtype=np.float32), glob_rotmats[:, :1].shape),
                    glob_rotmats[:, parent_indices],
                ],
                axis=1,
            )
            result['relative_orientations'] = matmul_transp_a(parent_glob_rotmats, glob_rotmats)

        if 'pose_rotvecs' in requested_keys:
            rotvecs = mat2rotvec(result['relative_orientations'])
            result['pose_rotvecs'] = rotvecs.reshape(rotvecs.shape[0], -1)

        return result

    def fit_with_known_pose(
        self,
        pose_rotvecs,
        target_vertices,
        target_joints=None,
        vertex_weights=None,
        joint_weights=None,
        beta_regularizer=1.0,
        beta_regularizer2=0.0,
        scale_regularizer=0.0,
        kid_regularizer=None,
        share_beta=False,
        scale_target=False,
        scale_fit=False,
        beta_regularizer_reference=None,
        kid_regularizer_reference=None,
        requested_keys=('shape_betas',),
    ):
        """
        Fits the body shape and translation with known output pose.

        Parameters:
            pose_rotvecs: The known output joint rotations as rotation vectors,
                shaped as (batch_size, num_joints * 3).
            target_vertices: Target mesh vertices, shaped as (batch_size, num_vertices, 3).
            target_joints: Optional target joint positions.
            vertex_weights: Optional importance weights for vertices.
            joint_weights: Optional importance weights for joints.
            beta_regularizer: L2 regularization weight for shape parameters.
            beta_regularizer2: Secondary regularization for first two shape params.
            scale_regularizer: Regularization for scale factor.
            kid_regularizer: Regularization for kid blendshape factor.
            share_beta: Whether to share shape params across batch.
            scale_target: Whether to estimate scale for target vertices.
            scale_fit: Whether to estimate scale for fitted mesh.
            beta_regularizer_reference: Reference values for beta regularization.
            kid_regularizer_reference: Reference values for kid factor regularization.
            requested_keys: List of result keys to return.

        Returns:
            Dictionary with shape_betas, trans, and optionally kid_factor, scale_corr.
        """
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
        glob_rotmats_ = [rel_rotmats[:, 0]]
        for i_joint in range(1, self.body_model.num_joints):
            i_parent = self.body_model.kintree_parents[i_joint]
            glob_rotmats_.append(glob_rotmats_[i_parent] @ rel_rotmats[:, i_joint])
        glob_rotmats = np.stack(glob_rotmats_, axis=1)

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

        # Add the mean back
        result['trans'] = result['trans'] + target_mean
        result.pop('vertices', None)
        result.pop('joints', None)

        return {k: v for k, v in result.items() if v is not None}

    def fit_with_known_shape(
        self,
        shape_betas,
        target_vertices,
        target_joints=None,
        vertex_weights=None,
        joint_weights=None,
        kid_factor=None,
        num_iter=1,
        final_adjust_rots=True,
        initial_pose_rotvecs=None,
        scale_fit=False,
        requested_keys=('pose_rotvecs',),
    ):
        """
        Fits the body pose and translation with known shape parameters.

        Parameters:
            shape_betas: Shape parameters (betas), shaped as (batch_size, num_betas).
            target_vertices: Target mesh vertices, shaped as (batch_size, num_vertices, 3).
            target_joints: Optional target joint positions.
            vertex_weights: Optional importance weights for vertices.
            joint_weights: Optional importance weights for joints.
            kid_factor: Optional kid blendshape factor.
            num_iter: Number of fitting iterations.
            final_adjust_rots: Whether to refine rotations after fitting.
            initial_pose_rotvecs: Optional initial pose as rotation vectors.
            scale_fit: Whether to estimate scale for fitted mesh.
            requested_keys: List of result keys to return.

        Returns:
            Dictionary with pose_rotvecs, trans, and optionally other keys.
        """
        if not requested_keys:
            requested_keys = ['pose_rotvecs']

        # Slice shape_betas to n_betas (matching PT behavior)
        shape_betas = shape_betas[:, : self.n_betas]

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

        glob_rotmats = (
            self._fit_global_rotations(
                target_vertices,
                target_joints,
                initial_vertices,
                initial_joints if target_joints is not None else None,
                vertex_weights,
                joint_weights,
            )
            @ initial_forw['orientations']
        )

        # Iterative refinement
        for _ in range(num_iter - 1):
            forw = self.body_model(
                glob_rotmats=glob_rotmats, shape_betas=shape_betas, kid_factor=kid_factor
            )
            ref_verts = forw['vertices']
            ref_joints = forw['joints'] if target_joints is not None else None
            glob_rotmats = (
                self._fit_global_rotations(
                    target_vertices,
                    target_joints,
                    ref_verts,
                    ref_joints,
                    vertex_weights,
                    joint_weights,
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
                    None,
                    trans,
                    kid_factor,
                )

        # Build result
        result = {
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
            parent_glob_rotmats = np.concatenate(
                [
                    np.broadcast_to(np.eye(3, dtype=np.float32), glob_rotmats[:, :1].shape),
                    glob_rotmats[:, parent_indices],
                ],
                axis=1,
            )
            result['relative_orientations'] = matmul_transp_a(parent_glob_rotmats, glob_rotmats)

        if 'pose_rotvecs' in requested_keys:
            rotvecs = mat2rotvec(result['relative_orientations'])
            result['pose_rotvecs'] = rotvecs.reshape(rotvecs.shape[0], -1)

        return result

    def _fit_shape(
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
        beta_regularizer_reference=None,
        kid_regularizer_reference=None,
        requested_keys=(),
    ):
        if scale_target and scale_fit:
            raise ValueError('Only one of estim_scale_target and estim_scale_fit can be True')

        batch_size = target_vertices.shape[0]
        parent_indices = self.body_model.kintree_parents[1:]

        parent_glob_rot_mats = np.concatenate(
            [
                np.broadcast_to(np.eye(3, dtype=np.float32), glob_rotmats[:, :1].shape),
                glob_rotmats[:, parent_indices],
            ],
            axis=1,
        )
        rel_rotmats = matmul_transp_a(parent_glob_rot_mats, glob_rotmats)

        glob_positions_ext = [np.repeat(self.J_template_ext[np.newaxis, 0], batch_size, axis=0)]
        for i_joint, i_parent in enumerate(self.body_model.kintree_parents[1:], start=1):
            glob_positions_ext.append(
                glob_positions_ext[i_parent]
                + np.einsum(
                    'bCc,cs->bCs',
                    glob_rotmats[:, i_parent],
                    self.J_template_ext[i_joint] - self.J_template_ext[i_parent],
                )
            )
        glob_positions_ext = np.stack(glob_positions_ext, axis=1)
        translations_ext = glob_positions_ext - np.einsum(
            'bjCc,jcs->bjCs', glob_rotmats, self.J_template_ext
        )

        rot_params = rel_rotmats[:, 1:].reshape(-1, (self.body_model.num_joints - 1) * 3 * 3)
        v_posed = self.v_template + np.einsum('vcp,bp->bvc', self.posedirs, rot_params)
        v_rotated = np.einsum('bjCc,vj,bvc->bvC', glob_rotmats, self.weights, v_posed)

        shapedirs = (
            np.concatenate(
                [self.shapedirs[:, :, : self.n_betas], self.kid_shapedir[:, :, np.newaxis]], axis=2
            )
            if self.enable_kid
            else self.shapedirs[:, :, : self.n_betas]
        )
        v_grad_rotated = np.einsum('bjCc,lj,lcs->blCs', glob_rotmats, self.weights, shapedirs)

        v_rotated_ext = np.concatenate([v_rotated[:, :, :, np.newaxis], v_grad_rotated], axis=3)
        v_translations_ext = np.einsum('vj,bjcs->bvcs', self.weights, translations_ext)
        v_posed_posed_ext = v_translations_ext + v_rotated_ext

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

        if scale_target:
            A = np.concatenate([jac_pos_both, -target_both[..., np.newaxis]], axis=3)
        elif scale_fit:
            A = np.concatenate([jac_pos_both, pos_both[..., np.newaxis]], axis=3)
        else:
            A = jac_pos_both

        b = target_both - pos_both
        mean_A = np.mean(A, axis=1, keepdims=True)
        mean_b = np.mean(b, axis=1, keepdims=True)
        A = A - mean_A
        b = b - mean_b

        if target_joints is not None and vertex_weights is not None and joint_weights is not None:
            weights = np.concatenate([vertex_weights, joint_weights], axis=1)
        elif target_joints is None and vertex_weights is not None:
            weights = vertex_weights
        else:
            weights = np.ones(A.shape[:2], dtype=np.float32)

        n_params = (
            self.n_betas + (1 if self.enable_kid else 0) + (1 if scale_target or scale_fit else 0)
        )
        A = A.reshape(batch_size, -1, n_params)
        b = b.reshape(batch_size, -1, 1)
        w = np.repeat(weights.reshape(batch_size, -1), 3, axis=1)

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

        l2_regularizer_rhs = (l2_regularizer_all * l2_regularizer_reference_all)[..., np.newaxis]

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
        new_trans = mean_b.squeeze(1) - np.matmul(mean_A.squeeze(1), x[..., np.newaxis]).squeeze(
            -1
        )
        new_shape = x[:, : self.n_betas]
        new_kid_factor = None
        new_scale_corr = None

        if self.enable_kid:
            new_kid_factor = x[:, self.n_betas]
        if scale_target or scale_fit:
            new_scale_corr = x[:, -1] + 1
            if scale_fit:
                new_shape /= new_scale_corr[..., np.newaxis]
                if self.enable_kid:
                    new_kid_factor /= new_scale_corr

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
            new_shape = np.concatenate([new_shape, new_kid_factor[:, np.newaxis]], axis=1)

        if 'joints' in requested_keys:
            result['joints'] = (
                glob_positions_ext[..., 0]
                + np.einsum('bvcs,bs->bvc', glob_positions_ext[..., 1:], new_shape)
                + new_trans[:, np.newaxis]
            )

        if 'vertices' in requested_keys:
            result['vertices'] = (
                v_posed_posed_ext[..., 0]
                + np.einsum('bvcs,bs->bvc', v_posed_posed_ext[..., 1:], new_shape)
                + new_trans[:, np.newaxis]
            )
        return result

    def _fit_global_rotations(
        self,
        target_vertices,
        target_joints,
        reference_vertices,
        reference_joints,
        vertex_weights,
        joint_weights,
    ):
        """Global orientation of each body part via swing-twist decomposition, batched.

        Parts containing >= 3 joints get a Kabsch fit from their joints alone; leaf parts
        (1 joint) from their vertices alone. Bone parts (2 joints) are
        solved in two exactly-determined steps: the swing aligns the bone direction, and
        the twist about the bone is recovered from the vertices in closed form. For the
        twist, with ``H = R_swing A^T`` where ``A = sum w t_bar a_bar^T`` is the part's
        centered cross-covariance, the optimal angle is
        ``atan2(b_hat . vee(H), tr(H) - b_hat^T H b_hat)``. This is the ``mesh_weight -> 0``
        limit of a weighted Kabsch fit, computed without the ill-conditioned SVD that such
        weighting would produce (stable gradients).
        """
        if target_joints is None or reference_joints is None:
            target_joints = self.J_regressor @ target_vertices
            reference_joints = self.J_regressor @ reference_vertices

        B = target_vertices.shape[0]

        # Per-part vertex cross-covariances about the children-mean centers, loop-free.
        raw, s_t, s_a, s_w = self._part_sums(target_vertices, reference_vertices, vertex_weights)
        mt = self.center_matrix @ target_joints  # (B, J, 3)
        ma = self.center_matrix @ reference_joints  # (B_ref, J, 3)
        A_vert = (
            raw
            - s_t[..., np.newaxis] * ma[..., np.newaxis, :]
            - mt[..., np.newaxis] * s_a[..., np.newaxis, :]
            + s_w[..., np.newaxis] * (mt[..., np.newaxis] * ma[..., np.newaxis, :])
        )  # (B, J, 3, 3)

        # Joint-point cross-covariances for the multi-joint parts, loop-free.
        rj = reference_joints
        if joint_weights is not None:
            rj = rj * joint_weights[..., np.newaxis]
            tj_sum_side = target_joints * joint_weights[..., np.newaxis]
            s_wj = self.mjp_joint_membership @ joint_weights[..., np.newaxis]
        else:
            tj_sum_side = target_joints
            s_wj = self.mjp_joint_counts
        outer_j = (target_joints[..., np.newaxis] * rj[..., np.newaxis, :]).reshape(
            B, target_joints.shape[1], 9
        )
        raw_j = (self.mjp_joint_membership @ outer_j).reshape(B, -1, 3, 3)
        mtj = self.mjp_center_matrix @ target_joints
        maj = self.mjp_center_matrix @ reference_joints
        s_tj = self.mjp_joint_membership @ tj_sum_side
        s_aj = self.mjp_joint_membership @ rj
        A_multi = (
            raw_j
            - s_tj[..., np.newaxis] * maj[..., np.newaxis, :]
            - mtj[..., np.newaxis] * s_aj[..., np.newaxis, :]
            + s_wj[..., np.newaxis] * (mtj[..., np.newaxis] * maj[..., np.newaxis, :])
        )

        # Kabsch bucket (multi-joint parts + leaf parts): one batched SVD.
        A_svd = np.concatenate([A_multi, A_vert[:, self.leaf_parts]], axis=1)
        R_svd = proj_SO3(A_svd)

        # Bone bucket: batched swing (bone alignment) + twist (from vertices).
        b_ref = (
            reference_joints[:, self.bone_pairs[:, 1]] - reference_joints[:, self.bone_pairs[:, 0]]
        )
        b_tgt = target_joints[:, self.bone_pairs[:, 1]] - target_joints[:, self.bone_pairs[:, 0]]
        b_ref_n = divide_no_nan(b_ref, np.linalg.norm(b_ref, axis=-1, keepdims=True))
        b_tgt_n = divide_no_nan(b_tgt, np.linalg.norm(b_tgt, axis=-1, keepdims=True))
        R_swing = align_unit_vectors(b_ref_n, b_tgt_n)  # (B, n_bones, 3, 3)

        H = R_swing @ np.swapaxes(A_vert[:, self.bone_parts], -1, -2)
        trH = np.trace(H, axis1=-2, axis2=-1)
        bHb = (b_tgt_n[..., np.newaxis, :] @ H @ b_tgt_n[..., np.newaxis])[..., 0, 0]
        # vee_i = eps_ijk H_jk: the vertex cross-product sums, extracted from H.
        vee = np.stack(
            [
                H[..., 1, 2] - H[..., 2, 1],
                H[..., 2, 0] - H[..., 0, 2],
                H[..., 0, 1] - H[..., 1, 0],
            ],
            axis=-1,
        )
        twist_angle = np.arctan2(np.sum(b_tgt_n * vee, axis=-1), trH - bHb)
        R_twist = rotvec2mat(b_tgt_n * twist_angle[..., np.newaxis])
        R_bone = R_twist @ R_swing

        # Scatter both buckets back to per-part order (toe parts take the feet slots).
        R_concat = np.concatenate([R_svd, R_bone], axis=1)
        return R_concat[:, self.assemble_indices]

    def _fit_global_rotations_dependent(
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
        trans,
        kid_factor,
    ):
        glob_rots = []

        true_reference_joints = reference_joints
        if target_joints is None or reference_joints is None:
            target_joints = self.J_regressor @ target_vertices
            reference_joints = self.J_regressor @ reference_vertices
        if true_reference_joints is None:
            true_reference_joints = reference_joints

        j = self.body_model.J_template + np.einsum(
            'jcs,...s->...jc', self.body_model.J_shapedirs[:, :, : self.n_betas], shape_betas
        )
        if kid_factor is not None:
            j += np.einsum('jc,...->...jc', self.body_model.kid_J_shapedir, kid_factor)

        if scale_corr is not None:
            j = j * scale_corr[:, np.newaxis, np.newaxis]

        parent_indices = self.body_model.kintree_parents[1:]
        j_parent = np.concatenate([np.zeros(3) * j[:, :1], j[:, parent_indices]], axis=1)
        bones = j - j_parent

        # Per-part vertex statistics, shared machinery with _fit_global_rotations. The
        # sequential loop below only needs 3x3 algebra per joint: the cross-covariance
        # about the dynamic centers follows algebraically from these fixed sums.
        raw, s_t, s_a, s_w = self._part_sums(target_vertices, reference_vertices, vertex_weights)

        glob_positions = []

        for i in range(self.body_model.num_joints):
            if i == 0:
                glob_position = j[:, i] + trans
            else:
                i_parent = self.body_model.kintree_parents[i]
                glob_position = glob_positions[i_parent] + np.matmul(
                    glob_rots[i_parent], bones[:, i][..., np.newaxis]
                ).squeeze(-1)
            glob_positions.append(glob_position)

            if self.is_smpl_family:
                if i == 10:
                    glob_rots.append(glob_rots[7])
                    continue
                elif i == 11:
                    glob_rots.append(glob_rots[8])
                    continue
            if i not in self.adjustable_parts:
                glob_rots.append(glob_rots_prev[:, i])
                continue

            # Vertex contribution: centered cross-covariance about the dynamic centers
            # (c_t = current global joint position, c_a = reference joint position).
            c_t = glob_position  # (B, 3)
            c_a = true_reference_joints[:, i]  # (B_ref, 3)
            A_vert = (
                raw[:, i]
                - s_t[:, i][..., np.newaxis] * c_a[..., np.newaxis, :]
                - c_t[..., np.newaxis] * s_a[:, i][..., np.newaxis, :]
                + s_w[:, i][..., np.newaxis] * (c_t[..., np.newaxis] * c_a[..., np.newaxis, :])
            )  # (B, 3, 3)

            # Joint contribution (children_and_self), same weighting as before.
            joint_selector = self.children_and_self[i]
            estim_joints = target_joints[:, joint_selector] - c_t[:, np.newaxis]
            default_joints = reference_joints[:, joint_selector] - c_a[:, np.newaxis]
            if joint_weights is not None:
                default_joints = default_joints * joint_weights[:, joint_selector][..., np.newaxis]
            A_joint = np.swapaxes(estim_joints, -1, -2) @ default_joints  # (B, 3, 3)

            glob_rot = proj_SO3(A_vert + A_joint) @ glob_rots_prev[:, i]
            glob_rots.append(glob_rot)

        return np.stack(glob_rots, axis=1)


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
