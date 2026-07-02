from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from .lstsq import lstsq, lstsq_partial_share
from .rotation import align_unit_vectors, divide_no_nan, mat2rotvec, proj_SO3, rotvec2mat

if TYPE_CHECKING:
    import smplfitter.pt


class BodyFitter(nn.Module):
    """Fits body model (SMPL/SMPL-X/SMPL+H) parameters to lists of target vertices and joints.

    Parameters:
        body_model: The body model instance we wish to fit, of a certain SMPL model variant and \
            gender.
        enable_kid: Enables the use of a kid blendshape, allowing for fitting kid shapes as in
            AGORA :footcite:`patel2021agora`.
    """

    def __init__(
        self,
        body_model: 'smplfitter.pt.BodyModel',
        enable_kid: bool = False,
    ):
        super(BodyFitter, self).__init__()
        self.body_model = body_model
        self.n_betas = self.body_model.shapedirs.shape[2]
        self.enable_kid = enable_kid
        self.is_smpl_family = body_model.model_name.startswith('smpl')

        part_assignment = torch.argmax(body_model.weights, dim=1)
        if self.is_smpl_family:
            part_assignment = torch.where(
                part_assignment == 10, torch.tensor(7, dtype=torch.int64), part_assignment
            )
            part_assignment = torch.where(
                part_assignment == 11, torch.tensor(8, dtype=torch.int64), part_assignment
            )
        self.part_assignment = nn.Buffer(part_assignment)
        self.part_vertex_selectors = [
            torch.where(part_assignment == i)[0] for i in range(body_model.num_joints)
        ]

        self.default_mesh_tf = nn.Buffer(body_model.single()['vertices'])

        # Template for joints with shape adjustments
        self.J_template_ext = nn.Buffer(
            torch.cat(
                [body_model.J_template.view(-1, 3, 1), body_model.J_shapedirs]
                + ([body_model.kid_J_shapedir.view(-1, 3, 1)] if enable_kid else []),
                dim=2,
            )
        )

        # Store joint hierarchy for each joint’s children and descendants
        self.children_and_self = [[i_joint] for i_joint in range(body_model.num_joints)]
        for i_joint in range(1, body_model.num_joints):
            i_parent = body_model.kintree_parents[i_joint]
            self.children_and_self[i_parent].append(i_joint)

        self.descendants_and_self = [[i_joint] for i_joint in range(body_model.num_joints)]
        for i_joint in range(body_model.num_joints - 1, 0, -1):
            i_parent = body_model.kintree_parents[i_joint]
            self.descendants_and_self[i_parent].extend(self.descendants_and_self[i_joint])

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
        multi_joint_parts: list[int] = []
        bone_parts: list[int] = []
        leaf_parts: list[int] = []
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

        # Body parts whose orientation the final adjustment pass refines (re-anchored at
        # the recomputed joint position; other parts keep their initial orientation).
        if self.is_smpl_family:
            self.adjustable_parts = [1, 2, 4, 5, 7, 8, 16, 17, 18, 19]
        else:
            self.adjustable_parts = [i for i in range(J)]

        # Vertices that participate in some vertex-based statistic (bone twist, leaf-part
        # Kabsch, or the final adjustment). Parts whose orientation comes from joints
        # alone are skipped where possible.
        stat_parts = sorted(set(bone_parts + leaf_parts + self.adjustable_parts))
        used_mask = torch.zeros(body_model.num_vertices, dtype=torch.bool)
        for i in stat_parts:
            used_mask[self.part_vertex_selectors[i]] = True
        used_vertex_indices = torch.where(used_mask)[0]
        self.used_vertex_indices = nn.Buffer(used_vertex_indices)

        # One-hot part membership over the used vertices: row j sums vertices of part j.
        part_matrix = torch.zeros(J, len(used_vertex_indices))
        part_matrix[
            part_assignment[used_vertex_indices], torch.arange(len(used_vertex_indices))
        ] = 1.0
        self.part_matrix = nn.Buffer(part_matrix)
        self.part_counts = nn.Buffer(part_matrix.sum(dim=1).view(1, J, 1))

        # Children-mean centering matrix: row i averages children_and_self[i] joint positions.
        center_matrix = torch.zeros(J, J)
        for i in range(J):
            js = self.children_and_self[i]
            center_matrix[i, js] = 1.0 / len(js)
        self.center_matrix = nn.Buffer(center_matrix)

        # Joint membership per multi-joint part (their orientation is Kabsch-fit from joints).
        mjp_joint_membership = torch.zeros(len(multi_joint_parts), J)
        for k, i in enumerate(multi_joint_parts):
            mjp_joint_membership[k, self.children_and_self[i]] = 1.0
        self.mjp_joint_membership = nn.Buffer(mjp_joint_membership)
        self.mjp_joint_counts = nn.Buffer(mjp_joint_membership.sum(dim=1).view(1, -1, 1))
        self.mjp_center_matrix = nn.Buffer(center_matrix[multi_joint_parts])

        # Bone endpoints (start joint, end joint) per bone part.
        self.bone_pairs = nn.Buffer(
            torch.tensor(
                [[self.children_and_self[i][0], self.children_and_self[i][1]] for i in bone_parts],
                dtype=torch.int64,
            ).reshape(len(bone_parts), 2)
        )

        # Assembly permutation: R_concat = cat([R_multi, R_leaf, R_bone]) is scattered back
        # to per-part order; SMPL toe parts take the feet slots directly (10 <- 7, 11 <- 8).
        concat_order = multi_joint_parts + leaf_parts + bone_parts
        inverse_perm = [0] * J
        for pos, j in enumerate(concat_order):
            inverse_perm[j] = pos
        if self.is_smpl_family:
            inverse_perm[10] = inverse_perm[7]
            inverse_perm[11] = inverse_perm[8]
        self.assemble_indices = nn.Buffer(torch.tensor(inverse_perm, dtype=torch.int64))

    def _part_sums(
        self,
        target_vertices: torch.Tensor,
        reference_vertices: torch.Tensor,
        vertex_weights: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            a = a * w.unsqueeze(-1)
            t_sum_side = t * w.unsqueeze(-1)
            s_w = self.part_matrix @ w.unsqueeze(-1)
        else:
            t_sum_side = t
            s_w = self.part_counts
        B = t.shape[0] if t.shape[0] >= a.shape[0] else a.shape[0]
        outer = (t.unsqueeze(-1) * a.unsqueeze(-2)).reshape(B, t.shape[1], 9)
        raw = (self.part_matrix @ outer).view(B, -1, 3, 3)
        s_t = self.part_matrix @ t_sum_side
        s_a = self.part_matrix @ a
        return raw, s_t, s_a, s_w

    @torch.jit.export
    def fit(
        self,
        target_vertices: torch.Tensor,
        target_joints: Optional[torch.Tensor] = None,
        vertex_weights: Optional[torch.Tensor] = None,
        joint_weights: Optional[torch.Tensor] = None,
        num_iter: int = 1,
        beta_regularizer: float = 1,
        beta_regularizer2: float = 0,
        scale_regularizer: float = 0,
        kid_regularizer: Optional[float] = None,
        share_beta: bool = False,
        final_adjust_rots: bool = True,
        scale_target: bool = False,
        scale_fit: bool = False,
        initial_pose_rotvecs: Optional[torch.Tensor] = None,
        initial_shape_betas: Optional[torch.Tensor] = None,
        initial_kid_factor: Optional[torch.Tensor] = None,
        requested_keys: Optional[list[str]] = None,
    ) -> dict[str, torch.Tensor]:
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
            beta_regularizer2: Regularization for the first two betas. Often zero works well.
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
            initial_pose_rotvecs: Optional initial pose rotations, if a good guess is available.
                Usually not necessary (experimental).
            initial_shape_betas: Optional initial shape parameters (betas), if a good guess is
                available. Usually not necessary (experimental).
            initial_kid_factor: Same as above, but for the kid blendshape factor.
            requested_keys: List of keys specifying which results to return.

        Returns:
            Dictionary
                - **pose_rotvecs** -- Estimated pose in concatenated rotation vector format.
                - **shape_betas** -- Estimated shape parameters (betas).
                - **trans** -- Estimated translation parameters.
                - **orientations** -- Global body part orientations as rotation matrices.
                - **relative_orientations** -- Parent-relative body part orientations as rotation \
                    matrices.
                - **kid_factor** -- Estimated kid blendshape factor, if `enable_kid` is True.
                - **scale_corr** -- Estimated scale correction factor, if `scale_target` or \
                    `scale_fit` is True.

        """

        if requested_keys is None:
            requested_keys = ['pose_rotvecs']

        # Subtract mean first for better numerical stability (and add it back later)
        if target_joints is None:
            target_mean = torch.mean(target_vertices, dim=1)
            target_vertices = target_vertices - target_mean[:, None]
        else:
            target_mean = torch.mean(torch.cat([target_vertices, target_joints], dim=1), dim=1)
            target_vertices = target_vertices - target_mean[:, None]
            target_joints = target_joints - target_mean[:, None]

        if initial_pose_rotvecs is not None or initial_shape_betas is not None:
            initial_forw = self.body_model(
                shape_betas=initial_shape_betas,
                kid_factor=initial_kid_factor,
                pose_rotvecs=initial_pose_rotvecs,
            )
            initial_joints = initial_forw['joints']
            initial_vertices = initial_forw['vertices']
            glob_rotmats = (
                self._fit_global_rotations(
                    target_vertices,
                    target_joints,
                    initial_vertices,
                    initial_joints,
                    vertex_weights,
                    joint_weights,
                )
                @ initial_forw['orientations']
            )
        else:
            initial_joints = self.body_model.J_template[None]
            initial_vertices = self.default_mesh_tf[None]
            glob_rotmats = self._fit_global_rotations(
                target_vertices,
                target_joints,
                initial_vertices,
                initial_joints,
                vertex_weights,
                joint_weights,
            )

        device = self.body_model.v_template.device
        parent_indices = self.body_model.kintree_parents_tensor[1:].to(device)

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
                requested_keys=(
                    ['vertices', 'joints'] if target_joints is not None else ['vertices']
                ),
            )
            ref_verts = result['vertices']
            ref_joints = result['joints'] if target_joints is not None else None

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
            requested_keys=(
                ['vertices', 'joints']
                if target_joints is not None or final_adjust_rots
                else ['vertices']
            ),
        )
        ref_verts = result['vertices']
        ref_joints = result['joints'] if target_joints is not None or final_adjust_rots else None
        ref_shape = result['shape_betas']
        ref_trans = result['trans']
        ref_kid_factor = result['kid_factor'] if self.enable_kid else None
        ref_scale_corr = result['scale_corr'][:, None, None] if scale_target or scale_fit else None

        if final_adjust_rots:
            assert ref_joints is not None
            if scale_target:
                assert ref_scale_corr is not None
                glob_rotmats = self._fit_global_rotations_dependent(
                    target_vertices * ref_scale_corr,
                    target_joints * ref_scale_corr if target_joints is not None else None,
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
            elif scale_fit:
                assert ref_scale_corr is not None
                glob_rotmats = self._fit_global_rotations_dependent(
                    target_vertices,
                    target_joints,
                    ref_scale_corr * ref_verts + (1 - ref_scale_corr) * ref_trans.unsqueeze(-2),
                    ref_scale_corr * ref_joints + (1 - ref_scale_corr) * ref_trans.unsqueeze(-2),
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

        # Add the mean back (scaled appropriately if using scale correction)
        if scale_target:
            result['trans'] = ref_trans + target_mean * result['scale_corr'][:, None]
        elif scale_fit:
            result['trans'] = ref_trans + target_mean / result['scale_corr'][:, None]
        else:
            result['trans'] = ref_trans + target_mean
        result['orientations'] = glob_rotmats

        # Provide other requested rotation formats
        if 'relative_orientations' in requested_keys or 'pose_rotvecs' in requested_keys:
            parent_glob_rotmats = torch.cat(
                [
                    torch.eye(3, device=device).expand(glob_rotmats.shape[0], 1, 3, 3),
                    torch.index_select(glob_rotmats, 1, parent_indices),
                ],
                dim=1,
            )
            result['relative_orientations'] = torch.matmul(
                parent_glob_rotmats.transpose(-1, -2), glob_rotmats
            )

        if 'pose_rotvecs' in requested_keys:
            rel_ori = result['relative_orientations']
            assert rel_ori is not None
            rotvecs = mat2rotvec(rel_ori)
            result['pose_rotvecs'] = rotvecs.view(rotvecs.shape[0], -1)

        if 'vertices' in result:
            result.pop('vertices')
        if 'joints' in result:
            result.pop('joints')
        result_non_none: dict[str, torch.Tensor] = {}
        for k, v in result.items():
            if v is not None:
                result_non_none[k] = v
        return result_non_none

    @torch.jit.export
    def fit_with_known_pose(
        self,
        pose_rotvecs: torch.Tensor,
        target_vertices: torch.Tensor,
        target_joints: Optional[torch.Tensor] = None,
        vertex_weights: Optional[torch.Tensor] = None,
        joint_weights: Optional[torch.Tensor] = None,
        beta_regularizer: float = 1,
        beta_regularizer2: float = 0,
        scale_regularizer: float = 0,
        kid_regularizer: Optional[float] = None,
        share_beta: bool = False,
        scale_target: bool = False,
        scale_fit: bool = False,
        beta_regularizer_reference: Optional[torch.Tensor] = None,
        kid_regularizer_reference: Optional[torch.Tensor] = None,
        requested_keys: Optional[list[str]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Fits the body shape and translation (and possibly scale), with known output pose.

        Parameters:
            pose_rotvecs: The known output joint rotations represented as rotation vectors,
                shaped as (batch_size, num_joints * 3).
            target_vertices: Target mesh vertices to fit, shaped as (batch_size, num_vertices, 3).
            target_joints: Optional target joint positions, shaped as (batch_size, num_joints, 3).
            vertex_weights: Optional importance weights for individual vertices during the
                fitting process.
            joint_weights: Optional importance weights for individual joints during the fitting
                process.
            beta_regularizer: L2 regularization weight for shape parameters (betas).
            beta_regularizer2: Secondary regularization applied to the first two shape parameters.
            scale_regularizer: Regularization term penalizing deviation of the scale factor from 1.
            kid_regularizer: Regularization weight for the kid blendshape factor.
            share_beta: Whether to share the shape parameters (betas) across instances in the batch.
            scale_target: Whether to estimate a scale factor for the target vertices to aid
                alignment.
            scale_fit: Whether to estimate a scale factor for the fitted mesh to aid alignment.
            beta_regularizer_reference: Optional reference values for beta regularization.
            kid_regularizer_reference: Optional reference values for kid factor regularization.
            requested_keys: List of result keys to return.

        Returns:
            Dictionary
                - **shape_betas** -- Estimated shape parameters (betas).
                - **trans** -- Estimated translation parameters.
                - **orientations** -- Global body part orientations as rotation matrices.
                - **relative_orientations** -- Parent-relative body part orientations as rotation \
                    matrices.
                - **kid_factor** -- Estimated kid blendshape factor, if enabled.
                - **scale_corr** -- Estimated scale correction factor, if scaling is enabled.

        """

        if requested_keys is None:
            requested_keys = []
        # Subtract mean first for better numerical stability (and add it back later)
        if target_joints is None:
            target_mean = torch.mean(target_vertices, dim=1)
            target_vertices = target_vertices - target_mean[:, None]
        else:
            target_mean = torch.mean(torch.cat([target_vertices, target_joints], dim=1), dim=1)
            target_vertices = target_vertices - target_mean[:, None]
            target_joints = target_joints - target_mean[:, None]

        rel_rotmats = rotvec2mat(pose_rotvecs.view(-1, self.body_model.num_joints, 3))
        glob_rotmats_ = [rel_rotmats[:, 0]]
        for i_joint in range(1, self.body_model.num_joints):
            i_parent = self.body_model.kintree_parents[i_joint]
            glob_rotmats_.append(glob_rotmats_[i_parent] @ rel_rotmats[:, i_joint])
        glob_rotmats = torch.stack(glob_rotmats_, dim=1)

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
        # ref_kid_factor = result['kid_factor'] if self.enable_kid else None
        # ref_scale_corr = result['scale_corr'][:, None, None] if scale_target or scale_fit else None

        # Add the mean back
        result['trans'] = result['trans'] + target_mean
        if 'vertices' in result:
            result.pop('vertices')
        if 'joints' in result:
            result.pop('joints')
        result_non_none: dict[str, torch.Tensor] = {}
        for k, v in result.items():
            if v is not None:
                result_non_none[k] = v
        return result_non_none

    @torch.jit.export
    def fit_with_known_shape(
        self,
        shape_betas: torch.Tensor,
        target_vertices: torch.Tensor,
        target_joints: Optional[torch.Tensor] = None,
        vertex_weights: Optional[torch.Tensor] = None,
        joint_weights: Optional[torch.Tensor] = None,
        kid_factor: Optional[torch.Tensor] = None,
        num_iter: int = 1,
        final_adjust_rots: bool = True,
        initial_pose_rotvecs: Optional[torch.Tensor] = None,
        scale_fit: bool = False,
        requested_keys: Optional[list[str]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Fits the body model pose and translation to target vertices and optionally target joints,
        given known shape parameters (betas).

        The method assumes the shape parameters (betas) are known and optimizes the pose and
        translation to fit the target vertices and joints. Initial pose rotations can
        optionally be provided to warm-start the optimization process.

        Parameters:
            shape_betas: Shape parameters (betas) for the body model, shaped as (batch_size,
                num_betas).
            target_vertices: Target mesh vertices to fit, shaped as (batch_size, num_vertices, 3).
            target_joints: Optional target joint positions, shaped as (batch_size, num_joints, 3).
            vertex_weights: Optional importance weights for individual vertices during the
                fitting process.
            joint_weights: Optional importance weights for individual joints during the fitting
                process.
            kid_factor: Optional adjustment factor for kid shapes, shaped as (batch_size, 1).
            num_iter: Number of iterations for the optimization process.
            final_adjust_rots: Whether to refine body part orientations after fitting for better
                alignment.
            initial_pose_rotvecs: Optional initial pose rotations in rotation vector format,
                shaped as (batch_size, num_joints * 3).
            scale_fit: Whether to estimate a scale factor to align the fitted mesh with the
                target vertices.
            requested_keys: List of result keys to return.

        Returns:
            Dictionary
                - **pose_rotvecs** -- Estimated pose rotation vectors in concatenated format.
                - **trans** -- Estimated translation parameters.
                - **orientations** -- Global body part orientations as rotation matrices.
                - **relative_orientations** -- Parent-relative body part orientations as rotation \
                    matrices.
                - **kid_factor** -- Estimated kid blendshape factor, if provided.
                - **scale_corr** -- Estimated scale correction factor, if scaling is enabled.

        """

        if requested_keys is None:
            requested_keys = ['pose_rotvecs']

        # Subtract mean first for better numerical stability (and add it back later)
        if target_joints is None:
            target_mean = torch.mean(target_vertices, dim=1)
            target_vertices = target_vertices - target_mean[:, None]
        else:
            target_mean = torch.mean(torch.cat([target_vertices, target_joints], dim=1), dim=1)
            target_vertices = target_vertices - target_mean[:, None]
            target_joints = target_joints - target_mean[:, None]

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
                initial_joints,
                vertex_weights,
                joint_weights,
            )
            @ initial_forw['orientations']
        )
        device = self.body_model.v_template.device
        parent_indices = self.body_model.kintree_parents_tensor[1:].to(device)

        for i in range(num_iter - 1):
            result = self.body_model(
                glob_rotmats=glob_rotmats, shape_betas=shape_betas, kid_factor=kid_factor
            )
            ref_verts = result['vertices']
            ref_joints = result['joints'] if target_joints is not None else None
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

        result = self.body_model(
            glob_rotmats=glob_rotmats, shape_betas=shape_betas, kid_factor=kid_factor
        )
        ref_verts = result['vertices']
        ref_joints = result['joints']
        ref_scale_corr, ref_trans = fit_scale_and_translation(
            target_vertices,
            ref_verts,
            target_joints,
            ref_joints,
            vertex_weights,
            joint_weights,
            scale=scale_fit,
        )

        if final_adjust_rots:
            if scale_fit:
                assert ref_scale_corr is not None
                glob_rotmats = self._fit_global_rotations_dependent(
                    target_vertices,
                    target_joints,
                    ref_scale_corr * ref_verts + ref_trans.unsqueeze(-2),
                    ref_scale_corr * ref_joints + ref_trans.unsqueeze(-2),
                    vertex_weights,
                    joint_weights,
                    glob_rotmats,
                    shape_betas,
                    ref_scale_corr,
                    ref_trans,
                    kid_factor,
                )
            else:
                glob_rotmats = self._fit_global_rotations_dependent(
                    target_vertices,
                    target_joints,
                    ref_verts + ref_trans.unsqueeze(-2),
                    ref_joints + ref_trans.unsqueeze(-2),
                    vertex_weights,
                    joint_weights,
                    glob_rotmats,
                    shape_betas,
                    None,
                    ref_trans,
                    kid_factor,
                )

        # Add the mean back
        result['trans'] = ref_trans + target_mean
        result['orientations'] = glob_rotmats

        if scale_fit:
            assert ref_scale_corr is not None
            result['scale_corr'] = ref_scale_corr

        # Provide other requested rotation formats
        if 'relative_orientations' in requested_keys or 'pose_rotvecs' in requested_keys:
            parent_glob_rotmats = torch.cat(
                [
                    torch.eye(3, device=device).expand(glob_rotmats.shape[0], 1, 3, 3),
                    torch.index_select(glob_rotmats, 1, parent_indices),
                ],
                dim=1,
            )
            result['relative_orientations'] = torch.matmul(
                parent_glob_rotmats.transpose(-1, -2), glob_rotmats
            )

        if 'pose_rotvecs' in requested_keys:
            rel_ori = result['relative_orientations']
            assert rel_ori is not None
            rotvecs = mat2rotvec(rel_ori)
            result['pose_rotvecs'] = rotvecs.view(rotvecs.shape[0], -1)

        result.pop('vertices')
        result.pop('joints')
        result_non_none: dict[str, torch.Tensor] = {}
        for k, v in result.items():
            if v is not None:
                result_non_none[k] = v
        return result_non_none

    def _fit_shape(
        self,
        glob_rotmats: torch.Tensor,
        target_vertices: torch.Tensor,
        target_joints: Optional[torch.Tensor] = None,
        vertex_weights: Optional[torch.Tensor] = None,
        joint_weights: Optional[torch.Tensor] = None,
        beta_regularizer: float = 1,
        beta_regularizer2: float = 0,
        scale_regularizer: float = 0,
        kid_regularizer: Optional[float] = None,
        share_beta: bool = False,
        scale_target: bool = False,
        scale_fit: bool = False,
        beta_regularizer_reference: Optional[torch.Tensor] = None,
        kid_regularizer_reference: Optional[torch.Tensor] = None,
        requested_keys: Optional[list[str]] = None,
    ) -> dict[str, torch.Tensor]:
        if scale_target and scale_fit:
            raise ValueError('Only one of estim_scale_target and estim_scale_fit can be True')
        if requested_keys is None:
            requested_keys = []

        glob_rotmats = glob_rotmats.float()
        batch_size = target_vertices.shape[0]

        device = self.body_model.v_template.device
        parent_indices = self.body_model.kintree_parents_tensor[1:].to(device)

        parent_glob_rot_mats = torch.cat(
            [
                torch.eye(3, device=device).expand(glob_rotmats.shape[0], 1, 3, 3),
                torch.index_select(glob_rotmats, 1, parent_indices),
            ],
            dim=1,
        )
        rel_rotmats = torch.matmul(parent_glob_rot_mats.transpose(-1, -2), glob_rotmats)

        glob_positions_ext_list = [self.J_template_ext[None, 0].expand(batch_size, -1, -1)]
        for i_joint, i_parent in enumerate(self.body_model.kintree_parents[1:], start=1):
            glob_positions_ext_list.append(
                glob_positions_ext_list[i_parent]
                + torch.einsum(
                    'bCc,cs->bCs',
                    glob_rotmats[:, i_parent],
                    self.J_template_ext[i_joint] - self.J_template_ext[i_parent],
                )
            )
        glob_positions_ext = torch.stack(glob_positions_ext_list, dim=1)
        translations_ext = glob_positions_ext - torch.einsum(
            'bjCc,jcs->bjCs', glob_rotmats, self.J_template_ext
        )

        rot_params = rel_rotmats[:, 1:].reshape(-1, (self.body_model.num_joints - 1) * 3 * 3)
        v_posed = self.body_model.v_template + torch.einsum(
            'vcp,bp->bvc', self.body_model.posedirs, rot_params
        )
        v_rotated = torch.einsum(
            'bjCc,vj,bvc->bvC', glob_rotmats, self.body_model.weights, v_posed
        )

        shapedirs = (
            torch.cat(
                [
                    self.body_model.shapedirs,
                    self.body_model.kid_shapedir[:, :, None],
                ],
                dim=2,
            )
            if self.enable_kid
            else self.body_model.shapedirs
        )
        v_grad_rotated = torch.einsum(
            'bjCc,lj,lcs->blCs', glob_rotmats, self.body_model.weights, shapedirs
        )

        v_rotated_ext = torch.cat([v_rotated.unsqueeze(-1), v_grad_rotated], dim=3)
        v_translations_ext = torch.einsum(
            'vj,bjcs->bvcs', self.body_model.weights, translations_ext
        )
        v_posed_posed_ext = v_translations_ext + v_rotated_ext

        if target_joints is None:
            target_both = target_vertices
            pos_both = v_posed_posed_ext[..., 0]
            jac_pos_both = v_posed_posed_ext[..., 1:]
        else:
            target_both = torch.cat([target_vertices, target_joints], dim=1)
            pos_both = torch.cat([v_posed_posed_ext[..., 0], glob_positions_ext[..., 0]], dim=1)
            jac_pos_both = torch.cat(
                [v_posed_posed_ext[..., 1:], glob_positions_ext[..., 1:]], dim=1
            )

        if scale_target:
            A = torch.cat([jac_pos_both, -target_both.unsqueeze(-1)], dim=3)
        elif scale_fit:
            A = torch.cat([jac_pos_both, pos_both.unsqueeze(-1)], dim=3)
        else:
            A = jac_pos_both

        b = target_both - pos_both
        # mean_A = torch.mean(A, dim=1, keepdim=True)
        # mean_b = torch.mean(b, dim=1, keepdim=True)
        # A = A - mean_A
        # b = b - mean_b

        if target_joints is not None and vertex_weights is not None and joint_weights is not None:
            weights = torch.cat([vertex_weights, joint_weights], dim=1)
        elif target_joints is None and vertex_weights is not None:
            weights = vertex_weights
        else:
            weights = torch.ones(A.shape[:2], dtype=torch.float32, device=device)

        n_params = (
            self.n_betas + (1 if self.enable_kid else 0) + (1 if scale_target or scale_fit else 0)
        )

        # print('A shape:', A.shape)
        # print('b shape:', b.shape)
        # print('weights shape:', weights.shape)

        w_sum_A = torch.sum(weights.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=True)
        w_sum_b = torch.sum(weights.unsqueeze(-1), dim=1, keepdim=True)
        mean_A = torch.where(
            w_sum_A == 0,
            torch.zeros_like(w_sum_A),
            torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * A, dim=1, keepdim=True) / w_sum_A,
        )
        mean_b = torch.where(
            w_sum_b == 0,
            torch.zeros_like(w_sum_b),
            torch.sum(weights.unsqueeze(-1) * b, dim=1, keepdim=True) / w_sum_b,
        )
        A = A - mean_A
        b = b - mean_b

        A = A.reshape(batch_size, -1, n_params)
        b = b.reshape(batch_size, -1, 1)
        w = torch.repeat_interleave(weights.reshape(batch_size, -1), 3, dim=1)

        l2_regularizer_all = torch.cat(
            [
                torch.full((2,), beta_regularizer2, device=device),
                torch.full((self.n_betas - 2,), beta_regularizer, device=device),
            ]
        )
        if beta_regularizer_reference is None:
            l2_regularizer_reference_all = torch.zeros([batch_size, self.n_betas], device=device)
        else:
            n_given = beta_regularizer_reference.shape[1]
            if n_given < self.n_betas:
                l2_regularizer_reference_all = torch.nn.functional.pad(
                    beta_regularizer_reference, (0, self.n_betas - n_given)
                )
            else:
                l2_regularizer_reference_all = beta_regularizer_reference

        if self.enable_kid:
            if kid_regularizer is None:
                kid_regularizer = beta_regularizer
            if kid_regularizer_reference is None:
                kid_regularizer_reference = torch.zeros(batch_size, device=device)
            l2_regularizer_all = torch.cat(
                [l2_regularizer_all, torch.tensor([kid_regularizer], device=device)]
            )
            l2_regularizer_reference_all = torch.cat(
                [l2_regularizer_reference_all, kid_regularizer_reference[:, np.newaxis]], dim=1
            )

        if scale_target or scale_fit:
            l2_regularizer_all = torch.cat(
                [l2_regularizer_all, torch.tensor([scale_regularizer], device=device)]
            )
            l2_regularizer_reference_all = torch.cat(
                [l2_regularizer_reference_all, torch.zeros([batch_size, 1], device=device)], dim=1
            )

        l2_regularizer_rhs = (l2_regularizer_all * l2_regularizer_reference_all).unsqueeze(-1)

        # l2_regularizer_all = torch.diag(l2_regularizer_all)
        # print('Loading penalty matrix')
        # penalty = np.load('/work/sarandi/data/projects/localizerfields
        # /smpl_beta_penalty_mat_new.npy')
        # penalty_bone = np.load('/work/sarandi/data/projects/localizerfields
        # /smpl_beta_penalty_mat_bone_new.npy')
        # l2_regularizer_all[:self.n_betas, :self.n_betas] = torch.tensor(penalty, device=device)
        # * beta_regularizer + torch.tensor(penalty_bone, device=device) * beta_regularizer2

        if share_beta:
            x = lstsq_partial_share(
                A,
                b,
                w,
                l2_regularizer_all,
                l2_regularizer_rhs=l2_regularizer_rhs,
                n_shared=self.n_betas + (1 if self.enable_kid else 0),
            )

        else:
            x = lstsq(A, b, w, l2_regularizer_all, l2_regularizer_rhs=l2_regularizer_rhs)

        x = x.squeeze(-1)
        new_trans = mean_b.squeeze(1) - torch.matmul(mean_A.squeeze(1), x.unsqueeze(-1)).squeeze(
            -1
        )
        new_shape = x[:, : self.n_betas]

        result = dict(shape_betas=new_shape, trans=new_trans, relative_orientations=rel_rotmats)

        if self.enable_kid:
            new_kid_factor = x[:, self.n_betas]
            result['kid_factor'] = new_kid_factor
        else:
            new_kid_factor = None

        if scale_target or scale_fit:
            new_scale_corr = x[:, -1] + 1
            if scale_fit and new_scale_corr is not None:
                new_shape = new_shape / new_scale_corr.unsqueeze(-1)
                if new_kid_factor is not None:
                    new_kid_factor = new_kid_factor / new_scale_corr
            result['scale_corr'] = new_scale_corr
        else:
            new_scale_corr = None

        if self.enable_kid and new_kid_factor is not None:
            new_shape = torch.cat([new_shape, new_kid_factor.unsqueeze(-1)], dim=1)

        if 'joints' in requested_keys:
            result['joints'] = (
                glob_positions_ext[..., 0]
                + torch.einsum('bvcs,bs->bvc', glob_positions_ext[..., 1:], new_shape)
                + new_trans.unsqueeze(1)
            )

        if 'vertices' in requested_keys:
            result['vertices'] = (
                v_posed_posed_ext[..., 0]
                + torch.einsum('bvcs,bs->bvc', v_posed_posed_ext[..., 1:], new_shape)
                + new_trans.unsqueeze(1)
            )
        return result

    def _fit_global_rotations(
        self,
        target_vertices: torch.Tensor,
        target_joints: Optional[torch.Tensor],
        reference_vertices: torch.Tensor,
        reference_joints: Optional[torch.Tensor],
        vertex_weights: Optional[torch.Tensor],
        joint_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Global orientation of each body part via swing-twist decomposition, batched.

        Parts containing >= 3 joints get a Kabsch fit from their joints alone; leaf parts
        (1 joint) from their vertices alone. Bone parts (2 joints) are
        solved in two exactly-determined steps: the swing aligns the bone direction, and
        the twist about the bone is recovered from the vertices in closed form. For the
        twist, with ``H = R_swing A^T`` where ``A = sum w t̄ ā^T`` is the part's centered
        cross-covariance, the optimal angle is ``atan2(b̂ . vee(H), tr(H) - b̂^T H b̂)``.
        This is the ``mesh_weight -> 0`` limit of a weighted Kabsch fit, computed without
        the ill-conditioned SVD that such weighting would produce (stable gradients).
        """
        if target_joints is None or reference_joints is None:
            target_joints = self.body_model.J_regressor_post_lbs @ target_vertices
            reference_joints = self.body_model.J_regressor_post_lbs @ reference_vertices

        B = target_vertices.shape[0]

        # Per-part vertex cross-covariances about the children-mean centers, loop-free.
        raw, s_t, s_a, s_w = self._part_sums(target_vertices, reference_vertices, vertex_weights)
        mt = self.center_matrix @ target_joints  # (B, J, 3)
        ma = self.center_matrix @ reference_joints  # (B_ref, J, 3)
        A_vert = (
            raw
            - s_t.unsqueeze(-1) * ma.unsqueeze(-2)
            - mt.unsqueeze(-1) * s_a.unsqueeze(-2)
            + s_w.unsqueeze(-1) * (mt.unsqueeze(-1) * ma.unsqueeze(-2))
        )  # (B, J, 3, 3)

        # Joint-point cross-covariances for the multi-joint parts, loop-free.
        rj = reference_joints
        if joint_weights is not None:
            rj = rj * joint_weights.unsqueeze(-1)
            tj_sum_side = target_joints * joint_weights.unsqueeze(-1)
            s_wj = self.mjp_joint_membership @ joint_weights.unsqueeze(-1)
        else:
            tj_sum_side = target_joints
            s_wj = self.mjp_joint_counts
        outer_j = (target_joints.unsqueeze(-1) * rj.unsqueeze(-2)).reshape(
            B, target_joints.shape[1], 9
        )
        raw_j = (self.mjp_joint_membership @ outer_j).view(B, -1, 3, 3)
        mtj = self.mjp_center_matrix @ target_joints
        maj = self.mjp_center_matrix @ reference_joints
        s_tj = self.mjp_joint_membership @ tj_sum_side
        s_aj = self.mjp_joint_membership @ rj
        A_multi = (
            raw_j
            - s_tj.unsqueeze(-1) * maj.unsqueeze(-2)
            - mtj.unsqueeze(-1) * s_aj.unsqueeze(-2)
            + s_wj.unsqueeze(-1) * (mtj.unsqueeze(-1) * maj.unsqueeze(-2))
        )

        # Kabsch bucket (multi-joint parts + leaf parts): one batched SVD.
        A_svd = torch.cat([A_multi, A_vert[:, self.leaf_parts]], dim=1)
        R_svd = proj_SO3(A_svd)

        # Bone bucket: batched swing (bone alignment) + twist (from vertices).
        b_ref = (
            reference_joints[:, self.bone_pairs[:, 1]] - reference_joints[:, self.bone_pairs[:, 0]]
        )
        b_tgt = target_joints[:, self.bone_pairs[:, 1]] - target_joints[:, self.bone_pairs[:, 0]]
        b_ref_n = divide_no_nan(b_ref, torch.linalg.norm(b_ref, dim=-1, keepdim=True))
        b_tgt_n = divide_no_nan(b_tgt, torch.linalg.norm(b_tgt, dim=-1, keepdim=True))
        R_swing = align_unit_vectors(b_ref_n, b_tgt_n)  # (B, n_bones, 3, 3)

        H = R_swing @ A_vert[:, self.bone_parts].mT
        trH = H.diagonal(dim1=-2, dim2=-1).sum(-1)
        bHb = (b_tgt_n.unsqueeze(-2) @ H @ b_tgt_n.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        # vee_i = eps_ijk H_jk: the vertex cross-product sums, extracted from H.
        vee = torch.stack(
            [
                H[..., 1, 2] - H[..., 2, 1],
                H[..., 2, 0] - H[..., 0, 2],
                H[..., 0, 1] - H[..., 1, 0],
            ],
            dim=-1,
        )
        twist_angle = torch.atan2((b_tgt_n * vee).sum(-1), trH - bHb)
        R_twist = rotvec2mat(b_tgt_n * twist_angle.unsqueeze(-1))
        R_bone = R_twist @ R_swing

        # Scatter both buckets back to per-part order (toe parts take the feet slots).
        R_concat = torch.cat([R_svd, R_bone], dim=1)
        return R_concat[:, self.assemble_indices]

    def _fit_global_rotations_dependent(
        self,
        target_vertices: torch.Tensor,
        target_joints: Optional[torch.Tensor],
        reference_vertices: torch.Tensor,
        reference_joints: torch.Tensor,
        vertex_weights: Optional[torch.Tensor],
        joint_weights: Optional[torch.Tensor],
        glob_rots_prev: torch.Tensor,
        shape_betas: torch.Tensor,
        scale_corr: Optional[torch.Tensor],
        trans: torch.Tensor,
        kid_factor: Optional[torch.Tensor],
    ) -> torch.Tensor:
        glob_rots: list[torch.Tensor] = []

        true_reference_joints = reference_joints
        if target_joints is None or reference_joints is None:
            target_joints = self.body_model.J_regressor_post_lbs @ target_vertices
            reference_joints = self.body_model.J_regressor_post_lbs @ reference_vertices
        if true_reference_joints is None:
            true_reference_joints = reference_joints

        device = self.body_model.v_template.device
        j = self.body_model.J_template + torch.einsum(
            'jcs,...s->...jc',
            self.body_model.J_shapedirs,
            shape_betas[:, : self.n_betas],
        )
        if kid_factor is not None:
            j = j + torch.einsum('jc,...->...jc', self.body_model.kid_J_shapedir, kid_factor)

        if scale_corr is not None:
            j = j * scale_corr

        parent_indices = self.body_model.kintree_parents_tensor[1:].to(device)
        j_parent = torch.cat(
            [
                torch.zeros(1, 3, device=device).expand(j.shape[0], -1, -1),
                torch.index_select(j, 1, parent_indices),
            ],
            dim=1,
        )
        bones = j - j_parent

        # Per-part vertex statistics, shared machinery with _fit_global_rotations. The
        # sequential loop below only needs 3x3 algebra per joint: the cross-covariance
        # about the dynamic centers follows algebraically from these fixed sums.
        raw, s_t, s_a, s_w = self._part_sums(target_vertices, reference_vertices, vertex_weights)

        glob_positions: list[torch.Tensor] = []

        for i in range(self.body_model.num_joints):
            if i == 0:
                glob_position = j[:, i] + trans
            else:
                i_parent = self.body_model.kintree_parents[i]
                glob_position = glob_positions[i_parent] + torch.matmul(
                    glob_rots[i_parent], bones[:, i].unsqueeze(-1)
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
                - s_t[:, i].unsqueeze(-1) * c_a.unsqueeze(-2)
                - c_t.unsqueeze(-1) * s_a[:, i].unsqueeze(-2)
                + s_w[:, i].unsqueeze(-1) * (c_t.unsqueeze(-1) * c_a.unsqueeze(-2))
            )  # (B, 3, 3)

            # Joint contribution (children_and_self), same weighting as before.
            joint_selector = self.children_and_self[i]
            estim_joints = target_joints[:, joint_selector] - c_t.unsqueeze(1)
            default_joints = reference_joints[:, joint_selector] - c_a.unsqueeze(1)
            if joint_weights is not None:
                default_joints = default_joints * joint_weights[:, joint_selector].unsqueeze(-1)
            A_joint = estim_joints.mT @ default_joints  # (B, 3, 3)

            glob_rot = proj_SO3(A_vert + A_joint) @ glob_rots_prev[:, i]
            glob_rots.append(glob_rot)

        return torch.stack(glob_rots, dim=1)


def fit_scale_and_translation(
    target_vertices: torch.Tensor,
    reference_vertices: torch.Tensor,
    target_joints: Optional[torch.Tensor],
    reference_joints: torch.Tensor,
    vertex_weights: Optional[torch.Tensor] = None,
    joint_weights: Optional[torch.Tensor] = None,
    scale: bool = False,
) -> tuple[Optional[torch.Tensor], torch.Tensor]:
    device = target_vertices.device

    if target_joints is None or reference_joints is None:
        target_both = target_vertices
        reference_both = reference_vertices
        if vertex_weights is not None:
            weights_both = vertex_weights
        else:
            weights_both = torch.ones(
                target_vertices.shape[0], target_vertices.shape[1], device=device
            )
    else:
        target_both = torch.cat([target_vertices, target_joints], dim=1)
        reference_both = torch.cat([reference_vertices, reference_joints], dim=1)

        if vertex_weights is not None and joint_weights is not None:
            weights_both = torch.cat([vertex_weights, joint_weights], dim=1)
        else:
            weights_both = torch.ones(
                target_vertices.shape[0],
                target_vertices.shape[1] + target_joints.shape[1],
                device=device,
            )

    weights_both = weights_both / torch.sum(weights_both, dim=1, keepdim=True)

    weighted_mean_target = torch.sum(target_both * weights_both.unsqueeze(-1), dim=1)
    weighted_mean_reference = torch.sum(reference_both * weights_both.unsqueeze(-1), dim=1)

    if scale:
        target_centered = target_both - weighted_mean_target[:, None]
        reference_centered = reference_both - weighted_mean_reference[:, None]

        ssq_reference = torch.sum(reference_centered**2 * weights_both.unsqueeze(-1), dim=(1, 2))
        ssq_target = torch.sum(target_centered**2 * weights_both.unsqueeze(-1), dim=(1, 2))

        # to make it unbiased, we could multiply by (1+2/(target_both.shape[1]))
        # but we are okay with the least squares solution
        scale_factor = torch.sqrt(ssq_target / ssq_reference)
        trans = weighted_mean_target - scale_factor * weighted_mean_reference
    else:
        scale_factor = None
        trans = weighted_mean_target - weighted_mean_reference

    return scale_factor, trans
