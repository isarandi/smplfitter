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

        # Index buffers for all part/joint gathers: indexing CUDA tensors with Python
        # lists would materialize the index tensor on CPU and copy it over at every call,
        # which also blocks CUDA graph capture. index_select on these buffers is free.
        self.leaf_part_indices = nn.Buffer(torch.tensor(leaf_parts, dtype=torch.int64))
        self.bone_part_indices = nn.Buffer(torch.tensor(bone_parts, dtype=torch.int64))
        self.root_index = nn.Buffer(torch.tensor([0], dtype=torch.int64))
        self.toe_part_indices = nn.Buffer(torch.tensor([10, 11], dtype=torch.int64))
        self.foot_part_indices = nn.Buffer(torch.tensor([7, 8], dtype=torch.int64))

        # Per-part joint indices (children_and_self), flattened with start offsets, for
        # the sequential final-adjust pass.
        cas_starts = [0]
        for i in range(J):
            cas_starts.append(cas_starts[-1] + len(self.children_and_self[i]))
        self.cas_starts = cas_starts
        self.cas_flat = nn.Buffer(
            torch.tensor([j for js in self.children_and_self for j in js], dtype=torch.int64)
        )

        # --- Kinematic-tree levels (root = level 0) for level-batched forward kinematics ---
        # The sequential per-joint FK loop is reformulated as one batched update per tree
        # level (8 levels for SMPL): joints of a level are independent given their parents,
        # which are final by then. Bit-exact reformulation of the per-joint loop.
        parents = body_model.kintree_parents
        depth = [0] * J
        for i in range(1, J):
            depth[i] = depth[parents[i]] + 1
        levels = [[i for i in range(J) if depth[i] == d] for d in range(1, max(depth) + 1)]
        self.num_fk_levels = len(levels)
        self.fk_level_sizes = [len(js) for js in levels]
        fk_js = [i for js in levels for i in js]
        self.fk_js = nn.Buffer(torch.tensor(fk_js, dtype=torch.int64))
        self.fk_ps = nn.Buffer(torch.tensor([parents[i] for i in fk_js], dtype=torch.int64))
        parents_with_root = [0] + list(parents[1:])
        self.bone_ext = nn.Buffer(self.J_template_ext - self.J_template_ext[parents_with_root])

        # --- Shape-Jacobian precompute for the split-Gramian solve in _fit_shape ---
        # jac_shapedirs[(j, c), (l, s)] = weights[l, j] * shapedirs[l, c, s]: one GEMM of
        # the flattened global rotations against this matrix yields the rotated shape
        # Jacobian directly in coordinate-major (B, 3, V, S) layout, avoiding einsum's
        # pathological contraction paths. Only precomputed when reasonably small (it
        # scales with num_betas); the kid blendshape needs the general solve anyway.
        V = body_model.num_vertices
        S = self.n_betas
        self.gram_supported = not enable_kid and J * 3 * V * S <= 2**26
        if self.gram_supported:
            jac_shapedirs = torch.einsum(
                'lj,lcs->jcls', body_model.weights, body_model.shapedirs
            ).reshape(J * 3, V * S)
        else:
            jac_shapedirs = torch.zeros(0)
        self.jac_shapedirs = nn.Buffer(jac_shapedirs)

        # --- Level-batched final-adjust precompute ---
        # The adjustable parts of SMPL-family models sit in tree levels of two independent
        # parts each (hips, knees, ankles, shoulders, elbows), so the final adjustment can
        # run level by level: batched FK position updates interleaved (in tree order) with
        # per-level part statistics and one batched proj_SO3 per level. This requires all
        # adjustable parts to contain the same number of joints so the joint statistics
        # form a fixed-width gather; models where that fails (e.g. MANO/FLAME, where every
        # part is adjustable) use the sequential per-joint pass instead.
        adjustable_set = set(self.adjustable_parts)
        joint_counts = {len(self.children_and_self[i]) for i in adjustable_set}
        self.leveladj_supported = self.is_smpl_family and len(joint_counts) == 1
        adj_levels = [[i for i in js if i in adjustable_set] for js in levels]
        self.adj_level_sizes = [len(a) for a in adj_levels]
        self.adj_last_level = max((k for k, a in enumerate(adj_levels) if len(a) > 0), default=-1)
        adj_flat = [i for a in adj_levels for i in a]
        self.adj_parts = nn.Buffer(torch.tensor(adj_flat, dtype=torch.int64))
        if self.leveladj_supported:
            self.adj_n_joints = joint_counts.pop()
            adj_joints = [j for i in adj_flat for j in self.children_and_self[i]]
        else:
            self.adj_n_joints = 0
            adj_joints = []
        self.adj_part_joints = nn.Buffer(torch.tensor(adj_joints, dtype=torch.int64))

    def _part_sums(
        self,
        target_vertices: torch.Tensor,
        reference_vertices: torch.Tensor,
        vertex_weights: Optional[torch.Tensor],
        share_beta: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-part sufficient statistics for cross-covariances, computed loop-free.

        Returns per-part weighted sums over each part's vertices:
        ``raw = sum w t a^T`` (B, J, 3, 3), ``s_t = sum w t`` (B, J, 3),
        ``s_a = sum w a`` (B_ref, J, 3), ``s_w = sum w`` (B or 1, J, 1).
        The centered cross-covariance about any centers (c_t, c_a) then follows as
        ``raw - s_t c_a^T - c_t s_a^T + s_w c_t c_a^T``.

        The ``s_t``/``s_a`` sums normally run as single flat GEMMs (batch folded into the
        columns), since batched (B, J, N) @ (B, N, 3) products underutilize the GPU at
        these shapes. Shared-beta fits keep the batched form instead: they must reproduce
        the reference reduction order exactly, because the shared-shape pipeline
        chaotically amplifies float-level reduction noise (to ~2e-3 in pose_rotvecs at
        the ankle parts).
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
        N = t.shape[1]
        outer = (t.unsqueeze(-1) * a.unsqueeze(-2)).reshape(B, N, 9)
        raw = (self.part_matrix @ outer).view(B, -1, 3, 3)
        if share_beta:
            s_t = self.part_matrix @ t_sum_side
            s_a = self.part_matrix @ a
        else:
            J = self.part_matrix.shape[0]
            t_flat = t_sum_side.permute(1, 0, 2).reshape(N, -1)  # (N, B*3)
            a_flat = a.permute(1, 0, 2).reshape(N, -1)  # (N, B_ref*3)
            s_t = (self.part_matrix @ t_flat).view(J, -1, 3).permute(1, 0, 2).contiguous()
            s_a = (self.part_matrix @ a_flat).view(J, -1, 3).permute(1, 0, 2).contiguous()
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
                    share_beta,
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
                share_beta,
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
                    share_beta,
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
                    share_beta,
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
                    share_beta,
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
                    share_beta,
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

        # Forward kinematics of the shape-dependent joint positions (position and its
        # Jacobian w.r.t. the betas), one batched update per kinematic-tree level.
        n_ext = self.J_template_ext.shape[2]
        glob_positions_ext = torch.empty(
            batch_size,
            self.body_model.num_joints,
            3,
            n_ext,
            device=device,
            dtype=self.J_template_ext.dtype,
        )
        glob_positions_ext.index_copy_(
            1, self.root_index, self.J_template_ext[:1].expand(batch_size, 1, 3, n_ext)
        )
        level_start = 0
        for k in range(self.num_fk_levels):
            level_size = self.fk_level_sizes[k]
            js = self.fk_js.narrow(0, level_start, level_size)
            ps = self.fk_ps.narrow(0, level_start, level_size)
            level_start += level_size
            glob_positions_ext.index_copy_(
                1,
                js,
                glob_positions_ext.index_select(1, ps)
                + torch.einsum(
                    'bnCc,ncs->bnCs',
                    glob_rotmats.index_select(1, ps),
                    self.bone_ext.index_select(0, js),
                ),
            )

        translations_ext = glob_positions_ext - torch.einsum(
            'bjCc,jcs->bjCs', glob_rotmats, self.J_template_ext
        )

        rot_params = rel_rotmats[:, 1:].reshape(-1, (self.body_model.num_joints - 1) * 3 * 3)
        v_posed = self.body_model.v_template + torch.einsum(
            'vcp,bp->bvc', self.body_model.posedirs, rot_params
        )

        # The shape solve has two implementations. The split-Gramian solve handles the
        # common case; the general solve covers the extra scale and kid unknowns, models
        # whose Jacobian precompute would be too large (see __init__), and share_beta,
        # which must keep the reference summation order exactly (see _part_sums).
        if self.gram_supported and not (share_beta or scale_target or scale_fit):
            return self._fit_shape_gram(
                glob_rotmats,
                glob_positions_ext,
                translations_ext,
                rel_rotmats,
                v_posed,
                target_vertices,
                target_joints,
                vertex_weights,
                joint_weights,
                beta_regularizer,
                beta_regularizer2,
                beta_regularizer_reference,
                requested_keys,
            )
        return self._fit_shape_general(
            glob_rotmats,
            glob_positions_ext,
            translations_ext,
            rel_rotmats,
            v_posed,
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
            beta_regularizer_reference,
            kid_regularizer_reference,
            requested_keys,
        )

    def _fit_shape_gram(
        self,
        glob_rotmats: torch.Tensor,
        glob_positions_ext: torch.Tensor,
        translations_ext: torch.Tensor,
        rel_rotmats: torch.Tensor,
        v_posed: torch.Tensor,
        target_vertices: torch.Tensor,
        target_joints: Optional[torch.Tensor],
        vertex_weights: Optional[torch.Tensor],
        joint_weights: Optional[torch.Tensor],
        beta_regularizer: float,
        beta_regularizer2: float,
        beta_regularizer_reference: Optional[torch.Tensor],
        requested_keys: list[str],
    ) -> dict[str, torch.Tensor]:
        """Shape and translation solve via per-block normal equations (Gramians).

        The vertex and joint blocks of the least-squares system are never concatenated;
        their normal equations are accumulated separately and added:
        ``G = Av^T Wv Av + Aj^T Wj Aj`` and ``r = Av^T Wv bv + Aj^T Wj bj``. The
        per-coordinate weighted-mean centering of the general solve is reproduced
        algebraically through the covariance identity (``mu_A = SA / W``,
        ``mu_b = Sb / W``): ``G_cen = G - SA^T SA / W`` and ``r_cen = r - SA^T Sb / W``,
        where ``SA = sum_n w_n A_n`` and ``Sb = sum_n w_n b_n``. The large GEMMs run in
        float32; the small (B, S, S) combination, centering and solve run in float64 to
        absorb the cancellation in this identity. The translation then follows exactly as
        in the general solve: ``trans = mean_b - mean_A @ x``.

        The vertex block is built in coordinate-major (B, 3, V, S) layout: the rotated
        shape Jacobian is one GEMM against the precomputed ``jac_shapedirs`` matrix, and
        the LBS rotation is an explicit blended-rotation multiply-reduce, avoiding
        einsum's pathological contraction paths on these shapes.
        """
        batch_size = target_vertices.shape[0]
        num_vertices = self.body_model.num_vertices
        n_betas = self.n_betas
        device = self.body_model.v_template.device

        # Blended per-vertex rotations and the position part of the LBS.
        rot_blend = torch.matmul(
            self.body_model.weights, glob_rotmats.reshape(batch_size, -1, 9)
        ).view(batch_size, num_vertices, 3, 3)
        pos_vmajor = (rot_blend * v_posed.unsqueeze(-2)).sum(-1)  # (B, V, 3)

        # Shape Jacobian via one GEMM:
        # jac[b, C, l, s] = sum_{j, c} rot[b, j, C, c] weights[l, j] shapedirs[l, c, s]
        rot_rows = glob_rotmats.permute(0, 2, 1, 3).reshape(batch_size * 3, -1)
        jac = torch.mm(rot_rows, self.jac_shapedirs).view(batch_size, 3, num_vertices, n_betas)

        # Skinned translation offsets for both the position and the Jacobian.
        trans_offsets = torch.matmul(
            self.body_model.weights, translations_ext.permute(0, 2, 1, 3)
        )  # (B, 3, V, S+1)
        jac += trans_offsets[..., 1:]
        pos = pos_vmajor.permute(0, 2, 1) + trans_offsets[..., 0]  # (B, 3, V)
        b = target_vertices.mT - pos  # (B, 3, V)

        # Weights only take effect if both vertex and joint weights are given (when
        # joints are used), or if vertex weights are given without joints.
        if target_joints is not None and vertex_weights is not None and joint_weights is not None:
            vw: Optional[torch.Tensor] = vertex_weights
            jw: Optional[torch.Tensor] = joint_weights
        elif target_joints is None and vertex_weights is not None:
            vw = vertex_weights
            jw = None
        else:
            vw = None
            jw = None

        # Vertex-block normal equations (rows = (coordinate, vertex); order irrelevant).
        jac_flat = jac.reshape(batch_size, 3 * num_vertices, n_betas)
        b_flat = b.reshape(batch_size, 3 * num_vertices, 1)
        if vw is None:
            gram = (jac_flat.mT @ jac_flat).double()
            rhs = (jac_flat.mT @ b_flat).double()
            sum_A = jac.sum(dim=2).double()  # (B, 3, S)
            sum_b = b.sum(dim=2).unsqueeze(-1).double()  # (B, 3, 1)
            w_sum = torch.full(
                (batch_size, 1, 1), float(num_vertices), device=device, dtype=torch.float64
            )
        else:
            wjac = jac * vw[:, None, :, None]
            wjac_flat = wjac.reshape(batch_size, 3 * num_vertices, n_betas)
            gram = (wjac_flat.mT @ jac_flat).double()
            rhs = (wjac_flat.mT @ b_flat).double()
            sum_A = wjac.sum(dim=2).double()
            sum_b = (b * vw[:, None, :]).sum(dim=2).unsqueeze(-1).double()
            w_sum = vw.sum(dim=1).view(batch_size, 1, 1).double()

        if target_joints is not None:
            gram_j, rhs_j, sum_A_j, sum_b_j, w_sum_j = _gram_block(
                glob_positions_ext[..., 1:], target_joints - glob_positions_ext[..., 0], jw
            )
            gram = gram + gram_j
            rhs = rhs + rhs_j
            sum_A = sum_A + sum_A_j
            sum_b = sum_b + sum_b_j
            w_sum = w_sum + w_sum_j

        w_sum_safe = torch.where(w_sum == 0, torch.ones_like(w_sum), w_sum)
        gram_cen = gram - sum_A.mT @ sum_A / w_sum_safe
        rhs_cen = rhs - sum_A.mT @ sum_b / w_sum_safe

        l2_regularizer_all = torch.cat(
            [
                torch.full((2,), float(beta_regularizer2), dtype=torch.float64, device=device),
                torch.full(
                    (n_betas - 2,), float(beta_regularizer), dtype=torch.float64, device=device
                ),
            ]
        )
        if beta_regularizer_reference is None:
            l2_reference = torch.zeros(batch_size, n_betas, dtype=torch.float64, device=device)
        else:
            l2_reference = beta_regularizer_reference.double()
            n_given = l2_reference.shape[1]
            if n_given < n_betas:
                l2_reference = torch.nn.functional.pad(l2_reference, [0, n_betas - n_given])
            else:
                l2_reference = l2_reference[:, :n_betas]
        l2_regularizer_rhs = (l2_regularizer_all * l2_reference).unsqueeze(-1)

        chol, _ = torch.linalg.cholesky_ex(gram_cen + torch.diag(l2_regularizer_all))
        x = torch.cholesky_solve(rhs_cen + l2_regularizer_rhs, chol)

        mean_A = sum_A / w_sum_safe  # (B, 3, S)
        mean_b = sum_b / w_sum_safe  # (B, 3, 1)
        new_trans = (mean_b - mean_A @ x).squeeze(-1).float()
        new_shape = x.squeeze(-1).float()

        result = dict(shape_betas=new_shape, trans=new_trans, relative_orientations=rel_rotmats)

        if 'joints' in requested_keys:
            result['joints'] = (
                glob_positions_ext[..., 0]
                + torch.einsum('bvcs,bs->bvc', glob_positions_ext[..., 1:], new_shape)
                + new_trans.unsqueeze(1)
            )
        if 'vertices' in requested_keys:
            verts = pos + (jac_flat @ new_shape.unsqueeze(-1)).view(batch_size, 3, num_vertices)
            result['vertices'] = (verts + new_trans.unsqueeze(-1)).mT.contiguous()
        return result

    def _fit_shape_general(
        self,
        glob_rotmats: torch.Tensor,
        glob_positions_ext: torch.Tensor,
        translations_ext: torch.Tensor,
        rel_rotmats: torch.Tensor,
        v_posed: torch.Tensor,
        target_vertices: torch.Tensor,
        target_joints: Optional[torch.Tensor],
        vertex_weights: Optional[torch.Tensor],
        joint_weights: Optional[torch.Tensor],
        beta_regularizer: float,
        beta_regularizer2: float,
        scale_regularizer: float,
        kid_regularizer: Optional[float],
        share_beta: bool,
        scale_target: bool,
        scale_fit: bool,
        beta_regularizer_reference: Optional[torch.Tensor],
        kid_regularizer_reference: Optional[torch.Tensor],
        requested_keys: list[str],
    ) -> dict[str, torch.Tensor]:
        """Shape and translation solve over the concatenated vertex and joint residuals.

        Handles all unknowns beyond the betas (kid factor, scale correction) and the
        share_beta coupling across the batch, at the cost of materializing the full
        stacked design matrix.
        """
        batch_size = target_vertices.shape[0]
        device = self.body_model.v_template.device

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

        # In-place add avoids allocating a third full-size (B, V, 3, S+1) tensor.
        v_posed_posed_ext = torch.cat([v_rotated.unsqueeze(-1), v_grad_rotated], dim=3)
        v_posed_posed_ext += torch.einsum(
            'vj,bjcs->bvcs', self.body_model.weights, translations_ext
        )

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
        share_beta: bool = False,
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
        raw, s_t, s_a, s_w = self._part_sums(
            target_vertices, reference_vertices, vertex_weights, share_beta
        )
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

        # Kabsch bucket (multi-joint parts + leaf parts): one batched projection.
        A_kabsch = torch.cat([A_multi, A_vert.index_select(1, self.leaf_part_indices)], dim=1)
        R_kabsch = proj_SO3(A_kabsch)

        # Bone bucket: batched swing (bone alignment) + twist (from vertices).
        b_ref = (
            reference_joints[:, self.bone_pairs[:, 1]] - reference_joints[:, self.bone_pairs[:, 0]]
        )
        b_tgt = target_joints[:, self.bone_pairs[:, 1]] - target_joints[:, self.bone_pairs[:, 0]]
        b_ref_n = divide_no_nan(b_ref, torch.linalg.norm(b_ref, dim=-1, keepdim=True))
        b_tgt_n = divide_no_nan(b_tgt, torch.linalg.norm(b_tgt, dim=-1, keepdim=True))
        R_swing = align_unit_vectors(b_ref_n, b_tgt_n)  # (B, n_bones, 3, 3)

        H = R_swing @ A_vert.index_select(1, self.bone_part_indices).mT
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
        R_concat = torch.cat([R_kabsch, R_bone], dim=1)
        return R_concat.index_select(1, self.assemble_indices)

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
        share_beta: bool = False,
    ) -> torch.Tensor:
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
        # loops below only need 3x3 algebra per part: the cross-covariance about the
        # dynamic centers follows algebraically from these fixed sums.
        raw, s_t, s_a, s_w = self._part_sums(
            target_vertices, reference_vertices, vertex_weights, share_beta
        )

        batch_size = target_vertices.shape[0]

        if self.leveladj_supported:
            # Level-batched adjustment: FK position updates run one kinematic-tree level
            # at a time, interleaved (in tree order) with the orientation refinement of
            # the adjustable parts on that level, so each level needs only one batched
            # proj_SO3 instead of one per part. Non-adjustable parts keep their previous
            # orientation.
            rots = glob_rots_prev.clone()
            positions = torch.empty(
                batch_size, self.body_model.num_joints, 3, device=device, dtype=j.dtype
            )
            positions.index_copy_(1, self.root_index, j[:, :1] + trans.unsqueeze(1))

            level_start = 0
            adj_start = 0
            for k in range(self.adj_last_level + 1):
                level_size = self.fk_level_sizes[k]
                js = self.fk_js.narrow(0, level_start, level_size)
                ps = self.fk_ps.narrow(0, level_start, level_size)
                level_start += level_size
                # FK for this level: parents (previous levels) are final by now.
                positions.index_copy_(
                    1,
                    js,
                    positions.index_select(1, ps)
                    + (rots.index_select(1, ps) @ bones.index_select(1, js).unsqueeze(-1)).squeeze(
                        -1
                    ),
                )

                n_adj = self.adj_level_sizes[k]
                cp_start = adj_start * self.adj_n_joints
                adj_start += n_adj
                if n_adj == 0:
                    continue
                adj = self.adj_parts.narrow(0, adj_start - n_adj, n_adj)
                cp = self.adj_part_joints.narrow(0, cp_start, n_adj * self.adj_n_joints)

                # Vertex contribution: centered cross-covariance about the dynamic
                # centers (c_t = current global joint position, c_a = reference one).
                c_t = positions.index_select(1, adj)  # (B, n, 3)
                c_a = true_reference_joints.index_select(1, adj)  # (B_ref, n, 3)
                A_vert = (
                    raw.index_select(1, adj)
                    - s_t.index_select(1, adj).unsqueeze(-1) * c_a.unsqueeze(-2)
                    - c_t.unsqueeze(-1) * s_a.index_select(1, adj).unsqueeze(-2)
                    + s_w.index_select(1, adj).unsqueeze(-1)
                    * (c_t.unsqueeze(-1) * c_a.unsqueeze(-2))
                )  # (B, n, 3, 3)

                # Joint contribution (children_and_self), same weighting as before.
                estim_joints = target_joints.index_select(1, cp).view(
                    batch_size, n_adj, self.adj_n_joints, 3
                ) - c_t.unsqueeze(-2)
                default_joints = reference_joints.index_select(1, cp).view(
                    -1, n_adj, self.adj_n_joints, 3
                ) - c_a.unsqueeze(-2)
                if joint_weights is not None:
                    default_joints = default_joints * joint_weights.index_select(1, cp).view(
                        -1, n_adj, self.adj_n_joints
                    ).unsqueeze(-1)
                A_joint = estim_joints.mT @ default_joints  # (B, n, 3, 3)

                rots.index_copy_(
                    1,
                    adj,
                    proj_SO3(A_vert + A_joint) @ glob_rots_prev.index_select(1, adj),
                )

            if self.is_smpl_family:
                # Toe parts copy the feet.
                rots.index_copy_(
                    1, self.toe_part_indices, rots.index_select(1, self.foot_part_indices)
                )
            return rots

        # Sequential fallback: parts are refined one at a time in tree order.
        glob_rots: list[torch.Tensor] = []
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
            joint_selector = self.cas_flat[self.cas_starts[i] : self.cas_starts[i + 1]]
            estim_joints = target_joints.index_select(1, joint_selector) - c_t.unsqueeze(1)
            default_joints = reference_joints.index_select(1, joint_selector) - c_a.unsqueeze(1)
            if joint_weights is not None:
                default_joints = default_joints * joint_weights.index_select(
                    1, joint_selector
                ).unsqueeze(-1)
            A_joint = estim_joints.mT @ default_joints  # (B, 3, 3)

            glob_rot = proj_SO3(A_vert + A_joint) @ glob_rots_prev[:, i]
            glob_rots.append(glob_rot)

        return torch.stack(glob_rots, dim=1)


def _gram_block(
    jac: torch.Tensor, b: torch.Tensor, w: Optional[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Raw (uncentered) normal-equation pieces of one point block, in float64.

    ``jac``: (B, N, 3, S), ``b``: (B, N, 3), ``w``: (B, N) or None (= all ones).
    Returns ``G`` (B, S, S), ``r`` (B, S, 1), ``SA = sum_n w_n A_n`` (B, 3, S),
    ``Sb = sum_n w_n b_n`` (B, 3, 1) and ``W = sum_n w_n`` (B, 1, 1). The large GEMMs run
    in float32; only the small results are converted.
    """
    B, N, _, S = jac.shape
    jac_flat = jac.reshape(B, N * 3, S)
    b_flat = b.reshape(B, N * 3, 1)
    if w is None:
        gram = jac_flat.mT @ jac_flat
        rhs = jac_flat.mT @ b_flat
        sum_A = jac.sum(dim=1)
        sum_b = b.sum(dim=1).unsqueeze(-1)
        w_sum = torch.full((B, 1, 1), float(N), device=jac.device, dtype=torch.float64)
    else:
        wjac = jac * w[:, :, None, None]
        wjac_flat = wjac.reshape(B, N * 3, S)
        gram = wjac_flat.mT @ jac_flat
        rhs = wjac_flat.mT @ b_flat
        sum_A = wjac.sum(dim=1)
        sum_b = (b * w.unsqueeze(-1)).sum(dim=1).unsqueeze(-1)
        w_sum = w.sum(dim=1).view(B, 1, 1).to(torch.float64)
    return gram.double(), rhs.double(), sum_A.double(), sum_b.double(), w_sum


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
