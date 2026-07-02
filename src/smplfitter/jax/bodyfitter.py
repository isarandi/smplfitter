from __future__ import annotations

from typing import Optional, TYPE_CHECKING
import jax.numpy as jnp
from .lstsq import lstsq, lstsq_partial_share
from .rotation import align_unit_vectors, divide_no_nan, mat2rotvec, proj_SO3, rotvec2mat

if TYPE_CHECKING:
    import smplfitter.jax


class BodyFitter:
    """Fits body model (SMPL/SMPL-X/SMPL+H) parameters to target vertices and joints.

    Args:
        body_model: The body model instance to fit.
        enable_kid: Enables the use of a kid blendshape for fitting kid shapes.
    """

    def __init__(
        self,
        body_model: 'smplfitter.jax.BodyModel',
        enable_kid: bool = False,
    ):
        self.body_model = body_model
        self.n_betas = body_model.shapedirs.shape[2]
        self.enable_kid = enable_kid
        self.is_smpl_family = body_model.model_name.startswith('smpl')

        # Compute default mesh in T-pose
        result = body_model(shape_betas=jnp.zeros((1, body_model.num_betas)))
        self.default_mesh_tf = result['vertices'][0]

        # Template for joints with shape adjustments
        kid_part = (
            body_model.kid_J_shapedir[:, :, None]
            if enable_kid
            else jnp.zeros((body_model.num_joints, 3, 0))
        )
        self.J_template_ext = jnp.concatenate(
            [body_model.J_template.reshape(-1, 3, 1), body_model.J_shapedirs, kid_part],
            axis=2,
        )

        # Store joint hierarchy for each joint's children and descendants
        self.children_and_self = [[i_joint] for i_joint in range(body_model.num_joints)]
        for i_joint in range(1, body_model.num_joints):
            i_parent = body_model.kintree_parents[i_joint]
            self.children_and_self[i_parent].append(i_joint)

        self.descendants_and_self = [[i_joint] for i_joint in range(body_model.num_joints)]
        for i_joint in range(body_model.num_joints - 1, 0, -1):
            i_parent = body_model.kintree_parents[i_joint]
            self.descendants_and_self[i_parent].extend(self.descendants_and_self[i_joint])

        # Precompute part assignment for rotation fitting
        part_assignment = jnp.argmax(body_model.weights, axis=1)
        # Disable the rotation of toes separately from the feet
        part_assignment = jnp.where(part_assignment == 10, 7, part_assignment)
        part_assignment = jnp.where(part_assignment == 11, 8, part_assignment)
        self.part_assignment = part_assignment

        # Precompute vertex selectors for each body part (as numpy for indexing)
        import numpy as np

        part_assignment_np = np.array(part_assignment)
        self.part_vertex_selectors = []
        for i in range(body_model.num_joints):
            selector = np.where(part_assignment_np == i)[0]
            self.part_vertex_selectors.append(selector)

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
        self.used_vertex_indices = jnp.array(used_vertex_indices)

        # One-hot part membership over the used vertices: row j sums vertices of part j.
        part_matrix = np.zeros((J, len(used_vertex_indices)), dtype=np.float32)
        part_matrix[
            part_assignment_np[used_vertex_indices], np.arange(len(used_vertex_indices))
        ] = 1.0
        self.part_matrix = jnp.array(part_matrix)
        self.part_counts = jnp.array(part_matrix.sum(axis=1).reshape(1, J, 1))

        # Children-mean centering matrix: row i averages children_and_self[i] joint positions.
        center_matrix = np.zeros((J, J), dtype=np.float32)
        for i in range(J):
            js = self.children_and_self[i]
            center_matrix[i, js] = 1.0 / len(js)
        self.center_matrix = jnp.array(center_matrix)

        # Joint membership per multi-joint part (their orientation is Kabsch-fit from joints).
        mjp_joint_membership = np.zeros((len(multi_joint_parts), J), dtype=np.float32)
        for k, i in enumerate(multi_joint_parts):
            mjp_joint_membership[k, self.children_and_self[i]] = 1.0
        self.mjp_joint_membership = jnp.array(mjp_joint_membership)
        self.mjp_joint_counts = jnp.array(mjp_joint_membership.sum(axis=1).reshape(1, -1, 1))
        self.mjp_center_matrix = jnp.array(center_matrix[multi_joint_parts])

        # Bone endpoints (start joint, end joint) per bone part.
        self.bone_pairs = jnp.array(
            np.array(
                [[self.children_and_self[i][0], self.children_and_self[i][1]] for i in bone_parts],
                dtype=np.int64,
            ).reshape(len(bone_parts), 2)
        )

        # Assembly permutation: R_concat = cat([R_multi, R_leaf, R_bone]) is scattered back
        # to per-part order; SMPL toe parts take the feet slots directly (10 <- 7, 11 <- 8).
        concat_order = multi_joint_parts + leaf_parts + bone_parts
        inverse_perm = [0] * J
        for pos, jj in enumerate(concat_order):
            inverse_perm[jj] = pos
        if self.is_smpl_family:
            inverse_perm[10] = inverse_perm[7]
            inverse_perm[11] = inverse_perm[8]
        self.assemble_indices = jnp.array(np.array(inverse_perm, dtype=np.int64))

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
            a = a * w[..., None]
            t_sum_side = t * w[..., None]
            s_w = self.part_matrix @ w[..., None]
        else:
            t_sum_side = t
            s_w = self.part_counts
        B = t.shape[0] if t.shape[0] >= a.shape[0] else a.shape[0]
        outer = (t[..., None] * a[..., None, :]).reshape(B, t.shape[1], 9)
        raw = (self.part_matrix @ outer).reshape(B, -1, 3, 3)
        s_t = self.part_matrix @ t_sum_side
        s_a = self.part_matrix @ a
        return raw, s_t, s_a, s_w

    def fit(
        self,
        target_vertices: jnp.ndarray,
        target_joints: Optional[jnp.ndarray] = None,
        vertex_weights: Optional[jnp.ndarray] = None,
        joint_weights: Optional[jnp.ndarray] = None,
        num_iter: int = 1,
        beta_regularizer: float = 1,
        beta_regularizer2: float = 0,
        scale_regularizer: float = 0,
        kid_regularizer: Optional[float] = None,
        share_beta: bool = False,
        final_adjust_rots: bool = True,
        scale_target: bool = False,
        scale_fit: bool = False,
        initial_pose_rotvecs: Optional[jnp.ndarray] = None,
        initial_shape_betas: Optional[jnp.ndarray] = None,
        initial_kid_factor: Optional[jnp.ndarray] = None,
        requested_keys: Optional[list[str]] = None,
    ) -> dict[str, jnp.ndarray]:
        """Fits the body model to target vertices and optionally joints.

        Args:
            target_vertices: Target mesh vertices, shape (batch_size, num_vertices, 3).
            target_joints: Target joint locations, shape (batch_size, num_joints, 3).
            vertex_weights: Importance weights for each vertex.
            joint_weights: Importance weights for each joint.
            num_iter: Number of optimization iterations (1-4 typical).
            beta_regularizer: L2 regularization weight for shape parameters.
            beta_regularizer2: Regularization for first two betas.
            scale_regularizer: Regularization for scale factor.
            kid_regularizer: Regularization for kid blendshape factor.
            share_beta: If True, shares shape parameters across batch.
            final_adjust_rots: Whether to refine body part orientations after fitting.
            scale_target: If True, estimates scale factor for target vertices.
            scale_fit: If True, estimates scale factor for fitted mesh.
            initial_pose_rotvecs: Optional initial pose rotations.
            initial_shape_betas: Optional initial shape parameters.
            initial_kid_factor: Optional initial kid factor.
            requested_keys: List of keys specifying which results to return.

        Returns:
            Dictionary with fitted parameters (pose_rotvecs, shape_betas, trans, etc.)
        """
        if requested_keys is None:
            requested_keys = ['pose_rotvecs']

        # Subtract mean first for better numerical stability
        if target_joints is None:
            target_mean = jnp.mean(target_vertices, axis=1)
            target_vertices = target_vertices - target_mean[:, None]
        else:
            target_mean = jnp.mean(
                jnp.concatenate([target_vertices, target_joints], axis=1), axis=1
            )
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

        parent_indices = jnp.array(self.body_model.kintree_parents[1:])

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
        ref_kid_factor = result.get('kid_factor') if self.enable_kid else None
        ref_scale_corr = result['scale_corr'][:, None, None] if scale_target or scale_fit else None

        if final_adjust_rots:
            if scale_target:
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
                    ref_scale_corr * ref_verts + (1 - ref_scale_corr) * ref_trans[:, None, :],
                    ref_scale_corr * ref_joints + (1 - ref_scale_corr) * ref_trans[:, None, :]
                    if ref_joints is not None
                    else None,
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
            batch_size = glob_rotmats.shape[0]
            eye = jnp.eye(3)[None, None, :, :]  # (1, 1, 3, 3)
            eye = jnp.broadcast_to(eye, (batch_size, 1, 3, 3))
            parent_glob_rotmats = jnp.concatenate([eye, glob_rotmats[:, parent_indices]], axis=1)
            result['relative_orientations'] = (
                jnp.swapaxes(parent_glob_rotmats, -1, -2) @ glob_rotmats
            )

        if 'pose_rotvecs' in requested_keys:
            rel_ori = result['relative_orientations']
            rotvecs = mat2rotvec(rel_ori)
            result['pose_rotvecs'] = rotvecs.reshape(rotvecs.shape[0], -1)

        # Remove intermediate results
        result.pop('vertices', None)
        result.pop('joints', None)

        return {k: v for k, v in result.items() if v is not None}

    def fit_with_known_pose(
        self,
        pose_rotvecs: jnp.ndarray,
        target_vertices: jnp.ndarray,
        target_joints: Optional[jnp.ndarray] = None,
        vertex_weights: Optional[jnp.ndarray] = None,
        joint_weights: Optional[jnp.ndarray] = None,
        beta_regularizer: float = 1,
        beta_regularizer2: float = 0,
        scale_regularizer: float = 0,
        kid_regularizer: Optional[float] = None,
        share_beta: bool = False,
        scale_target: bool = False,
        scale_fit: bool = False,
        beta_regularizer_reference: Optional[jnp.ndarray] = None,
        kid_regularizer_reference: Optional[jnp.ndarray] = None,
        requested_keys: Optional[list[str]] = None,
    ) -> dict[str, jnp.ndarray]:
        """Fits body shape and translation with known output pose.

        Args:
            pose_rotvecs: Known joint rotations as rotation vectors, shape (batch, num_joints*3).
            target_vertices: Target mesh vertices, shape (batch, num_vertices, 3).
            target_joints: Optional target joint positions, shape (batch, num_joints, 3).
            vertex_weights: Optional importance weights for vertices.
            joint_weights: Optional importance weights for joints.
            beta_regularizer: L2 regularization weight for shape parameters.
            beta_regularizer2: Secondary regularization for first two betas.
            scale_regularizer: Regularization for scale factor.
            kid_regularizer: Regularization for kid blendshape factor.
            share_beta: Whether to share shape parameters across batch.
            scale_target: Whether to estimate scale factor for target vertices.
            scale_fit: Whether to estimate scale factor for fitted mesh.
            beta_regularizer_reference: Reference values for beta regularization.
            kid_regularizer_reference: Reference values for kid factor regularization.
            requested_keys: List of result keys to return.

        Returns:
            Dictionary with shape_betas, trans, and optionally kid_factor and scale_corr.
        """
        if requested_keys is None:
            requested_keys = []

        # Subtract mean for numerical stability
        if target_joints is None:
            target_mean = jnp.mean(target_vertices, axis=1)
            target_vertices = target_vertices - target_mean[:, None]
        else:
            target_mean = jnp.mean(
                jnp.concatenate([target_vertices, target_joints], axis=1), axis=1
            )
            target_vertices = target_vertices - target_mean[:, None]
            target_joints = target_joints - target_mean[:, None]

        # Convert pose_rotvecs to global rotation matrices
        rel_rotmats = rotvec2mat(pose_rotvecs.reshape(-1, self.body_model.num_joints, 3))

        # Forward kinematics: relative to global
        glob_rotmats_list = [rel_rotmats[:, 0]]
        for i_joint in range(1, self.body_model.num_joints):
            i_parent = self.body_model.kintree_parents[i_joint]
            glob_rotmats_list.append(glob_rotmats_list[i_parent] @ rel_rotmats[:, i_joint])
        glob_rotmats = jnp.stack(glob_rotmats_list, axis=1)

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

        # Add mean back to translation
        result['trans'] = result['trans'] + target_mean

        # Remove intermediate results
        result.pop('vertices', None)
        result.pop('joints', None)
        result.pop('relative_orientations', None)

        return {k: v for k, v in result.items() if v is not None}

    def fit_with_known_shape(
        self,
        shape_betas: jnp.ndarray,
        target_vertices: jnp.ndarray,
        target_joints: Optional[jnp.ndarray] = None,
        vertex_weights: Optional[jnp.ndarray] = None,
        joint_weights: Optional[jnp.ndarray] = None,
        kid_factor: Optional[jnp.ndarray] = None,
        num_iter: int = 1,
        final_adjust_rots: bool = True,
        initial_pose_rotvecs: Optional[jnp.ndarray] = None,
        scale_fit: bool = False,
        requested_keys: Optional[list[str]] = None,
    ) -> dict[str, jnp.ndarray]:
        """Fits body pose and translation with known shape parameters.

        Args:
            shape_betas: Known shape parameters, shape (batch, num_betas).
            target_vertices: Target mesh vertices, shape (batch, num_vertices, 3).
            target_joints: Optional target joint positions, shape (batch, num_joints, 3).
            vertex_weights: Optional importance weights for vertices.
            joint_weights: Optional importance weights for joints.
            kid_factor: Optional kid blendshape factor.
            num_iter: Number of optimization iterations.
            final_adjust_rots: Whether to refine body part orientations after fitting.
            initial_pose_rotvecs: Optional initial pose rotations.
            scale_fit: Whether to estimate scale factor for fitted mesh.
            requested_keys: List of result keys to return.

        Returns:
            Dictionary with pose_rotvecs, trans, orientations, and optionally scale_corr.
        """
        if requested_keys is None:
            requested_keys = ['pose_rotvecs']

        # Slice shape_betas if needed
        if shape_betas.shape[1] > self.n_betas:
            shape_betas = shape_betas[:, : self.n_betas]

        # Subtract mean for numerical stability
        if target_joints is None:
            target_mean = jnp.mean(target_vertices, axis=1)
            target_vertices = target_vertices - target_mean[:, None]
        else:
            target_mean = jnp.mean(
                jnp.concatenate([target_vertices, target_joints], axis=1), axis=1
            )
            target_vertices = target_vertices - target_mean[:, None]
            target_joints = target_joints - target_mean[:, None]

        # Get initial reference mesh from body model
        initial_forw = self.body_model(
            shape_betas=shape_betas, kid_factor=kid_factor, pose_rotvecs=initial_pose_rotvecs
        )
        initial_joints = initial_forw['joints']
        initial_vertices = initial_forw['vertices']

        # Initial rotation fitting
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

        # Iterative refinement
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

        # Final forward pass
        result = self.body_model(
            glob_rotmats=glob_rotmats, shape_betas=shape_betas, kid_factor=kid_factor
        )
        ref_verts = result['vertices']
        ref_joints = result['joints']

        # Compute translation (and optionally scale)
        ref_scale_corr, ref_trans = fit_scale_and_translation(
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
                    ref_scale_corr[:, None, None] * ref_verts + ref_trans[:, None],
                    ref_scale_corr[:, None, None] * ref_joints + ref_trans[:, None],
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
                    ref_verts + ref_trans[:, None],
                    ref_joints + ref_trans[:, None],
                    vertex_weights,
                    joint_weights,
                    glob_rotmats,
                    shape_betas,
                    None,  # scale_corr
                    ref_trans,
                    kid_factor,
                )

        # Add mean back to translation
        final_trans = ref_trans + target_mean

        result_out: dict[str, jnp.ndarray] = {
            'orientations': glob_rotmats,
            'trans': final_trans,
        }
        if scale_fit and ref_scale_corr is not None:
            result_out['scale_corr'] = ref_scale_corr

        # Compute relative orientations and rotvecs if requested
        parent_indices = jnp.array(self.body_model.kintree_parents[1:])
        if 'relative_orientations' in requested_keys or 'pose_rotvecs' in requested_keys:
            batch_size = glob_rotmats.shape[0]
            eye = jnp.eye(3)[None, None, :, :]
            eye = jnp.broadcast_to(eye, (batch_size, 1, 3, 3))
            parent_glob_rotmats = jnp.concatenate([eye, glob_rotmats[:, parent_indices]], axis=1)
            result_out['relative_orientations'] = (
                jnp.swapaxes(parent_glob_rotmats, -1, -2) @ glob_rotmats
            )

        if 'pose_rotvecs' in requested_keys:
            rel_ori = result_out['relative_orientations']
            rotvecs = mat2rotvec(rel_ori)
            result_out['pose_rotvecs'] = rotvecs.reshape(rotvecs.shape[0], -1)

        return result_out

    def _fit_shape(
        self,
        glob_rotmats: jnp.ndarray,
        target_vertices: jnp.ndarray,
        target_joints: Optional[jnp.ndarray] = None,
        vertex_weights: Optional[jnp.ndarray] = None,
        joint_weights: Optional[jnp.ndarray] = None,
        beta_regularizer: float = 1,
        beta_regularizer2: float = 0,
        scale_regularizer: float = 0,
        kid_regularizer: Optional[float] = None,
        share_beta: bool = False,
        scale_target: bool = False,
        scale_fit: bool = False,
        beta_regularizer_reference: Optional[jnp.ndarray] = None,
        kid_regularizer_reference: Optional[jnp.ndarray] = None,
        requested_keys: Optional[list[str]] = None,
    ) -> dict[str, jnp.ndarray]:
        if scale_target and scale_fit:
            raise ValueError('Only one of scale_target and scale_fit can be True')
        if requested_keys is None:
            requested_keys = []

        glob_rotmats = glob_rotmats.astype(jnp.float32)
        batch_size = target_vertices.shape[0]

        parent_indices = jnp.array(self.body_model.kintree_parents[1:])

        eye = jnp.eye(3)[None, None, :, :]
        eye = jnp.broadcast_to(eye, (batch_size, 1, 3, 3))
        parent_glob_rot_mats = jnp.concatenate([eye, glob_rotmats[:, parent_indices]], axis=1)
        rel_rotmats = jnp.swapaxes(parent_glob_rot_mats, -1, -2) @ glob_rotmats

        # Compute global positions with shape gradients
        glob_positions_ext_list = [
            jnp.broadcast_to(
                self.J_template_ext[None, 0], (batch_size, 3, self.J_template_ext.shape[2])
            )
        ]
        for i_joint in range(1, self.body_model.num_joints):
            i_parent = self.body_model.kintree_parents[i_joint]
            glob_positions_ext_list.append(
                glob_positions_ext_list[i_parent]
                + jnp.einsum(
                    'bCc,cs->bCs',
                    glob_rotmats[:, i_parent],
                    self.J_template_ext[i_joint] - self.J_template_ext[i_parent],
                )
            )
        glob_positions_ext = jnp.stack(glob_positions_ext_list, axis=1)
        translations_ext = glob_positions_ext - jnp.einsum(
            'bjCc,jcs->bjCs', glob_rotmats, self.J_template_ext
        )

        rot_params = rel_rotmats[:, 1:].reshape(-1, (self.body_model.num_joints - 1) * 3 * 3)
        v_posed = self.body_model.v_template + jnp.einsum(
            'vcp,bp->bvc', self.body_model.posedirs, rot_params
        )
        v_rotated = jnp.einsum('bjCc,vj,bvc->bvC', glob_rotmats, self.body_model.weights, v_posed)

        shapedirs = (
            jnp.concatenate(
                [self.body_model.shapedirs, self.body_model.kid_shapedir[:, :, None]],
                axis=2,
            )
            if self.enable_kid
            else self.body_model.shapedirs
        )
        v_grad_rotated = jnp.einsum(
            'bjCc,lj,lcs->blCs', glob_rotmats, self.body_model.weights, shapedirs
        )

        v_rotated_ext = jnp.concatenate([v_rotated[..., None], v_grad_rotated], axis=3)
        v_translations_ext = jnp.einsum('vj,bjcs->bvcs', self.body_model.weights, translations_ext)
        v_posed_posed_ext = v_translations_ext + v_rotated_ext

        if target_joints is None:
            target_both = target_vertices
            pos_both = v_posed_posed_ext[..., 0]
            jac_pos_both = v_posed_posed_ext[..., 1:]
        else:
            target_both = jnp.concatenate([target_vertices, target_joints], axis=1)
            pos_both = jnp.concatenate(
                [v_posed_posed_ext[..., 0], glob_positions_ext[..., 0]], axis=1
            )
            jac_pos_both = jnp.concatenate(
                [v_posed_posed_ext[..., 1:], glob_positions_ext[..., 1:]], axis=1
            )

        if scale_target:
            A = jnp.concatenate([jac_pos_both, -target_both[..., None]], axis=3)
        elif scale_fit:
            A = jnp.concatenate([jac_pos_both, pos_both[..., None]], axis=3)
        else:
            A = jac_pos_both

        b = target_both - pos_both

        if target_joints is not None and vertex_weights is not None and joint_weights is not None:
            weights = jnp.concatenate([vertex_weights, joint_weights], axis=1)
        elif target_joints is None and vertex_weights is not None:
            weights = vertex_weights
        else:
            weights = jnp.ones(A.shape[:2], dtype=jnp.float32)

        n_params = (
            self.n_betas + (1 if self.enable_kid else 0) + (1 if scale_target or scale_fit else 0)
        )

        # Center with weighted mean for better numerical stability
        weights_sum = jnp.sum(weights, axis=1, keepdims=True)
        mean_A = jnp.where(
            weights_sum[..., None, None] == 0,
            0.0,
            jnp.sum(weights[..., None, None] * A, axis=1, keepdims=True)
            / weights_sum[..., None, None],
        )
        mean_b = jnp.where(
            weights_sum[..., None] == 0,
            0.0,
            jnp.sum(weights[..., None] * b, axis=1, keepdims=True) / weights_sum[..., None],
        )
        A = A - mean_A
        b = b - mean_b

        A = A.reshape(batch_size, -1, n_params)
        b = b.reshape(batch_size, -1, 1)
        w = jnp.repeat(weights.reshape(batch_size, -1), 3, axis=1)

        l2_regularizer_all = jnp.concatenate(
            [
                jnp.full((2,), beta_regularizer2),
                jnp.full((self.n_betas - 2,), beta_regularizer),
            ]
        )

        if beta_regularizer_reference is None:
            l2_regularizer_reference_all = jnp.zeros((batch_size, self.n_betas))
        else:
            l2_regularizer_reference_all = beta_regularizer_reference

        if self.enable_kid:
            if kid_regularizer is None:
                kid_regularizer = beta_regularizer
            if kid_regularizer_reference is None:
                kid_regularizer_reference = jnp.zeros(batch_size)
            l2_regularizer_all = jnp.concatenate(
                [l2_regularizer_all, jnp.array([kid_regularizer])]
            )
            l2_regularizer_reference_all = jnp.concatenate(
                [l2_regularizer_reference_all, kid_regularizer_reference[:, None]], axis=1
            )

        if scale_target or scale_fit:
            l2_regularizer_all = jnp.concatenate(
                [l2_regularizer_all, jnp.array([scale_regularizer])]
            )
            l2_regularizer_reference_all = jnp.concatenate(
                [l2_regularizer_reference_all, jnp.zeros((batch_size, 1))], axis=1
            )

        l2_regularizer_rhs = (l2_regularizer_all * l2_regularizer_reference_all)[..., None]

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
        new_trans = mean_b.squeeze(1) - (mean_A.squeeze(1) @ x[..., None]).squeeze(-1)
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
                new_shape = new_shape / new_scale_corr[:, None]
                if new_kid_factor is not None:
                    new_kid_factor = new_kid_factor / new_scale_corr
            result['scale_corr'] = new_scale_corr
        else:
            new_scale_corr = None

        if self.enable_kid and new_kid_factor is not None:
            new_shape_with_kid = jnp.concatenate([new_shape, new_kid_factor[:, None]], axis=1)
        else:
            new_shape_with_kid = new_shape

        if 'joints' in requested_keys:
            result['joints'] = (
                glob_positions_ext[..., 0]
                + jnp.einsum('bvcs,bs->bvc', glob_positions_ext[..., 1:], new_shape_with_kid)
                + new_trans[:, None]
            )

        if 'vertices' in requested_keys:
            result['vertices'] = (
                v_posed_posed_ext[..., 0]
                + jnp.einsum('bvcs,bs->bvc', v_posed_posed_ext[..., 1:], new_shape_with_kid)
                + new_trans[:, None]
            )
        return result

    def _fit_global_rotations(
        self,
        target_vertices: jnp.ndarray,
        target_joints: Optional[jnp.ndarray],
        reference_vertices: jnp.ndarray,
        reference_joints: Optional[jnp.ndarray],
        vertex_weights: Optional[jnp.ndarray],
        joint_weights: Optional[jnp.ndarray],
    ) -> jnp.ndarray:
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
            J_regressor = getattr(self.body_model, 'J_regressor_post_lbs', None)
            if J_regressor is not None:
                target_joints = J_regressor @ target_vertices
                reference_joints = J_regressor @ reference_vertices
            else:
                # Fallback: use template joints
                target_joints = jnp.broadcast_to(
                    self.body_model.J_template[None],
                    (target_vertices.shape[0], self.body_model.num_joints, 3),
                )
                reference_joints = jnp.broadcast_to(
                    self.body_model.J_template[None],
                    (reference_vertices.shape[0], self.body_model.num_joints, 3),
                )

        B = target_vertices.shape[0]

        # Per-part vertex cross-covariances about the children-mean centers, loop-free.
        raw, s_t, s_a, s_w = self._part_sums(target_vertices, reference_vertices, vertex_weights)
        mt = self.center_matrix @ target_joints  # (B, J, 3)
        ma = self.center_matrix @ reference_joints  # (B_ref, J, 3)
        A_vert = (
            raw
            - s_t[..., None] * ma[..., None, :]
            - mt[..., None] * s_a[..., None, :]
            + s_w[..., None] * (mt[..., None] * ma[..., None, :])
        )  # (B, J, 3, 3)

        # Joint-point cross-covariances for the multi-joint parts, loop-free.
        rj = reference_joints
        if joint_weights is not None:
            rj = rj * joint_weights[..., None]
            tj_sum_side = target_joints * joint_weights[..., None]
            s_wj = self.mjp_joint_membership @ joint_weights[..., None]
        else:
            tj_sum_side = target_joints
            s_wj = self.mjp_joint_counts
        outer_j = (target_joints[..., None] * rj[..., None, :]).reshape(
            B, target_joints.shape[1], 9
        )
        raw_j = (self.mjp_joint_membership @ outer_j).reshape(B, -1, 3, 3)
        mtj = self.mjp_center_matrix @ target_joints
        maj = self.mjp_center_matrix @ reference_joints
        s_tj = self.mjp_joint_membership @ tj_sum_side
        s_aj = self.mjp_joint_membership @ rj
        A_multi = (
            raw_j
            - s_tj[..., None] * maj[..., None, :]
            - mtj[..., None] * s_aj[..., None, :]
            + s_wj[..., None] * (mtj[..., None] * maj[..., None, :])
        )

        # Kabsch bucket (multi-joint parts + leaf parts): one batched SVD.
        A_svd = jnp.concatenate([A_multi, A_vert[:, self.leaf_parts]], axis=1)
        R_svd = proj_SO3(A_svd)

        # Bone bucket: batched swing (bone alignment) + twist (from vertices).
        b_ref = (
            reference_joints[:, self.bone_pairs[:, 1]] - reference_joints[:, self.bone_pairs[:, 0]]
        )
        b_tgt = target_joints[:, self.bone_pairs[:, 1]] - target_joints[:, self.bone_pairs[:, 0]]
        b_ref_n = divide_no_nan(b_ref, jnp.linalg.norm(b_ref, axis=-1, keepdims=True))
        b_tgt_n = divide_no_nan(b_tgt, jnp.linalg.norm(b_tgt, axis=-1, keepdims=True))
        R_swing = align_unit_vectors(b_ref_n, b_tgt_n)  # (B, n_bones, 3, 3)

        H = R_swing @ jnp.swapaxes(A_vert[:, self.bone_parts], -1, -2)
        trH = jnp.trace(H, axis1=-2, axis2=-1)
        bHb = (b_tgt_n[..., None, :] @ H @ b_tgt_n[..., None])[..., 0, 0]
        # vee_i = eps_ijk H_jk: the vertex cross-product sums, extracted from H.
        vee = jnp.stack(
            [
                H[..., 1, 2] - H[..., 2, 1],
                H[..., 2, 0] - H[..., 0, 2],
                H[..., 0, 1] - H[..., 1, 0],
            ],
            axis=-1,
        )
        twist_angle = jnp.arctan2(jnp.sum(b_tgt_n * vee, axis=-1), trH - bHb)
        R_twist = rotvec2mat(b_tgt_n * twist_angle[..., None])
        R_bone = R_twist @ R_swing

        # Scatter both buckets back to per-part order (toe parts take the feet slots).
        R_concat = jnp.concatenate([R_svd, R_bone], axis=1)
        return R_concat[:, self.assemble_indices]

    def _fit_global_rotations_dependent(
        self,
        target_vertices: jnp.ndarray,
        target_joints: Optional[jnp.ndarray],
        reference_vertices: jnp.ndarray,
        reference_joints: Optional[jnp.ndarray],
        vertex_weights: Optional[jnp.ndarray],
        joint_weights: Optional[jnp.ndarray],
        glob_rots_prev: jnp.ndarray,
        shape_betas: jnp.ndarray,
        scale_corr: Optional[jnp.ndarray],
        trans: jnp.ndarray,
        kid_factor: Optional[jnp.ndarray],
    ) -> jnp.ndarray:
        glob_rots: list[jnp.ndarray] = []

        true_reference_joints = reference_joints
        if target_joints is None or reference_joints is None:
            J_regressor = getattr(self.body_model, 'J_regressor_post_lbs', None)
            if J_regressor is not None:
                target_joints = J_regressor @ target_vertices
                reference_joints = J_regressor @ reference_vertices
            else:
                target_joints = jnp.broadcast_to(
                    self.body_model.J_template[None],
                    (target_vertices.shape[0], self.body_model.num_joints, 3),
                )
                reference_joints = jnp.broadcast_to(
                    self.body_model.J_template[None],
                    (reference_vertices.shape[0], self.body_model.num_joints, 3),
                )
        if true_reference_joints is None:
            true_reference_joints = reference_joints

        j = self.body_model.J_template + jnp.einsum(
            'jcs,...s->...jc',
            self.body_model.J_shapedirs,
            shape_betas[:, : self.n_betas],
        )
        if kid_factor is not None:
            j = j + jnp.einsum('jc,...->...jc', self.body_model.kid_J_shapedir, kid_factor)

        if scale_corr is not None:
            j = j * scale_corr

        parent_indices = jnp.array(self.body_model.kintree_parents[1:])
        j_parent = jnp.concatenate([jnp.zeros((j.shape[0], 1, 3)), j[:, parent_indices]], axis=1)
        bones = j - j_parent

        # Per-part vertex statistics, shared machinery with _fit_global_rotations. The
        # sequential loop below only needs 3x3 algebra per joint: the cross-covariance
        # about the dynamic centers follows algebraically from these fixed sums.
        raw, s_t, s_a, s_w = self._part_sums(target_vertices, reference_vertices, vertex_weights)

        glob_positions: list[jnp.ndarray] = []

        for i in range(self.body_model.num_joints):
            if i == 0:
                glob_position = j[:, i] + trans
            else:
                i_parent = self.body_model.kintree_parents[i]
                glob_position = glob_positions[i_parent] + (
                    glob_rots[i_parent] @ bones[:, i, :, None]
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
                - s_t[:, i][..., None] * c_a[..., None, :]
                - c_t[..., None] * s_a[:, i][..., None, :]
                + s_w[:, i][..., None] * (c_t[..., None] * c_a[..., None, :])
            )  # (B, 3, 3)

            # Joint contribution (children_and_self), same weighting as before.
            joint_indices = jnp.array(self.children_and_self[i])
            estim_joints = target_joints[:, joint_indices] - c_t[:, None]
            default_joints = reference_joints[:, joint_indices] - c_a[:, None]
            if joint_weights is not None:
                default_joints = default_joints * joint_weights[:, joint_indices][..., None]
            A_joint = jnp.swapaxes(estim_joints, -1, -2) @ default_joints  # (B, 3, 3)

            glob_rot = proj_SO3(A_vert + A_joint) @ glob_rots_prev[:, i]
            glob_rots.append(glob_rot)

        return jnp.stack(glob_rots, axis=1)


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
            weights_both = jnp.ones(target_vertices.shape[:2], dtype=jnp.float32)
    else:
        target_both = jnp.concatenate([target_vertices, target_joints], axis=1)
        reference_both = jnp.concatenate([reference_vertices, reference_joints], axis=1)
        if vertex_weights is not None and joint_weights is not None:
            weights_both = jnp.concatenate([vertex_weights, joint_weights], axis=1)
        else:
            weights_both = jnp.ones(
                (target_vertices.shape[0], target_vertices.shape[1] + target_joints.shape[1]),
                dtype=jnp.float32,
            )

    weights_both = weights_both / jnp.sum(weights_both, axis=1, keepdims=True)

    weighted_mean_target = jnp.sum(target_both * weights_both[..., None], axis=1)
    weighted_mean_reference = jnp.sum(reference_both * weights_both[..., None], axis=1)

    if scale:
        target_centered = target_both - weighted_mean_target[:, None]
        reference_centered = reference_both - weighted_mean_reference[:, None]
        ssq_reference = jnp.sum(reference_centered**2 * weights_both[..., None], axis=(1, 2))
        ssq_target = jnp.sum(target_centered**2 * weights_both[..., None], axis=(1, 2))
        scale_factor = jnp.sqrt(ssq_target / ssq_reference)
        trans = weighted_mean_target - scale_factor[:, None] * weighted_mean_reference
    else:
        scale_factor = None
        trans = weighted_mean_target - weighted_mean_reference

    return scale_factor, trans
