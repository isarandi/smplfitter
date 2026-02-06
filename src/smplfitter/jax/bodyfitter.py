from __future__ import annotations

from typing import Optional, TYPE_CHECKING
import jax.numpy as jnp
from .lstsq import lstsq, lstsq_partial_share
from .rotation import kabsch, mat2rotvec, rotvec2mat

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
        glob_rots: list[jnp.ndarray] = []
        mesh_weight = 1e-6
        joint_weight = 1 - mesh_weight

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

        for i in range(self.body_model.num_joints):
            # Disable the rotation of toes separately from the feet
            if i == 10:
                glob_rots.append(glob_rots[7])
                continue
            elif i == 11:
                glob_rots.append(glob_rots[8])
                continue

            selector = self.part_vertex_selectors[i]
            default_body_part = reference_vertices[:, selector]
            estim_body_part = target_vertices[:, selector]
            weights_body_part = (
                vertex_weights[:, selector, None] * mesh_weight
                if vertex_weights is not None
                else mesh_weight
            )

            joint_indices = jnp.array(self.children_and_self[i])
            default_joints = reference_joints[:, joint_indices]
            estim_joints = target_joints[:, joint_indices]
            weights_joints = (
                joint_weights[:, joint_indices, None] * joint_weight
                if joint_weights is not None
                else joint_weight
            )

            body_part_mean_reference = jnp.mean(default_joints, axis=1, keepdims=True)
            default_points = jnp.concatenate(
                [
                    (default_body_part - body_part_mean_reference) * weights_body_part,
                    (default_joints - body_part_mean_reference) * weights_joints,
                ],
                axis=1,
            )

            body_part_mean_target = jnp.mean(estim_joints, axis=1, keepdims=True)
            estim_points = jnp.concatenate(
                [
                    estim_body_part - body_part_mean_target,
                    estim_joints - body_part_mean_target,
                ],
                axis=1,
            )

            glob_rot = kabsch(estim_points, default_points)
            glob_rots.append(glob_rot)

        return jnp.stack(glob_rots, axis=1)

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

            if i == 10:
                glob_rots.append(glob_rots[7])
                continue
            elif i == 11:
                glob_rots.append(glob_rots[8])
                continue
            elif i not in [1, 2, 4, 5, 7, 8, 16, 17, 18, 19]:
                glob_rots.append(glob_rots_prev[:, i])
                continue

            vertex_selector = self.part_vertex_selectors[i]
            joint_indices = jnp.array(self.children_and_self[i])

            default_body_part = reference_vertices[:, vertex_selector]
            estim_body_part = target_vertices[:, vertex_selector]
            weights_body_part = (
                vertex_weights[:, vertex_selector, None] if vertex_weights is not None else 1.0
            )

            default_joints = reference_joints[:, joint_indices]
            estim_joints = target_joints[:, joint_indices]
            weights_joints = (
                joint_weights[:, joint_indices, None] if joint_weights is not None else 1.0
            )

            reference_point = glob_position[:, None, :]
            assert true_reference_joints is not None
            default_reference_point = true_reference_joints[:, i : i + 1]
            default_points = jnp.concatenate(
                [
                    (default_body_part - default_reference_point) * weights_body_part,
                    (default_joints - default_reference_point) * weights_joints,
                ],
                axis=1,
            )
            estim_points = jnp.concatenate(
                [(estim_body_part - reference_point), (estim_joints - reference_point)], axis=1
            )
            glob_rot = kabsch(estim_points, default_points) @ glob_rots_prev[:, i]
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
