from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from smplfitter.pt.lstsq import lstsq, lstsq_partial_share
from smplfitter.pt.rotation import kabsch, mat2rotvec, rotvec2mat
import smplfitter.pt.lstsq
import importlib


# importlib.reload(smplfitter.pt.lstsq)
# lstsq = smplfitter.pt.lstsq.lstsq

class SMPLFitter(nn.Module):
    """
    Class for fitting body model (SMPL/SMPL-X/SMPL+H) parameters to lists of target vertices and
    joints.

    Parameters:
        body_model (nn.Module): The SMPL model instance we wish to fit, of a certain model
        variant and gender.
        num_betas (int, optional): Number of shape parameters (betas) to use when fitting.
        Default is 10.
        enable_kid (bool, optional): If True, enables the use of a kid blendshape, allowing for
        fitting kid shapes as in AGORA. Default is False.
        vertex_subset (Optional[torch.Tensor], optional): A tensor specifying a subset of
        vertices to use
                                                          in the fitting process, allowing
                                                          partial fitting.
                                                          Default is None, meaning all vertices
                                                          are used. The subset of vertices should
                                                          cover all body parts to provide enough
                                                          constraints.
        joint_regressor (Optional[torch.Tensor], optional): A regression matrix of shape (
        num_joints, num_vertices) for obtaining joint locations, in case the target joints are
        not specified when fitting. Defaults to the joint regressor of the body model, however a
        custom one must be supplied if `vertex_subset` is partial and target joint locations will
        not be provided.

    """

    def __init__(
            self,
            body_model: nn.Module,
            num_betas: int = 10,
            enable_kid: bool = False,
            vertex_subset: Optional[torch.Tensor] = None,
            joint_regressor: Optional[torch.Tensor] = None
    ):
        super(SMPLFitter, self).__init__()
        self.body_model = body_model
        self.n_betas = num_betas if num_betas is not None else self.body_model.shapedirs.shape[2]
        self.enable_kid = enable_kid
        device = body_model.v_template.device

        # Initialize vertex subset
        if vertex_subset is None:
            vertex_subset = torch.arange(body_model.num_vertices, dtype=torch.int64, device=device)
        else:
            vertex_subset = torch.as_tensor(vertex_subset, dtype=torch.int64, device=device)
        self.register_buffer('vertex_subset', vertex_subset)

        # Register other static buffers
        self.register_buffer('default_mesh_tf', body_model.single()['vertices'][self.vertex_subset])

        # Template for joints with shape adjustments
        J_template_ext = torch.cat(
            [body_model.J_template.view(-1, 3, 1),
             body_model.J_shapedirs[:, :, :self.n_betas]] +
            ([body_model.kid_J_shapedir.view(-1, 3, 1)] if enable_kid else []),
            dim=2)
        self.register_buffer('J_template_ext', J_template_ext)

        # Store joint hierarchy for each joint’s children and descendants
        self.children_and_self = [[i_joint] for i_joint in range(body_model.num_joints)]
        for i_joint in range(1, body_model.num_joints):
            i_parent = body_model.kintree_parents[i_joint]
            self.children_and_self[i_parent].append(i_joint)

        self.descendants_and_self = [[i_joint] for i_joint in range(body_model.num_joints)]
        for i_joint in range(body_model.num_joints - 1, 0, -1):
            i_parent = body_model.kintree_parents[i_joint]
            self.descendants_and_self[i_parent].extend(self.descendants_and_self[i_joint])

        # Register buffers from the body model with subset indexing
        self.register_buffer('shapedirs', body_model.shapedirs.index_select(0, self.vertex_subset))
        self.register_buffer('kid_shapedir',
                             body_model.kid_shapedir.index_select(0, self.vertex_subset))
        self.register_buffer('v_template',
                             body_model.v_template.index_select(0, self.vertex_subset))
        self.register_buffer('weights', body_model.weights.index_select(0, self.vertex_subset))
        self.register_buffer('posedirs', body_model.posedirs.index_select(0, self.vertex_subset))

        # Save additional configurations
        self.num_vertices = self.v_template.shape[0]

        # Joint regressor setup
        self.J_regressor = joint_regressor if joint_regressor is not None else (
            body_model.J_regressor)

    @torch.jit.export
    def fit(
            self,
            target_vertices: torch.Tensor,
            target_joints: Optional[torch.Tensor] = None,
            vertex_weights: Optional[torch.Tensor] = None,
            joint_weights: Optional[torch.Tensor] = None,
            n_iter: int = 1,
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
            requested_keys: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Fit the body model to target vertices and optionally joints by optimizing for shape and
        pose, and optionally others.

        Parameters:
            target_vertices (torch.Tensor): Target mesh vertices, shaped as (batch_size,
            num_vertices, 3).
            target_joints (Optional[torch.Tensor], optional): Target joint locations, shaped as (
            batch_size, num_joints, 3).
            vertex_weights (Optional[torch.Tensor], optional): Importance weights for each vertex
            during the fitting process.
            joint_weights (Optional[torch.Tensor], optional): Importance weights for each joint
            during the fitting process.
            n_iter (int, optional): Number of iterations for the optimization process. Reasonable
            values are in the range of 1-4. Default is 1.
            beta_regularizer (float, optional): L2 regularization weight for shape parameters (
            betas). Default is 1.
            beta_regularizer2 (float, optional): Secondary regularization for betas, affecting
            the first two parameters. Default is 0.
            scale_regularizer (float, optional): Regularization term to penalize the scale factor
            deviating from 1. Default is 0. Has no effect unless `scale_target` or `scale_fit` is
            True.
            kid_regularizer (Optional[float], optional): Regularization weight for the kid
            blendshape factor. Default is None. Has no effect unless `enable_kid` on the object
            is True.
            share_beta (bool, optional): If True, shares the shape parameters (betas) across
            instances in the batch. Default is False.
            final_adjust_rots (bool, optional): Whether to perform a final, dependent refinement
            of the body part orientations to improve alignment. Default is True.
            scale_target (bool, optional): If True, estimates a scale factor to apply to the
            target vertices for alignment. Default is False.
            scale_fit (bool, optional): If True, estimates a scale factor to apply to the fitted
            mesh for alignment. Default is False.
            requested_keys (Optional[List[str]], optional): List of keys specifying which results
            to return. Default is ['pose_rotvecs']. Other options are ['relative_orientations',
            'joints', 'vertices'].

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the following items, based on
            requested keys:
                - 'pose_rotvecs' (torch.Tensor): Estimated pose in concatenated rotation vector
                format.
                - 'shape_betas' (torch.Tensor): Estimated shape parameters (betas).
                - 'trans' (torch.Tensor): Estimated translation parameters.
                - 'joints' (torch.Tensor): Estimated joint positions, if requested.
                - 'vertices' (torch.Tensor): Fitted mesh vertices, if requested.
                - 'orientations' (torch.Tensor): Global body part orientations as rotation matrices.
                - 'relative_orientations' (torch.Tensor): Parent-relative body part orientations
                as rotation matrices.
                - 'kid_factor' (torch.Tensor): Estimated kid blendshape factor, if 'enable_kid'
                is True.
                - 'scale_corr' (torch.Tensor): Estimated scale correction factor,
                if 'scale_target' or 'scale_fit' is True.
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
                shape_betas=initial_shape_betas, kid_factor=initial_kid_factor,
                pose_rotvecs=initial_pose_rotvecs)
            initial_joints = initial_forw['joints']
            initial_vertices = initial_forw['vertices'][:, self.vertex_subset]
            glob_rotmats = self._fit_global_rotations(
                target_vertices, target_joints, initial_vertices, initial_joints, vertex_weights,
                joint_weights) @ initial_forw['orientations']
        else:
            initial_joints = self.body_model.J_template[None]
            initial_vertices = self.default_mesh_tf[None]
            glob_rotmats = self._fit_global_rotations(
                target_vertices, target_joints, initial_vertices, initial_joints, vertex_weights,
                joint_weights)

        device = self.body_model.v_template.device
        parent_indices = self.body_model.kintree_parents_tensor[1:].to(device)

        for i in range(n_iter - 1):
            result = self._fit_shape(
                glob_rotmats, target_vertices, target_joints, vertex_weights, joint_weights,
                beta_regularizer, beta_regularizer2, scale_regularizer=0.0,
                kid_regularizer=kid_regularizer, share_beta=share_beta, scale_target=False,
                scale_fit=False,
                beta_regularizer_reference=initial_shape_betas,
                kid_regularizer_reference=initial_kid_factor,
                requested_keys=['vertices', 'joints'] if target_joints is not None else [
                    'vertices'])
            ref_verts = result['vertices']
            ref_joints = result['joints'] if target_joints is not None else None
            glob_rotmats = self._fit_global_rotations(
                target_vertices, target_joints, ref_verts, ref_joints,
                vertex_weights, joint_weights) @ glob_rotmats

        result = self._fit_shape(
            glob_rotmats, target_vertices, target_joints, vertex_weights,
            joint_weights, beta_regularizer, beta_regularizer2, scale_regularizer,
            kid_regularizer, share_beta, scale_target, scale_fit,
            beta_regularizer_reference=initial_shape_betas,
            kid_regularizer_reference=initial_kid_factor,
            requested_keys=['vertices', 'joints']
            if target_joints is not None or final_adjust_rots else ['vertices'])
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
                    ref_verts, ref_joints, vertex_weights, joint_weights, glob_rotmats, ref_shape,
                    None, ref_trans, ref_kid_factor)
            elif scale_fit:
                assert ref_scale_corr is not None
                glob_rotmats = self._fit_global_rotations_dependent(
                    target_vertices, target_joints,
                    ref_scale_corr * ref_verts +
                    (1 - ref_scale_corr) * ref_trans.unsqueeze(-2),
                    ref_scale_corr * ref_joints +
                    (1 - ref_scale_corr) * ref_trans.unsqueeze(-2),
                    vertex_weights, joint_weights,
                    glob_rotmats, ref_shape, ref_scale_corr, ref_trans, ref_kid_factor)
            else:
                glob_rotmats = self._fit_global_rotations_dependent(
                    target_vertices, target_joints, ref_verts, ref_joints,
                    vertex_weights, joint_weights, glob_rotmats, ref_shape, None,
                    ref_trans, ref_kid_factor)

        if 'joints' in requested_keys or 'vertices' in requested_keys:
            forw = self.body_model(
                glob_rotmats=glob_rotmats, shape_betas=result['shape_betas'], trans=result['trans'],
                kid_factor=ref_kid_factor)
        else:
            forw = {}

        # Add the mean back
        result['trans'] = ref_trans + target_mean
        if 'joints' in requested_keys:
            result['joints'] = forw['joints'] + target_mean[:, None]
        if 'vertices' in requested_keys:
            result['vertices'] = forw['vertices'] + target_mean[:, None]

        result['orientations'] = glob_rotmats

        # Provide other requested rotation formats
        if 'relative_orientations' in requested_keys or 'pose_rotvecs' in requested_keys:
            parent_glob_rotmats = torch.cat([
                torch.eye(3, device=device).expand(glob_rotmats.shape[0], 1, 3, 3),
                torch.index_select(glob_rotmats, 1, parent_indices)], dim=1)
            result['relative_orientations'] = torch.matmul(
                parent_glob_rotmats.transpose(-1, -2), glob_rotmats)

        if 'pose_rotvecs' in requested_keys:
            rel_ori = result['relative_orientations']
            assert rel_ori is not None
            rotvecs = mat2rotvec(rel_ori)
            result['pose_rotvecs'] = rotvecs.view(rotvecs.shape[0], -1)

        result_non_none: Dict[str, torch.Tensor] = {}
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
            requested_keys: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
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

        device = self.body_model.v_template.device
        parent_indices = self.body_model.kintree_parents_tensor[1:].to(device)

        rel_rotmats = rotvec2mat(pose_rotvecs.view(-1, self.body_model.num_joints, 3))
        glob_rotmats_ = [rel_rotmats[:, 0]]
        for i_joint in range(1, self.body_model.num_joints):
            i_parent = self.body_model.kintree_parents[i_joint]
            glob_rotmats_.append(glob_rotmats_[i_parent] @ rel_rotmats[:, i_joint])
        glob_rotmats = torch.stack(glob_rotmats_, dim=1)

        result = self._fit_shape(
            glob_rotmats, target_vertices, target_joints, vertex_weights,
            joint_weights, beta_regularizer, beta_regularizer2, scale_regularizer,
            kid_regularizer, share_beta, scale_target, scale_fit,
            beta_regularizer_reference=beta_regularizer_reference,
            kid_regularizer_reference=kid_regularizer_reference)
        ref_shape = result['shape_betas']
        ref_trans = result['trans']
        ref_kid_factor = result['kid_factor'] if self.enable_kid else None
        ref_scale_corr = result['scale_corr'][:, None, None] if scale_target or scale_fit else None

        if 'joints' in requested_keys or 'vertices' in requested_keys:
            forw = self.body_model(
                glob_rotmats=glob_rotmats, shape_betas=result['shape_betas'], trans=result['trans'],
                kid_factor=ref_kid_factor)
        else:
            forw = {}

        # Add the mean back
        result['trans'] = ref_trans + target_mean
        if 'joints' in requested_keys:
            result['joints'] = forw['joints'] + target_mean[:, None]
        if 'vertices' in requested_keys:
            result['vertices'] = forw['vertices'] + target_mean[:, None]

        result_non_none: Dict[str, torch.Tensor] = {}
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
            n_iter: int = 1,
            final_adjust_rots: bool = True,
            initial_pose_rotvecs: Optional[torch.Tensor] = None,
            scale_fit: bool = False,
            requested_keys: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
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
            shape_betas=shape_betas, kid_factor=kid_factor, pose_rotvecs=initial_pose_rotvecs)
        initial_joints = initial_forw['joints']
        initial_vertices = initial_forw['vertices'][:, self.vertex_subset]

        glob_rotmats = self._fit_global_rotations(
            target_vertices, target_joints, initial_vertices, initial_joints, vertex_weights,
            joint_weights) @ initial_forw['orientations']
        device = self.body_model.v_template.device
        parent_indices = self.body_model.kintree_parents_tensor[1:].to(device)

        for i in range(n_iter - 1):
            result = self.body_model(
                glob_rotmats=glob_rotmats, shape_betas=shape_betas, kid_factor=kid_factor)
            ref_verts = result['vertices'][:, self.vertex_subset]
            ref_joints = result['joints'] if target_joints is not None else None
            glob_rotmats = self._fit_global_rotations(
                target_vertices, target_joints, ref_verts, ref_joints,
                vertex_weights, joint_weights) @ glob_rotmats

        result = self.body_model(
            glob_rotmats=glob_rotmats, shape_betas=shape_betas, kid_factor=kid_factor)
        ref_verts = result['vertices'][:, self.vertex_subset]
        ref_joints = result['joints']
        ref_scale_corr, ref_trans = fit_scale_and_translation(
            target_vertices, ref_verts, target_joints, ref_joints, vertex_weights, joint_weights,
            scale=scale_fit)

        if final_adjust_rots:
            if scale_fit:
                assert ref_scale_corr is not None
                glob_rotmats = self._fit_global_rotations_dependent(
                    target_vertices, target_joints,
                    ref_scale_corr * ref_verts + ref_trans.unsqueeze(-2),
                    ref_scale_corr * ref_joints + ref_trans.unsqueeze(-2),
                    vertex_weights, joint_weights,
                    glob_rotmats, shape_betas, ref_scale_corr, ref_trans, kid_factor)
            else:
                glob_rotmats = self._fit_global_rotations_dependent(
                    target_vertices, target_joints, ref_verts + ref_trans, ref_joints + ref_trans,
                    vertex_weights, joint_weights, glob_rotmats, shape_betas, None,
                    ref_trans, kid_factor)

        if 'joints' in requested_keys or 'vertices' in requested_keys:
            forw = self.body_model(
                glob_rotmats=glob_rotmats, shape_betas=shape_betas, trans=ref_trans,
                kid_factor=kid_factor)
        else:
            forw = {}

        # Add the mean back
        result['trans'] = ref_trans + target_mean
        if 'joints' in requested_keys:
            result['joints'] = forw['joints'] + target_mean[:, None]
        if 'vertices' in requested_keys:
            result['vertices'] = forw['vertices'] + target_mean[:, None]

        result['orientations'] = glob_rotmats

        if scale_fit:
            assert ref_scale_corr is not None
            result['scale_corr'] = ref_scale_corr

        # Provide other requested rotation formats
        if 'relative_orientations' in requested_keys or 'pose_rotvecs' in requested_keys:
            parent_glob_rotmats = torch.cat([
                torch.eye(3, device=device).expand(glob_rotmats.shape[0], 1, 3, 3),
                torch.index_select(glob_rotmats, 1, parent_indices)], dim=1)
            result['relative_orientations'] = torch.matmul(
                parent_glob_rotmats.transpose(-1, -2), glob_rotmats)

        if 'pose_rotvecs' in requested_keys:
            rel_ori = result['relative_orientations']
            assert rel_ori is not None
            rotvecs = mat2rotvec(rel_ori)
            result['pose_rotvecs'] = rotvecs.view(rotvecs.shape[0], -1)

        result_non_none: Dict[str, torch.Tensor] = {}
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
            requested_keys: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        if scale_target and scale_fit:
            raise ValueError("Only one of estim_scale_target and estim_scale_fit can be True")
        if requested_keys is None:
            requested_keys = []

        glob_rotmats = glob_rotmats.float()
        batch_size = target_vertices.shape[0]

        device = self.body_model.v_template.device
        parent_indices = self.body_model.kintree_parents_tensor[1:].to(device)

        parent_glob_rot_mats = torch.cat([
            torch.eye(3, device=device).expand(glob_rotmats.shape[0], 1, 3, 3),
            torch.index_select(glob_rotmats, 1, parent_indices)], dim=1)
        rel_rotmats = torch.matmul(parent_glob_rot_mats.transpose(-1, -2), glob_rotmats)

        glob_positions_ext = [self.J_template_ext[None, 0].expand(batch_size, -1, -1)]
        for i_joint, i_parent in enumerate(self.body_model.kintree_parents[1:], start=1):
            glob_positions_ext.append(
                glob_positions_ext[i_parent] +
                torch.einsum('bCc,cs->bCs', glob_rotmats[:, i_parent],
                             self.J_template_ext[i_joint] - self.J_template_ext[i_parent]))
        glob_positions_ext = torch.stack(glob_positions_ext, dim=1)
        translations_ext = glob_positions_ext - torch.einsum(
            'bjCc,jcs->bjCs', glob_rotmats, self.J_template_ext)

        rot_params = rel_rotmats[:, 1:].reshape(-1, (self.body_model.num_joints - 1) * 3 * 3)
        v_posed = self.v_template + torch.einsum('vcp,bp->bvc', self.posedirs, rot_params)
        v_rotated = torch.einsum('bjCc,vj,bvc->bvC', glob_rotmats, self.weights, v_posed)

        shapedirs = (
            torch.cat([
                self.shapedirs[:, :, :self.n_betas],
                self.kid_shapedir[:, :, None]], dim=2) if self.enable_kid
            else self.shapedirs[:, :, :self.n_betas])
        v_grad_rotated = torch.einsum('bjCc,lj,lcs->blCs', glob_rotmats, self.weights, shapedirs)

        v_rotated_ext = torch.cat([v_rotated.unsqueeze(-1), v_grad_rotated], dim=3)
        v_translations_ext = torch.einsum('vj,bjcs->bvcs', self.weights, translations_ext)
        v_posed_posed_ext = v_translations_ext + v_rotated_ext

        if target_joints is None:
            target_both = target_vertices
            pos_both = v_posed_posed_ext[..., 0]
            jac_pos_both = v_posed_posed_ext[..., 1:]
        else:
            target_both = torch.cat([target_vertices, target_joints], dim=1)
            pos_both = torch.cat([v_posed_posed_ext[..., 0], glob_positions_ext[..., 0]], dim=1)
            jac_pos_both = torch.cat(
                [v_posed_posed_ext[..., 1:], glob_positions_ext[..., 1:]], dim=1)

        if scale_target:
            A = torch.cat([jac_pos_both, -target_both.unsqueeze(-1)], dim=3)
        elif scale_fit:
            A = torch.cat([jac_pos_both, pos_both.unsqueeze(-1)], dim=3)
        else:
            A = jac_pos_both

        b = target_both - pos_both
        mean_A = torch.mean(A, dim=1, keepdim=True)
        mean_b = torch.mean(b, dim=1, keepdim=True)
        A = A - mean_A
        b = b - mean_b

        if target_joints is not None and vertex_weights is not None and joint_weights is not None:
            weights = torch.cat([vertex_weights, joint_weights], dim=1)
        elif target_joints is None and vertex_weights is not None:
            weights = vertex_weights
        else:
            weights = torch.ones(A.shape[:2], dtype=torch.float32, device=device)

        n_params = (
                self.n_betas + (1 if self.enable_kid else 0) +
                (1 if scale_target or scale_fit else 0))
        A = A.reshape(batch_size, -1, n_params)
        b = b.reshape(batch_size, -1, 1)
        w = weights.reshape(batch_size, -1).repeat(1, 3)

        l2_regularizer_all = torch.cat([
            torch.full((2,), beta_regularizer2, device=device),
            torch.full((self.n_betas - 2,), beta_regularizer, device=device)
        ])
        if beta_regularizer_reference is None:
            l2_regularizer_reference_all = torch.zeros([batch_size, self.n_betas], device=device)
        else:
            l2_regularizer_reference_all = beta_regularizer_reference

        if self.enable_kid:
            if kid_regularizer is None:
                kid_regularizer = beta_regularizer
            if kid_regularizer_reference is None:
                kid_regularizer_reference = torch.zeros(batch_size, device=device)
            l2_regularizer_all = torch.cat(
                [l2_regularizer_all, torch.tensor([kid_regularizer], device=device)])
            l2_regularizer_reference_all = torch.cat(
                [l2_regularizer_reference_all, kid_regularizer_reference[:, np.newaxis]], dim=1)

        if scale_target or scale_fit:
            l2_regularizer_all = torch.cat(
                [l2_regularizer_all, torch.tensor([scale_regularizer], device=device)])

        l2_regularizer_rhs = (l2_regularizer_all * l2_regularizer_reference_all).unsqueeze(-1)

        l2_regularizer_all = torch.diag(l2_regularizer_all)
        # print('Loading penalty matrix')
        # penalty = np.load('/work_uncached/sarandi/data/projects/localizerfields
        # /smpl_beta_penalty_mat_new.npy')
        # penalty_bone = np.load('/work_uncached/sarandi/data/projects/localizerfields
        # /smpl_beta_penalty_mat_bone_new.npy')
        # l2_regularizer_all[:self.n_betas, :self.n_betas] = torch.tensor(penalty, device=device)
        # * beta_regularizer + torch.tensor(penalty_bone, device=device) * beta_regularizer2

        if share_beta:
            x = lstsq_partial_share(
                A, b, w, l2_regularizer_all, n_shared=self.n_betas + (1 if self.enable_kid else 0))
        else:
            x = lstsq(A, b, w, l2_regularizer_all, l2_regularizer_rhs=l2_regularizer_rhs)

        x = x.squeeze(-1)
        new_trans = mean_b.squeeze(1) - torch.matmul(mean_A.squeeze(1), x.unsqueeze(-1)).squeeze(-1)
        new_shape = x[:, :self.n_betas]

        result = dict(shape_betas=new_shape, trans=new_trans, relative_orientations=rel_rotmats)

        if self.enable_kid:
            new_kid_factor = x[:, self.n_betas]
            result['kid_factor'] = new_kid_factor
        else:
            new_kid_factor = None

        if scale_target or scale_fit:
            new_scale_corr = x[:, -1] + 1
            if scale_fit and new_scale_corr is not None:
                new_shape /= new_scale_corr.unsqueeze(-1)
            result['scale_corr'] = new_scale_corr
        else:
            new_scale_corr = None

        if self.enable_kid and new_kid_factor is not None:
            new_shape = torch.cat([new_shape, new_kid_factor.unsqueeze(-1)], dim=1)

        if 'joints' in requested_keys:
            result['joints'] = (
                    glob_positions_ext[..., 0] +
                    torch.einsum('bvcs,bs->bvc', glob_positions_ext[..., 1:], new_shape) +
                    new_trans.unsqueeze(1))

        if 'vertices' in requested_keys:
            result['vertices'] = (
                    v_posed_posed_ext[..., 0] +
                    torch.einsum('bvcs,bs->bvc', v_posed_posed_ext[..., 1:], new_shape) +
                    new_trans.unsqueeze(1))
        return result

    def _fit_global_rotations(
            self,
            target_vertices: torch.Tensor,
            target_joints: Optional[torch.Tensor],
            reference_vertices: torch.Tensor,
            reference_joints: Optional[torch.Tensor],
            vertex_weights: Optional[torch.Tensor],
            joint_weights: Optional[torch.Tensor]
    ) -> torch.Tensor:
        device = target_vertices.device
        glob_rots = []
        mesh_weight = torch.tensor(1e-6, device=device, dtype=torch.float32)
        joint_weight = 1 - mesh_weight

        if target_joints is None or reference_joints is None:
            target_joints = self.J_regressor @ target_vertices
            reference_joints = self.J_regressor @ reference_vertices

        part_assignment = torch.argmax(self.weights, dim=1)
        # Disable the rotation of toes separately from the feet
        part_assignment = torch.where(
            part_assignment == 10,
            torch.tensor(7, dtype=torch.int64, device=device), part_assignment)
        part_assignment = torch.where(
            part_assignment == 11,
            torch.tensor(8, dtype=torch.int64, device=device), part_assignment)

        for i in range(self.body_model.num_joints):
            # Disable the rotation of toes separately from the feet
            if i == 10:
                glob_rots.append(glob_rots[7])
                continue
            elif i == 11:
                glob_rots.append(glob_rots[8])
                continue

            selector = torch.where(part_assignment == i)[0]
            default_body_part = reference_vertices[:, selector]
            estim_body_part = target_vertices[:, selector]
            weights_body_part = (
                vertex_weights[:, selector].unsqueeze(-1) * mesh_weight
                if vertex_weights is not None else mesh_weight)

            default_joints = reference_joints[:, self.children_and_self[i]]
            estim_joints = target_joints[:, self.children_and_self[i]]
            weights_joints = (
                joint_weights[:, self.children_and_self[i]].unsqueeze(-1) * joint_weight
                if joint_weights is not None else joint_weight)

            body_part_mean_reference = torch.mean(default_joints, dim=1, keepdim=True)
            default_points = torch.cat([
                (default_body_part - body_part_mean_reference) * weights_body_part,
                (default_joints - body_part_mean_reference) * weights_joints], dim=1)

            body_part_mean_target = torch.mean(estim_joints, dim=1, keepdim=True)
            estim_points = torch.cat([
                (estim_body_part - body_part_mean_target),
                (estim_joints - body_part_mean_target)], dim=1)

            glob_rot = kabsch(estim_points, default_points)
            glob_rots.append(glob_rot)

        return torch.stack(glob_rots, dim=1)

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
            kid_factor: Optional[torch.Tensor]
    ) -> torch.Tensor:
        glob_rots = []

        true_reference_joints = reference_joints
        if target_joints is None or reference_joints is None:
            target_joints = self.J_regressor @ target_vertices
            reference_joints = self.J_regressor @ reference_vertices

        device = self.body_model.v_template.device
        part_assignment = torch.argmax(self.weights, dim=1)
        part_assignment = torch.where(
            part_assignment == 10, torch.tensor(7, dtype=torch.int64, device=device),
            part_assignment)
        part_assignment = torch.where(
            part_assignment == 11, torch.tensor(8, dtype=torch.int64, device=device),
            part_assignment)

        j = (self.body_model.J_template +
             torch.einsum(
                 'jcs,...s->...jc', self.body_model.J_shapedirs[:, :, :self.n_betas],
                 shape_betas[:, :self.n_betas]))
        if kid_factor is not None:
            j += torch.einsum('jc,...->...jc', self.body_model.kid_J_shapedir, kid_factor)

        if scale_corr is not None:
            j *= scale_corr[:, np.newaxis, np.newaxis]

        parent_indices = self.body_model.kintree_parents_tensor[1:].to(device)
        j_parent = torch.cat([
            torch.zeros(1, 3, device=device).expand(j.shape[0], -1, -1),
            torch.index_select(j, 1, parent_indices)], dim=1)
        bones = j - j_parent

        glob_positions = []

        for i in range(self.body_model.num_joints):
            if i == 0:
                glob_position = j[:, i] + trans
            else:
                i_parent = self.body_model.kintree_parents[i]
                glob_position = (
                        glob_positions[i_parent] +
                        torch.matmul(glob_rots[i_parent], bones[:, i].unsqueeze(-1)).squeeze(-1))
            glob_positions.append(glob_position)

            if i == 10:
                glob_rots.append(glob_rots[7])
                continue
            elif i == 11:
                glob_rots.append(glob_rots[8])
                continue

            vertex_selector = torch.where(part_assignment == i)[0]
            joint_selector = self.children_and_self[i]

            default_body_part = reference_vertices[:, vertex_selector]
            estim_body_part = target_vertices[:, vertex_selector]
            weights_body_part = (
                vertex_weights[:, vertex_selector].unsqueeze(-1)
                if vertex_weights is not None else torch.tensor(
                    1.0, dtype=torch.float32, device=device))

            default_joints = reference_joints[:, joint_selector]
            estim_joints = target_joints[:, joint_selector]
            weights_joints = (
                joint_weights[:, joint_selector].unsqueeze(-1)
                if joint_weights is not None else torch.tensor(
                    1.0, dtype=torch.float32, device=device))

            reference_point = glob_position.unsqueeze(1)
            default_reference_point = true_reference_joints[:, i:i + 1]
            default_points = torch.cat([
                (default_body_part - default_reference_point) * weights_body_part,
                (default_joints - default_reference_point) * weights_joints
            ], dim=1)
            estim_points = torch.cat([
                (estim_body_part - reference_point),
                (estim_joints - reference_point)
            ], dim=1)
            glob_rot = kabsch(estim_points, default_points) @ glob_rots_prev[:, i]
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
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    device = target_vertices.device

    if target_joints is None or reference_joints is None:
        target_both = target_vertices
        reference_both = reference_vertices
        if vertex_weights is not None:
            weights_both = vertex_weights
        else:
            weights_both = torch.ones(
                target_vertices.shape[0], target_vertices.shape[1], device=device)
    else:
        target_both = torch.cat([target_vertices, target_joints], dim=1)
        reference_both = torch.cat([reference_vertices, reference_joints], dim=1)

        if vertex_weights is not None and joint_weights is not None:
            weights_both = torch.cat([vertex_weights, joint_weights], dim=1)
        else:
            weights_both = torch.ones(
                target_vertices.shape[0], target_vertices.shape[1] + target_joints.shape[1],
                device=device)

    weights_both /= torch.sum(weights_both, dim=1, keepdim=True)

    weighted_mean_target = torch.sum(target_both * weights_both.unsqueeze(-1), dim=1)
    weighted_mean_reference = torch.sum(reference_both * weights_both.unsqueeze(-1), dim=1)

    if scale:
        target_centered = target_both - weighted_mean_target[:, None]
        reference_centered = reference_both - weighted_mean_reference[:, None]

        ssq_reference = torch.sum(
            reference_centered ** 2 * weights_both.unsqueeze(-1), dim=(1, 2))
        ssq_target = torch.sum(
            target_centered ** 2 * weights_both.unsqueeze(-1), dim=(1, 2))

        # to make it unbiased, we could multiply by (1+2/(target_both.shape[1]))
        # but we are okay with the least squares solution
        scale_factor = torch.sqrt(ssq_target / ssq_reference)
        trans = weighted_mean_target - scale_factor * weighted_mean_reference
    else:
        scale_factor = None
        trans = weighted_mean_target - weighted_mean_reference

    return scale_factor, trans
