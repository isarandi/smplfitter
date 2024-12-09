from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

import smplfitter.common
from smplfitter.pt.rotation import rotvec2mat, mat2rotvec


class SMPLBodyModel(nn.Module):
    """
    Represents a statistical body model of the SMPL family.

    The SMPL (Skinned Multi-Person Linear) model provides a way to represent articulated 3D human
    meshes through a compact shape vector (beta) and pose (body part rotation) parameters.

    Parameters:
        model_name (str, optional): Name of the model type, typically 'smpl'. Default is 'smpl'.
        gender (str, optional): Gender of the model, which can be 'neutral', 'f' (female),
        or 'm' (male). Default is 'neutral'.
        model_root (str, optional): Path to the directory containing model files. By default,
        {DATA_ROOT}/body_models/{model_name} is used, with the DATA_ROOT environment variable.
    """

    def __init__(self, model_name='smpl', gender='neutral', model_root=None, unit='m', num_betas=None):
        super().__init__()
        self.gender = gender
        self.model_name = model_name
        tensors, nontensors = smplfitter.common.initialize(model_name, gender, model_root, num_betas)

        # Register buffers and parameters
        self.register_buffer('v_template', torch.tensor(tensors['v_template'], dtype=torch.float32))
        self.register_buffer('shapedirs', torch.tensor(tensors['shapedirs'], dtype=torch.float32))
        self.register_buffer('posedirs', torch.tensor(tensors['posedirs'], dtype=torch.float32))
        self.register_buffer('J_regressor',
                             torch.tensor(tensors['J_regressor'], dtype=torch.float32))
        self.register_buffer('J_template', torch.tensor(tensors['J_template'], dtype=torch.float32))
        self.register_buffer('J_shapedirs',
                             torch.tensor(tensors['J_shapedirs'], dtype=torch.float32))
        self.register_buffer('kid_shapedir',
                             torch.tensor(tensors['kid_shapedir'], dtype=torch.float32))
        self.register_buffer('kid_J_shapedir',
                             torch.tensor(tensors['kid_J_shapedir'], dtype=torch.float32))
        self.register_buffer('weights', torch.tensor(tensors['weights'], dtype=torch.float32))
        self.register_buffer('kintree_parents_tensor',
                             torch.tensor(nontensors['kintree_parents'], dtype=torch.int64))

        self.kintree_parents = nontensors['kintree_parents']
        self.faces = nontensors['faces']
        self.num_joints = nontensors['num_joints']
        self.num_vertices = nontensors['num_vertices']
        self.unit_factor = dict(mm=1000, cm=100, m=1)[unit]

    def forward(
            self,
            pose_rotvecs: Optional[torch.Tensor] = None,
            shape_betas: Optional[torch.Tensor] = None,
            trans: Optional[torch.Tensor] = None,
            kid_factor: Optional[torch.Tensor] = None,
            rel_rotmats: Optional[torch.Tensor] = None,
            glob_rotmats: Optional[torch.Tensor] = None,
            return_vertices: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the body model vertices, joint positions, and orientations for a batch of
        instances given the input pose, shape, and translation parameters. The rotation may be
        specified as one of three options:
          * parent-relative rotation vectors,
          * parent-relative rotation matrices, or
          * global rotation matrices

        Parameters:
            pose_rotvecs (Optional[torch.Tensor]): Rotation vectors per joint, shaped as (
            batch_size, num_joints, 3) or flattened as (batch_size, num_joints * 3).
            shape_betas (Optional[torch.Tensor]): Shape coefficients (betas) for the body shape,
            shaped as (batch_size, num_betas).
            trans (Optional[torch.Tensor]): Translation vector to apply after posing, shaped as (
            batch_size, 3).
            kid_factor (Optional[torch.Tensor]): Adjustment factor for child shapes, shaped as (
            batch_size, 1). Default is None.
            rel_rotmats (Optional[torch.Tensor]): Parent-relative rotation matrices per joint,
            shaped as (batch_size, num_joints, 3, 3).
            glob_rotmats (Optional[torch.Tensor]): Global rotation matrices per joint, shaped as
            (batch_size, num_joints, 3, 3).
            return_vertices (bool): Flag indicating whether to compute and return the body model
            vertices. Default is True. If only joints and orientations are needed, setting this
            to False is faster.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - 'vertices' (torch.Tensor): 3D body model vertices, shaped as (batch_size,
                num_vertices, 3), if `return_vertices` is True.
                - 'joints' (torch.Tensor): 3D joint positions, shaped as (batch_size, num_joints,
                3).
                - 'orientations' (torch.Tensor): Global orientation matrices for each joint,
                shaped as (batch_size, num_joints, 3, 3).
        """

        batch_size = 0
        for arg in [pose_rotvecs, shape_betas, trans, rel_rotmats, glob_rotmats]:
            if arg is not None:
                batch_size = arg.shape[0]
                break

        device = self.v_template.device
        if rel_rotmats is not None:
            rel_rotmats = rel_rotmats.float()
        elif pose_rotvecs is not None:
            pose_rotvecs = pose_rotvecs.float()
            rel_rotmats = rotvec2mat(pose_rotvecs.view(batch_size, self.num_joints, 3))
        elif glob_rotmats is None:
            rel_rotmats = torch.eye(3, device=device).repeat(batch_size, self.num_joints, 1, 1)

        if glob_rotmats is None:
            if rel_rotmats is None:
                raise ValueError('Rotation info missing.')
            glob_rotmats_ = [rel_rotmats[:, 0]]
            for i_joint in range(1, self.num_joints):
                i_parent = self.kintree_parents[i_joint]
                glob_rotmats_.append(glob_rotmats_[i_parent] @ rel_rotmats[:, i_joint])
            glob_rotmats = torch.stack(glob_rotmats_, dim=1)

        parent_indices = self.kintree_parents_tensor[1:].to(glob_rotmats.device)
        parent_glob_rotmats = torch.cat([
            torch.eye(3, device=device).expand(glob_rotmats.shape[0], 1, 3, 3),
            glob_rotmats.index_select(1, parent_indices)],
            dim=1)

        if rel_rotmats is None:
            rel_rotmats = torch.matmul(parent_glob_rotmats.transpose(-1, -2), glob_rotmats)

        shape_betas = shape_betas.float() if shape_betas is not None else torch.zeros(
            (batch_size, 0), dtype=torch.float32, device=device)
        num_betas = min(shape_betas.shape[1], self.shapedirs.shape[2])

        kid_factor = torch.zeros(
            (1,), dtype=torch.float32, device=device) if kid_factor is None else torch.tensor(
            kid_factor, dtype=torch.float32, device=device)
        j = (self.J_template +
             torch.einsum('jcs,bs->bjc', self.J_shapedirs[:, :, :num_betas],
                          shape_betas[:, :num_betas]) +
             torch.einsum('jc,b->bjc', self.kid_J_shapedir, kid_factor))

        j_parent = torch.cat([
            torch.zeros(3, device=device).expand(j.shape[0], 1, 3),
            j[:, parent_indices]], dim=1)
        bones = j - j_parent
        rotated_bones = torch.einsum('bjCc,bjc->bjC', parent_glob_rotmats, bones)

        glob_positions = [j[:, 0]]
        for i_joint in range(1, self.num_joints):
            i_parent = self.kintree_parents[i_joint]
            glob_positions.append(glob_positions[i_parent] + rotated_bones[:, i_joint])
        glob_positions = torch.stack(glob_positions, dim=1)

        trans = torch.zeros(
            (1, 3), dtype=torch.float32, device=device) if trans is None else trans.float()

        if not return_vertices:
            return dict(
                joints=(glob_positions + trans[:, None]) * self.unit_factor,
                orientations=glob_rotmats)

        pose_feature = rel_rotmats[:, 1:].reshape(-1, (self.num_joints - 1) * 3 * 3)
        v_posed = (
                self.v_template +
                torch.einsum('vcp,bp->bvc', self.shapedirs[:, :, :num_betas],
                             shape_betas[:, :num_betas]) +
                torch.einsum('vcp,bp->bvc', self.posedirs, pose_feature) +
                torch.einsum('vc,b->bvc', self.kid_shapedir, kid_factor))

        translations = glob_positions - torch.einsum('bjCc,bjc->bjC', glob_rotmats, j)
        vertices = (
                torch.einsum('bjCc,vj,bvc->bvC', glob_rotmats, self.weights, v_posed) +
                self.weights @ translations)

        return dict(
            joints=(glob_positions + trans[:, None]) * self.unit_factor,
            vertices=(vertices + trans[:, None]) * self.unit_factor,
            orientations=glob_rotmats)

    @torch.jit.export
    def single(
            self,
            pose_rotvecs: Optional[torch.Tensor] = None,
            shape_betas: Optional[torch.Tensor] = None,
            trans: Optional[torch.Tensor] = None,
            kid_factor: Optional[torch.Tensor] = None,
            rel_rotmats: Optional[torch.Tensor] = None,
            glob_rotmats: Optional[torch.Tensor] = None,
            return_vertices: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the body model vertices, joint positions, and orientations for a single
        instance given the input pose, shape, and translation parameters. The rotation may be
        specified as one of three options:
          * parent-relative rotation vectors,
          * parent-relative rotation matrices, or
          * global rotation matrices

        Parameters:
            pose_rotvecs (Optional[torch.Tensor]): Rotation vectors per joint, shaped as (
            num_joints, 3) or (num_joints * 3,).
            shape_betas (Optional[torch.Tensor]): Shape coefficients (betas) for the body shape,
            shaped as (num_betas,).
            trans (Optional[torch.Tensor]): Translation vector to apply after posing, shaped as (
            3,).
            kid_factor (Optional[torch.Tensor]): Adjustment factor for child shapes, shaped as (
            1,). Default is None.
            rel_rotmats (Optional[torch.Tensor]): Parent-relative rotation matrices per joint,
            shaped as (num_joints, 3, 3).
            glob_rotmats (Optional[torch.Tensor]): Global rotation matrices per joint, shaped as
            (num_joints, 3, 3).
            return_vertices (bool): Flag indicating whether to compute and return the body model
            vertices. Default is True. If only joints and orientations are needed, False is much
            faster.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - 'vertices' (torch.Tensor): 3D body model vertices, shaped as (num_vertices, 3),
                if `return_vertices` is True.
                - 'joints' (torch.Tensor): 3D joint positions, shaped as (num_joints, 3).
                - 'orientations' (torch.Tensor): Global orientation matrices for each joint,
                shaped as (num_joints, 3, 3).
        """

        # Add batch dimension by unsqueezing to shape (1, ...)
        pose_rotvecs = pose_rotvecs.unsqueeze(0) if pose_rotvecs is not None else None
        shape_betas = shape_betas.unsqueeze(0) if shape_betas is not None else None
        trans = trans.unsqueeze(0) if trans is not None else None
        rel_rotmats = rel_rotmats.unsqueeze(0) if rel_rotmats is not None else None
        glob_rotmats = glob_rotmats.unsqueeze(0) if glob_rotmats is not None else None

        # if all are None, then shape_betas is made to be zeros(1,0)
        if (pose_rotvecs is None and shape_betas is None and trans is None and rel_rotmats is None
                and glob_rotmats is None):
            shape_betas = torch.zeros(
                (1, 0), dtype=torch.float32, device=self.v_template.device)

        # Call forward with the adjusted arguments
        result = self.forward(
            pose_rotvecs=pose_rotvecs,
            shape_betas=shape_betas,
            trans=trans,
            kid_factor=kid_factor,
            rel_rotmats=rel_rotmats,
            glob_rotmats=glob_rotmats,
            return_vertices=return_vertices
        )

        # Squeeze out the batch dimension in the result
        return {k: v.squeeze(0) for k, v in result.items()}

    @torch.jit.export
    def rototranslate(
            self,
            R: torch.Tensor,
            t: torch.Tensor,
            pose_rotvecs: torch.Tensor,
            shape_betas: torch.Tensor,
            trans: torch.Tensor,
            kid_factor: Optional[torch.Tensor] = None,
            post_translate: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rotate and translate the body in parametric form.

        Parameters:
            R (torch.Tensor): Rotation matrix, shaped as (3, 3).
            t (torch.Tensor): Translation vector, shaped as (3,).
            pose_rotvecs (torch.Tensor): Initial rotation vectors per joint, shaped as (
            num_joints * 3,).
            shape_betas (torch.Tensor): Shape coefficients (betas) for body shape, shaped as (
            num_betas,).
            trans (torch.Tensor): Initial translation vector, shaped as (3,).
            kid_factor (Optional[torch.Tensor]): Optional in case of kid shapes like in AGORA.
            Shaped as (1,). Default is None.
            post_translate (bool): Flag indicating whether to apply the translation after rotation.
                If True, `t` is added after rotation by `R`; if False, `t` is subtracted before
                rotation by `R`. Default is True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - `new_pose_rotvec` (torch.Tensor): Updated pose rotation vectors, shaped as (
                num_joints * 3,).
                - `new_trans` (torch.Tensor): Updated translation vector, shaped as (3,).

        Notes:
            Rotating a parametric representation is nontrivial because the global orientation (
            first three rotation parameters) perform the rotation around the pelvis joint instad
            of the origin of the canonical coordinate system.
            This method takes into accound the offset between the pelvis joint in the shaped
            T-pose and the origin of the canonical coordinate system.

        """
        current_rotmat = rotvec2mat(pose_rotvecs[:3])
        new_rotmat = torch.matmul(R, current_rotmat)
        new_pose_rotvec = torch.cat([mat2rotvec(new_rotmat), pose_rotvecs[3:]], dim=0)

        pelvis = (
                self.J_template[0]
                + self.J_shapedirs[0, :, :shape_betas.shape[0]] @ shape_betas)
        if kid_factor is not None:
            pelvis += self.kid_J_shapedir[0] * kid_factor

        pelvis *= self.unit_factor

        eye3 = torch.eye(3, device=R.device, dtype=R.dtype)
        if post_translate:
            new_trans = trans @ R.mT + t + pelvis @ (R.mT - eye3)
        else:
            new_trans = (trans - t) @ R.mT + pelvis @ (R.mT - eye3)

        return new_pose_rotvec, new_trans
