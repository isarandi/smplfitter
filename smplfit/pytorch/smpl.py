import torch
import torch.nn as nn

import smplfit.common
from smplfit.pytorch.rotation import rotvec2mat


class SMPL(nn.Module):
    def __init__(self, model_root=None, gender='neutral', model_type='basic', model_name='smpl'):
        """
        Args:
            model_root: path to pickle files for the model (see https://smpl.is.tue.mpg.de).
            gender: 'neutral', 'f' or 'm'
        """
        super(SMPL, self).__init__()
        self.gender = gender
        self.model_name = model_name
        tensors, nontensors = smplfit.common.initialize(
            model_root, model_name, gender, model_type)

        # Register buffers and parameters
        self.register_buffer('v_template', torch.tensor(tensors.v_template, dtype=torch.float32))
        self.register_buffer('shapedirs', torch.tensor(tensors.shapedirs, dtype=torch.float32))
        self.register_buffer('posedirs', torch.tensor(tensors.posedirs, dtype=torch.float32))
        self.register_buffer('v_dirs', torch.tensor(tensors.v_dirs, dtype=torch.float32))
        self.register_buffer('J_regressor', torch.tensor(tensors.J_regressor, dtype=torch.float32))
        self.register_buffer('J_template', torch.tensor(tensors.J_template, dtype=torch.float32))
        self.register_buffer('J_shapedirs', torch.tensor(tensors.J_shapedirs, dtype=torch.float32))
        self.register_buffer('kid_shapedir',
                             torch.tensor(tensors.kid_shapedir, dtype=torch.float32))
        self.register_buffer('kid_J_shapedir',
                             torch.tensor(tensors.kid_J_shapedir, dtype=torch.float32))
        self.register_buffer('weights', torch.tensor(tensors.weights, dtype=torch.float32))
        self.register_buffer('kintree_parents_tensor',
                             torch.tensor(nontensors.kintree_parents, dtype=torch.int64))

        self.kintree_parents = nontensors.kintree_parents
        self.faces = nontensors.faces
        self.num_joints = nontensors.num_joints
        self.num_vertices = nontensors.num_vertices

    def forward(
            self, pose_rotvecs=None, shape_betas=None, trans=None, kid_factor=None,
            rel_rotmats=None, glob_rotmats=None, *, return_vertices=True):
        """Calculate the SMPL body model vertices, joint positions, and orientations given the input
         pose, shape parameters.

        Args:
            pose_rotvecs (torch.Tensor, optional): Tensor representing rotation vectors for each
            joint pose.
            shape_betas (torch.Tensor, optional): Tensor representing shape coefficients (betas) for
            the body shape.
            trans (torch.Tensor, optional): Tensor representing the translation vector to apply
            after posing.
            kid_factor (float, optional): Adjustment factor for child shapes. Defaults to 0.
            rel_rotmats (torch.Tensor, optional): Tensor representing the rotation matrices for each
            joint in the pose.
            glob_rotmats (torch.Tensor, optional): Tensor representing global rotation matrices for
            the pose.
            return_vertices (bool, optional): Flag indicating whether to return the body model
            vertices. Defaults to True.

        Returns:
            dict: Dictionary containing 3D body model vertices ('vertices'), joint positions (
            'joints'),
                  and orientation matrices for each joint ('orientations').
        """
        batch_size = check_batch_size(pose_rotvecs, shape_betas, trans, rel_rotmats,
                                              glob_rotmats)
        device = self.v_template.device
        if rel_rotmats is not None:
            rel_rotmats = rel_rotmats.float()
        elif pose_rotvecs is not None:
            pose_rotvecs = pose_rotvecs.float()
            rel_rotmats = rotvec2mat(pose_rotvecs.view(batch_size, self.num_joints, 3))
        elif glob_rotmats is None:
            rel_rotmats = torch.eye(3, device=device).repeat(batch_size, self.num_joints, 1, 1)

        if glob_rotmats is None:
            glob_rotmats = [rel_rotmats[:, 0]]
            for i_joint in range(1, self.num_joints):
                i_parent = self.kintree_parents[i_joint]
                glob_rotmats.append(glob_rotmats[i_parent] @ rel_rotmats[:, i_joint])
            glob_rotmats = torch.stack(glob_rotmats, dim=1)

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
            j[:, parent_indices]],
            dim=1)
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
            return dict(joints=glob_positions + trans[:, None], orientations=glob_rotmats)

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
            joints=glob_positions + trans[:, None],
            vertices=vertices + trans[:, None],
            orientations=glob_rotmats)

    def single(self, *args, return_vertices=True, **kwargs):
        args = [arg.unsqueeze(0) for arg in args]
        kwargs = {k: v.unsqueeze(0) for k, v in kwargs.items()}
        if len(args) == 0 and len(kwargs) == 0:
            kwargs['shape_betas'] = torch.zeros(
                (1, 0), dtype=torch.float32, device=self.v_template.device)
        result = self(*args, return_vertices=return_vertices, **kwargs)
        return {k: v.squeeze(0) for k, v in result.items()}


def check_batch_size(pose_rotvecs, shape_betas, trans, rel_rotmats, glob_rotmats):
    batch_sizes = [
        x.shape[0] for x in [pose_rotvecs, shape_betas, trans, rel_rotmats, glob_rotmats]
        if x is not None]

    if len(batch_sizes) == 0:
        raise RuntimeError(
            'At least one argument must be given among pose_rotvecs, shape_betas, trans, '
            'rel_rotmats.')

    return batch_sizes[0]
