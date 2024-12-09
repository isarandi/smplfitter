import os
import pickle
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from smplfitter.pt.bodymodel import SMPLBodyModel
from smplfitter.pt.fitter import SMPLFitter
import scipy.spatial.distance
import scipy.optimize


from smplfitter.pt.converter import load_pickle, scipy2torch_csr

class SMPLFlipper(nn.Module):
    """
    Class to horizontally (along the x axis) flip SMPL-like body model parameters, to mirror the pose.

    Parameters:
        body_model (SMPLBodyModel): A body model whose parameters are to be transformed.
    """
    def __init__(self, body_model: SMPLBodyModel):
        super().__init__()
        self.body_model = body_model
        self.fitter = SMPLFitter(self.body_model, enable_kid=True, num_betas=10)

        res = self.body_model.single()
        self.mirror_csr = get_mirror_csr(body_model.num_vertices, res['joints'].device)
        self.mirror_inds_joints = get_mirror_mapping(res['joints'])
        self.mirror_inds = get_mirror_mapping(res['vertices'])

    def naive_flip(self, pose_rotvecs):
        hflip_multiplier = torch.tensor(
            [1, -1, -1], dtype=pose_rotvecs.dtype, device=pose_rotvecs.device)

        reshaped = pose_rotvecs.reshape(-1, self.body_model.num_joints, 3)
        reshaped_flipped = reshaped[:, self.mirror_inds_joints] * hflip_multiplier
        return reshaped_flipped.reshape(-1, self.body_model.num_joints*3)

    def convert_to_mirror(self, inp_vertices):
        v = inp_vertices.permute(1, 0, 2).reshape(self.body_model.num_vertices, -1)
        r = torch.sparse.mm(self.mirror_csr, v)
        return r.reshape(self.body_model.num_vertices, -1, 3).permute(1, 0, 2)


    @torch.jit.export
    def flip(
            self,
            pose_rotvecs: torch.Tensor,
            shape_betas: torch.Tensor,
            trans: torch.Tensor,
            kid_factor: Optional[torch.Tensor] = None,
            num_iter: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Converts the input body parameters to the output body model's parametrization.

        Parameters:
            pose_rotvecs (torch.Tensor): Input body part orientations expressed as rotation vectors concatenated to shape (batch_size, num_joints*3).
            shape_betas (torch.Tensor): Input beta coefficients representing body shape.
            trans (torch.Tensor): Input translation parameters (meters).
            kid_factor (Optional[torch.Tensor], optional): Coefficient for the kid blendshape which is the difference of the SMIL infant mesh and the adult tempate mesh. Default is None, which disables the use of the kid factor. See the AGORA paper :cite:`Patel21CVPR` for more information.
            num_iter (int, optional): Number of iterations for fitting. Default is 1.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the conversion results.
        """
        inp = self.body_model(pose_rotvecs, shape_betas, trans)
        hflip_multiplier = torch.tensor(
            [-1, 1, 1], dtype=inp['vertices'].dtype, device=inp['vertices'].device)
        vertices = self.convert_to_mirror(inp['vertices']) * hflip_multiplier

        fit = self.fitter.fit(
            target_vertices=vertices,
            n_iter=num_iter, beta_regularizer=0.0,
            beta_regularizer2=0,
            final_adjust_rots=False,
            kid_regularizer=1e9 if kid_factor is None else 0.0,
            initial_pose_rotvecs=self.naive_flip(pose_rotvecs),
            #initial_shape_betas=shape_betas,
            requested_keys=['pose_rotvecs', 'shape_betas'])
        fit_out = dict(pose_rotvecs=fit['pose_rotvecs'], shape_betas=fit['shape_betas'],
                       trans=fit['trans'])

        return fit_out


def get_mirror_mapping(points):
    points_np = points.cpu().numpy()
    dist = scipy.spatial.distance.cdist(points_np, points_np * [-1, 1, 1])
    v_inds, mirror_inds = scipy.optimize.linear_sum_assignment(dist)
    return torch.tensor(
        mirror_inds[np.argsort(v_inds)], dtype=torch.int,
        device=points.device)

def get_mirror_csr(num_verts, device):
    DATA_ROOT = os.environ['DATA_ROOT']

    m = np.load(f'{DATA_ROOT}/body_models/smplx/smplx_flip_correspondences.npz')
    smplx2mirror = scipy.sparse.coo_matrix(
        (m['bc'].flatten(),
         (np.repeat(np.arange(m['bc'].shape[0]), 3), m['closest_faces'].flatten())
         ),
        shape=(m['bc'].shape[0], m['bc'].shape[0])).tocsr().astype(np.float32)

    if num_verts == 6890:
        smpl2smplx_csr = load_pickle(
            f'{DATA_ROOT}/body_models/smpl2smplx_deftrafo_setup.pkl'
        )['mtx'].tocsr().astype(np.float32)[:, :6890]
        smplx2smpl_csr = load_pickle(
            f'{DATA_ROOT}/body_models/smplx2smpl_deftrafo_setup.pkl'
        )['mtx'].tocsr().astype(np.float32)[:, :10475]
        smplx2mirror = scipy.sparse.coo_matrix(
            (m['bc'].flatten(),
             (np.repeat(np.arange(m['bc'].shape[0]), 3), m['closest_faces'].flatten())
             ),
            shape=(m['bc'].shape[0], m['bc'].shape[0])).tocsr().astype(np.float32)
        smpl2mirror = smplx2smpl_csr @ smplx2mirror @ smpl2smplx_csr
        return scipy2torch_csr(smpl2mirror)
    elif num_verts == 10475:
        return scipy2torch_csr(smplx2mirror)
    else:
        raise ValueError(f'Unsupported number of vertices: {num_verts}')
