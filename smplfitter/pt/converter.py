import os
import pickle
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from smplfitter.pt.bodymodel import SMPLBodyModel
from smplfitter.pt.fitter import SMPLFitter


class SMPLConverter(nn.Module):
    """
    Class to convert between different SMPL-like body model parameters.

    Parameters:
        body_model_in (str): Name of the input body model, one of 'smpl', 'smplx', 'smplh', 'smplh16'.
        gender_in (str): Gender of the input body model, one of 'female', 'male' or 'neutral'.
        body_model_out (str): Name of the output body model, one of 'smpl', 'smplx', 'smplh', 'smplh16'.
        gender_out (str): Gender of the output body model, one of 'female', 'male' or 'neutral'.
        num_betas_out (int): Number of estimated shape betas for output model. Default is 10.
    """
    def __init__(
            self,
            body_model_in: str,
            gender_in: str,
            body_model_out: str,
            gender_out: str,
            num_betas_out: int = 10
    ):
        super().__init__()
        self.body_model_in = SMPLBodyModel(model_name=body_model_in, gender=gender_in)
        self.body_model_out = SMPLBodyModel(model_name=body_model_out, gender=gender_out)
        self.fitter = SMPLFitter(self.body_model_out, num_betas=num_betas_out, enable_kid=True)

        DATA_ROOT = os.environ['DATA_ROOT']
        if self.body_model_in.num_vertices == 6890 and self.body_model_out.num_vertices == 10475:
            vertex_converter_path = f'{DATA_ROOT}/body_models/smpl2smplx_deftrafo_setup.pkl'
        elif self.body_model_in.num_vertices == 10475 and self.body_model_out.num_vertices == 6890:
            vertex_converter_path = f'{DATA_ROOT}/body_models/smplx2smpl_deftrafo_setup.pkl'
        else:
            vertex_converter_path = None

        if vertex_converter_path is not None:
            self.vertex_converter_csr = scipy2torch_csr(
                load_vertex_converter_csr(vertex_converter_path))
        else:
            self.vertex_converter_csr = None



    @torch.jit.export
    def convert_vertices(self, inp_vertices: torch.Tensor) -> torch.Tensor:
        """
        Converts body mesh vertices from the input model to the output body model's topology, via barycentric interpolation. If no conversion is needed (i.e., same body mesh topology in both input and output model, such as with SMPL and SMPL+H), the input vertices are returned as is.

        Parameters:
            inp_vertices (torch.Tensor): Input tensor of vertices to convert, shape (batch_size, num_vertices_in, 3).

        Returns:
            torch.Tensor: Converted vertices tensor, shape (batch_size, num_vertices_out, 3).
        """
        if self.vertex_converter_csr is None:
            return inp_vertices

        v = inp_vertices.permute(1, 0, 2).reshape(self.body_model_in.num_vertices, -1)
        r = torch.sparse.mm(self.vertex_converter_csr, v)
        return r.reshape(self.body_model_out.num_vertices, -1, 3).permute(1, 0, 2)

    @torch.jit.export
    def convert(
            self,
            pose_rotvecs: torch.Tensor,
            shape_betas: torch.Tensor,
            trans: torch.Tensor,
            kid_factor: Optional[torch.Tensor] = None,
            known_output_pose_rotvecs: Optional[torch.Tensor] = None,
            known_output_shape_betas: Optional[torch.Tensor] = None,
            known_output_kid_factor: Optional[torch.Tensor] = None,
            num_iter: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Converts the input body parameters to the output body model's parametrization.

        Parameters:
            pose_rotvecs (torch.Tensor): Input body part orientations expressed as rotation vectors concatenated to shape (batch_size, num_joints*3).
            shape_betas (torch.Tensor): Input beta coefficients representing body shape.
            trans (torch.Tensor): Input translation parameters (meters).
            kid_factor (Optional[torch.Tensor], optional): Coefficient for the kid blendshape which is the difference of the SMIL infant mesh and the adult tempate mesh. Default is None, which disables the use of the kid factor. See the AGORA paper :cite:`Patel21CVPR` for more information.
            known_output_pose_rotvecs (Optional[torch.Tensor], optional): If the output pose is already known and only the shape and translation need to be estimated, supply it here. Default is None.
            known_output_shape_betas (Optional[torch.Tensor], optional): If the output body shape betas are already known and only the pose and translation need to be estimated, supply it here. Default is None.
            known_output_kid_factor (Optional[torch.Tensor], optional): You may supply a known kid factor similar to known_output_shape_betas. Default is None.
            num_iter (int, optional): Number of iterations for fitting. Default is 1.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the conversion results.
        """
        inp_vertices = self.body_model_in(pose_rotvecs, shape_betas, trans)['vertices']
        verts = self.convert_vertices(inp_vertices)

        if known_output_shape_betas is not None:
            fit = self.fitter.fit_with_known_shape(
                shape_betas=known_output_shape_betas, kid_factor=known_output_kid_factor,
                target_vertices=verts, n_iter=num_iter, final_adjust_rots=False,
                requested_keys=['pose_rotvecs'])
            fit_out = dict(pose_rotvecs=fit['pose_rotvecs'], trans=fit['trans'])
        elif known_output_pose_rotvecs is not None:
            fit = self.fitter.fit_with_known_pose(
                pose_rotvecs=known_output_pose_rotvecs, target_vertices=verts,
                beta_regularizer=0.0, kid_regularizer=1e9 if kid_factor is None else 0.0)
            fit_out = dict(shape_betas=fit['shape_betas'], trans=fit['trans'])
            if kid_factor is not None:
                fit_out['kid_factor'] = fit['kid_factor']
        else:
            fit = self.fitter.fit(
                target_vertices=verts, n_iter=num_iter, beta_regularizer=0.0,
                final_adjust_rots=False, kid_regularizer=1e9 if kid_factor is None else 0.0,
                requested_keys=['pose_rotvecs', 'shape_betas'])
            fit_out = dict(
                pose_rotvecs=fit['pose_rotvecs'], shape_betas=fit['shape_betas'], trans=fit['trans'])
            if kid_factor is not None:
                fit_out['kid_factor'] = fit['kid_factor']

        return fit_out


def load_vertex_converter_csr(vertex_converter_path):
    scipy_csr = load_pickle(vertex_converter_path)['mtx'].tocsr().astype(np.float32)
    return scipy_csr[:, :scipy_csr.shape[1]//2]


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def scipy2torch_csr(sparse_matrix):
    return torch.sparse_csr_tensor(
        torch.from_numpy(sparse_matrix.indptr),
        torch.from_numpy(sparse_matrix.indices),
        torch.from_numpy(sparse_matrix.data),
        sparse_matrix.shape)
