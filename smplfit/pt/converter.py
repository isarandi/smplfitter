import os
import pickle
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import numpy as np
from smplfit.pt.smpl import SMPL
from smplfit.pt.fitting import SMPLfit
import torch


class SMPLConverter(nn.Module):
    def __init__(
            self,
            body_model_in: str,
            gender_in: str,
            body_model_out: str,
            gender_out: str,
            num_betas_out: int = 10
    ):
        super(SMPLConverter, self).__init__()
        self.body_model_in = SMPL(model_name=body_model_in, gender=gender_in)
        self.body_model_out = SMPL(model_name=body_model_out, gender=gender_out)
        self.fitter = SMPLfit(self.body_model_out, num_betas=num_betas_out, enable_kid=True)
        self.fitter = torch.jit.script(self.fitter)

        DATA_ROOT = os.environ['DATA_ROOT']
        if self.body_model_in.num_vertices == 6890 and self.body_model_out.num_vertices == 10475:
            vertex_converter_path = f'{DATA_ROOT}/body_models/smpl2smplx_deftrafo_setup.pkl'
        elif self.body_model_in.num_vertices == 10475 and self.body_model_out.num_vertices == 6890:
            vertex_converter_path = f'{DATA_ROOT}/body_models/smplx2smpl_deftrafo_setup.pkl'
        else:
            vertex_converter_path = None

        if vertex_converter_path is not None:
            scipy_csr = load_pickle(vertex_converter_path)['mtx'].tocsr().astype(np.float32)
            self.vertex_converter_csr = scipy2torch_csr(
                scipy_csr[:, :self.body_model_in.num_vertices])
        else:
            self.vertex_converter_csr = None

    def convert_vertices(self, inp_vertices: torch.Tensor) -> torch.Tensor:
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




def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def scipy2torch_csr(sparse_matrix):
    return torch.sparse_csr_tensor(
        torch.from_numpy(sparse_matrix.indptr),
        torch.from_numpy(sparse_matrix.indices),
        torch.from_numpy(sparse_matrix.data),
        sparse_matrix.shape)
