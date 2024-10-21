import os
import pickle

import numpy as np
import smplfit.pytorch
import smplfit.pytorch.fitting
import torch

DATA_ROOT = os.environ['DATA_ROOT']


class BodyModelConverter:
    def __init__(self, body_model_in, gender_in, body_model_out, gender_out, num_betas_out=10):
        self.body_model_in = smplfit.pytorch.SMPL(model_name=body_model_in, gender=gender_in)
        self.body_model_out = smplfit.pytorch.SMPL(model_name=body_model_out, gender=gender_out)
        self.fitter = smplfit.pytorch.fitting.Fitter(self.body_model_out, num_betas=num_betas_out)

        if self.body_model_in.num_vertices == 6890 and self.body_model_out.num_vertices == 10475:
            vertex_converter_path = f'{DATA_ROOT}/body_models/smpl2smplx_deftrafo_setup.pkl'
        elif self.body_model_in.num_vertices == 10475 and self.body_model_out.num_vertices == 6890:
            vertex_converter_path = f'{DATA_ROOT}/body_models/smplx2smpl_deftrafo_setup.pkl'
        else:
            vertex_converter_path = None

        if vertex_converter_path is not None:
            scipy_csr = (load_pickle(vertex_converter_path)['mtx'].tocsr()).astype(np.float32)
            self.vertex_converter_csr = scipy2torch_csr(
                scipy_csr[:, :self.body_model_in.num_vertices])
        else:
            self.vertex_converter_csr = None

    def convert_vertices(self, inp_vertices):
        if self.vertex_converter_csr is None:
            return inp_vertices

        v = inp_vertices.permute(1, 0, 2).reshape(self.body_model_in.num_vertices, -1)
        r = torch.sparse.mm(self.vertex_converter_csr, v)
        return r.reshape(self.body_model_out.num_vertices, -1, 3).permute(1, 0, 2)

    def convert(self, pose, betas, trans):
        res = self.body_model_in(pose, betas, trans)
        verts = self.convert_vertices(res['vertices'])
        fit = self.fitter.fit(
            to_fit=verts, n_iter=1, l2_regularizer=0, final_adjust_rots=True,
            requested_keys=['pose_rotvecs', 'shape_betas'])
        return fit['pose_rotvecs'], fit['shape_betas'], fit['trans']


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def scipy2torch_csr(sparse_matrix):
    return torch.sparse_csr_tensor(
        torch.from_numpy(sparse_matrix.indptr),
        torch.from_numpy(sparse_matrix.indices),
        torch.from_numpy(sparse_matrix.data),
        sparse_matrix.shape)
