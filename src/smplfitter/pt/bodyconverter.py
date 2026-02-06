"""Body model parameter converter (PyTorch implementation)."""

from __future__ import annotations
import os
import pickle
from typing import Optional, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from . import bodyfitter as _bodyfitter

if TYPE_CHECKING:
    import smplfitter.pt


class BodyConverter(nn.Module):
    """
    Converts between different SMPL-family body model parameters.

    Parameters:
        body_model_in: Input body model to convert from.
        body_model_out: Output body model to convert to.
    """

    def __init__(
        self,
        body_model_in: 'smplfitter.pt.BodyModel',
        body_model_out: 'smplfitter.pt.BodyModel',
    ):
        super().__init__()
        self.body_model_in = body_model_in
        self.body_model_out = body_model_out
        self.fitter = _bodyfitter.BodyFitter(self.body_model_out, enable_kid=True)

        DATA_ROOT = os.getenv('DATA_ROOT', '.')
        if self.body_model_in.num_vertices == 6890 and self.body_model_out.num_vertices == 10475:
            csr_path = f'{DATA_ROOT}/body_models/smpl2smplx_deftrafo_setup.pkl'
        elif self.body_model_in.num_vertices == 10475 and self.body_model_out.num_vertices == 6890:
            csr_path = f'{DATA_ROOT}/body_models/smplx2smpl_deftrafo_setup.pkl'
        else:
            csr_path = None

        self.vertex_converter_csr: Optional[nn.Buffer]
        if csr_path is not None:
            self.vertex_converter_csr = nn.Buffer(
                scipy2torch_csr(load_vertex_converter_csr(csr_path))
            )
        else:
            self.vertex_converter_csr = None

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
        num_iter: int = 1,
    ) -> dict[str, torch.Tensor]:
        """
        Converts the input body parameters to the output body model's parametrization.

        Parameters:
            pose_rotvecs: Input body part orientations expressed as rotation vectors
                concatenated to shape (batch_size, num_joints*3).
            shape_betas: Input beta coefficients representing body shape.
            trans: Input translation parameters (meters).
            kid_factor: Coefficient for the kid blendshape.
            known_output_pose_rotvecs: If the output pose is already known and only the
                shape and translation need to be estimated, supply it here.
            known_output_shape_betas: If the output body shape betas are already known
                and only the pose and translation need to be estimated, supply it here.
            known_output_kid_factor: You may supply a known kid factor similar to
                known_output_shape_betas.
            num_iter: Number of iterations for fitting.

        Returns:
            Dictionary containing the conversion results
                - **pose_rotvecs** (*torch.Tensor*) -- Converted body part orientations expressed \
                    as rotation vectors concatenated to shape (batch_size, num_joints*3).
                - **shape_betas** (*torch.Tensor*) -- Converted beta coefficients representing \
                    body shape.
                - **trans** (*torch.Tensor*) -- Converted translation parameters (meters).

        """

        inp_vertices = self.body_model_in(pose_rotvecs, shape_betas, trans)['vertices']
        verts = self.convert_vertices(inp_vertices)

        if known_output_shape_betas is not None:
            fit = self.fitter.fit_with_known_shape(
                shape_betas=known_output_shape_betas,
                kid_factor=known_output_kid_factor,
                target_vertices=verts,
                num_iter=num_iter,
                final_adjust_rots=False,
                requested_keys=['pose_rotvecs'],
            )
            fit_out = dict(pose_rotvecs=fit['pose_rotvecs'], trans=fit['trans'])
        elif known_output_pose_rotvecs is not None:
            fit = self.fitter.fit_with_known_pose(
                pose_rotvecs=known_output_pose_rotvecs,
                target_vertices=verts,
                beta_regularizer=0.0,
                kid_regularizer=1e9 if kid_factor is None else 0.0,
            )
            fit_out = dict(shape_betas=fit['shape_betas'], trans=fit['trans'])
            if kid_factor is not None:
                fit_out['kid_factor'] = fit['kid_factor']
        else:
            fit = self.fitter.fit(
                target_vertices=verts,
                num_iter=num_iter,
                beta_regularizer=0.0,
                final_adjust_rots=False,
                kid_regularizer=1e9 if kid_factor is None else 0.0,
                requested_keys=['pose_rotvecs', 'shape_betas'],
            )
            fit_out = dict(
                pose_rotvecs=fit['pose_rotvecs'],
                shape_betas=fit['shape_betas'],
                trans=fit['trans'],
            )
            if kid_factor is not None:
                fit_out['kid_factor'] = fit['kid_factor']

        return fit_out

    @torch.jit.export
    def convert_vertices(self, inp_vertices: torch.Tensor) -> torch.Tensor:
        """
        Converts body mesh vertices from the input model to the output body model's topology
        using barycentric coordinates. If no conversion is needed (e.g., same body mesh
        topology in both input and output models, such as between SMPL and SMPL+H), the input
        vertices are returned without change.

        Parameters:
            inp_vertices: Input vertices to convert, with shape (batch_size, num_vertices_in, 3).

        Returns:
            Converted vertices, with shape (batch_size, num_vertices_out, 3).
        """

        if self.vertex_converter_csr is None:
            return inp_vertices

        v = inp_vertices.permute(1, 0, 2).reshape(self.body_model_in.num_vertices, -1)
        r = torch.sparse.mm(self.vertex_converter_csr, v)
        return r.reshape(self.body_model_out.num_vertices, -1, 3).permute(1, 0, 2)


def load_vertex_converter_csr(vertex_converter_path):
    scipy_csr = load_pickle(vertex_converter_path)['mtx'].tocsr().astype(np.float32)
    return scipy_csr[:, : scipy_csr.shape[1] // 2]


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def scipy2torch_csr(sparse_matrix):
    return torch.sparse_csr_tensor(
        torch.from_numpy(sparse_matrix.indptr),
        torch.from_numpy(sparse_matrix.indices),
        torch.from_numpy(sparse_matrix.data),
        sparse_matrix.shape,
    )
