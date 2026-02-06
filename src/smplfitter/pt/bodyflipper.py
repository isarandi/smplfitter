from __future__ import annotations
import os
from typing import Optional, TYPE_CHECKING

import numpy as np
import scipy.optimize
import scipy.spatial.distance
import torch
import torch.nn as nn
from .bodyconverter import load_vertex_converter_csr, scipy2torch_csr
from . import bodyfitter as _bodyfitter

if TYPE_CHECKING:
    import smplfitter.pt


class BodyFlipper(nn.Module):
    """
    Horizontally (x axis) flips SMPL-like body model parameters, to mirror the body.

    Parameters:
        body_model: A body model whose parameters are to be transformed.
    """

    def __init__(self, body_model: 'smplfitter.pt.BodyModel'):
        super().__init__()
        self.body_model = body_model
        self.fitter = _bodyfitter.BodyFitter(self.body_model, enable_kid=True)

        res = self.body_model.single()
        self.mirror_csr = nn.Buffer(get_mirror_csr(body_model.num_vertices))
        self.mirror_inds_joints = nn.Buffer(get_mirror_mapping(res['joints']))
        self.mirror_inds = nn.Buffer(get_mirror_mapping(res['vertices']))

    @torch.jit.export
    def flip(
        self,
        pose_rotvecs: torch.Tensor,
        shape_betas: torch.Tensor,
        trans: torch.Tensor,
        kid_factor: Optional[torch.Tensor] = None,
        num_iter: int = 1,
    ) -> dict[str, torch.Tensor]:
        """
        Returns the body model parameters that represent the horizontally flipped 3D human, i.e.,
        flipped along the x axis.

        Internally, this function flips and reorders the vertices and joints then fits
        parameters to the flipped input.

        Parameters:
            pose_rotvecs: Input body part orientations expressed as rotation vectors
                concatenated to shape (batch_size, num_joints*3).
            shape_betas: Input beta coefficients representing body shape.
            trans: Input translation parameters (meters).
            kid_factor: Coefficient for the kid blendshape.
                Default is None, which disables the use of the kid factor.
                See the AGORA paper :footcite:`patel2021agora` for more information.
            num_iter: Number of iterations for fitting.

        Returns:
            Dictionary
                - **pose_rotvecs** (*torch.Tensor*) -- Rotation vectors for the flipped mesh.
                - **shape_betas** (*torch.Tensor*) -- Body shape beta parameters for the flipped \
                    mesh.
                - **trans** (*torch.Tensor*) -- Translation parameters for the flipped mesh.

        """
        inp = self.body_model(pose_rotvecs, shape_betas, trans, kid_factor=kid_factor)
        flipped_vertices = self.flip_vertices(inp['vertices'])

        fit = self.fitter.fit(
            target_vertices=flipped_vertices,
            num_iter=num_iter,
            beta_regularizer=1e-2,
            beta_regularizer2=1e-2,
            final_adjust_rots=True,
            kid_regularizer=1e9 if kid_factor is None else 0.0,
            initial_pose_rotvecs=self.naive_flip_rotvecs(pose_rotvecs),
            initial_shape_betas=shape_betas,
            requested_keys=['pose_rotvecs', 'shape_betas'],
        )
        return dict(
            pose_rotvecs=fit['pose_rotvecs'],
            shape_betas=fit['shape_betas'],
            trans=fit['trans'],
            kid_factor=fit.get('kid_factor'),
        )

    @torch.jit.export
    def flip_vertices(self, inp_vertices: torch.Tensor) -> torch.Tensor:
        """Converts the input vertices to the mirrored version by reordering and flipping
        along the x axis.

        Parameters:
            inp_vertices: A batch of input vertices to be flipped, shaped \
                (batch_size, num_vertices, 3).

        Returns:
            Flipped vertices, shaped (batch_size, num_vertices, 3).

        """

        hflip_multiplier = torch.tensor(
            [-1, 1, 1], dtype=inp_vertices.dtype, device=inp_vertices.device
        )
        v = inp_vertices.permute(1, 0, 2).reshape(self.body_model.num_vertices, -1)
        r = torch.sparse.mm(self.mirror_csr, v)
        return r.reshape(self.body_model.num_vertices, -1, 3).permute(1, 0, 2) * hflip_multiplier

    @torch.jit.export
    def naive_flip_rotvecs(self, pose_rotvecs: torch.Tensor) -> torch.Tensor:
        """Flips each pose rotation vector along the x axis and reorders them to exchange
        left and right body parts. Does not take into account that the body model is slighly
        asymmetric.

        Parameters:
            pose_rotvecs: Input body part orientations expressed as rotation vectors, shaped \
                (batch_size, num_joints*3).

        Returns:
            Flipped pose rotation vectors, shaped (batch_size, num_joints*3).
        """
        hflip_multiplier = torch.tensor(
            [1, -1, -1], dtype=pose_rotvecs.dtype, device=pose_rotvecs.device
        )

        reshaped = pose_rotvecs.reshape(-1, self.body_model.num_joints, 3)
        reshaped_flipped = reshaped[:, self.mirror_inds_joints] * hflip_multiplier
        return reshaped_flipped.reshape(-1, self.body_model.num_joints * 3)


def get_mirror_mapping(points):
    points_np = points.cpu().numpy()
    dist = scipy.spatial.distance.cdist(points_np, points_np * [-1, 1, 1])
    v_inds, mirror_inds = scipy.optimize.linear_sum_assignment(dist)
    return torch.tensor(mirror_inds[np.argsort(v_inds)], dtype=torch.int, device=points.device)


def get_mirror_csr(num_verts):
    DATA_ROOT = os.getenv('DATA_ROOT', '.')
    smplx2mirror = load_mirror_csr(f'{DATA_ROOT}/body_models/smplx/smplx_flip_correspondences.npz')

    if num_verts == 6890:
        smpl2smplx_csr = load_vertex_converter_csr(
            f'{DATA_ROOT}/body_models/smpl2smplx_deftrafo_setup.pkl'
        )
        smplx2smpl_csr = load_vertex_converter_csr(
            f'{DATA_ROOT}/body_models/smplx2smpl_deftrafo_setup.pkl'
        )
        smpl2mirror = smplx2smpl_csr @ smplx2mirror @ smpl2smplx_csr
        return scipy2torch_csr(smpl2mirror)
    elif num_verts == 10475:
        return scipy2torch_csr(smplx2mirror)
    else:
        raise ValueError(f'Unsupported number of vertices: {num_verts}')


def load_mirror_csr(path):
    m = np.load(path)
    faces = m['closest_faces']
    barycentrics = m['bc']
    n_verts = barycentrics.shape[0]
    n_faces = faces.shape[0]
    data = barycentrics.flatten()
    row = np.repeat(np.arange(n_faces), 3)
    col = faces.flatten()
    coo_mat = scipy.sparse.coo_matrix((data, (row, col)), shape=(n_faces, n_verts))
    return coo_mat.tocsr().astype(np.float32)
