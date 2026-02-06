"""Body model parameter converter (TensorFlow implementation)."""

from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING, Optional

import numpy as np
import scipy.sparse
import tensorflow as tf

from . import bodyfitter as _bodyfitter

if TYPE_CHECKING:
    import smplfitter.tf


class BodyConverter:
    """
    Converts between different SMPL-family body model parameters.

    Parameters:
        body_model_in: Input body model to convert from.
        body_model_out: Output body model to convert to.
    """

    def __init__(
        self,
        body_model_in: 'smplfitter.tf.BodyModel',
        body_model_out: 'smplfitter.tf.BodyModel',
    ):
        self.body_model_in = body_model_in
        self.body_model_out = body_model_out
        self.fitter = _bodyfitter.BodyFitter(self.body_model_out, enable_kid=True)

        DATA_ROOT = os.getenv('DATA_ROOT', '.')
        if self.body_model_in.num_vertices == 6890 and self.body_model_out.num_vertices == 10475:
            csr_path = f'{DATA_ROOT}/body_models/smpl2smplx_deftrafo_setup.pkl'
        elif (
            self.body_model_in.num_vertices == 10475 and self.body_model_out.num_vertices == 6890
        ):
            csr_path = f'{DATA_ROOT}/body_models/smplx2smpl_deftrafo_setup.pkl'
        else:
            csr_path = None

        self.vertex_converter_sparse: Optional[tf.SparseTensor]
        if csr_path is not None:
            scipy_csr = load_vertex_converter_csr(csr_path)
            self.vertex_converter_sparse = scipy2tf_sparse(scipy_csr)
        else:
            self.vertex_converter_sparse = None

    def convert(
        self,
        pose_rotvecs: tf.Tensor,
        shape_betas: tf.Tensor,
        trans: tf.Tensor,
        kid_factor: Optional[tf.Tensor] = None,
        known_output_pose_rotvecs: Optional[tf.Tensor] = None,
        known_output_shape_betas: Optional[tf.Tensor] = None,
        known_output_kid_factor: Optional[tf.Tensor] = None,
        num_iter: int = 1,
    ) -> dict[str, tf.Tensor]:
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
            Dictionary containing the conversion results:
                - **pose_rotvecs** -- Converted body part orientations.
                - **shape_betas** -- Converted beta coefficients.
                - **trans** -- Converted translation parameters.
                - **kid_factor** -- Kid factor (if input had kid_factor).
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

    def convert_vertices(self, inp_vertices: tf.Tensor) -> tf.Tensor:
        """
        Converts body mesh vertices from the input model to the output body model's topology
        using barycentric coordinates.

        Parameters:
            inp_vertices: Input vertices to convert, shape (batch_size, num_vertices_in, 3).

        Returns:
            Converted vertices, shape (batch_size, num_vertices_out, 3).
        """
        if self.vertex_converter_sparse is None:
            return inp_vertices

        batch_size = tf.shape(inp_vertices)[0]
        # Reshape to (num_vertices_in, batch_size * 3) for sparse matmul
        v = tf.reshape(
            tf.transpose(inp_vertices, [1, 0, 2]), [self.body_model_in.num_vertices, -1]
        )
        r = tf.sparse.sparse_dense_matmul(self.vertex_converter_sparse, v)
        return tf.transpose(
            tf.reshape(r, [self.body_model_out.num_vertices, batch_size, 3]), [1, 0, 2]
        )


def load_vertex_converter_csr(vertex_converter_path: str) -> scipy.sparse.csr_matrix:
    scipy_csr = load_pickle(vertex_converter_path)['mtx'].tocsr().astype(np.float32)
    return scipy_csr[:, : scipy_csr.shape[1] // 2]


def load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def scipy2tf_sparse(sparse_matrix: scipy.sparse.csr_matrix) -> tf.SparseTensor:
    """Convert scipy sparse matrix to TensorFlow SparseTensor."""
    coo = sparse_matrix.tocoo()
    indices = np.column_stack([coo.row, coo.col])
    return tf.SparseTensor(
        indices=indices, values=coo.data.astype(np.float32), dense_shape=coo.shape
    )
