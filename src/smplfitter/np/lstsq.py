import numpy as np
import scipy
from smplfitter.np.util import matmul_transp_a


def lstsq(matrix, rhs, weights, l2_regularizer=None, shared=False):
    weighted_matrix = weights[..., np.newaxis] * matrix
    regularized_gramian = matmul_transp_a(weighted_matrix, matrix)
    if l2_regularizer is not None:
        regularized_gramian += np.diag(l2_regularizer)

    ATb = matmul_transp_a(weighted_matrix, rhs)

    if shared:
        regularized_gramian = np.sum(regularized_gramian, axis=0, keepdims=True)
        ATb = np.sum(ATb, axis=0, keepdims=True)

    chol = np.linalg.cholesky(regularized_gramian)
    return cholesky_solve(chol, ATb)


def lstsq_partial_share(matrix, rhs, weights, l2_regularizer, n_shared=0):
    n_params = matrix.shape[-1]
    n_rhs_outputs = rhs.shape[-1]
    n_indep = n_params - n_shared

    matrix = np.concatenate(
        [matrix, np.eye(n_params)[np.newaxis, ...].repeat(matrix.shape[0], axis=0)], axis=1
    )
    rhs = np.pad(rhs, ((0, 0), (0, n_params), (0, 0)))
    weights = np.concatenate(
        [weights, np.repeat(l2_regularizer[np.newaxis], matrix.shape[0], axis=0)], axis=1
    )
    matrix_shared, matrix_indep = np.split(matrix, [n_shared], axis=-1)

    coeff_indep2shared, coeff_indep2rhs = np.split(
        lstsq(matrix_indep, np.concatenate([matrix_shared, rhs], axis=-1), weights),
        [n_shared],
        axis=-1,
    )

    coeff_shared2rhs = lstsq(
        matrix_shared - matrix_indep @ coeff_indep2shared,
        rhs - matrix_indep @ coeff_indep2rhs,
        weights,
        shared=True,
    )

    coeff_indep2rhs = coeff_indep2rhs - coeff_indep2shared @ coeff_shared2rhs
    coeff_shared2rhs = np.repeat(coeff_shared2rhs, matrix.shape[0], axis=0)
    return np.concatenate([coeff_shared2rhs, coeff_indep2rhs], axis=1)


def cholesky_solve(chol, rhs):
    y = solve_triangular(chol, rhs, transpose=False)
    return solve_triangular(chol, y, transpose=True)


def solve_triangular(a, b, transpose=False):
    return np.stack(
        [
            scipy.linalg.solve_triangular(a_, b_, lower=True, trans=transpose)
            for a_, b_ in zip(a, b)
        ]
    )
