from __future__ import annotations
import numpy as np
import scipy
from .util import matmul_transp_a


def lstsq(matrix, rhs, weights, l2_regularizer=None, l2_regularizer_reference=None, shared=False):
    """Weighted least squares via the normal equations, with a diagonal L2 penalty.

    Minimizes ``sum_n w_n |A_n x - b_n|^2 + sum_i lambda_i (x_i - r_i)^2`` per batch
    element, or the sum of these over the batch if ``shared`` is True.

    Args:
        matrix: Design matrix A, shape (batch, n_points, n_params).
        rhs: Right-hand side b, shape (batch, n_points, n_outputs).
        weights: Per-point weights w, shape (batch, n_points).
        l2_regularizer: Penalty weights lambda, shape (n_params,).
        l2_regularizer_reference: Penalty reference values r, shape
            (batch, n_params, n_outputs). Zero if None. Ignored without ``l2_regularizer``.
        shared: If True, solve one system for the whole batch.

    Returns:
        Solution x, shape (batch, n_params, n_outputs), or (1, n_params, n_outputs) if shared.
    """
    weighted_matrix = weights[..., np.newaxis] * matrix
    regularized_gramian = matmul_transp_a(weighted_matrix, matrix)
    ATb = matmul_transp_a(weighted_matrix, rhs)
    if l2_regularizer is not None:
        regularized_gramian += np.diag(l2_regularizer)
        if l2_regularizer_reference is not None:
            ATb = ATb + l2_regularizer[:, np.newaxis] * l2_regularizer_reference

    if shared:
        regularized_gramian = np.sum(regularized_gramian, axis=0, keepdims=True)
        ATb = np.sum(ATb, axis=0, keepdims=True)

    return _solve_each_or_nan(regularized_gramian, ATb)


def lstsq_partial_share(
    matrix, rhs, weights, l2_regularizer, l2_regularizer_reference=None, n_shared=0
):
    """Weighted least squares where the first ``n_shared`` parameters are shared across the batch.

    Same objective as :func:`lstsq`, but the first ``n_shared`` entries of x are a single
    vector common to all batch elements while the remaining ones are per-element.
    """
    n_params = matrix.shape[-1]
    n_indep = n_params - n_shared

    if n_indep == 0:
        result = lstsq(matrix, rhs, weights, l2_regularizer, l2_regularizer_reference, shared=True)
        return np.repeat(result, matrix.shape[0], axis=0)

    # Append the penalty as extra equations: row i states x_i = r_i with weight lambda_i.
    matrix = np.concatenate(
        [matrix, np.eye(n_params)[np.newaxis, ...].repeat(matrix.shape[0], axis=0)], axis=1
    )
    if l2_regularizer_reference is not None:
        rhs = np.concatenate([rhs, l2_regularizer_reference], axis=1)
    else:
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


def _solve_each_or_nan(gramian, rhs):
    """Solves ``gramian @ x = rhs`` per batch element. An unsolvable element gives NaN.

    One unsolvable element stops the batched calls, so only then the elements run one by one.
    """
    try:
        return cholesky_solve(np.linalg.cholesky(gramian), rhs)
    except (np.linalg.LinAlgError, ValueError):
        pass

    solutions = []
    for gramian_element, rhs_element in zip(gramian, rhs):
        try:
            chol = np.linalg.cholesky(gramian_element[np.newaxis])
            solutions.append(cholesky_solve(chol, rhs_element[np.newaxis])[0])
        except (np.linalg.LinAlgError, ValueError):
            solutions.append(np.full_like(rhs_element, np.nan))
    return np.stack(solutions)


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
