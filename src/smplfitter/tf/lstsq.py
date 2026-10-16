from __future__ import annotations

from typing import Optional

import tensorflow as tf


def lstsq(
    matrix: tf.Tensor,
    rhs: tf.Tensor,
    weights: tf.Tensor,
    l2_regularizer: Optional[tf.Tensor] = None,
    l2_regularizer_reference: Optional[tf.Tensor] = None,
    shared: bool = False,
) -> tf.Tensor:
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
    weighted_matrix = weights[..., tf.newaxis] * matrix
    regularized_gramian = tf.linalg.matmul(weighted_matrix, matrix, transpose_a=True)
    ATb = tf.linalg.matmul(weighted_matrix, rhs, transpose_a=True)
    if l2_regularizer is not None:
        regularized_gramian += tf.linalg.diag(l2_regularizer)
        if l2_regularizer_reference is not None:
            ATb += l2_regularizer[:, tf.newaxis] * l2_regularizer_reference

    if shared:
        regularized_gramian = tf.reduce_sum(regularized_gramian, axis=0, keepdims=True)
        ATb = tf.reduce_sum(ATb, axis=0, keepdims=True)

    chol = tf.linalg.cholesky(regularized_gramian)
    return tf.linalg.cholesky_solve(chol, ATb)


def lstsq_partial_share(
    matrix: tf.Tensor,
    rhs: tf.Tensor,
    weights: tf.Tensor,
    l2_regularizer: tf.Tensor,
    l2_regularizer_reference: Optional[tf.Tensor] = None,
    n_shared: int = 0,
) -> tf.Tensor:
    """Weighted least squares where the first ``n_shared`` parameters are shared across the batch.

    Same objective as :func:`lstsq`, but the first ``n_shared`` entries of x are a single
    vector common to all batch elements while the remaining ones are per-element.
    """
    n_params = tf.shape(matrix)[-1]
    n_rhs_outputs = tf.shape(rhs)[-1]
    n_indep = n_params - n_shared

    if n_indep == 0:
        result = lstsq(matrix, rhs, weights, l2_regularizer, l2_regularizer_reference, shared=True)
        return tf.repeat(result, tf.shape(matrix)[0], axis=0)

    # Append the penalty as extra equations: row i states x_i = r_i with weight lambda_i.
    # The steps below then only need to handle the unregularized case.
    matrix = tf.concat([matrix, tf.eye(n_params, batch_shape=[tf.shape(matrix)[0]])], axis=1)

    if l2_regularizer_reference is not None:
        rhs = tf.concat([rhs, l2_regularizer_reference], axis=1)
    else:
        rhs = tf.pad(rhs, [[0, 0], [0, n_params], [0, 0]])
    weights = tf.concat(
        [weights, tf.repeat(l2_regularizer[tf.newaxis], tf.shape(matrix)[0], axis=0)], axis=1
    )

    # Split the shared and independent parts of the matrices
    matrix_shared, matrix_indep = tf.split(matrix, [n_shared, n_indep], axis=-1)

    # First solve for the independent params only (~shared params are forced to 0)
    # Also regress the shared columns on the independent columns
    # Since we regress the rhs from the independent columns, any part of the shared
    # columns that are linearly predictable from the indep columns needs to be removed,
    # so we can solve for the shared params while considering only the information that's
    # unaccounted for so far.
    coeff_indep2shared, coeff_indep2rhs = tf.split(
        lstsq(matrix_indep, tf.concat([matrix_shared, rhs], axis=-1), weights),
        [n_shared, n_rhs_outputs],
        axis=-1,
    )

    # Now solve for the shared params using the residuals
    coeff_shared2rhs = lstsq(
        matrix_shared - matrix_indep @ coeff_indep2shared,
        rhs - matrix_indep @ coeff_indep2rhs,
        weights,
        shared=True,
    )

    # Finally, update the estimate for the independent params
    coeff_indep2rhs = coeff_indep2rhs - coeff_indep2shared @ coeff_shared2rhs

    # Repeat the shared coefficients for each sample and concatenate them with the independent ones
    coeff_shared2rhs = tf.repeat(coeff_shared2rhs, tf.shape(matrix)[0], axis=0)
    return tf.concat([coeff_shared2rhs, coeff_indep2rhs], axis=1)
