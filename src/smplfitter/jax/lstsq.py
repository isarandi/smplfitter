from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import jax.scipy.linalg


def lstsq(
    matrix: jnp.ndarray,
    rhs: jnp.ndarray,
    weights: jnp.ndarray,
    l2_regularizer: Optional[jnp.ndarray] = None,
    l2_regularizer_reference: Optional[jnp.ndarray] = None,
    shared: bool = False,
) -> jnp.ndarray:
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
    weighted_matrix = weights[..., None] * matrix
    regularized_gramian = jnp.swapaxes(weighted_matrix, -2, -1) @ matrix
    ATb = jnp.swapaxes(weighted_matrix, -2, -1) @ rhs

    if l2_regularizer is not None:
        diag_indices = jnp.arange(regularized_gramian.shape[-1])
        regularized_gramian = regularized_gramian.at[..., diag_indices, diag_indices].add(
            l2_regularizer
        )
        if l2_regularizer_reference is not None:
            ATb = ATb + l2_regularizer[:, None] * l2_regularizer_reference

    if shared:
        regularized_gramian = jnp.sum(regularized_gramian, axis=0, keepdims=True)
        ATb = jnp.sum(ATb, axis=0, keepdims=True)

    chol = jnp.linalg.cholesky(regularized_gramian)
    return jax.scipy.linalg.cho_solve((chol, True), ATb)


def lstsq_partial_share(
    matrix: jnp.ndarray,
    rhs: jnp.ndarray,
    weights: jnp.ndarray,
    l2_regularizer: jnp.ndarray,
    l2_regularizer_reference: Optional[jnp.ndarray] = None,
    n_shared: int = 0,
) -> jnp.ndarray:
    """Weighted least squares where the first ``n_shared`` parameters are shared across the batch.

    Same objective as :func:`lstsq`, but the first ``n_shared`` entries of x are a single
    vector common to all batch elements (e.g. shape betas) while the remaining ones are
    per-element (e.g. a scale correction).

    Args:
        matrix: Design matrix A, shape (batch, n_points, n_params).
        rhs: Right-hand side b, shape (batch, n_points, n_outputs).
        weights: Per-point weights w, shape (batch, n_points).
        l2_regularizer: Penalty weights lambda, shape (n_params,).
        l2_regularizer_reference: Penalty reference values r, shape
            (batch, n_params, n_outputs). Zero if None.
        n_shared: Number of leading parameters shared across the batch.

    Returns:
        Solution x, shape (batch, n_params, n_outputs), with the shared part repeated.
    """
    batch_size = matrix.shape[0]
    n_params = matrix.shape[-1]
    n_rhs_outputs = rhs.shape[-1]
    n_indep = n_params - n_shared

    if n_indep == 0:
        result = lstsq(matrix, rhs, weights, l2_regularizer, l2_regularizer_reference, shared=True)
        return jnp.broadcast_to(result, (batch_size, n_params, n_rhs_outputs))

    # Append the penalty as extra equations: row i states x_i = r_i with weight lambda_i.
    eye = jnp.eye(n_params)
    eye_expanded = jnp.broadcast_to(eye, (batch_size, n_params, n_params))
    matrix = jnp.concatenate([matrix, eye_expanded], axis=1)

    if l2_regularizer_reference is not None:
        rhs = jnp.concatenate([rhs, l2_regularizer_reference], axis=1)
    else:
        rhs = jnp.pad(rhs, ((0, 0), (0, n_params), (0, 0)))

    l2_reg_expanded = jnp.broadcast_to(l2_regularizer, (batch_size, n_params))
    weights = jnp.concatenate([weights, l2_reg_expanded], axis=1)

    # Split the shared and independent parts of the matrices
    matrix_shared = matrix[..., :n_shared]
    matrix_indep = matrix[..., n_shared:]

    # First solve for the independent params only (~shared params are forced to 0)
    # Also regress the shared columns on the independent columns
    combined_rhs = jnp.concatenate([matrix_shared, rhs], axis=-1)
    combined_solution = lstsq(matrix_indep, combined_rhs, weights)
    coeff_indep2shared = combined_solution[..., :n_shared]
    coeff_indep2rhs = combined_solution[..., n_shared:]

    # Now solve for the shared params using the residuals
    residual_matrix = matrix_shared - matrix_indep @ coeff_indep2shared
    residual_rhs = rhs - matrix_indep @ coeff_indep2rhs
    coeff_shared2rhs = lstsq(residual_matrix, residual_rhs, weights, shared=True)

    # Finally, update the estimate for the independent params
    coeff_indep2rhs = coeff_indep2rhs - coeff_indep2shared @ coeff_shared2rhs

    # Repeat the shared coefficients for each sample and concatenate
    coeff_shared2rhs = jnp.broadcast_to(coeff_shared2rhs, (batch_size, n_shared, n_rhs_outputs))
    return jnp.concatenate([coeff_shared2rhs, coeff_indep2rhs], axis=1)
