from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import jax.scipy.linalg


def lstsq(
    matrix: jnp.ndarray,
    rhs: jnp.ndarray,
    weights: jnp.ndarray,
    l2_regularizer: Optional[jnp.ndarray] = None,
    l2_regularizer_rhs: Optional[jnp.ndarray] = None,
    shared: bool = False,
) -> jnp.ndarray:
    """Weighted least squares with L2 regularization.

    Args:
        matrix: Design matrix, shape (batch, n_points, n_params)
        rhs: Right-hand side, shape (batch, n_points, n_outputs)
        weights: Per-point weights, shape (batch, n_points)
        l2_regularizer: L2 regularization coefficients, shape (n_params,)
        l2_regularizer_rhs: Regularization target, shape (batch, n_params, n_outputs)
        shared: If True, solve a single shared problem across the batch

    Returns:
        Solution, shape (batch, n_params, n_outputs) or (1, n_params, n_outputs) if shared
    """
    weighted_matrix = weights[..., None] * matrix
    regularized_gramian = jnp.swapaxes(weighted_matrix, -2, -1) @ matrix

    if l2_regularizer is not None:
        # Add regularization to diagonal
        diag_indices = jnp.arange(regularized_gramian.shape[-1])
        regularized_gramian = regularized_gramian.at[..., diag_indices, diag_indices].add(
            l2_regularizer
        )

    ATb = jnp.swapaxes(weighted_matrix, -2, -1) @ rhs
    if l2_regularizer_rhs is not None:
        ATb = ATb + l2_regularizer_rhs

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
    l2_regularizer_rhs: Optional[jnp.ndarray] = None,
    n_shared: int = 0,
) -> jnp.ndarray:
    """Weighted least squares with partial parameter sharing.

    Some parameters are shared across the batch (e.g., shape betas),
    while others are independent (e.g., scale correction per sample).

    Args:
        matrix: Design matrix, shape (batch, n_points, n_params)
        rhs: Right-hand side, shape (batch, n_points, n_outputs)
        weights: Per-point weights, shape (batch, n_points)
        l2_regularizer: L2 regularization coefficients, shape (n_params,)
        l2_regularizer_rhs: Regularization target, shape (batch, n_params, n_outputs)
        n_shared: Number of shared parameters (first n_shared params are shared)

    Returns:
        Solution, shape (batch, n_params, n_outputs)
    """
    batch_size = matrix.shape[0]
    n_params = matrix.shape[-1]
    n_rhs_outputs = rhs.shape[-1]
    n_indep = n_params - n_shared

    if n_indep == 0:
        result = lstsq(matrix, rhs, weights, l2_regularizer, l2_regularizer_rhs, shared=True)
        return jnp.broadcast_to(result, (batch_size, n_params, n_rhs_outputs))

    # Add the regularization equations into the design matrix
    eye = jnp.eye(n_params)
    eye_expanded = jnp.broadcast_to(eye, (batch_size, n_params, n_params))
    matrix = jnp.concatenate([matrix, eye_expanded], axis=1)

    if l2_regularizer_rhs is not None:
        rhs = jnp.concatenate([rhs, l2_regularizer_rhs], axis=1)
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
