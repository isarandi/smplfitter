from __future__ import annotations

from typing import Optional
import torch


def lstsq(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    weights: torch.Tensor,
    l2_regularizer: Optional[torch.Tensor] = None,
    l2_regularizer_reference: Optional[torch.Tensor] = None,
    shared: bool = False,
) -> torch.Tensor:
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
    weighted_matrix = weights.unsqueeze(-1) * matrix
    regularized_gramian = weighted_matrix.mT @ matrix
    ATb = weighted_matrix.mT @ rhs
    if l2_regularizer is not None:
        regularized_gramian = regularized_gramian + torch.diag(l2_regularizer)
        if l2_regularizer_reference is not None:
            ATb = ATb + l2_regularizer.unsqueeze(-1) * l2_regularizer_reference

    if shared:
        regularized_gramian = regularized_gramian.sum(dim=0, keepdim=True)
        ATb = ATb.sum(dim=0, keepdim=True)

    chol, _ = torch.linalg.cholesky_ex(regularized_gramian)
    return torch.cholesky_solve(ATb, chol)


def lstsq_partial_share(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    weights: torch.Tensor,
    l2_regularizer: torch.Tensor,
    l2_regularizer_reference: Optional[torch.Tensor] = None,
    n_shared: int = 0,
) -> torch.Tensor:
    """Weighted least squares where the first ``n_shared`` parameters are shared across the batch.

    Same objective as :func:`lstsq`, but the first ``n_shared`` entries of x are a single
    vector common to all batch elements while the remaining ones are per-element.

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
    n_params = matrix.shape[-1]
    n_rhs_outputs = rhs.shape[-1]
    n_indep = n_params - n_shared

    if n_indep == 0:
        result = lstsq(matrix, rhs, weights, l2_regularizer, l2_regularizer_reference, shared=True)
        return result.expand(matrix.shape[0], -1, -1)

    # Append the penalty as extra equations: row i states x_i = r_i with weight lambda_i.
    # The steps below then only need to handle the unregularized case.
    matrix = torch.cat(
        [matrix, torch.eye(n_params, device=matrix.device).expand(matrix.shape[0], -1, -1)], dim=1
    )
    if l2_regularizer_reference is not None:
        rhs = torch.cat([rhs, l2_regularizer_reference], dim=1)
    else:
        rhs = torch.nn.functional.pad(rhs, (0, 0, 0, n_params))
    weights = torch.cat([weights, l2_regularizer.expand(matrix.shape[0], -1)], dim=1)

    # Split the shared and independent parts of the matrices
    matrix_shared, matrix_indep = torch.split(matrix, [n_shared, n_indep], dim=-1)

    # First solve for the independent params only (~shared params are forced to 0)
    # Also regress the shared columns on the independent columns
    # Since we regress the rhs from the independent columns, any part of the shared
    # columns that are linearly predictable from the indep columns needs to be removed,
    # so we can solve for the shared params while considering only the information that's
    # unaccounted for so far.
    coeff_indep2shared, coeff_indep2rhs = torch.split(
        lstsq(matrix_indep, torch.cat([matrix_shared, rhs], dim=-1), weights),
        [n_shared, n_rhs_outputs],
        dim=-1,
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
    coeff_shared2rhs = coeff_shared2rhs.expand(matrix.shape[0], -1, -1)
    return torch.cat([coeff_shared2rhs, coeff_indep2rhs], dim=1)
