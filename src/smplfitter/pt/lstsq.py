import torch
from typing import Optional


def lstsq(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    weights: torch.Tensor,
    l2_regularizer: Optional[torch.Tensor] = None,
    l2_regularizer_rhs: Optional[torch.Tensor] = None,
    shared: bool = False,
) -> torch.Tensor:
    weighted_matrix = weights.unsqueeze(-1) * matrix
    regularized_gramian = weighted_matrix.mT @ matrix
    if l2_regularizer is not None:
        regularized_gramian += torch.diag(l2_regularizer)

    ATb = weighted_matrix.mT @ rhs
    if l2_regularizer_rhs is not None:
        ATb += l2_regularizer_rhs

    if shared:
        regularized_gramian = regularized_gramian.sum(dim=0, keepdim=True)
        ATb = ATb.sum(dim=0, keepdim=True)

    chol = torch.linalg.cholesky(regularized_gramian)
    return torch.cholesky_solve(ATb, chol)


def lstsq_partial_share(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    weights: torch.Tensor,
    l2_regularizer: torch.Tensor,
    l2_regularizer_rhs: Optional[torch.Tensor] = None,
    n_shared: int = 0,
) -> torch.Tensor:
    n_params = matrix.shape[-1]
    n_rhs_outputs = rhs.shape[-1]
    n_indep = n_params - n_shared

    if n_indep == 0:
        result = lstsq(matrix, rhs, weights, l2_regularizer, shared=True)
        return result.expand(matrix.shape[0], -1, -1)

    # Add the regularization equations into the design matrix
    # This way it's simpler to handle all these steps,
    # we only need to implement the unregularized case,
    # and regularization is just adding more rows to the matrix.
    matrix = torch.cat([matrix, batch_eye(n_params, matrix.shape[0], matrix.device)], dim=1)

    if l2_regularizer_rhs is not None:
        rhs =  torch.cat([rhs, l2_regularizer_rhs], dim=1)
    else:
        rhs = torch.nn.functional.pad(rhs, (0, 0, 0, n_params))
    weights = torch.cat(
        [weights, l2_regularizer.unsqueeze(0).expand(matrix.shape[0], -1)], dim=1
    )

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


def batch_eye(
    n_params: int, batch_size: int, device: Optional[torch.device] = None
) -> torch.Tensor:
    return (
        torch.eye(n_params, device=device)
        .reshape(1, n_params, n_params)
        .expand(batch_size, -1, -1)
    )
