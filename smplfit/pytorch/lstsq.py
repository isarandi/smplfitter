import torch


def lstsq(matrix, rhs, weights, l2_regularizer):
    weighted_matrix = weights.unsqueeze(-1) * matrix
    regularized_gramian = weighted_matrix.mT @ matrix + torch.diag(l2_regularizer)
    chol = torch.linalg.cholesky(regularized_gramian)
    return torch.cholesky_solve(weighted_matrix.mT @ rhs, chol)


def lstsq_partial_share(matrix, rhs, weights, l2_regularizer, n_shared=0):
    n_params = matrix.shape[-1]
    n_rhs_outputs = rhs.shape[-1]
    n_indep = n_params - n_shared

    # Add the regularization equations into the design matrix
    # This way it's simpler to handle all these steps,
    # we only need to implement the unregularized case,
    # and regularization is just adding more rows to the matrix.
    matrix = torch.cat([matrix, batch_eye(n_params, matrix.shape[0])], dim=1)
    rhs = torch.nn.functional.pad(rhs, (0, 0, 0, n_params))
    weights = torch.cat([weights, l2_regularizer.unsqueeze(0).expand(matrix.shape[0], -1)], dim=1)

    # Split the shared and independent parts of the matrices
    matrix_shared, matrix_indep = torch.split(matrix, [n_shared, n_indep], dim=-1)

    # First solve for the independent params only (~shared params are forced to 0)
    # Also regress the shared columns on the independent columns
    # Since we regress the rhs from the independent columns, any part of the shared
    # columns that are linearly predictable from the indep columns needs to be removed,
    # so we can solve for the shared params while considering only the information that's
    # unaccounted for so far.
    solve_indep_fn = get_lstsq_solver_fn(matrix_indep, weights)
    coeff_indep2shared, coeff_indep2rhs = torch.split(
        solve_indep_fn(torch.cat([matrix_shared, rhs], dim=-1).mT),
        [n_shared, n_rhs_outputs], dim=-1)

    # Now solve for the shared params using the residuals
    solve_shared_fn = get_lstsq_solver_fn(matrix_shared - matrix_indep @ coeff_indep2shared,
                                          weights, shared=True)
    coeff_shared2rhs = solve_shared_fn((rhs - matrix_indep @ coeff_indep2rhs).mT)

    # Finally, update the estimate for the independent params, reusing the Cholesky decomposition.
    coeff_indep2rhs = solve_indep_fn((rhs - matrix_shared @ coeff_shared2rhs).mT)

    # Repeat the shared coefficients for each sample and concatenate them with the independent ones
    coeff_shared2rhs = coeff_shared2rhs.expand(matrix.shape[0], -1, -1)
    return torch.cat([coeff_shared2rhs, coeff_indep2rhs], dim=1)


def get_lstsq_solver_fn(matrix, weights, shared=False):
    # The solver function saves the Cholesky decomposition of the Gramian, so it can be reused.
    weights = weights.unsqueeze(-1)
    w_matrix = matrix * weights
    gramian = w_matrix.mT @ matrix
    if shared:
        gramian = gramian.sum(dim=0, keepdim=True)

    chol = torch.linalg.cholesky(gramian)

    def solver(rhs):
        ATb = w_matrix.mT @ rhs
        if shared:
            ATb = ATb.sum(dim=0, keepdim=True)

        return torch.cholesky_solve(ATb, chol)

    return solver


def batch_eye(n_params, batch_size):
    return torch.eye(n_params).reshape(1, n_params, n_params).expand(batch_size, -1, -1)
