from __future__ import annotations

import numpy as np
import numba


@numba.njit(error_model='numpy', cache=True)
def lstsq(matrix, rhs, weights, l2_regularizer=None, l2_regularizer_rhs=None, shared=False):
    """Weighted least squares with L2 regularization.

    Args:
        matrix: (batch, n_points, n_params)
        rhs: (batch, n_points, n_outputs)
        weights: (batch, n_points)
        l2_regularizer: (n_params,) diagonal regularization
        l2_regularizer_rhs: (batch, n_params, n_outputs) regularization reference
        shared: If True, solve shared parameters across batch

    Returns:
        (batch, n_params, n_outputs) solution
    """
    batch_size = matrix.shape[0]
    n_params = matrix.shape[2]
    n_outputs = rhs.shape[2]

    # Use np.dot for fast matrix operations
    regularized_gramian = np.zeros((batch_size, n_params, n_params), dtype=np.float32)
    ATb = np.zeros((batch_size, n_params, n_outputs), dtype=np.float32)

    for b in range(batch_size):
        # weighted_matrix = diag(weights) @ matrix, then gramian = matrix.T @ weighted_matrix
        # Equivalent to: (weights[:, None] * matrix).T @ matrix
        w = weights[b]
        m = matrix[b]
        r = rhs[b]

        # Compute weighted matrix (n_points, n_params)
        weighted_m = np.empty_like(m)
        for p in range(m.shape[0]):
            for i in range(n_params):
                weighted_m[p, i] = w[p] * m[p, i]

        # Gramian: weighted_m.T @ m using np.dot
        regularized_gramian[b] = np.dot(weighted_m.T, m)

        # ATb: weighted_m.T @ rhs using np.dot
        ATb[b] = np.dot(weighted_m.T, r)

    # Add L2 regularization to diagonal
    if l2_regularizer is not None:
        for b in range(batch_size):
            for i in range(n_params):
                regularized_gramian[b, i, i] += l2_regularizer[i]

    if l2_regularizer_rhs is not None:
        for b in range(batch_size):
            for i in range(n_params):
                for o in range(n_outputs):
                    ATb[b, i, o] += l2_regularizer_rhs[b, i, o]

    if shared:
        # Sum across batch
        gramian_sum = np.zeros((n_params, n_params), dtype=np.float32)
        ATb_sum = np.zeros((n_params, n_outputs), dtype=np.float32)
        for b in range(batch_size):
            for i in range(n_params):
                for j in range(n_params):
                    gramian_sum[i, j] += regularized_gramian[b, i, j]
                for o in range(n_outputs):
                    ATb_sum[i, o] += ATb[b, i, o]

        # Solve single system
        result_single = _solve_cholesky(gramian_sum, ATb_sum)

        # Broadcast to batch
        result = np.empty((batch_size, n_params, n_outputs), dtype=np.float32)
        for b in range(batch_size):
            for i in range(n_params):
                for o in range(n_outputs):
                    result[b, i, o] = result_single[i, o]
        return result
    else:
        # Solve per batch element
        result = np.empty((batch_size, n_params, n_outputs), dtype=np.float32)
        for b in range(batch_size):
            result[b] = _solve_cholesky(regularized_gramian[b], ATb[b])
        return result


@numba.njit(error_model='numpy', cache=True)
def _solve_cholesky(A, b):
    """Solve Ax = b using Cholesky decomposition.

    Args:
        A: (n, n) positive definite matrix
        b: (n, m) right-hand side

    Returns:
        (n, m) solution
    """
    # Hand-rolled is faster for small matrices, LAPACK wins for n > 16
    if A.shape[0] <= 16:
        return _solve_cholesky_small(A, b)
    else:
        return _solve_cholesky_lapack(A, b)


@numba.njit(error_model='numpy', cache=True)
def _solve_cholesky_small(A, b):
    """Hand-rolled Cholesky solver, faster for small matrices (n <= 16)."""
    n = A.shape[0]
    m = b.shape[1]

    # Cholesky decomposition: A = L @ L.T
    L = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1):
            s = np.float32(0.0)
            for k in range(j):
                s += L[i, k] * L[j, k]
            if i == j:
                val = A[i, i] - s
                if val <= 0:
                    val = np.float32(1e-6)  # Regularize if not positive definite
                L[i, j] = np.sqrt(val)
            else:
                if L[j, j] > 0:
                    L[i, j] = (A[i, j] - s) / L[j, j]

    # Forward substitution: L @ y = b
    y = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        for o in range(m):
            val = b[i, o]
            for j in range(i):
                val -= L[i, j] * y[j, o]
            if L[i, i] > 0:
                y[i, o] = val / L[i, i]

    # Backward substitution: L.T @ x = y
    x = np.zeros((n, m), dtype=np.float32)
    for i in range(n - 1, -1, -1):
        for o in range(m):
            val = y[i, o]
            for j in range(i + 1, n):
                val -= L[j, i] * x[j, o]
            if L[i, i] > 0:
                x[i, o] = val / L[i, i]

    return x


@numba.njit(error_model='numpy', cache=True)
def _solve_cholesky_lapack(A, b):
    """LAPACK-based Cholesky solver, faster for larger matrices (n > 16)."""
    n = A.shape[0]
    m = b.shape[1]

    try:
        L = np.linalg.cholesky(A)
    except Exception:
        # Fallback to hand-rolled with regularization for non-positive-definite
        return _solve_cholesky_small(A, b)

    # Forward substitution: L @ y = b
    y = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        for o in range(m):
            val = b[i, o]
            for j in range(i):
                val -= L[i, j] * y[j, o]
            y[i, o] = val / L[i, i]

    # Backward substitution: L.T @ x = y
    x = np.zeros((n, m), dtype=np.float32)
    for i in range(n - 1, -1, -1):
        for o in range(m):
            val = y[i, o]
            for j in range(i + 1, n):
                val -= L[j, i] * x[j, o]
            x[i, o] = val / L[i, i]

    return x


@numba.njit(error_model='numpy', cache=True)
def lstsq_partial_share(matrix, rhs, weights, l2_regularizer, l2_regularizer_rhs=None, n_shared=0):
    """Least squares with some parameters shared across batch.

    Args:
        matrix: (batch, n_points, n_params)
        rhs: (batch, n_points, n_outputs)
        weights: (batch, n_points)
        l2_regularizer: (n_params,) diagonal regularization
        l2_regularizer_rhs: (batch, n_params, n_outputs) regularization reference
        n_shared: Number of parameters shared across batch (first n_shared params)

    Returns:
        (batch, n_params, n_outputs) solution
    """
    batch_size = matrix.shape[0]
    n_points = matrix.shape[1]
    n_params = matrix.shape[2]
    n_rhs_outputs = rhs.shape[2]
    n_indep = n_params - n_shared

    if n_indep == 0:
        result = lstsq(matrix, rhs, weights, l2_regularizer, l2_regularizer_rhs, shared=True)
        return result

    # Add regularization equations to design matrix
    # New matrix shape: (batch, n_points + n_params, n_params)
    new_n_points = n_points + n_params
    new_matrix = np.zeros((batch_size, new_n_points, n_params), dtype=np.float32)
    for b in range(batch_size):
        for p in range(n_points):
            for i in range(n_params):
                new_matrix[b, p, i] = matrix[b, p, i]
        # Add identity for regularization
        for i in range(n_params):
            new_matrix[b, n_points + i, i] = np.float32(1.0)

    # New rhs: (batch, n_points + n_params, n_outputs)
    new_rhs = np.zeros((batch_size, new_n_points, n_rhs_outputs), dtype=np.float32)
    for b in range(batch_size):
        for p in range(n_points):
            for o in range(n_rhs_outputs):
                new_rhs[b, p, o] = rhs[b, p, o]
        if l2_regularizer_rhs is not None:
            for i in range(n_params):
                for o in range(n_rhs_outputs):
                    new_rhs[b, n_points + i, o] = l2_regularizer_rhs[b, i, o]

    # New weights: (batch, n_points + n_params)
    new_weights = np.zeros((batch_size, new_n_points), dtype=np.float32)
    for b in range(batch_size):
        for p in range(n_points):
            new_weights[b, p] = weights[b, p]
        for i in range(n_params):
            new_weights[b, n_points + i] = l2_regularizer[i]

    # Split shared and independent parts
    matrix_shared = np.empty((batch_size, new_n_points, n_shared), dtype=np.float32)
    matrix_indep = np.empty((batch_size, new_n_points, n_indep), dtype=np.float32)
    for b in range(batch_size):
        for p in range(new_n_points):
            for i in range(n_shared):
                matrix_shared[b, p, i] = new_matrix[b, p, i]
            for i in range(n_indep):
                matrix_indep[b, p, i] = new_matrix[b, p, n_shared + i]

    # First solve: regress both matrix_shared and rhs from matrix_indep
    combined_rhs = np.empty((batch_size, new_n_points, n_shared + n_rhs_outputs), dtype=np.float32)
    for b in range(batch_size):
        for p in range(new_n_points):
            for i in range(n_shared):
                combined_rhs[b, p, i] = matrix_shared[b, p, i]
            for o in range(n_rhs_outputs):
                combined_rhs[b, p, n_shared + o] = new_rhs[b, p, o]

    coeff_indep = lstsq(matrix_indep, combined_rhs, new_weights)

    # Split the coefficients
    coeff_indep2shared = np.empty((batch_size, n_indep, n_shared), dtype=np.float32)
    coeff_indep2rhs = np.empty((batch_size, n_indep, n_rhs_outputs), dtype=np.float32)
    for b in range(batch_size):
        for i in range(n_indep):
            for j in range(n_shared):
                coeff_indep2shared[b, i, j] = coeff_indep[b, i, j]
            for o in range(n_rhs_outputs):
                coeff_indep2rhs[b, i, o] = coeff_indep[b, i, n_shared + o]

    # Compute residual matrix: matrix_shared - matrix_indep @ coeff_indep2shared
    residual_matrix = np.empty((batch_size, new_n_points, n_shared), dtype=np.float32)
    for b in range(batch_size):
        for p in range(new_n_points):
            for j in range(n_shared):
                val = matrix_shared[b, p, j]
                for i in range(n_indep):
                    val -= matrix_indep[b, p, i] * coeff_indep2shared[b, i, j]
                residual_matrix[b, p, j] = val

    # Compute residual rhs: rhs - matrix_indep @ coeff_indep2rhs
    residual_rhs = np.empty((batch_size, new_n_points, n_rhs_outputs), dtype=np.float32)
    for b in range(batch_size):
        for p in range(new_n_points):
            for o in range(n_rhs_outputs):
                val = new_rhs[b, p, o]
                for i in range(n_indep):
                    val -= matrix_indep[b, p, i] * coeff_indep2rhs[b, i, o]
                residual_rhs[b, p, o] = val

    # Solve for shared params using residuals (shared across batch)
    coeff_shared2rhs = lstsq(residual_matrix, residual_rhs, new_weights, shared=True)

    # Update independent coefficients: coeff_indep2rhs -= coeff_indep2shared @ coeff_shared2rhs
    for b in range(batch_size):
        for i in range(n_indep):
            for o in range(n_rhs_outputs):
                for j in range(n_shared):
                    coeff_indep2rhs[b, i, o] -= (
                        coeff_indep2shared[b, i, j] * coeff_shared2rhs[0, j, o]
                    )

    # Combine results
    result = np.empty((batch_size, n_params, n_rhs_outputs), dtype=np.float32)
    for b in range(batch_size):
        for i in range(n_shared):
            for o in range(n_rhs_outputs):
                result[b, i, o] = coeff_shared2rhs[0, i, o]
        for i in range(n_indep):
            for o in range(n_rhs_outputs):
                result[b, n_shared + i, o] = coeff_indep2rhs[b, i, o]

    return result
