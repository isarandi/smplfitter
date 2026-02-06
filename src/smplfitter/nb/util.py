from __future__ import annotations

import numpy as np
import numba


@numba.njit(error_model='numpy', cache=True)
def matvec(mat, vec):
    # mat: (batch, n, 3, 3), vec: (batch, n, 3)
    # Output: (batch, n, 3)
    batch_size = mat.shape[0]
    n = mat.shape[1]
    result = np.empty((batch_size, n, 3), dtype=np.float32)
    for bi in range(batch_size):
        for ni in range(n):
            for i in range(3):
                val = np.float32(0.0)
                for j in range(3):
                    val += mat[bi, ni, i, j] * vec[bi, ni, j]
                result[bi, ni, i] = val
    return result


@numba.njit(error_model='numpy', cache=True)
def matmul_transp_a(a, b):
    # Computes a.T @ b for batched 3x3 matrices
    # a: (batch, n, 3, 3), b: (batch, n, 3, 3)
    # Output: (batch, n, 3, 3)
    batch_size = a.shape[0]
    n = a.shape[1]
    result = np.empty((batch_size, n, 3, 3), dtype=np.float32)
    for bi in range(batch_size):
        for ni in range(n):
            for i in range(3):
                for j in range(3):
                    val = np.float32(0.0)
                    for k in range(3):
                        val += a[bi, ni, k, i] * b[bi, ni, k, j]
                    result[bi, ni, i, j] = val
    return result


@numba.njit(error_model='numpy', cache=True)
def matmul_transp_a_3d(a, b):
    # Computes a.T @ b for batched point sets (used in kabsch)
    # a: (batch, n_points, 3), b: (batch, n_points, 3)
    # Output: (batch, 3, 3) - computes sum over n_points of outer products
    batch_size = a.shape[0]
    n_points = a.shape[1]
    result = np.empty((batch_size, 3, 3), dtype=np.float32)
    for bi in range(batch_size):
        for i in range(3):
            for j in range(3):
                val = np.float32(0.0)
                for k in range(n_points):
                    val += a[bi, k, i] * b[bi, k, j]
                result[bi, i, j] = val
    return result
