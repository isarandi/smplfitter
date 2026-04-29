from __future__ import annotations

import math

import numpy as np
import numba
from numba import prange


@numba.njit(error_model='numpy', cache=True)
def kabsch(X, Y):
    # X, Y: (batch, n_points, 3) - supports broadcasting (one can have batch=1)
    # Output: (batch, 3, 3) rotation matrices
    batch_x = X.shape[0]
    batch_y = Y.shape[0]
    batch_size = max(batch_x, batch_y)

    T = np.empty((batch_size, 3, 3), dtype=np.float32)
    for bi in range(batch_size):
        xi = bi if batch_x > 1 else 0
        yi = bi if batch_y > 1 else 0

        # Compute X[xi].T @ Y[yi]
        A = np.zeros((3, 3), dtype=np.float32)
        n_points = X.shape[1]
        for k in range(n_points):
            for i in range(3):
                for j in range(3):
                    A[i, j] += X[xi, k, i] * Y[yi, k, j]

        U, _, Vh = np.linalg.svd(A)
        Ti = U @ Vh
        if np.linalg.det(Ti) < 0:
            Ti = Ti - np.float32(2.0) * U[:, -1:] @ Vh[-1:, :]
        T[bi] = Ti
    return T


@numba.njit(error_model='numpy', cache=True)
def rotvec2mat(rotvec):
    """Convert a single rotation vector (3,) to rotation matrix (3, 3)."""
    rx = rotvec[0]
    ry = rotvec[1]
    rz = rotvec[2]
    out = np.empty((3, 3), dtype=np.float32)
    angle_sq = rx * rx + ry * ry + rz * rz
    if angle_sq < 1e-16:
        out[0, 0] = 1
        out[0, 1] = 0
        out[0, 2] = 0
        out[1, 0] = 0
        out[1, 1] = 1
        out[1, 2] = 0
        out[2, 0] = 0
        out[2, 1] = 0
        out[2, 2] = 1
    else:
        angle = math.sqrt(angle_sq)
        inv_a = np.float32(1.0) / np.float32(angle)
        ax = rx * inv_a
        ay = ry * inv_a
        az = rz * inv_a
        s = np.float32(math.sin(angle))
        c = np.float32(math.cos(angle))
        c1 = np.float32(1.0) - c
        out[0, 0] = c + c1 * ax * ax
        out[0, 1] = c1 * ax * ay - s * az
        out[0, 2] = c1 * ax * az + s * ay
        out[1, 0] = c1 * ax * ay + s * az
        out[1, 1] = c + c1 * ay * ay
        out[1, 2] = c1 * ay * az - s * ax
        out[2, 0] = c1 * ax * az - s * ay
        out[2, 1] = c1 * ay * az + s * ax
        out[2, 2] = c + c1 * az * az
    return out


@numba.njit(error_model='numpy', cache=True, parallel=True)
def rotvec2mat_batch(rotvecs):
    """Convert batch of rotation vectors (B, N, 3) to rotation matrices (B, N, 3, 3)."""
    B = rotvecs.shape[0]
    N = rotvecs.shape[1]
    out = np.empty((B, N, 3, 3), dtype=np.float32)
    for b in prange(B):
        for n in range(N):
            rx = rotvecs[b, n, 0]
            ry = rotvecs[b, n, 1]
            rz = rotvecs[b, n, 2]
            angle_sq = rx * rx + ry * ry + rz * rz
            if angle_sq < 1e-16:
                out[b, n, 0, 0] = 1
                out[b, n, 0, 1] = 0
                out[b, n, 0, 2] = 0
                out[b, n, 1, 0] = 0
                out[b, n, 1, 1] = 1
                out[b, n, 1, 2] = 0
                out[b, n, 2, 0] = 0
                out[b, n, 2, 1] = 0
                out[b, n, 2, 2] = 1
            else:
                angle = math.sqrt(angle_sq)
                inv_a = np.float32(1.0) / np.float32(angle)
                ax = rx * inv_a
                ay = ry * inv_a
                az = rz * inv_a
                s = np.float32(math.sin(angle))
                c = np.float32(math.cos(angle))
                c1 = np.float32(1.0) - c
                out[b, n, 0, 0] = c + c1 * ax * ax
                out[b, n, 0, 1] = c1 * ax * ay - s * az
                out[b, n, 0, 2] = c1 * ax * az + s * ay
                out[b, n, 1, 0] = c1 * ax * ay + s * az
                out[b, n, 1, 1] = c + c1 * ay * ay
                out[b, n, 1, 2] = c1 * ay * az - s * ax
                out[b, n, 2, 0] = c1 * ax * az - s * ay
                out[b, n, 2, 1] = c1 * ay * az + s * ax
                out[b, n, 2, 2] = c + c1 * az * az
    return out


@numba.njit(error_model='numpy', cache=True)
def mat2rotvec(rotmat):
    r00 = rotmat[..., 0, 0]
    r01 = rotmat[..., 0, 1]
    r02 = rotmat[..., 0, 2]
    r10 = rotmat[..., 1, 0]
    r11 = rotmat[..., 1, 1]
    r12 = rotmat[..., 1, 2]
    r20 = rotmat[..., 2, 0]
    r21 = rotmat[..., 2, 1]
    r22 = rotmat[..., 2, 2]

    p10p01 = r10 + r01
    p10m01 = r10 - r01
    p02p20 = r02 + r20
    p02m20 = r02 - r20
    p21p12 = r21 + r12
    p21m12 = r21 - r12
    p00p11 = r00 + r11
    p00m11 = r00 - r11
    _1p22 = np.float32(1.0) + r22
    _1m22 = np.float32(1.0) - r22

    trace = p00p11 + r22
    cond0 = np.stack((p21m12, p02m20, p10m01, np.float32(1.0) + trace), axis=-1)
    cond1 = np.stack((_1m22 + p00m11, p10p01, p02p20, p21m12), axis=-1)
    cond2 = np.stack((p10p01, _1m22 - p00m11, p21p12, p02m20), axis=-1)
    cond3 = np.stack((p02p20, p21p12, _1p22 - p00p11, p10m01), axis=-1)

    trace_pos = trace > 0
    d00_large = np.logical_and(r00 > r11, r00 > r22)
    d11_large = r11 > r22

    q = np.where(
        trace_pos.reshape(trace_pos.shape + (1,)),
        cond0,
        np.where(
            d00_large.reshape(d00_large.shape + (1,)),
            cond1,
            np.where(d11_large.reshape(d11_large.shape + (1,)), cond2, cond3),
        ),
    )

    xyz = q[..., :3]
    w = q[..., 3:4]
    norm = np.sqrt(np.sum(xyz * xyz, axis=-1)).reshape(xyz.shape[:-1] + (1,))
    factor = np.float32(2.0) / norm * np.arctan2(norm, w)
    factor = np.where(norm == 0, np.zeros_like(factor), factor)
    return (factor * xyz).astype(np.float32)
