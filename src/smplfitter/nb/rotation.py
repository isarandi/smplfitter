from __future__ import annotations

import numpy as np
import numba


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
    angle = np.sqrt(np.sum(rotvec * rotvec, axis=-1)).reshape(rotvec.shape[:-1] + (1,))
    axis = rotvec / angle
    axis = np.where(angle == 0, np.zeros_like(axis), axis)

    sin_axis = np.sin(angle) * axis
    cos_angle = np.cos(angle)
    cos1_axis = (np.float32(1.0) - cos_angle) * axis
    axis_y = axis[..., 1]
    axis_z = axis[..., 2]
    cos1_axis_x = cos1_axis[..., 0]
    cos1_axis_y = cos1_axis[..., 1]
    sin_axis_x = sin_axis[..., 0]
    sin_axis_y = sin_axis[..., 1]
    sin_axis_z = sin_axis[..., 2]

    tmp = cos1_axis_x * axis_y
    m01 = tmp - sin_axis_z
    m10 = tmp + sin_axis_z
    tmp = cos1_axis_x * axis_z
    m02 = tmp + sin_axis_y
    m20 = tmp - sin_axis_y
    tmp = cos1_axis_y * axis_z
    m12 = tmp - sin_axis_x
    m21 = tmp + sin_axis_x

    diag = cos1_axis * axis + cos_angle
    m00 = diag[..., 0]
    m11 = diag[..., 1]
    m22 = diag[..., 2]

    matrix = np.stack((m00, m01, m02, m10, m11, m12, m20, m21, m22), axis=-1)
    return matrix.reshape(axis.shape[:-1] + (3, 3)).astype(np.float32)


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

    trace = r00 + r11 + r22
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
