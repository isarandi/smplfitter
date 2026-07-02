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
def proj_SO3_single(A):
    """Project a single 3x3 matrix onto SO(3) (closest rotation in Frobenius norm)."""
    U, _, Vh = np.linalg.svd(A)
    T = U @ Vh
    if np.linalg.det(T) < 0:
        T = T - np.float32(2.0) * U[:, -1:] @ Vh[-1:, :]
    return T


@numba.njit(error_model='numpy', cache=True)
def align_unit_vectors_single(a, b):
    """Closed-form rotation (3, 3) that maps unit vector ``a`` to unit vector ``b``.

    Rodrigues on the axis-angle ``angle * (a x b) / |a x b|`` with
    ``angle = atan2(|a x b|, a . b)``. Parallel/antiparallel and zero-input limits
    return the identity (the zero cross-product gives a zero rotation vector).
    """
    cx = a[1] * b[2] - a[2] * b[1]
    cy = a[2] * b[0] - a[0] * b[2]
    cz = a[0] * b[1] - a[1] * b[0]
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    sin_a = math.sqrt(cx * cx + cy * cy + cz * cz)
    rotvec = np.zeros(3, dtype=np.float32)
    if sin_a > 0.0:
        f = np.float32(math.atan2(sin_a, dot)) / np.float32(sin_a)
        rotvec[0] = np.float32(cx) * f
        rotvec[1] = np.float32(cy) * f
        rotvec[2] = np.float32(cz) * f
    return rotvec2mat(rotvec)


@numba.njit(error_model='numpy', cache=True)
def swing_twist_bone(A, b_ref, b_tgt):
    """Swing-twist rotation for a two-joint (bone) part, in closed form.

    Parameters:
        A: (3, 3) part vertex cross-covariance ``sum w (t - c_t)(a - c_a)^T``
            (target on the rows, reference on the columns), centered at the
            children-mean joint positions.
        b_ref: (3,) un-normalized reference bone direction.
        b_tgt: (3,) un-normalized target bone direction.

    Returns:
        (3, 3) rotation ``R_twist @ R_swing``: the swing aligns the reference bone
        direction to the target one, the twist about the aligned bone is recovered
        from the vertices via ``theta = atan2(b_hat . vee(H), tr(H) - b_hat^T H b_hat)``
        with ``H = R_swing @ A^T``.
    """
    nref = math.sqrt(b_ref[0] * b_ref[0] + b_ref[1] * b_ref[1] + b_ref[2] * b_ref[2])
    ntgt = math.sqrt(b_tgt[0] * b_tgt[0] + b_tgt[1] * b_tgt[1] + b_tgt[2] * b_tgt[2])
    b_ref_n = np.zeros(3, dtype=np.float32)
    b_tgt_n = np.zeros(3, dtype=np.float32)
    if nref > 0.0:
        inv = np.float32(1.0) / np.float32(nref)
        for c in range(3):
            b_ref_n[c] = np.float32(b_ref[c]) * inv
    if ntgt > 0.0:
        inv = np.float32(1.0) / np.float32(ntgt)
        for c in range(3):
            b_tgt_n[c] = np.float32(b_tgt[c]) * inv

    R_swing = align_unit_vectors_single(b_ref_n, b_tgt_n)

    # H = R_swing @ A^T
    H = np.empty((3, 3), dtype=np.float32)
    for r in range(3):
        for c in range(3):
            val = np.float32(0.0)
            for k in range(3):
                val += R_swing[r, k] * A[c, k]
            H[r, c] = val

    trH = H[0, 0] + H[1, 1] + H[2, 2]
    bHb = np.float32(0.0)
    for r in range(3):
        for c in range(3):
            bHb += b_tgt_n[r] * H[r, c] * b_tgt_n[c]

    vee0 = H[1, 2] - H[2, 1]
    vee1 = H[2, 0] - H[0, 2]
    vee2 = H[0, 1] - H[1, 0]
    num = b_tgt_n[0] * vee0 + b_tgt_n[1] * vee1 + b_tgt_n[2] * vee2
    den = trH - bHb
    twist_angle = np.float32(math.atan2(num, den))

    rotvec = np.empty(3, dtype=np.float32)
    rotvec[0] = b_tgt_n[0] * twist_angle
    rotvec[1] = b_tgt_n[1] * twist_angle
    rotvec[2] = b_tgt_n[2] * twist_angle
    R_twist = rotvec2mat(rotvec)

    # R_bone = R_twist @ R_swing
    R_bone = np.empty((3, 3), dtype=np.float32)
    for r in range(3):
        for c in range(3):
            val = np.float32(0.0)
            for k in range(3):
                val += R_twist[r, k] * R_swing[k, c]
            R_bone[r, c] = val
    return R_bone


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
