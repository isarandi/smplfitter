from __future__ import annotations

import jax.numpy as jnp


def kabsch(X, Y):
    """Compute optimal rotation matrix from X to Y using Kabsch algorithm.

    Args:
        X: Source points, shape (batch, n_points, 3)
        Y: Target points, shape (batch, n_points, 3)

    Returns:
        Rotation matrices, shape (batch, 3, 3)
    """
    A = jnp.swapaxes(X, -2, -1) @ Y
    U, _, Vh = jnp.linalg.svd(A)
    T = U @ Vh
    has_reflection = (jnp.linalg.det(T) < 0)[..., None, None]
    T_mirror = T - 2 * U[..., -1:] @ Vh[..., -1:, :]
    return jnp.where(has_reflection, T_mirror, T)


def rotvec2mat(rotvec):
    angle = jnp.linalg.norm(rotvec, axis=-1, keepdims=True)
    axis = jnp.where(angle == 0, jnp.zeros_like(rotvec), rotvec / angle)

    sin_axis = jnp.sin(angle) * axis
    cos_angle = jnp.cos(angle)
    cos1_axis = (1.0 - cos_angle) * axis
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

    matrix = jnp.stack((m00, m01, m02, m10, m11, m12, m20, m21, m22), axis=-1)
    return matrix.reshape(axis.shape[:-1] + (3, 3))


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
    _1p22 = 1.0 + r22
    _1m22 = 1.0 - r22

    trace = r00 + r11 + r22
    cond0 = jnp.stack((p21m12, p02m20, p10m01, 1.0 + trace), axis=-1)
    cond1 = jnp.stack((_1m22 + p00m11, p10p01, p02p20, p21m12), axis=-1)
    cond2 = jnp.stack((p10p01, _1m22 - p00m11, p21p12, p02m20), axis=-1)
    cond3 = jnp.stack((p02p20, p21p12, _1p22 - p00p11, p10m01), axis=-1)

    trace_pos = (trace > 0)[..., None]
    d00_large = ((r00 > r11) & (r00 > r22))[..., None]
    d11_large = (r11 > r22)[..., None]

    q = jnp.where(
        trace_pos, cond0, jnp.where(d00_large, cond1, jnp.where(d11_large, cond2, cond3))
    )

    xyz = q[..., :3]
    w = q[..., 3:4]
    norm = jnp.linalg.norm(xyz, axis=-1, keepdims=True)
    factor = jnp.where(norm == 0, jnp.zeros_like(norm), 2.0 / norm) * jnp.arctan2(norm, w)
    return factor * xyz
