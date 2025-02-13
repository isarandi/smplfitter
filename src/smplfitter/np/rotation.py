import numpy as np
from smplfitter.np.util import matmul_transp_a


def kabsch(X, Y):
    A = matmul_transp_a(X, Y)
    U, _, Vh = np.linalg.svd(A)
    T = U @ Vh
    has_reflection = (np.linalg.det(T) < 0)[..., np.newaxis, np.newaxis]
    T_mirror = T - 2 * U[..., -1:] @ Vh[..., -1:, :]
    return np.where(has_reflection, T_mirror, T)


def rotvec2mat(rotvec):
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        axis = np.nan_to_num(rotvec / angle)

    sin_axis = np.sin(angle) * axis
    cos_angle = np.cos(angle)
    cos1_axis = (1.0 - cos_angle) * axis
    axis_x, axis_y, axis_z = unstack(axis, axis=-1)
    cos1_axis_x, cos1_axis_y, _ = unstack(cos1_axis, axis=-1)
    sin_axis_x, sin_axis_y, sin_axis_z = unstack(sin_axis, axis=-1)

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
    m00, m11, m22 = unstack(diag, axis=-1)

    matrix = np.stack((m00, m01, m02, m10, m11, m12, m20, m21, m22), axis=-1)
    return matrix.reshape(*axis.shape[:-1], 3, 3)


def mat2rotvec(rotmat):
    rows = unstack(rotmat, axis=-2)
    r = [unstack(row, axis=-1) for row in rows]

    p10p01 = r[1][0] + r[0][1]
    p10m01 = r[1][0] - r[0][1]
    p02p20 = r[0][2] + r[2][0]
    p02m20 = r[0][2] - r[2][0]
    p21p12 = r[2][1] + r[1][2]
    p21m12 = r[2][1] - r[1][2]
    p00p11 = r[0][0] + r[1][1]
    p00m11 = r[0][0] - r[1][1]
    _1p22 = 1.0 + r[2][2]
    _1m22 = 1.0 - r[2][2]

    trace = np.trace(rotmat, axis1=-2, axis2=-1)
    cond0 = np.stack((p21m12, p02m20, p10m01, 1.0 + trace), axis=-1)
    cond1 = np.stack((_1m22 + p00m11, p10p01, p02p20, p21m12), axis=-1)
    cond2 = np.stack((p10p01, _1m22 - p00m11, p21p12, p02m20), axis=-1)
    cond3 = np.stack((p02p20, p21p12, _1p22 - p00p11, p10m01), axis=-1)

    trace_pos = (trace > 0)[..., np.newaxis]
    d00_large = np.logical_and(r[0][0] > r[1][1], r[0][0] > r[2][2])[..., np.newaxis]
    d11_large = (r[1][1] > r[2][2])[..., np.newaxis]

    q = np.where(trace_pos, cond0, np.where(d00_large, cond1, np.where(d11_large, cond2, cond3)))

    xyz, w = np.split(q, (3,), axis=-1)
    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        return (np.nan_to_num(2.0 / norm) * np.arctan2(norm, w)) * xyz


def unstack(x, axis=-1):
    return tuple(np.moveaxis(x, axis, 0))
