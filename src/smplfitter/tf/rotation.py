from __future__ import annotations
import tensorflow as tf


def proj_SO3(A):
    """Project (..., 3, 3) matrices onto SO(3) — closest rotation in Frobenius norm."""
    _, U, V = tf.linalg.svd(A)
    T = tf.matmul(U, V, transpose_b=True)
    has_reflection = (_det3x3(T) < 0)[..., tf.newaxis, tf.newaxis]
    T_mirror = T - 2 * tf.matmul(U[..., -1:], V[..., -1:], transpose_b=True)
    return tf.where(has_reflection, T_mirror, T)


def _det3x3(m):
    """Determinant of (..., 3, 3) matrices via cofactor expansion.

    Used instead of ``tf.linalg.det`` because ``MatrixDeterminant`` has no XLA CPU
    kernel, so it breaks ``tf.function(jit_compile=True)``; the explicit 3x3 form is
    XLA-compatible and numerically identical. It only feeds a sign comparison, so it
    does not affect gradients.
    """
    return (
        m[..., 0, 0] * (m[..., 1, 1] * m[..., 2, 2] - m[..., 1, 2] * m[..., 2, 1])
        - m[..., 0, 1] * (m[..., 1, 0] * m[..., 2, 2] - m[..., 1, 2] * m[..., 2, 0])
        + m[..., 0, 2] * (m[..., 1, 0] * m[..., 2, 1] - m[..., 1, 1] * m[..., 2, 0])
    )


def kabsch(X, Y):
    return proj_SO3(tf.matmul(X, Y, transpose_a=True))


def align_unit_vectors(a, b):
    """Closed-form rotation that maps unit vector ``a`` to unit vector ``b``.

    Returns (..., 3, 3). Built from Rodrigues on the axis-angle
    ``angle * (a x b) / |a x b|`` with ``angle = atan2(|a x b|, a . b)``.
    The parallel (a == b) and antiparallel (a == -b) limits stay finite —
    ``tf.math.divide_no_nan`` returns a zero rotvec there, giving the identity matrix.
    The antiparallel choice is arbitrary (no canonical 180-deg rotation).

    Unlike ``tf.linalg.cross``, this helper broadcasts ``a`` and ``b`` against
    each other (e.g. (1, 3) against (B, 3)).
    """
    broadcast_shape = tf.broadcast_dynamic_shape(tf.shape(a), tf.shape(b))
    a = tf.broadcast_to(a, broadcast_shape)
    b = tf.broadcast_to(b, broadcast_shape)
    cross = tf.linalg.cross(a, b)
    dot = tf.reduce_sum(a * b, axis=-1, keepdims=True)
    sin_a = tf.linalg.norm(cross, axis=-1, keepdims=True)
    angle = tf.math.atan2(sin_a, dot)
    rotvec = tf.math.divide_no_nan(cross * angle, sin_a)
    return rotvec2mat(rotvec)


def project_onto_plane(v, n_hat):
    """Component of ``v`` perpendicular to the unit vector ``n_hat``.

    Batched over leading dims; ``n_hat`` broadcasts against ``v``.
    """
    parallel = tf.reduce_sum(v * n_hat, axis=-1, keepdims=True) * n_hat
    return v - parallel


def rotvec2mat(rotvec):
    angle = tf.linalg.norm(rotvec, axis=-1, keepdims=True)
    axis = tf.math.divide_no_nan(rotvec, angle)

    sin_axis = tf.sin(angle) * axis
    cos_angle = tf.cos(angle)
    cos1_axis = (1.0 - cos_angle) * axis
    _, axis_y, axis_z = tf.unstack(axis, axis=-1)
    cos1_axis_x, cos1_axis_y, _ = tf.unstack(cos1_axis, axis=-1, num=3)
    sin_axis_x, sin_axis_y, sin_axis_z = tf.unstack(sin_axis, axis=-1, num=3)
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
    m00, m11, m22 = tf.unstack(diag, axis=-1, num=3)
    matrix = tf.stack((m00, m01, m02, m10, m11, m12, m20, m21, m22), axis=-1)
    return tf.reshape(matrix, tf.concat((tf.shape(axis)[:-1], (3, 3)), axis=-1))


def mat2rotvec(rotmat):
    (r00, r01, r02, r10, r11, r12, r20, r21, r22) = tf.unstack(
        tf.reshape(rotmat, tf.concat((tf.shape(rotmat)[:-2], (9,)), axis=-1)), axis=-1, num=9
    )
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

    trace = tf.linalg.trace(rotmat)
    cond0 = tf.stack((p21m12, p02m20, p10m01, 1.0 + trace), axis=-1)
    cond1 = tf.stack((_1m22 + p00m11, p10p01, p02p20, p21m12), axis=-1)
    cond2 = tf.stack((p10p01, _1m22 - p00m11, p21p12, p02m20), axis=-1)
    cond3 = tf.stack((p02p20, p21p12, _1p22 - p00p11, p10m01), axis=-1)

    trace_pos = tf.expand_dims(trace > 0, -1)
    d00_large = tf.expand_dims(tf.logical_and(r00 > r11, r00 > r22), -1)
    d11_large = tf.expand_dims(r11 > r22, -1)
    q = tf.where(trace_pos, cond0, tf.where(d00_large, cond1, tf.where(d11_large, cond2, cond3)))
    xyz, w = tf.split(q, (3, 1), axis=-1)
    norm = tf.norm(xyz, axis=-1, keepdims=True)
    return (tf.math.divide_no_nan(2.0, norm) * tf.atan2(norm, w)) * xyz
