import tensorflow as tf


def kabsch(X, Y):
    A = tf.matmul(X, Y, transpose_a=True)
    _, U, V = tf.linalg.svd(A)
    T = tf.matmul(U, V, transpose_b=True)
    has_reflection = (tf.linalg.det(T) < 0)[..., tf.newaxis, tf.newaxis]
    T_mirror = T - 2 * tf.matmul(U[..., -1:], V[..., -1:], transpose_b=True)
    return tf.where(has_reflection, T_mirror, T)


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
    rows = tf.unstack(rotmat, axis=-2, num=3)
    r = [tf.unstack(row, axis=-1, num=3) for row in rows]

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

    trace = tf.linalg.trace(rotmat)
    cond0 = tf.stack((p21m12, p02m20, p10m01, 1.0 + trace), axis=-1)
    cond1 = tf.stack((_1m22 + p00m11, p10p01, p02p20, p21m12), axis=-1)
    cond2 = tf.stack((p10p01, _1m22 - p00m11, p21p12, p02m20), axis=-1)
    cond3 = tf.stack((p02p20, p21p12, _1p22 - p00p11, p10m01), axis=-1)

    trace_pos = tf.expand_dims(trace > 0, -1)
    d00_large = tf.expand_dims(tf.logical_and(r[0][0] > r[1][1], r[0][0] > r[2][2]), -1)
    d11_large = tf.expand_dims(r[1][1] > r[2][2], -1)
    q = tf.where(trace_pos, cond0, tf.where(d00_large, cond1, tf.where(d11_large, cond2, cond3)))
    xyz, w = tf.split(q, (3, 1), axis=-1)
    norm = tf.norm(xyz, axis=-1, keepdims=True)
    return (tf.math.divide_no_nan(2.0, norm) * tf.atan2(norm, w)) * xyz
