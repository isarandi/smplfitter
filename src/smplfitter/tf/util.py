from __future__ import annotations
import tensorflow as tf


@tf.custom_gradient
def safe_nan_to_zero(x):
    isnan = tf.math.is_nan(x)

    def grad(upstream):
        return tf.where(isnan, tf.cast(0, x.dtype), upstream)

    return tf.where(isnan, tf.cast(0, x.dtype), x), grad


def batched_matmul(a: tf.Tensor, b: tf.Tensor, transpose_a: bool = False) -> tf.Tensor:
    """Matrix product of batches for long inner dimensions, such as a mesh-sized design matrix.

    Use this instead of ``tf.linalg.matmul`` or ``tf.einsum`` when the summed-over dimension can
    reach about 20000. On CPUs with AVX-512, these dispatch to the oneDNN ``brg_matmul`` kernel,
    which then returns wrong values or NaN without any error if the batch has more than one
    element and more than one thread runs (measured with TF 2.17 to 2.21, not with TF 2.16).
    ``BatchMatMulV3`` has no oneDNN variant. It is otherwise the same operation, with kernels
    on the CPU and the GPU, gradients and XLA support.
    """
    return tf.raw_ops.BatchMatMulV3(x=a, y=b, Tout=a.dtype, adj_x=transpose_a)
