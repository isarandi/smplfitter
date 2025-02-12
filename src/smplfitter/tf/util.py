import tensorflow as tf


@tf.custom_gradient
def safe_nan_to_zero(x):
    isnan = tf.math.is_nan(x)

    def grad(upstream):
        return tf.where(isnan, tf.cast(0, x.dtype), upstream)

    return tf.where(isnan, tf.cast(0, x.dtype), x), grad
