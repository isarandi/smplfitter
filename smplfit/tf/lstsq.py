import tensorflow as tf


def lstsq(matrix, rhs, weights, l2_regularizer):
    weighted_matrix = weights[..., tf.newaxis] * matrix
    regularized_gramian = tf.linalg.matmul(
        weighted_matrix, matrix, transpose_a=True) + tf.linalg.diag(l2_regularizer)
    chol = tf.linalg.cholesky(regularized_gramian)
    return tf.linalg.cholesky_solve(chol, tf.linalg.matmul(weighted_matrix, rhs, transpose_a=True))


def lstsq_partial_share(matrix, rhs, weights, l2_regularizer, n_shared=0):
    n_params = tf.shape(matrix)[-1]
    n_rhs_outputs = tf.shape(rhs)[-1]
    n_indep = n_params - n_shared

    # Add the regularization equations into the design matrix
    # This way it's simpler to handle all these steps,
    # we only need to implement the unregularized case,
    # and regularization is just adding more rows to the matrix.
    matrix = tf.concat([matrix, tf.eye(n_params, batch_shape=[tf.shape(matrix)[0]])], axis=1)
    rhs = tf.pad(rhs, [[0, 0], [0, n_params], [0, 0]])
    weights = tf.concat(
        [weights, tf.repeat(l2_regularizer[tf.newaxis], tf.shape(matrix)[0], axis=0)], axis=1)

    # Split the shared and independent parts of the matrices
    matrix_shared, matrix_indep = tf.split(matrix, [n_shared, n_indep], axis=-1)

    # First solve for the independent params only (~shared params are forced to 0)
    # Also regress the shared columns on the independent columns
    # Since we regress the rhs from the independent columns, any part of the shared
    # columns that are linearly predictable from the indep columns needs to be removed,
    # so we can solve for the shared params while considering only the information that's
    # unaccounted for so far.
    solve_indep_fn = get_lstsq_solver_fn(matrix_indep, weights)
    coeff_indep2shared, coeff_indep2rhs = tf.split(
        solve_indep_fn(tf.concat([matrix_shared, rhs], axis=-1)),
        [n_shared, n_rhs_outputs], axis=-1)

    # Now solve for the shared params using the residuals
    solve_shared_fn = get_lstsq_solver_fn(
        matrix_shared - matrix_indep @ coeff_indep2shared, weights, shared=True)
    coeff_shared2rhs = solve_shared_fn(rhs - matrix_indep @ coeff_indep2rhs)

    # Finally, update the estimate for the independent params, reusing the Cholesky decomposition.
    coeff_indep2rhs = solve_indep_fn(rhs - matrix_shared @ coeff_shared2rhs)

    # Repeat the shared coefficients for each sample and concatenate them with the independent ones
    coeff_shared2rhs = tf.repeat(coeff_shared2rhs, tf.shape(matrix)[0], axis=0)
    return tf.concat([coeff_shared2rhs, coeff_indep2rhs], axis=1)


def get_lstsq_solver_fn(matrix, weights, shared=False):
    # The solver function saves the Cholesky decomposition of the Gramian, so it can be reused.
    weights = tf.expand_dims(weights, axis=-1)
    w_matrix = matrix * weights
    gramian = tf.linalg.matmul(w_matrix, matrix, transpose_a=True)
    if shared:
        gramian = tf.reduce_sum(gramian, axis=0, keepdims=True)

    chol = tf.linalg.cholesky(gramian)

    def solver(rhs):
        ATb = tf.linalg.matmul(w_matrix, rhs, transpose_a=True)
        if shared:
            ATb = tf.reduce_sum(ATb, axis=0, keepdims=True)

        return tf.linalg.cholesky_solve(chol, ATb)

    return solver
