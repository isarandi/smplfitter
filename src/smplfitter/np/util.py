import numpy as np


def matvec(mat, vec):
    return (mat @ vec[..., np.newaxis]).squeeze(-1)


def unstack(x, axis=-1):
    return tuple(np.moveaxis(x, axis, 0))


def matrix_transpose(mat):
    return np.swapaxes(mat, -2, -1)


def matmul_transp_a(a, b):
    return np.einsum('...ji,...jk->...ik', a, b)
