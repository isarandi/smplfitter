from __future__ import annotations
import os

import numpy as np
from .. import np as smplfitter_np
import trimesh
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def main():
    DATA_ROOT = os.getenv('DATA_ROOT', default='.')
    for model_name in ['smpl', 'smplx']:
        model_root = f'{DATA_ROOT}/body_models/{model_name}'
        bm = smplfitter_np.get_cached_body_model(model_name)
        verts = bm.single()['vertices']

        for n in [32, 64, 128, 256, 512, 1024]:
            i_verts, faces = decimate(verts, bm.faces, n)
            if len(i_verts) != n:
                print(f'Failed to decimate to {n} vertices')
                continue
            else:
                print(f'Decimated to {n} vertices')
            np.savez(f'{model_root}/vertex_subset_{n}.npz', i_verts=i_verts, faces=faces)


def _decimate(verts, faces, n_verts_out=128):
    n_faces = 2 * n_verts_out - 4
    decimated_mesh = trimesh.Trimesh(verts, faces).simplify_quadric_decimation(face_count=n_faces)
    row_ind, col_ind = linear_sum_assignment(cdist(verts, decimated_mesh.vertices))
    i_verts = row_ind[np.argsort(col_ind)]
    return i_verts, decimated_mesh.faces


def decimate(verts, faces, n_verts_out=128, n_trials=100):
    # We may have to try it multiple times to get the desired number of vertices
    # since the decimation algorithm may not always return the exact number of vertices we ask.
    # The documentation explicitly states this.

    n_verts_out_arg = n_verts_out
    i_verts, new_faces = _decimate(verts, faces, n_verts_out_arg)

    for _ in range(n_trials):
        if i_verts.shape[0] == n_verts_out:
            break

        n_verts_out_arg += n_verts_out - i_verts.shape[0]
        i_verts, new_faces = _decimate(verts, faces, n_verts_out_arg)

    return i_verts, new_faces


if __name__ == '__main__':
    main()
