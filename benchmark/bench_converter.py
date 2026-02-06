"""Benchmark SMPL-to-SMPLX conversion across backends.

Reproduces Table 10 from the NLF paper (arXiv:2407.07532).

The official converter outputs were pre-generated using smplx's transfer_model tool
with various iteration counts (25, 37, 50, 100 iterations).
"""

import time
from pathlib import Path

import numpy as np
import scipy.sparse


def load_sparse_matrix(path: str) -> scipy.sparse.csr_matrix:
    """Load sparse matrix from pickle."""
    import pickle

    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['mtx'].tocsr()


def convert_vertices_np(
    vertices: np.ndarray, conversion_matrix: scipy.sparse.csr_matrix
) -> np.ndarray:
    """Convert vertices using numpy/scipy sparse matrix multiplication."""
    # vertices: (batch, num_verts_in, 3) -> (batch, num_verts_out, 3)
    batch_size = vertices.shape[0]
    results = np.zeros((batch_size, conversion_matrix.shape[0], 3), dtype=np.float32)
    for i in range(batch_size):
        results[i] = conversion_matrix @ vertices[i]
    return results


def benchmark_pytorch(
    smpl_verts: np.ndarray,
    smplx2smpl_mat: scipy.sparse.csr_matrix,
    smpl2smplx_mat: scipy.sparse.csr_matrix,
) -> dict:
    """Benchmark PyTorch backend."""
    import torch
    import smplfitter.pt as smplfitter

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    body_model_smplx = smplfitter.BodyModel('smplx', 'neutral', num_betas=10).to(device)
    fitter = smplfitter.BodyFitter(body_model_smplx, enable_kid=False).to(device)
    fitter_jit = torch.jit.script(fitter)

    # Convert SMPL vertices to SMPLX topology
    smplx_target_verts_np = convert_vertices_np(smpl_verts, smpl2smplx_mat)
    smplx_target_verts = torch.from_numpy(smplx_target_verts_np).to(device)
    smplx_target_joints = torch.einsum(
        'jv,bvc->bjc', body_model_smplx.J_regressor_post_lbs, smplx_target_verts
    )

    # Warmup
    for _ in range(3):
        with torch.inference_mode():
            _ = fitter_jit.fit(
                target_vertices=smplx_target_verts,
                target_joints=smplx_target_joints,
                num_iter=1,
                beta_regularizer=0.0,
            )
    if device == 'cuda':
        torch.cuda.synchronize()

    results = {}
    for num_iter in [1, 2, 3, 4, 5]:
        times = []
        for _ in range(10):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.inference_mode():
                result = fitter_jit.fit(
                    target_vertices=smplx_target_verts,
                    target_joints=smplx_target_joints,
                    num_iter=num_iter,
                    beta_regularizer=0.0,
                    requested_keys=['pose_rotvecs', 'shape_betas'],
                )
                output = body_model_smplx(
                    pose_rotvecs=result['pose_rotvecs'],
                    shape_betas=result['shape_betas'],
                    trans=result['trans'],
                )

            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

        smplx_verts = output['vertices'].cpu().numpy()
        smplx_verts_as_smpl = convert_vertices_np(smplx_verts, smplx2smpl_mat)
        error = np.mean(np.linalg.norm(smplx_verts_as_smpl - smpl_verts, axis=-1))

        results[num_iter] = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'error': error * 1000,  # mm
        }

    return results


def benchmark_tensorflow(
    smpl_verts: np.ndarray,
    smplx2smpl_mat: scipy.sparse.csr_matrix,
    smpl2smplx_mat: scipy.sparse.csr_matrix,
) -> dict:
    """Benchmark TensorFlow backend."""
    import tensorflow as tf
    import smplfitter.tf as smplfitter

    body_model_smplx = smplfitter.BodyModel('smplx', 'neutral')
    fitter = smplfitter.BodyFitter(body_model_smplx, num_betas=10, enable_kid=False)

    # Convert SMPL vertices to SMPLX topology
    smplx_target_verts_np = convert_vertices_np(smpl_verts, smpl2smplx_mat)
    smplx_target_verts = tf.constant(smplx_target_verts_np)
    smplx_target_joints = tf.einsum(
        'jv,bvc->bjc', body_model_smplx.J_regressor_post_lbs, smplx_target_verts
    )

    # Create tf.function for each num_iter to avoid retracing
    @tf.function(reduce_retracing=True)
    def forward_fn(pose_rotvecs, shape_betas, trans):
        return body_model_smplx(
            pose_rotvecs=pose_rotvecs,
            shape_betas=shape_betas,
            trans=trans,
        )

    def make_fit_fn(n_iter):
        @tf.function
        def fit_fn(verts, joints):
            return fitter.fit(
                target_vertices=verts,
                target_joints=joints,
                num_iter=n_iter,
                beta_regularizer=0.0,
                requested_keys=['pose_rotvecs', 'shape_betas'],
            )

        return fit_fn

    fit_fns = {n: make_fit_fn(n) for n in [1, 2, 3, 4, 5]}

    # Warmup - trace and run each function multiple times
    for num_iter in [1, 2, 3, 4, 5]:
        for _ in range(3):
            _ = fit_fns[num_iter](smplx_target_verts, smplx_target_joints)
            _ = forward_fn(_['pose_rotvecs'], _['shape_betas'], _['trans'])['vertices'].numpy()

    results = {}
    for num_iter in [1, 2, 3, 4, 5]:
        fit_fn = fit_fns[num_iter]
        times = []
        for _ in range(10):
            start = time.perf_counter()

            result = fit_fn(smplx_target_verts, smplx_target_joints)
            output = forward_fn(
                result['pose_rotvecs'],
                result['shape_betas'],
                result['trans'],
            )
            # Force execution
            _ = output['vertices'].numpy()

            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

        smplx_verts = output['vertices'].numpy()
        smplx_verts_as_smpl = convert_vertices_np(smplx_verts, smplx2smpl_mat)
        error = np.mean(np.linalg.norm(smplx_verts_as_smpl - smpl_verts, axis=-1))

        results[num_iter] = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'error': error * 1000,  # mm
        }

    return results


def benchmark_numba(
    smpl_verts: np.ndarray,
    smplx2smpl_mat: scipy.sparse.csr_matrix,
    smpl2smplx_mat: scipy.sparse.csr_matrix,
) -> dict:
    """Benchmark Numba backend."""
    import smplfitter.nb as smplfitter

    body_model_smplx = smplfitter.BodyModel('smplx', 'neutral', num_betas=10)
    fitter = smplfitter.BodyFitter(body_model_smplx, enable_kid=False)

    # Convert SMPL vertices to SMPLX topology
    smplx_target_verts = convert_vertices_np(smpl_verts, smpl2smplx_mat)
    smplx_target_joints = np.einsum(
        'jv,bvc->bjc', body_model_smplx.J_regressor_post_lbs, smplx_target_verts
    )

    # Warmup
    for _ in range(3):
        _ = fitter.fit(
            target_vertices=smplx_target_verts,
            target_joints=smplx_target_joints,
            num_iter=1,
            beta_regularizer=0.0,
        )

    results = {}
    for num_iter in [1, 2, 3, 4, 5]:
        times = []
        for _ in range(10):
            start = time.perf_counter()

            result = fitter.fit(
                target_vertices=smplx_target_verts,
                target_joints=smplx_target_joints,
                num_iter=num_iter,
                beta_regularizer=0.0,
                requested_keys=['pose_rotvecs', 'shape_betas'],
            )
            output = body_model_smplx(
                pose_rotvecs=result['pose_rotvecs'],
                shape_betas=result['shape_betas'],
                trans=result['trans'],
            )

            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

        smplx_verts = np.asarray(output['vertices'])
        smplx_verts_as_smpl = convert_vertices_np(smplx_verts, smplx2smpl_mat)
        error = np.mean(np.linalg.norm(smplx_verts_as_smpl - smpl_verts, axis=-1))

        results[num_iter] = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'error': error * 1000,  # mm
        }

    return results


def main():
    import argparse
    import os
    import trimesh

    parser = argparse.ArgumentParser(description='Benchmark SMPL-to-SMPLX conversion')
    parser.add_argument(
        '--backends',
        nargs='+',
        default=['pt', 'tf', 'nb'],
        choices=['pt', 'tf', 'nb'],
        help='Backends to benchmark',
    )
    args = parser.parse_args()

    data_root = Path(os.environ.get('DATA_ROOT', '/work/sarandi/data'))
    mesh_dir = data_root / 'bedlam/smplx/transfer_data/meshes/smpl'
    official_output_dir = data_root / 'bedlam/smplx/output'

    # Load SMPL meshes (33 samples used in paper)
    ply_paths = sorted(mesh_dir.glob('*.ply'))[:33]
    meshes = [trimesh.load(p) for p in ply_paths]
    smpl_verts = np.array([m.vertices for m in meshes], dtype=np.float32)
    print(f'Loaded {len(smpl_verts)} SMPL meshes')

    # Load conversion matrices
    smplx2smpl_mat = load_sparse_matrix(f'{data_root}/body_models/smplx2smpl_deftrafo_setup.pkl')[
        :, :10475
    ].astype(np.float32)

    smpl2smplx_mat = load_sparse_matrix(f'{data_root}/body_models/smpl2smplx_deftrafo_setup.pkl')[
        :, :6890
    ].astype(np.float32)

    # Load official converter outputs for comparison
    official_paths = sorted(official_output_dir.glob('*.obj'))[:33]
    if len(official_paths) == 33:
        official_verts = np.array(
            [trimesh.load(p).vertices for p in official_paths], dtype=np.float32
        )
        official_verts_as_smpl = convert_vertices_np(official_verts, smplx2smpl_mat)
        official_error = np.mean(np.linalg.norm(official_verts_as_smpl - smpl_verts, axis=-1))
        print(f'\nOfficial converter error: {official_error * 1000:.2f} mm')
        print('(Note: official converter takes 3-33 min depending on iterations)')
    else:
        print(f'\nOfficial outputs not found (expected 33, found {len(official_paths)})')

    # Run benchmarks
    all_results = {}

    for backend in args.backends:
        print(f'\n{"=" * 60}')
        print(f'{backend.upper()} Backend Benchmark')
        print('=' * 60)

        try:
            if backend == 'pt':
                results = benchmark_pytorch(smpl_verts, smplx2smpl_mat, smpl2smplx_mat)
            elif backend == 'tf':
                results = benchmark_tensorflow(smpl_verts, smplx2smpl_mat, smpl2smplx_mat)
            elif backend == 'nb':
                results = benchmark_numba(smpl_verts, smplx2smpl_mat, smpl2smplx_mat)

            all_results[backend] = results

            print(f'{"Iterations":<12} {"Time (ms)":<15} {"Error (mm)":<12} {"Per-mesh (ms)"}')
            print('-' * 60)

            for num_iter in [1, 2, 3, 4, 5]:
                r = results[num_iter]
                print(
                    f'{num_iter:<12} {r["mean_time"]:>6.1f} ± {r["std_time"]:>4.1f}   '
                    f'{r["error"]:>6.2f}        {r["mean_time"] / len(smpl_verts):.2f}'
                )

        except Exception as e:
            print(f'Error running {backend} benchmark: {e}')
            import traceback

            traceback.print_exc()

    # Summary comparison
    if len(all_results) > 1:
        print(f'\n{"=" * 60}')
        print('Summary Comparison (3 iterations)')
        print('=' * 60)
        print(f'{"Backend":<10} {"Time (ms)":<15} {"Error (mm)":<12}')
        print('-' * 40)
        for backend, results in all_results.items():
            r = results[3]
            print(
                f'{backend.upper():<10} {r["mean_time"]:>6.1f} ± {r["std_time"]:>4.1f}   {r["error"]:>6.2f}'
            )

    print('\n' + '-' * 60)
    print('Comparison to official converter (from Table 10 in paper):')
    print('  Official 25 iter:  3 min 18 s, 14.0 mm error')
    print('  Official 50 iter: 16 min 35 s,  6.2 mm error')
    print('  Official 100 iter: 33 min,      5.0 mm error')


if __name__ == '__main__':
    main()
