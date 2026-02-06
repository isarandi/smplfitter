"""Benchmark full body model: NumPy vs Numba vs Cython."""

import numpy as np
import time


def benchmark(func, inputs, n_warmup=3, n_iter=50):
    """Run benchmark and return mean time in ms."""
    # Warmup
    for _ in range(n_warmup):
        func(**inputs)

    # Timed runs
    start = time.perf_counter()
    for _ in range(n_iter):
        func(**inputs)
    elapsed = time.perf_counter() - start
    return elapsed / n_iter * 1000  # ms


def main():
    import smplfitter.np as smpl_np
    import smplfitter.nb as smpl_nb

    try:
        import smplfitter.cy.bodymodel as smpl_cy

        has_cython = True
    except ImportError as e:
        print(f'Cython not built: {e}')
        print('Run: python setup_cy.py build_ext --inplace')
        has_cython = False

    print('Benchmarking BodyModel forward pass')
    print('=' * 70)

    for model_name in ['smpl']:
        print(f'\nModel: {model_name.upper()}')

        model_np = smpl_np.BodyModel(model_name=model_name, num_betas=10)
        model_nb = smpl_nb.BodyModel(model_name=model_name, num_betas=10)
        if has_cython:
            model_cy = smpl_cy.BodyModel(model_name=model_name, num_betas=10)

        num_joints = model_np.num_joints

        for return_vertices in [False, True]:
            mode = 'with_vertices' if return_vertices else 'joints_only'
            print(f'\n  Mode: {mode}')

            for batch_size in [1, 8, 32, 128, 512]:
                pose = np.random.randn(batch_size, num_joints * 3).astype(np.float32) * 0.1
                shape = np.random.randn(batch_size, 10).astype(np.float32) * 0.5
                trans = np.random.randn(batch_size, 3).astype(np.float32)

                inputs = dict(
                    pose_rotvecs=pose,
                    shape_betas=shape,
                    trans=trans,
                    return_vertices=return_vertices,
                )

                time_np = benchmark(model_np, inputs)
                time_nb = benchmark(model_nb, inputs)

                print(f'\n    batch={batch_size:3d}')
                print(
                    f'      NumPy:  {time_np:8.3f} ms  ({batch_size/time_np*1000:8.0f} items/sec)'
                )
                print(
                    f'      Numba:  {time_nb:8.3f} ms  ({batch_size/time_nb*1000:8.0f} items/sec)  {time_np/time_nb:5.1f}x vs NumPy'
                )

                if has_cython:
                    time_cy = benchmark(model_cy, inputs)
                    print(
                        f'      Cython: {time_cy:8.3f} ms  ({batch_size/time_cy*1000:8.0f} items/sec)  {time_np/time_cy:5.1f}x vs NumPy, {time_nb/time_cy:5.2f}x vs Numba'
                    )

                # Verify correctness
                result_np = model_np(**inputs)
                result_nb = model_nb(**inputs)

                if not np.allclose(result_np['joints'], result_nb['joints'], atol=1e-4):
                    print('      WARNING: Numba joints differ!')
                if return_vertices and not np.allclose(
                    result_np['vertices'], result_nb['vertices'], atol=1e-4
                ):
                    print('      WARNING: Numba vertices differ!')

                if has_cython:
                    result_cy = model_cy(**inputs)
                    if not np.allclose(result_np['joints'], result_cy['joints'], atol=1e-4):
                        print('      WARNING: Cython joints differ!')
                        diff = np.abs(result_np['joints'] - result_cy['joints']).max()
                        print(f'               Max diff: {diff}')
                    if return_vertices and not np.allclose(
                        result_np['vertices'], result_cy['vertices'], atol=1e-4
                    ):
                        print('      WARNING: Cython vertices differ!')
                        diff = np.abs(result_np['vertices'] - result_cy['vertices']).max()
                        print(f'               Max diff: {diff}')


if __name__ == '__main__':
    main()
