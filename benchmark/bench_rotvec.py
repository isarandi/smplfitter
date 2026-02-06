"""Benchmark rotvec2mat: NumPy vs Numba vs Cython."""

import numpy as np
import time


def benchmark(func, rotvec, n_warmup=3, n_iter=100):
    """Run benchmark and return mean time in ms."""
    # Warmup
    for _ in range(n_warmup):
        func(rotvec)

    # Timed runs
    start = time.perf_counter()
    for _ in range(n_iter):
        func(rotvec)
    elapsed = time.perf_counter() - start
    return elapsed / n_iter * 1000  # ms


def main():
    from smplfitter.np.rotation import rotvec2mat as rotvec2mat_np
    from smplfitter.nb.rotation import rotvec2mat as rotvec2mat_nb

    try:
        from smplfitter.cy.rotation import rotvec2mat as rotvec2mat_cy

        has_cython = True
    except ImportError:
        print('Cython not built. Run: python setup_cy.py build_ext --inplace')
        has_cython = False

    print('Benchmarking rotvec2mat')
    print('=' * 60)

    for batch_size in [1, 8, 32, 128, 512]:
        for n_joints in [24, 55]:  # SMPL, SMPLX
            rotvec = np.random.randn(batch_size, n_joints, 3).astype(np.float32) * 0.1

            time_np = benchmark(rotvec2mat_np, rotvec)
            time_nb = benchmark(rotvec2mat_nb, rotvec)

            print(f'\nbatch={batch_size:3d}, joints={n_joints:2d}')
            print(f'  NumPy:  {time_np:8.3f} ms')
            print(f'  Numba:  {time_nb:8.3f} ms  ({time_np/time_nb:5.1f}x vs NumPy)')

            if has_cython:
                time_cy = benchmark(rotvec2mat_cy, rotvec)
                print(
                    f'  Cython: {time_cy:8.3f} ms  ({time_np/time_cy:5.1f}x vs NumPy, {time_nb/time_cy:5.2f}x vs Numba)'
                )

            # Verify correctness
            result_np = rotvec2mat_np(rotvec)
            result_nb = rotvec2mat_nb(rotvec)
            if not np.allclose(result_np, result_nb, atol=1e-5):
                print('  WARNING: Numba result differs!')

            if has_cython:
                result_cy = rotvec2mat_cy(rotvec)
                if not np.allclose(result_np, result_cy, atol=1e-5):
                    print('  WARNING: Cython result differs!')


if __name__ == '__main__':
    main()
