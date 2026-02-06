"""Precompile all Numba functions to avoid JIT overhead on first call.

Run with: python -m smplfitter.nb.precompile
"""

from __future__ import annotations

import numpy as np


def precompile():
    """Precompile all Numba functions to avoid JIT overhead on first call."""
    from . import BodyFitter, get_cached_body_model, rotation

    print('Precompiling smplfitter.nb Numba functions...')

    bm = get_cached_body_model()
    num_joints = bm.num_joints
    num_betas = bm.num_betas

    # Test different batch sizes to compile for various shapes
    for batch_size in [1, 4]:
        pose_rotvecs = np.random.randn(batch_size, num_joints * 3).astype(np.float32) * 0.1
        shape_betas = np.random.randn(batch_size, num_betas).astype(np.float32) * 0.5
        trans = np.random.randn(batch_size, 3).astype(np.float32)
        kid_factor = np.zeros((1,), np.float32)

        # Forward pass with pose_rotvecs
        bm(pose_rotvecs=pose_rotvecs, shape_betas=shape_betas, trans=trans, kid_factor=kid_factor)
        bm(pose_rotvecs=pose_rotvecs, shape_betas=shape_betas, trans=trans, return_vertices=False)

        # Forward pass with rel_rotmats
        rel_rotmats = rotation.rotvec2mat(pose_rotvecs.reshape(batch_size, num_joints, 3))
        bm(rel_rotmats=rel_rotmats, shape_betas=shape_betas, trans=trans)

        # Forward pass with glob_rotmats
        result = bm(rel_rotmats=rel_rotmats, shape_betas=shape_betas, trans=trans)
        bm(glob_rotmats=result['orientations'], shape_betas=shape_betas, trans=trans)

    # Rotation functions
    rotvec = np.random.randn(4, 3).astype(np.float32)
    rotmat = rotation.rotvec2mat(rotvec)
    rotation.mat2rotvec(rotmat)

    # Kabsch
    X = np.random.randn(4, 10, 3).astype(np.float32)
    Y = np.random.randn(4, 10, 3).astype(np.float32)
    rotation.kabsch(X, Y)

    # BodyFitter
    print('Precompiling BodyFitter...')
    fitter = BodyFitter(bm)

    for batch_size in [1, 4]:
        pose_rotvecs = np.random.randn(batch_size, num_joints * 3).astype(np.float32) * 0.1
        shape_betas = np.random.randn(batch_size, num_betas).astype(np.float32) * 0.5
        trans = np.random.randn(batch_size, 3).astype(np.float32)

        result = bm(pose_rotvecs=pose_rotvecs, shape_betas=shape_betas, trans=trans)
        vertices = result['vertices']
        joints = result['joints']

        # Basic fit with vertices and joints
        fitter.fit(
            target_vertices=vertices,
            target_joints=joints,
            num_iter=1,
            requested_keys=('pose_rotvecs', 'shape_betas'),
        )

        # Fit with vertices only
        fitter.fit(
            target_vertices=vertices,
            num_iter=1,
            requested_keys=('pose_rotvecs', 'shape_betas'),
        )

        # Fit with scale_target
        fitter.fit(
            target_vertices=vertices,
            target_joints=joints,
            num_iter=1,
            scale_target=True,
            requested_keys=('pose_rotvecs', 'shape_betas', 'scale_corr'),
        )

        # Fit with scale_fit
        fitter.fit(
            target_vertices=vertices,
            target_joints=joints,
            num_iter=1,
            scale_fit=True,
            requested_keys=('pose_rotvecs', 'shape_betas', 'scale_corr'),
        )

        # Fit with share_beta
        fitter.fit(
            target_vertices=vertices,
            target_joints=joints,
            num_iter=1,
            share_beta=True,
            requested_keys=('pose_rotvecs', 'shape_betas'),
        )

    print('Precompilation complete.')


if __name__ == '__main__':
    precompile()
