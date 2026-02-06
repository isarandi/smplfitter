"""Cross-backend BodyConverter tests using random SMPL parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from conftest import Backend


def _make_random_smpl_params(batch_size: int = 33, num_betas: int = 10, seed: int = 42):
    """Generate random SMPL parameters (mild poses, moderate shapes)."""
    rng = np.random.RandomState(seed)
    pose_rotvecs = (rng.uniform(-0.1, 0.1, (batch_size, 24 * 3))).astype(np.float32)
    shape_betas = (rng.uniform(-1, 1, (batch_size, num_betas))).astype(np.float32)
    trans = (rng.uniform(-1, 1, (batch_size, 3))).astype(np.float32)
    return pose_rotvecs, shape_betas, trans


def _make_converter(backend, model_in, model_out):
    """Create a BodyConverter, moving to correct device for PT."""
    conv = backend.module.BodyConverter(model_in, model_out)
    return backend.prepare_model(conv)


class TestConverterRoundtrip:
    """Test SMPL -> SMPLX -> SMPL roundtrip conversion."""

    def test_roundtrip_vertices(self, backend: Backend):
        """Convert SMPL->SMPLX->SMPL and check vertex error."""
        pose_rotvecs, shape_betas, trans = _make_random_smpl_params()

        model_smpl = backend.prepare_model(
            backend.module.BodyModel('smpl', 'neutral', num_betas=10))
        model_smplx = backend.prepare_model(
            backend.module.BodyModel('smplx', 'neutral', num_betas=10))

        converter = _make_converter(backend, model_smpl, model_smplx)
        reverse_converter = _make_converter(backend, model_smplx, model_smpl)

        with backend.context():
            res_smpl = model_smpl(
                pose_rotvecs=backend.to_tensor(pose_rotvecs),
                shape_betas=backend.to_tensor(shape_betas),
                trans=backend.to_tensor(trans),
            )

            conv = converter.convert(
                pose_rotvecs=backend.to_tensor(pose_rotvecs),
                shape_betas=backend.to_tensor(shape_betas),
                trans=backend.to_tensor(trans),
                num_iter=1,
            )

            res_smplx = model_smplx(
                pose_rotvecs=conv['pose_rotvecs'],
                shape_betas=conv['shape_betas'],
                trans=conv['trans'],
            )

            # Convert SMPLX vertices back to SMPL topology
            back_verts = reverse_converter.convert_vertices(res_smplx['vertices'])

        orig_verts = backend.to_numpy(res_smpl['vertices'])
        roundtrip_verts = backend.to_numpy(back_verts)

        err = np.mean(np.linalg.norm(orig_verts - roundtrip_verts, axis=-1))
        assert err < 2e-2, f'Roundtrip vertex error {err * 1000:.1f} mm >= 20 mm'


class TestConverterVertexAccuracy:
    """Test that BodyConverter.convert produces vertices close to the topology-converted input."""

    def test_smpl_to_smplx(self, backend: Backend):
        """Convert SMPL params to SMPLX, compare fit vertices vs topology-converted vertices."""
        pose_rotvecs, shape_betas, trans = _make_random_smpl_params()

        model_smpl = backend.prepare_model(
            backend.module.BodyModel('smpl', 'neutral', num_betas=10))
        model_smplx = backend.prepare_model(
            backend.module.BodyModel('smplx', 'neutral', num_betas=10))

        converter = _make_converter(backend, model_smpl, model_smplx)

        with backend.context():
            res_smpl = model_smpl(
                pose_rotvecs=backend.to_tensor(pose_rotvecs),
                shape_betas=backend.to_tensor(shape_betas),
                trans=backend.to_tensor(trans),
            )

            # Topology conversion (ground truth for SMPLX vertex positions)
            target_smplx_verts = converter.convert_vertices(res_smpl['vertices'])

            # Parameter conversion (fit SMPLX params, then forward)
            conv = converter.convert(
                pose_rotvecs=backend.to_tensor(pose_rotvecs),
                shape_betas=backend.to_tensor(shape_betas),
                trans=backend.to_tensor(trans),
                num_iter=1,
            )

            res_smplx = model_smplx(
                pose_rotvecs=conv['pose_rotvecs'],
                shape_betas=conv['shape_betas'],
                trans=conv['trans'],
            )

        target = backend.to_numpy(target_smplx_verts)
        fitted = backend.to_numpy(res_smplx['vertices'])

        err = np.mean(np.linalg.norm(target - fitted, axis=-1))
        # Parametric approximation error: SMPLX can't perfectly represent SMPL
        assert err < 7e-3, f'SMPL->SMPLX vertex error {err * 1000:.2f} mm >= 7 mm'