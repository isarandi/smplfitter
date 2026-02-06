"""Unified fitter tests that run across all backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from conftest import Backend


def _make_model_and_fitter(backend: 'Backend', num_betas: int = 10, enable_kid: bool = False):
    """Create BodyModel and BodyFitter with correct num_betas placement."""
    if backend.num_betas_on_model:
        model = backend.module.BodyModel('smpl', 'neutral', num_betas=num_betas)
        model = backend.prepare_model(model)
        fitter = backend.module.BodyFitter(model, enable_kid=enable_kid)
    else:
        model = backend.module.BodyModel('smpl', 'neutral')
        model = backend.prepare_model(model)
        fitter = backend.module.BodyFitter(model, enable_kid=enable_kid, num_betas=num_betas)

    fitter = backend.prepare_fitter(fitter)
    return model, fitter


class TestFitter:
    """Basic fitter tests across backends."""

    def test_fitter_basic(self, fitter_backend: Backend):
        """Test basic fitting recovers original pose/shape."""
        np.random.seed(42)
        model, fitter = _make_model_and_fitter(fitter_backend)

        pose_rotvecs = np.random.randn(2, 24 * 3).astype(np.float32) * 0.1
        shape_betas = np.random.randn(2, 10).astype(np.float32)
        trans = np.random.randn(2, 3).astype(np.float32)

        with fitter_backend.context():
            res = model(
                pose_rotvecs=fitter_backend.to_tensor(pose_rotvecs),
                shape_betas=fitter_backend.to_tensor(shape_betas),
                trans=fitter_backend.to_tensor(trans),
            )

            fit = fitter.fit(
                target_vertices=res['vertices'],
                target_joints=res['joints'],
                num_iter=3,
                beta_regularizer=0.0,
                requested_keys=['pose_rotvecs', 'shape_betas'],
            )

            res_fit = model(
                pose_rotvecs=fit['pose_rotvecs'],
                shape_betas=fit['shape_betas'],
                trans=fit['trans'],
            )

        verts = fitter_backend.to_numpy(res['vertices'])
        verts_fit = fitter_backend.to_numpy(res_fit['vertices'])
        joints = fitter_backend.to_numpy(res['joints'])
        joints_fit = fitter_backend.to_numpy(res_fit['joints'])

        verts_err = np.linalg.norm(verts - verts_fit, axis=-1)
        joints_err = np.linalg.norm(joints - joints_fit, axis=-1)

        mean_verts_err = np.mean(verts_err)
        mean_joints_err = np.mean(joints_err)
        assert mean_verts_err < 5e-3, f'Vertex error {mean_verts_err:.4f} >= 5e-3'
        assert mean_joints_err < 5e-3, f'Joint error {mean_joints_err:.4f} >= 5e-3'

    def test_fitter_share_beta(self, fitter_backend: Backend):
        """Test fitting with shared betas across batch."""
        np.random.seed(43)
        model, fitter = _make_model_and_fitter(fitter_backend)

        # Same shape, different poses
        pose_rotvecs = np.random.randn(4, 24 * 3).astype(np.float32) * 0.1
        shape_betas = np.random.randn(1, 10).astype(np.float32)
        shape_betas = np.broadcast_to(shape_betas, (4, 10)).copy()
        trans = np.random.randn(4, 3).astype(np.float32)

        with fitter_backend.context():
            res = model(
                pose_rotvecs=fitter_backend.to_tensor(pose_rotvecs),
                shape_betas=fitter_backend.to_tensor(shape_betas),
                trans=fitter_backend.to_tensor(trans),
            )

            fit = fitter.fit(
                target_vertices=res['vertices'],
                target_joints=res['joints'],
                num_iter=3,
                beta_regularizer=0.0,
                share_beta=True,
                requested_keys=['pose_rotvecs', 'shape_betas'],
            )

            res_fit = model(
                pose_rotvecs=fit['pose_rotvecs'],
                shape_betas=fit['shape_betas'],
                trans=fit['trans'],
            )

        # Check that all betas are the same
        fit_betas = fitter_backend.to_numpy(fit['shape_betas'])
        beta_std = np.std(fit_betas, axis=0)
        assert np.all(beta_std < 1e-6), 'Betas should be shared (identical) across batch'

        verts = fitter_backend.to_numpy(res['vertices'])
        verts_fit = fitter_backend.to_numpy(res_fit['vertices'])
        verts_err = np.linalg.norm(verts - verts_fit, axis=-1)
        mean_verts_err = np.mean(verts_err)
        assert mean_verts_err < 5e-3, f'Vertex error {mean_verts_err:.4f} >= 5e-3'


class TestFitterWithScale:
    """Scale estimation tests across backends."""

    def test_fitter_scale_target(self, fitter_backend: Backend):
        """Test fitting with scale_target=True."""
        np.random.seed(44)
        model, fitter = _make_model_and_fitter(fitter_backend)

        pose_rotvecs = np.random.randn(2, 24 * 3).astype(np.float32) * 0.1
        shape_betas = np.random.randn(2, 10).astype(np.float32)
        trans = np.random.randn(2, 3).astype(np.float32)

        scale_factor = 1.1
        expected_scale = 1.0 / scale_factor

        with fitter_backend.context():
            res = model(
                pose_rotvecs=fitter_backend.to_tensor(pose_rotvecs),
                shape_betas=fitter_backend.to_tensor(shape_betas),
                trans=fitter_backend.to_tensor(trans),
            )

            # Scale the target
            scaled_vertices = res['vertices'] * scale_factor
            scaled_joints = res['joints'] * scale_factor

            fit = fitter.fit(
                target_vertices=scaled_vertices,
                target_joints=scaled_joints,
                num_iter=3,
                beta_regularizer=0.0,
                scale_target=True,
                requested_keys=['pose_rotvecs', 'shape_betas', 'scale_corr'],
            )

            res_fit = model(
                pose_rotvecs=fit['pose_rotvecs'],
                shape_betas=fit['shape_betas'],
                trans=fit['trans'],
            )

        # scale_target=True means scale_corr is applied to target to match fit
        scale_corr = fitter_backend.to_numpy(fit['scale_corr'])
        scaled_verts = fitter_backend.to_numpy(scaled_vertices)
        scaled_jnts = fitter_backend.to_numpy(scaled_joints)
        verts_fit = fitter_backend.to_numpy(res_fit['vertices'])
        joints_fit = fitter_backend.to_numpy(res_fit['joints'])

        verts_err = np.linalg.norm(scaled_verts * scale_corr[:, None, None] - verts_fit, axis=-1)
        joints_err = np.linalg.norm(scaled_jnts * scale_corr[:, None, None] - joints_fit, axis=-1)

        mean_verts_err = np.mean(verts_err)
        mean_joints_err = np.mean(joints_err)
        assert mean_verts_err < 5e-3, f'Vertex error {mean_verts_err:.4f} >= 5e-3'
        assert mean_joints_err < 5e-3, f'Joint error {mean_joints_err:.4f} >= 5e-3'

        # Check scale correction is approximately correct
        mean_scale = np.mean(scale_corr)
        assert (
            abs(mean_scale - expected_scale) < 0.05
        ), f'Scale {mean_scale:.3f} != {expected_scale:.3f}'

    def test_fitter_scale_fit(self, fitter_backend: Backend):
        """Test fitting with scale_fit=True."""
        np.random.seed(45)
        model, fitter = _make_model_and_fitter(fitter_backend)

        pose_rotvecs = np.random.randn(2, 24 * 3).astype(np.float32) * 0.1
        shape_betas = np.random.randn(2, 10).astype(np.float32)
        trans = np.random.randn(2, 3).astype(np.float32)

        scale_factor = 1.1

        with fitter_backend.context():
            res = model(
                pose_rotvecs=fitter_backend.to_tensor(pose_rotvecs),
                shape_betas=fitter_backend.to_tensor(shape_betas),
                trans=fitter_backend.to_tensor(trans),
            )

            # Scale the target
            scaled_vertices = res['vertices'] * scale_factor
            scaled_joints = res['joints'] * scale_factor

            fit = fitter.fit(
                target_vertices=scaled_vertices,
                target_joints=scaled_joints,
                num_iter=5,
                beta_regularizer=0.0,
                scale_fit=True,
                requested_keys=['pose_rotvecs', 'shape_betas', 'scale_corr'],
            )

            res_fit = model(
                pose_rotvecs=fit['pose_rotvecs'],
                shape_betas=fit['shape_betas'],
                trans=fit['trans'],
            )

        # scale_fit=True means scale_corr is applied to fit to match target
        scale_corr = fitter_backend.to_numpy(fit['scale_corr'])
        scaled_verts = fitter_backend.to_numpy(scaled_vertices)
        scaled_jnts = fitter_backend.to_numpy(scaled_joints)
        verts_fit = fitter_backend.to_numpy(res_fit['vertices'])
        joints_fit = fitter_backend.to_numpy(res_fit['joints'])

        verts_err = np.linalg.norm(scaled_verts - verts_fit * scale_corr[:, None, None], axis=-1)
        joints_err = np.linalg.norm(scaled_jnts - joints_fit * scale_corr[:, None, None], axis=-1)

        mean_verts_err = np.mean(verts_err)
        mean_joints_err = np.mean(joints_err)
        assert mean_verts_err < 1e-2, f'Vertex error {mean_verts_err:.4f} >= 1e-2'
        assert mean_joints_err < 1e-2, f'Joint error {mean_joints_err:.4f} >= 1e-2'

        # Check scale correction is approximately correct
        mean_scale = np.mean(scale_corr)
        assert (
            abs(mean_scale - scale_factor) < 0.05
        ), f'Scale {mean_scale:.3f} != {scale_factor:.3f}'


class TestFitterKnownParams:
    """Tests for fitting with known pose or shape."""

    def test_fitter_with_known_shape(self, fitter_backend: Backend):
        """Test fitting when shape (betas) is known, only fit pose and translation."""
        np.random.seed(46)
        model, fitter = _make_model_and_fitter(fitter_backend)

        pose_rotvecs = np.random.randn(2, 24 * 3).astype(np.float32) * 0.1
        shape_betas = np.random.randn(2, 10).astype(np.float32)
        trans = np.random.randn(2, 3).astype(np.float32)

        shape_betas_t = fitter_backend.to_tensor(shape_betas)

        with fitter_backend.context():
            res = model(
                pose_rotvecs=fitter_backend.to_tensor(pose_rotvecs),
                shape_betas=shape_betas_t,
                trans=fitter_backend.to_tensor(trans),
            )

            fit = fitter.fit_with_known_shape(
                shape_betas=shape_betas_t,
                target_vertices=res['vertices'],
                target_joints=res['joints'],
                num_iter=3,
                requested_keys=['pose_rotvecs'],
            )

            res_fit = model(
                pose_rotvecs=fit['pose_rotvecs'],
                shape_betas=shape_betas_t,
                trans=fit['trans'],
            )

        verts = fitter_backend.to_numpy(res['vertices'])
        verts_fit = fitter_backend.to_numpy(res_fit['vertices'])
        joints = fitter_backend.to_numpy(res['joints'])
        joints_fit = fitter_backend.to_numpy(res_fit['joints'])

        verts_err = np.linalg.norm(verts - verts_fit, axis=-1)
        joints_err = np.linalg.norm(joints - joints_fit, axis=-1)

        mean_verts_err = np.mean(verts_err)
        mean_joints_err = np.mean(joints_err)
        assert mean_verts_err < 5e-3, f'Vertex error {mean_verts_err:.4f} >= 5e-3'
        assert mean_joints_err < 5e-3, f'Joint error {mean_joints_err:.4f} >= 5e-3'

    def test_fitter_with_known_pose(self, fitter_backend: Backend):
        """Test fitting when pose is known, only fit shape and translation."""
        np.random.seed(47)
        model, fitter = _make_model_and_fitter(fitter_backend)

        pose_rotvecs = np.random.randn(2, 24 * 3).astype(np.float32) * 0.1
        shape_betas = np.random.randn(2, 10).astype(np.float32)
        trans = np.random.randn(2, 3).astype(np.float32)

        pose_rotvecs_t = fitter_backend.to_tensor(pose_rotvecs)

        with fitter_backend.context():
            res = model(
                pose_rotvecs=pose_rotvecs_t,
                shape_betas=fitter_backend.to_tensor(shape_betas),
                trans=fitter_backend.to_tensor(trans),
            )

            fit = fitter.fit_with_known_pose(
                pose_rotvecs=pose_rotvecs_t,
                target_vertices=res['vertices'],
                target_joints=res['joints'],
                beta_regularizer=0.0,
                requested_keys=['shape_betas'],
            )

            res_fit = model(
                pose_rotvecs=pose_rotvecs_t,
                shape_betas=fit['shape_betas'],
                trans=fit['trans'],
            )

        verts = fitter_backend.to_numpy(res['vertices'])
        verts_fit = fitter_backend.to_numpy(res_fit['vertices'])
        joints = fitter_backend.to_numpy(res['joints'])
        joints_fit = fitter_backend.to_numpy(res_fit['joints'])

        verts_err = np.linalg.norm(verts - verts_fit, axis=-1)
        joints_err = np.linalg.norm(joints - joints_fit, axis=-1)

        mean_verts_err = np.mean(verts_err)
        mean_joints_err = np.mean(joints_err)
        assert mean_verts_err < 5e-3, f'Vertex error {mean_verts_err:.4f} >= 5e-3'
        assert mean_joints_err < 5e-3, f'Joint error {mean_joints_err:.4f} >= 5e-3'
