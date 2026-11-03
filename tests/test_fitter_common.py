"""Unified fitter tests that run across all backends."""

from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING, Optional

import numpy as np
import pytest

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


class TestShareBetaReference:
    """The shape reference (initial_shape_betas) must act identically with and without share_beta.

    Oracle: a batch of identical copies of one target. Sharing the betas across identical
    samples is the same least-squares problem as fitting them independently, so both
    paths must return the same betas. The regularizer weight is set away from 1 so that
    a solver which scales the reference by the weight (or drops it) is detected.
    """

    @staticmethod
    def _identical_batch_fit(backend: 'Backend', share_beta: bool, scale_fit: bool):
        np.random.seed(45)
        model, fitter = _make_model_and_fitter(backend)
        batch_size = 3
        pose_rotvecs = np.random.randn(1, 24 * 3).astype(np.float32) * 0.1
        shape_betas = np.random.randn(1, 10).astype(np.float32)
        trans = np.random.randn(1, 3).astype(np.float32)
        reference = np.random.randn(1, 10).astype(np.float32)
        pose_rotvecs, shape_betas, trans, reference = (
            np.broadcast_to(x, (batch_size,) + x.shape[1:]).copy()
            for x in (pose_rotvecs, shape_betas, trans, reference)
        )
        with backend.context():
            res = model(
                pose_rotvecs=backend.to_tensor(pose_rotvecs),
                shape_betas=backend.to_tensor(shape_betas),
                trans=backend.to_tensor(trans),
            )
            fit = fitter.fit(
                target_vertices=res['vertices'],
                target_joints=res['joints'],
                num_iter=1,
                beta_regularizer=0.3,
                beta_regularizer2=0.3,
                share_beta=share_beta,
                scale_fit=scale_fit,
                final_adjust_rots=False,
                initial_shape_betas=backend.to_tensor(reference),
                requested_keys=['shape_betas'],
            )
        return {k: backend.to_numpy(v) for k, v in fit.items()}

    def test_share_beta_reference(self, fitter_backend: Backend):
        indep = self._identical_batch_fit(fitter_backend, share_beta=False, scale_fit=False)
        shared = self._identical_batch_fit(fitter_backend, share_beta=True, scale_fit=False)
        diff = np.max(np.abs(indep['shape_betas'] - shared['shape_betas']))
        assert diff < 1e-3, f'Shared and independent betas differ by {diff:.2e}'

    def test_share_beta_reference_with_scale(self, fitter_backend: Backend):
        indep = self._identical_batch_fit(fitter_backend, share_beta=False, scale_fit=True)
        shared = self._identical_batch_fit(fitter_backend, share_beta=True, scale_fit=True)
        diff = np.max(np.abs(indep['shape_betas'] - shared['shape_betas']))
        assert diff < 1e-3, f'Shared and independent betas differ by {diff:.2e}'
        diff_scale = np.max(np.abs(indep['scale_corr'] - shared['scale_corr']))
        assert diff_scale < 1e-4, f'Shared and independent scale differ by {diff_scale:.2e}'


class TestKidTemplatePlaceholder:
    """An all-zero kid_template.npy (written by ``download --no-kid-template``) disables kids.

    The model must load and fit with enable_kid=False, and enable_kid=True must raise an
    error that points to the AGORA registration and the download command.
    """

    @pytest.fixture
    def placeholder_models(self, tmp_path, monkeypatch):
        from smplfitter.common import body_models_dir

        real_smpl = os.path.join(body_models_dir(), 'smpl')
        if not os.path.isdir(real_smpl):
            pytest.skip('SMPL body model files not available')
        smpl = tmp_path / 'smpl'
        smpl.mkdir()
        for name in os.listdir(real_smpl):
            if name != 'kid_template.npy':
                (smpl / name).symlink_to(os.path.join(os.path.abspath(real_smpl), name))
        np.save(smpl / 'kid_template.npy', np.zeros((6890, 3)))
        monkeypatch.setenv('SMPLFITTER_BODY_MODELS', str(tmp_path))

    def test_model_works_without_kid(self, fitter_backend: Backend, placeholder_models):
        model, fitter = _make_model_and_fitter(fitter_backend)
        assert not model._has_kid_template
        assert np.all(fitter_backend.to_numpy(model.kid_shapedir) == 0)

    def test_enable_kid_raises(self, fitter_backend: Backend, placeholder_models):
        with pytest.raises(ValueError, match='agora.is.tue.mpg.de.*--models smpl'):
            _make_model_and_fitter(fitter_backend, enable_kid=True)

    def test_real_template_is_detected(self, fitter_backend: Backend):
        model, _ = _make_model_and_fitter(fitter_backend, enable_kid=True)
        assert model._has_kid_template


# Backend agreement: all backends must give the same fit for the same inputs, up to float
# rounding. They implement the same algorithm independently, so this catches algorithmic
# drift (for example a step that one backend weights and another does not). The reference
# is the NumPy backend; fits are compared through the vertices of the fitted bodies.

# Rounding alone moves the fitted vertices by up to ~0.03 mm between backends; algorithmic
# differences moved them by millimeters.
TOLERANCE_M = 2e-4

SETTINGS = {
    'joints_weights_iter3': dict(joints=True, weights=True, num_iter=3),
    'joints_weights_iter1': dict(joints=True, weights=True, num_iter=1),
    'joints_noweights_noadjust': dict(
        joints=True, weights=False, num_iter=3, final_adjust_rots=False
    ),
    'nojoints_weights_iter3': dict(joints=False, weights=True, num_iter=3),
    'nojoints_noweights_iter1': dict(joints=False, weights=False, num_iter=1),
    'joints_weights_scale_target': dict(joints=True, weights=True, num_iter=2, scale_target=True),
}


@pytest.mark.parametrize('setting', list(SETTINGS))
def test_fit_matches_numpy_backend(fitter_backend: Backend, setting: str):
    if fitter_backend.name == 'np':
        pytest.skip('the NumPy backend is the reference')
    reference = _numpy_fit_vertices(setting)
    vertices = _fit_vertices(fitter_backend, setting)
    max_diff = np.max(np.linalg.norm(vertices - reference, axis=-1))
    assert (
        max_diff < TOLERANCE_M
    ), f'{fitter_backend.name}: max vertex diff {max_diff * 1000:.3f} mm'


@pytest.mark.parametrize('batch_size', [2, 3, 6])
def test_fit_does_not_depend_on_batch_size(fitter_backend: Backend, batch_size: int):
    """Each batch element gets the fit it would get alone (up to rounding).

    Batched linear algebra kernels can fail only above a certain batch or matrix size, which
    a comparison of one batch size across backends does not necessarily reveal.
    """
    setting = 'joints_weights_iter3'
    batched = _fit_vertices(fitter_backend, setting, batch_size)
    for i in range(batch_size):
        alone = _fit_vertices(fitter_backend, setting, 1, first=i)
        max_diff = np.max(np.linalg.norm(batched[i] - alone[0], axis=-1))
        assert (
            max_diff < TOLERANCE_M
        ), f'{fitter_backend.name}: element {i} differs by {max_diff * 1000:.3f} mm'


@functools.lru_cache
def _numpy_fit_vertices(setting: str) -> np.ndarray:
    from conftest import _make_np_backend

    return _fit_vertices(_make_np_backend(), setting)


def _fit_vertices(
    backend: Backend, setting: str, batch_size: Optional[int] = None, first: int = 0
) -> np.ndarray:
    """Vertices (B, V, 3) of the bodies fitted by the backend, posed by the NumPy model.

    Fits ``batch_size`` of the test targets starting at ``first``, or all of them if None.
    """
    import smplfitter.np

    s = dict(SETTINGS[setting])
    use_joints = s.pop('joints')
    use_weights = s.pop('weights')
    end = None if batch_size is None else first + batch_size
    verts, joints, vertex_weights, joint_weights = (x[first:end] for x in _targets())
    model = backend.prepare_model(backend.module.BodyModel('smpl', 'neutral', num_betas=10))
    fitter = backend.prepare_fitter(backend.module.BodyFitter(model))
    t = backend.to_tensor
    with backend.context():
        fit = fitter.fit(
            t(verts),
            t(joints) if use_joints else None,
            t(vertex_weights) if use_weights else None,
            t(joint_weights) if use_weights and use_joints else None,
            beta_regularizer=1.0,
            requested_keys=['pose_rotvecs', 'shape_betas'],
            **s,
        )
    fit = {k: backend.to_numpy(v) for k, v in fit.items() if v is not None}
    np_model = smplfitter.np.BodyModel('smpl', 'neutral', num_betas=10)
    vertices = np_model(
        pose_rotvecs=fit['pose_rotvecs'], shape_betas=fit['shape_betas'], trans=fit['trans']
    )['vertices']
    if 'scale_corr' in fit:
        # scale_target: the fit matches the target scaled by scale_corr
        vertices = vertices / fit['scale_corr'][:, None, None]
    return vertices


@functools.lru_cache
def _targets():
    """Noisy SMPL vertices and joints of varied poses, with random weights."""
    import smplfitter.np

    rng = np.random.default_rng(0)
    batch_size = 6
    model = smplfitter.np.BodyModel('smpl', 'neutral', num_betas=10)
    res = model(
        pose_rotvecs=rng.normal(0, 0.4, (batch_size, 72)).astype(np.float32),
        shape_betas=rng.normal(0, 1, (batch_size, 10)).astype(np.float32),
        trans=rng.normal(0, 1, (batch_size, 3)).astype(np.float32),
    )
    verts = res['vertices'] + rng.normal(0, 0.02, res['vertices'].shape)
    joints = res['joints'] + rng.normal(0, 0.02, res['joints'].shape)
    vertex_weights = rng.uniform(0.2, 2, verts.shape[:2])
    joint_weights = rng.uniform(0.2, 2, joints.shape[:2])
    return tuple(x.astype(np.float32) for x in (verts, joints, vertex_weights, joint_weights))
