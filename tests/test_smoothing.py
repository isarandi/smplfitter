"""Tests of the online smoother on synthetic motion with noise matched to per-frame estimators."""

from __future__ import annotations

import numpy as np
import pytest

import smplfitter.np
from smplfitter.smoothing import CausalPoseSmoother

NUM_FRAMES = 30


@pytest.fixture(params=['nb', 'np'], ids=lambda x: x)
def body_model(request):
    if request.param == 'nb':
        nb = pytest.importorskip('smplfitter.nb')
        return nb.BodyModel('smpl', 'neutral', num_betas=10)
    return smplfitter.np.BodyModel('smpl', 'neutral', num_betas=10)


def test_static_camera_reduces_jitter_and_error(body_model):
    truth = _motion()
    cameras = np.zeros((NUM_FRAMES, 3), np.float32)
    frames = _noisy_frames(truth, cameras)
    outputs = _run(CausalPoseSmoother(body_model), frames)
    raw = np.stack([vertices for vertices, _ in frames])
    # Measured with this noise: 0.18 of the raw jitter and 0.92 of the raw error. The filter lags
    # behind the motion, so the error drops much less than the jitter.
    assert _jitter(outputs, truth) < 0.25 * _jitter(raw, truth)
    assert _error(outputs, truth) < 0.95 * _error(raw, truth)


def test_moving_camera_centers_keep_a_jerky_camera_out_of_the_motion(body_model):
    truth = _motion()
    cameras = _jerky_cameras()
    frames = _noisy_frames(truth, cameras)
    with_centers = _run(CausalPoseSmoother(body_model), frames, cameras)
    fixed_center = _run(CausalPoseSmoother(body_model, camera_center=cameras.mean(axis=0)), frames)
    raw = np.stack([vertices for vertices, _ in frames])
    # Measured with this noise: 0.19 of the raw jitter, 0.79 of the jitter with a fixed center,
    # and 0.75 of the raw error.
    assert _jitter(with_centers, truth) < 0.25 * _jitter(raw, truth)
    assert _jitter(with_centers, truth) < 0.85 * _jitter(fixed_center, truth)
    assert _error(with_centers, truth) < 0.8 * _error(raw, truth)


def test_clean_input_stays_close_to_the_true_motion(body_model):
    truth = _motion()
    frames = [(truth['vertices'][i], truth['joints'][i]) for i in range(NUM_FRAMES)]
    outputs = _run(CausalPoseSmoother(body_model), frames)
    # The causal filter lags behind the motion. Measured: 34 mm for this motion.
    assert _error(outputs, truth) < 40e-3


def test_single_frame_outlier_is_mostly_ignored(body_model):
    truth = _motion()
    frames = [(truth['vertices'][i], truth['joints'][i]) for i in range(NUM_FRAMES)]
    clean = _run(CausalPoseSmoother(body_model), frames)
    index = NUM_FRAMES // 2
    outlier = 0.5
    bad_frames = list(frames)
    bad_frames[index] = (frames[index][0] + outlier, frames[index][1] + outlier)
    with_outlier = _run(CausalPoseSmoother(body_model), bad_frames)
    full_shift = np.linalg.norm([outlier] * 3)
    moved = np.mean(np.linalg.norm(with_outlier - clean, axis=-1), axis=1) / full_shift
    # A plain weighted mean would move by the kernel share of the outlier frame: 0.50 at that
    # frame, and 0.34 at the next one. Measured here: 0.36 and 0.075.
    assert moved[index] < 0.42
    assert moved[index + 1] < 0.1


def test_past_results_do_not_depend_on_future_frames(body_model):
    truth = _motion()
    frames = [(truth['vertices'][i], truth['joints'][i]) for i in range(8)]
    other_future = frames[:4] + [(v + 0.3, j + 0.3) for v, j in frames[4:]]
    first = _run(CausalPoseSmoother(body_model), frames)
    second = _run(CausalPoseSmoother(body_model), other_future)
    np.testing.assert_array_equal(first[:4], second[:4])
    assert not np.array_equal(first[4:], second[4:])


def test_reset_restarts_the_stream(body_model):
    truth = _motion()
    frames = [(truth['vertices'][i], truth['joints'][i]) for i in range(4)]
    fresh = _run(CausalPoseSmoother(body_model), frames)
    smoother = CausalPoseSmoother(body_model)
    _run(smoother, [(v + 1.0, j + 1.0) for v, j in frames])
    smoother.reset()
    np.testing.assert_array_equal(_run(smoother, frames), fresh)


def test_non_finite_frame_gives_nan_and_then_recovers(body_model):
    truth = _motion()
    frames = [(truth['vertices'][i], truth['joints'][i]) for i in range(16)]
    index = 3
    bad_joints = frames[index][1].copy()
    bad_joints[3, 1] = np.nan
    frames[index] = (frames[index][0], bad_joints)
    outputs = _run(CausalPoseSmoother(body_model, window=5), frames)
    assert np.all(np.isfinite(outputs[:index]))
    assert np.all(np.isnan(outputs[index]))
    # It leaves the window of raw frames and then that of the smoothed frames.
    assert np.all(np.isfinite(outputs[index + 2 * 5 :]))


def test_returned_values_are_consistent_with_the_model(body_model):
    truth = _motion()
    smoother = CausalPoseSmoother(body_model)
    for i in range(7):
        result = smoother.update(
            truth['vertices'][i],
            truth['joints'][i],
            *_uncertainties(truth),
        )
    assert result['pose_rotvecs'].shape == (72,)
    assert result['shape_betas'].shape == (10,)
    assert result['trans'].shape == (3,)
    posed = body_model(
        pose_rotvecs=result['pose_rotvecs'][np.newaxis],
        shape_betas=result['shape_betas'][np.newaxis],
        trans=result['trans'][np.newaxis],
    )
    np.testing.assert_allclose(posed['vertices'][0], result['vertices'], atol=1e-4)
    np.testing.assert_allclose(posed['joints'][0], result['joints'], atol=1e-4)


@pytest.mark.parametrize(
    'kwargs',
    [
        dict(window=7),
        dict(window=1),
        dict(fit_vertex_subset_size=None),
        dict(do_scale=False),
        dict(do_root=False),
        dict(do_offset=False),
    ],
    ids=lambda x: next(iter(x)) + '=' + str(next(iter(x.values()))),
)
def test_options_keep_clean_input_close(body_model, kwargs):
    truth = _motion()
    frames = [(truth['vertices'][i], truth['joints'][i]) for i in range(8)]
    outputs = _run(CausalPoseSmoother(body_model, **kwargs), frames)
    errors = np.linalg.norm(outputs - truth['vertices'][:8], axis=-1)
    # Measured: at most 32 mm, for the longest window.
    assert np.mean(errors) < 36e-3


def test_smplx():
    nb = pytest.importorskip('smplfitter.nb')
    model = nb.BodyModel('smplx', 'neutral', num_betas=10)
    rng = np.random.default_rng(0)
    pose = rng.normal(0, 0.05, (4, model.num_joints * 3)).astype(np.float32)
    trans = np.tile(np.array([[0, 0, 3]], np.float32), (4, 1))
    posed = model(pose_rotvecs=pose, shape_betas=np.zeros((4, 10), np.float32), trans=trans)
    smoother = CausalPoseSmoother(model)
    for i in range(4):
        result = smoother.update(
            posed['vertices'][i],
            posed['joints'][i],
            np.ones(model.num_vertices, np.float32),
            np.ones(model.num_joints, np.float32),
        )
    assert result['vertices'].shape == (model.num_vertices, 3)
    # The pose here changes randomly per frame, which a temporal filter averages. Measured: 24 mm.
    error = np.mean(np.linalg.norm(result['vertices'] - posed['vertices'][3], axis=-1))
    assert error < 30e-3


def test_other_backends_are_rejected():
    pt = pytest.importorskip('smplfitter.pt')
    with pytest.raises(ValueError, match='smplfitter.np or smplfitter.nb'):
        CausalPoseSmoother(pt.BodyModel('smpl', 'neutral', num_betas=10))


def _run(smoother, frames, cameras=None):
    """The output vertices (T, V, 3) of the smoother over the frames."""
    outputs = []
    for i, (vertices, joints) in enumerate(frames):
        center = None if cameras is None else cameras[i]
        vertex_uncertainties = np.ones(len(vertices), np.float32)
        joint_uncertainties = np.ones(len(joints), np.float32)
        result = smoother.update(
            vertices, joints, vertex_uncertainties, joint_uncertainties, center
        )
        outputs.append(result['vertices'])
    return np.stack(outputs)


def _uncertainties(truth):
    return (
        np.ones(truth['vertices'].shape[1], np.float32),
        np.ones(truth['joints'].shape[1], np.float32),
    )


def _motion():
    """Vertices and joints of a body that walks in front of the camera and swings its arms."""
    time = np.arange(NUM_FRAMES, dtype=np.float32)
    pose = np.zeros((NUM_FRAMES, 72), np.float32)
    pose[:, 3 * 1] = 0.3 * np.sin(0.1 * time)
    pose[:, 3 * 2] = -0.3 * np.sin(0.1 * time)
    pose[:, 3 * 16 + 2] = -0.8 + 0.5 * np.sin(0.15 * time)
    pose[:, 3 * 17 + 2] = 0.8 + 0.5 * np.sin(0.13 * time)
    betas = np.random.default_rng(0).normal(0, 0.7, 10).astype(np.float32)
    trans = np.stack(
        [
            0.6 * np.sin(0.05 * time),
            0.05 * np.sin(0.11 * time),
            3.0 + 0.4 * np.sin(0.03 * time),
        ],
        axis=1,
    ).astype(np.float32)
    model = smplfitter.np.BodyModel('smpl', 'neutral', num_betas=10)
    return model(pose_rotvecs=pose, shape_betas=np.tile(betas, (NUM_FRAMES, 1)), trans=trans)


def _jerky_cameras():
    """A camera that travels, with jumps every few frames and jitter on every frame."""
    rng = np.random.default_rng(2)
    time = np.arange(NUM_FRAMES, dtype=np.float32)
    jumps = np.repeat(rng.normal(0, 0.25, (NUM_FRAMES // 5 + 1, 3)), 5, axis=0)[:NUM_FRAMES]
    travel = (time[:, np.newaxis] / NUM_FRAMES) * np.array([3.0, 0.3, 1.5])
    cameras = np.array([0.0, -1.2, 0.0]) + jumps + travel + rng.normal(0, 0.03, (NUM_FRAMES, 3))
    return cameras.astype(np.float32)


def _noisy_frames(truth, cameras):
    """Scale ambiguity about the camera, depth noise along the ray, and noise per point."""
    rng = np.random.default_rng(1)
    frames = []
    for i in range(NUM_FRAMES):
        center = cameras[i]
        vertices, joints = truth['vertices'][i], truth['joints'][i]
        ray = joints[0] - center
        ray = ray / np.linalg.norm(ray)
        scale = 1.0 + rng.normal(0, 0.03)
        depth = rng.normal(0, 0.04) * ray
        frames.append(
            (
                (
                    (vertices - center) * scale
                    + center
                    + depth
                    + rng.normal(0, 0.01, vertices.shape)
                ).astype(np.float32),
                (
                    (joints - center) * scale + center + depth + rng.normal(0, 0.01, joints.shape)
                ).astype(np.float32),
            )
        )
    return frames


def _jitter(vertices, truth):
    """Mean deviation of the frame-to-frame motion from the true one."""
    motion = np.diff(vertices, axis=0) - np.diff(truth['vertices'], axis=0)
    return np.mean(np.linalg.norm(motion, axis=-1))


def _error(vertices, truth):
    return np.mean(np.linalg.norm(vertices - truth['vertices'], axis=-1))
