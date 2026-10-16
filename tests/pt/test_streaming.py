"""fit_streaming must reproduce BodyFitter.fit(share_beta=True) chunk-independently."""

import numpy as np
import pytest
import torch

import smplfitter.pt


def _synthetic(n, seed, same_shape=True):
    rng = np.random.RandomState(seed)
    pose = torch.tensor(rng.randn(n, 24 * 3) * 0.1, dtype=torch.float32)
    betas = torch.tensor(rng.randn(1, 10), dtype=torch.float32).expand(n, 10).contiguous()
    trans = torch.tensor(rng.randn(n, 3), dtype=torch.float32)
    return pose, betas, trans


@pytest.fixture(scope='module')
def model_fitter():
    model = smplfitter.pt.BodyModel('smpl', 'neutral', num_betas=10)
    return model, smplfitter.pt.BodyFitter(model)


def _targets(model, n=6, seed=7):
    pose, betas, trans = _synthetic(n, seed)
    with torch.no_grad():
        res = model(pose_rotvecs=pose, shape_betas=betas, trans=trans)
    return res['vertices'], res['joints'], (pose, betas, trans)


def _chunks_of(verts, joints, sizes):
    parts = []
    i = 0
    for s in sizes:
        parts.append((verts[i : i + s], joints[i : i + s]))
        i += s
    assert i == verts.shape[0]
    return lambda: iter(parts)


@pytest.mark.parametrize('num_iter', [1, 3])
def test_single_chunk_matches_batch(model_fitter, num_iter):
    model, fitter = model_fitter
    verts, joints, _ = _targets(model)
    keys = ['pose_rotvecs', 'shape_betas', 'trans', 'orientations']
    with torch.no_grad():
        ref = fitter.fit(
            verts,
            joints,
            num_iter=num_iter,
            beta_regularizer=0.5,
            share_beta=True,
            requested_keys=keys,
        )
        out = smplfitter.pt.fit_streaming(
            fitter,
            lambda: [(verts, joints)],
            num_iter=num_iter,
            beta_regularizer=0.5,
            requested_keys=keys,
        )
    for k in keys:
        assert out[k].shape == ref[k].shape, k
        torch.testing.assert_close(out[k], ref[k], rtol=1e-4, atol=1e-4, msg=k)


@pytest.mark.parametrize('sizes', [[2, 4], [1, 2, 3], [3, 3]])
def test_chunked_matches_batch(model_fitter, sizes):
    model, fitter = model_fitter
    verts, joints, _ = _targets(model)
    keys = ['pose_rotvecs', 'shape_betas', 'trans']
    with torch.no_grad():
        ref = fitter.fit(
            verts, joints, num_iter=3, beta_regularizer=0.5, share_beta=True, requested_keys=keys
        )
        out = smplfitter.pt.fit_streaming(
            fitter,
            _chunks_of(verts, joints, sizes),
            num_iter=3,
            beta_regularizer=0.5,
            requested_keys=keys,
        )
    for k in keys:
        torch.testing.assert_close(out[k], ref[k], rtol=1e-3, atol=1e-3, msg=k)
    # betas are one shared vector
    assert torch.allclose(out['shape_betas'], out['shape_betas'][:1].expand_as(out['shape_betas']))


def test_recovers_ground_truth(model_fitter):
    model, fitter = model_fitter
    verts, joints, (pose, betas, trans) = _targets(model, n=8, seed=3)
    with torch.no_grad():
        out = smplfitter.pt.fit_streaming(
            fitter,
            _chunks_of(verts, joints, [3, 5]),
            num_iter=3,
            beta_regularizer=0.0,
            requested_keys=['pose_rotvecs', 'shape_betas'],
        )
        rec = model(
            pose_rotvecs=out['pose_rotvecs'], shape_betas=out['shape_betas'], trans=out['trans']
        )
    err = torch.linalg.norm(rec['vertices'] - verts, dim=-1).mean().item()
    assert err < 5e-3, err


def test_vertices_only_and_weights(model_fitter):
    model, fitter = model_fitter
    verts, joints, _ = _targets(model)
    vw = torch.rand(verts.shape[:2]) + 0.5
    jw = torch.rand(joints.shape[:2]) + 0.5
    with torch.no_grad():
        ref = fitter.fit(
            verts,
            joints,
            vw,
            jw,
            num_iter=2,
            share_beta=True,
            requested_keys=['pose_rotvecs', 'shape_betas'],
        )
        out = smplfitter.pt.fit_streaming(
            fitter,
            lambda: [
                (verts[:2], joints[:2], vw[:2], jw[:2]),
                (verts[2:], joints[2:], vw[2:], jw[2:]),
            ],
            num_iter=2,
            requested_keys=['pose_rotvecs', 'shape_betas'],
        )
        ref_v = fitter.fit(
            verts, num_iter=2, share_beta=True, requested_keys=['pose_rotvecs', 'shape_betas']
        )
        out_v = smplfitter.pt.fit_streaming(
            fitter,
            lambda: [(verts[:3],), (verts[3:],)],
            num_iter=2,
            requested_keys=['pose_rotvecs', 'shape_betas'],
        )
    torch.testing.assert_close(out['shape_betas'], ref['shape_betas'], rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(out_v['shape_betas'], ref_v['shape_betas'], rtol=1e-3, atol=1e-3)


def test_inconsistent_replay_raises(model_fitter):
    model, fitter = model_fitter
    verts, joints, _ = _targets(model)
    calls = {'n': 0}

    def bad_chunks():
        calls['n'] += 1
        n = verts.shape[0] if calls['n'] == 1 else verts.shape[0] - 1
        return [(verts[:n], joints[:n])]

    with pytest.raises(ValueError), torch.no_grad():
        smplfitter.pt.fit_streaming(fitter, bad_chunks, num_iter=2)


def test_method_form_and_scripting(model_fitter):
    """fitter.fit_streaming(...) is the same as the function, and adding a Python-callable
    method must not break TorchScript scripting of BodyFitter (used by get_cached_fit_fn)."""
    model, fitter = model_fitter
    verts, joints, _ = _targets(model)
    with torch.no_grad():
        a = fitter.fit_streaming(
            _chunks_of(verts, joints, [2, 4]), num_iter=2, requested_keys=['shape_betas']
        )
        b = smplfitter.pt.fit_streaming(
            fitter, _chunks_of(verts, joints, [2, 4]), num_iter=2, requested_keys=['shape_betas']
        )
    torch.testing.assert_close(a['shape_betas'], b['shape_betas'])
    scripted = torch.jit.script(fitter)
    assert hasattr(scripted, 'fit') and hasattr(scripted, 'fit_with_known_pose')


def test_max_chunk_frames_splitting_is_transparent(model_fitter):
    """Internal slicing of oversized chunks must not change the result."""
    model, fitter = model_fitter
    verts, joints, _ = _targets(model, n=9, seed=5)
    keys = ['pose_rotvecs', 'shape_betas', 'trans']
    with torch.no_grad():
        ref = fitter.fit_streaming(
            lambda: [(verts, joints)], num_iter=2, requested_keys=keys, max_chunk_frames=None
        )
        out = fitter.fit_streaming(
            lambda: [(verts, joints)], num_iter=2, requested_keys=keys, max_chunk_frames=4
        )
    for k in keys:
        torch.testing.assert_close(out[k], ref[k], rtol=1e-4, atol=1e-4, msg=k)
