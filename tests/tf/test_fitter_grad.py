"""Gradient tests for the TensorFlow fitter (mirror of tests/pt/test_fitter_grad.py)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

tf = pytest.importorskip('tensorflow')
smpl_tf = pytest.importorskip('smplfitter.tf')

import smplfitter.np as smpl_np  # noqa: E402  (import must follow importorskip)


def _make_targets(model_name: str, batch_size: int, seed: int = 0):
    np.random.seed(seed)
    bm_t = smpl_tf.BodyModel(model_name=model_name, num_betas=10)
    bm_n = smpl_np.BodyModel(model_name=model_name, num_betas=10)
    pose = np.random.randn(batch_size, bm_n.num_joints * 3).astype(np.float32) * 0.1
    shape = np.random.randn(batch_size, 10).astype(np.float32) * 0.5
    trans = np.random.randn(batch_size, 3).astype(np.float32)
    out = bm_n(pose_rotvecs=pose, shape_betas=shape, trans=trans)
    return bm_t, out['vertices'].astype(np.float32), out['joints'].astype(np.float32)


def _loss(fit):
    return sum(tf.reduce_sum(fit[k] ** 2) for k in ('pose_rotvecs', 'shape_betas', 'trans'))


@pytest.mark.parametrize('model_name', ['smpl', 'smplx'])
@pytest.mark.parametrize('num_iter', [1, 3])
def test_fitter_grad_finite(model_name, num_iter):
    """Backprop through the fit must yield finite, non-zero gradients."""
    bm, target_v, target_j = _make_targets(model_name, batch_size=2)
    fitter = smpl_tf.BodyFitter(bm)

    tv = tf.Variable(target_v)
    tj = tf.Variable(target_j)
    with tf.GradientTape() as tape:
        fit = fitter.fit(
            target_vertices=tv,
            target_joints=tj,
            num_iter=num_iter,
            beta_regularizer=1.0,
            requested_keys=['pose_rotvecs', 'shape_betas', 'trans'],
        )
        loss = _loss(fit)
    gv, gj = tape.gradient(loss, [tv, tj])

    for name, g in [('target_vertices', gv), ('target_joints', gj)]:
        assert g is not None, f'{name} grad is None'
        arr = g.numpy()
        assert np.isfinite(arr).all(), f'{name} grad has NaN/Inf'
        assert np.abs(arr).max() > 0, f'{name} grad is all-zero'


@pytest.mark.parametrize('model_name', ['smpl', 'smplx'])
@pytest.mark.parametrize('num_iter', [1, 3])
def test_fitter_grad_of_batch_matches_single_elements(model_name, num_iter):
    """The gradient of a batched fit must equal the gradients of the elements fitted alone.

    A failing batched kernel in the backward pass can give finite but wrong gradients, which
    only a comparison of values reveals. The loss is linear in the rotation vectors, so that
    identity rotations (the SMPL toes follow the feet) also receive a non-zero upstream gradient.
    """
    batch_size = 3
    bm, target_v, target_j = _make_targets(model_name, batch_size=batch_size)
    fitter = smpl_tf.BodyFitter(bm)
    loss_weights = np.random.default_rng(0).normal(size=(batch_size, bm.num_joints * 3))
    loss_weights = tf.constant(loss_weights.astype(np.float32))

    def gradient(indices):
        tv = tf.Variable(target_v[indices])
        tj = tf.Variable(target_j[indices])
        with tf.GradientTape() as tape:
            fit = fitter.fit(
                target_vertices=tv,
                target_joints=tj,
                num_iter=num_iter,
                beta_regularizer=1.0,
                requested_keys=['pose_rotvecs'],
            )
            loss = tf.reduce_sum(fit['pose_rotvecs'] * tf.gather(loss_weights, indices))
        return tape.gradient(loss, tv).numpy()

    batched = gradient(list(range(batch_size)))
    for i in range(batch_size):
        alone = gradient([i])[0]
        rel = np.max(np.abs(batched[i] - alone)) / np.max(np.abs(alone))
        assert rel < 1e-2, f'element {i}: relative gradient difference {rel:.2e}'


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_fitter_grad_vs_finite_diff(seed):
    """Directional finite-difference check on the SMPL fit gradient.

    Probes a random unit-direction so the FD averages over many coordinates
    (more stable than per-coordinate FD in float32). The integrated swing-twist
    solver is the default, so this runs directly on ``BodyFitter``.
    """
    bm, target_v, target_j = _make_targets('smpl', batch_size=1, seed=seed)
    fitter = smpl_tf.BodyFitter(bm)

    # Two iterations: with one, the target joints enter only the shape solve and the final
    # adjustment (the first rotation estimate uses regressed joints), so the directional
    # derivative is small and the finite difference is dominated by float32 rounding.
    def loss_fn(tv_np, tj_np):
        fit = fitter.fit(
            target_vertices=tf.constant(tv_np),
            target_joints=tf.constant(tj_np),
            num_iter=2,
            beta_regularizer=1.0,
            requested_keys=['pose_rotvecs', 'shape_betas', 'trans'],
        )
        return _loss(fit)

    tv = tf.Variable(target_v)
    tj = tf.Variable(target_j)
    with tf.GradientTape() as tape:
        fit = fitter.fit(
            target_vertices=tv,
            target_joints=tj,
            num_iter=2,
            beta_regularizer=1.0,
            requested_keys=['pose_rotvecs', 'shape_betas', 'trans'],
        )
        loss = _loss(fit)
    gv, gj = tape.gradient(loss, [tv, tj])

    g = torch.Generator().manual_seed(seed + 100)
    dv_t = torch.randn(target_v.shape, generator=g)
    dj_t = torch.randn(target_j.shape, generator=g)
    dv = (dv_t / dv_t.norm()).numpy().astype(np.float32)
    dj = (dj_t / dj_t.norm()).numpy().astype(np.float32)

    ag_dir = float(tf.reduce_sum(gv * dv).numpy()) + float(tf.reduce_sum(gj * dj).numpy())

    eps = 1e-2
    lp = float(
        loss_fn(
            (target_v + eps * dv).astype(np.float32),
            (target_j + eps * dj).astype(np.float32),
        ).numpy()
    )
    lm = float(
        loss_fn(
            (target_v - eps * dv).astype(np.float32),
            (target_j - eps * dj).astype(np.float32),
        ).numpy()
    )
    fd_dir = (lp - lm) / (2 * eps)

    denom = max(abs(ag_dir), abs(fd_dir), 1e-3)
    rel = abs(ag_dir - fd_dir) / denom
    assert rel < 5e-2, f'seed={seed}: autograd={ag_dir:.4e}, fd={fd_dir:.4e}, rel={rel:.3e}'
