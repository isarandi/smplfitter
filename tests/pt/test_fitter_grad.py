"""Gradient tests for the PyTorch fitter.

These guard against regressions where in-place ops or degenerate-SVD vjps
make backprop through the fit produce None / NaN / Inf gradients.
"""

from __future__ import annotations

import pytest
import torch

import smplfitter.pt as smpl_pt


def _make_targets(model_name: str, batch_size: int, seed: int = 0):
    """Run the model forward on random params to get on-manifold targets."""
    torch.manual_seed(seed)
    bm = smpl_pt.BodyModel(model_name=model_name, num_betas=10)
    pose = torch.randn(batch_size, bm.num_joints * 3) * 0.1
    shape = torch.randn(batch_size, 10) * 0.5
    trans = torch.randn(batch_size, 3)
    with torch.no_grad():
        out = bm(pose_rotvecs=pose, shape_betas=shape, trans=trans)
    return bm, out['vertices'].detach(), out['joints'].detach()


def _loss(fit):
    return sum(fit[k].pow(2).sum() for k in ['pose_rotvecs', 'shape_betas', 'trans'])


@pytest.mark.parametrize('model_name', ['smpl', 'smplx'])
@pytest.mark.parametrize('num_iter', [1, 3])
def test_fitter_grad_finite(model_name, num_iter):
    """Backprop through the fit must yield finite, non-zero gradients."""
    bm, target_v, target_j = _make_targets(model_name, batch_size=2)
    fitter = smpl_pt.BodyFitter(bm)

    tv = target_v.clone().requires_grad_(True)
    tj = target_j.clone().requires_grad_(True)
    fit = fitter.fit(
        target_vertices=tv,
        target_joints=tj,
        num_iter=num_iter,
        beta_regularizer=1.0,
        requested_keys=['pose_rotvecs', 'shape_betas', 'trans'],
    )
    _loss(fit).backward()

    for name, g in [('target_vertices', tv.grad), ('target_joints', tj.grad)]:
        assert g is not None, f'{name} grad is None'
        assert torch.isfinite(g).all(), f'{name} grad has NaN/Inf'
        assert g.abs().max().item() > 0, f'{name} grad is all-zero'


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_fitter_grad_vs_finite_diff(seed):
    """Directional finite-difference check: (grad · d) should match (L(x+eps*d) - L(x-eps*d)) / (2*eps).

    Per-coordinate FD is too noisy in float32 on small gradient components; a random
    unit-direction probe averages over many coordinates and is much more stable.
    """
    bm, target_v, target_j = _make_targets('smpl', batch_size=1, seed=seed)
    fitter = smpl_pt.BodyFitter(bm)

    def loss_fn(tv, tj):
        fit = fitter.fit(
            target_vertices=tv,
            target_joints=tj,
            num_iter=1,
            beta_regularizer=1.0,
            requested_keys=['pose_rotvecs', 'shape_betas', 'trans'],
        )
        return _loss(fit)

    # Autograd directional derivative.
    tv = target_v.clone().requires_grad_(True)
    tj = target_j.clone().requires_grad_(True)
    loss_fn(tv, tj).backward()

    g = torch.Generator().manual_seed(seed + 100)
    dv = torch.randn(target_v.shape, generator=g)
    dj = torch.randn(target_j.shape, generator=g)
    dv = dv / dv.norm()
    dj = dj / dj.norm()
    ag_dir = (tv.grad * dv).sum().item() + (tj.grad * dj).sum().item()

    # Central finite difference at a moderate step (float32 sweet spot).
    eps = 1e-2
    with torch.no_grad():
        lp = loss_fn((target_v + eps * dv).clone(), (target_j + eps * dj).clone()).item()
        lm = loss_fn((target_v - eps * dv).clone(), (target_j - eps * dj).clone()).item()
    fd_dir = (lp - lm) / (2 * eps)

    denom = max(abs(ag_dir), abs(fd_dir), 1e-3)
    rel = abs(ag_dir - fd_dir) / denom
    # 5% tolerance: looser because (a) the body model is float32 and (b) proj_SO3 has
    # a reflection-branch `where` that can be non-smooth near the boundary, biasing FD.
    # Still tight enough to catch sign flips, magnitude errors, or NaN regressions.
    assert rel < 5e-2, f'seed={seed}: autograd={ag_dir:.4e}, fd={fd_dir:.4e}, rel={rel:.3e}'
