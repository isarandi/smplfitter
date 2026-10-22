"""The analytic rotation projection: same fit as the SVD one, and capturable in a CUDA graph."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import smplfitter.pt as smpl_pt

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')


def _noisy_targets(batch_size, seed, device='cuda'):
    """Noisy SMPL vertices and joints with random weights."""
    bm = smpl_pt.BodyModel('smpl', num_betas=10)
    g = torch.Generator().manual_seed(seed)
    pose = torch.randn(batch_size, bm.num_joints * 3, generator=g) * 0.3
    shape = torch.randn(batch_size, 10, generator=g)
    trans = torch.randn(batch_size, 3, generator=g)
    with torch.inference_mode():
        out = bm(pose_rotvecs=pose, shape_betas=shape, trans=trans)
    verts = out['vertices'] + 0.01 * torch.randn(out['vertices'].shape, generator=g)
    joints = out['joints'] + 0.01 * torch.randn(out['joints'].shape, generator=g)
    vertex_weights = torch.rand(verts.shape[:2], generator=g) + 0.5
    joint_weights = torch.rand(joints.shape[:2], generator=g) + 0.5
    return [x.to(device) for x in (verts, joints, vertex_weights, joint_weights)]


def _fit(fitter, inputs, rotation_projection):
    verts, joints, vw, jw = inputs
    return fitter.fit(
        verts,
        joints,
        vw,
        jw,
        num_iter=3,
        beta_regularizer=1.0,
        requested_keys=['pose_rotvecs', 'shape_betas', 'trans'],
        rotation_projection=rotation_projection,
    )


def _max_vertex_diff_mm(bm, fit1, fit2):
    with torch.inference_mode():
        v1 = bm(fit1['pose_rotvecs'], fit1['shape_betas'], fit1['trans'])['vertices']
        v2 = bm(fit2['pose_rotvecs'], fit2['shape_betas'], fit2['trans'])['vertices']
    return (v1 - v2).norm(dim=-1).max().item() * 1000


def test_analytic_matches_svd():
    bm = smpl_pt.BodyModel('smpl', num_betas=10).cuda()
    fitter = torch.jit.script(smpl_pt.BodyFitter(bm).cuda())
    inputs = _noisy_targets(8, seed=0)
    with torch.inference_mode():
        fit_svd = _fit(fitter, inputs, 'svd')
        fit_analytic = _fit(fitter, inputs, 'analytic')
    assert _max_vertex_diff_mm(bm, fit_svd, fit_analytic) < 0.05


def test_analytic_fit_captures_in_cuda_graph():
    # cuSOLVER for the batched Cholesky: with some torch versions the default (MAGMA)
    # allocates memory during the call, which a capture does not allow.
    previous = torch.backends.cuda.preferred_linalg_library()
    torch.backends.cuda.preferred_linalg_library('cusolver')
    try:
        bm = smpl_pt.BodyModel('smpl', num_betas=10).cuda()
        fitter = torch.jit.script(smpl_pt.BodyFitter(bm).cuda())
        static_inputs = _noisy_targets(4, seed=1)
        with torch.inference_mode():
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    _fit(fitter, static_inputs, 'analytic')
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_out = _fit(fitter, static_inputs, 'analytic')

            # Replay on new inputs copied into the static input buffers
            new_inputs = _noisy_targets(4, seed=2)
            for static, new in zip(static_inputs, new_inputs):
                static.copy_(new)
            graph.replay()
            torch.cuda.synchronize()
            replayed = {k: v.clone() for k, v in graph_out.items()}
            eager = _fit(fitter, new_inputs, 'analytic')
        assert np.isfinite(replayed['pose_rotvecs'].cpu().numpy()).all()
        assert _max_vertex_diff_mm(bm, replayed, eager) < 0.01
    finally:
        torch.backends.cuda.preferred_linalg_library(previous)
