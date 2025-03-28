import os

import numpy as np
from smplfitter.np import BodyFitter, BodyModel

def test_fitter():
    body_model = BodyModel('smpl', 'neutral')
    fitter = BodyFitter(body_model, enable_kid=False, num_betas=10)

    pose_rotvecs = np.random.randn(2, 24 * 3).astype(np.float32) * 0.1
    shape_betas = np.random.randn(2, 10).astype(np.float32)
    trans = np.random.randn(2, 3).astype(np.float32)

    res = body_model(pose_rotvecs=pose_rotvecs, shape_betas=shape_betas, trans=trans)

    fit = fitter.fit(
        target_vertices=res['vertices'], target_joints=res['joints'], num_iter=3,
        beta_regularizer=0.0, requested_keys=('pose_rotvecs', 'shape_betas'))

    res_fit = body_model(
        pose_rotvecs=fit['pose_rotvecs'], shape_betas=fit['shape_betas'], trans=fit['trans'])

    verts_err = np.linalg.norm(res['vertices'] - res_fit['vertices'], axis=-1)
    joints_err = np.linalg.norm(res['joints'] - res_fit['joints'], axis=-1)

    mean_verts_err = np.mean(verts_err)
    mean_joints_err = np.mean(joints_err)
    assert mean_verts_err < 5e-3
    assert mean_joints_err < 5e-3
