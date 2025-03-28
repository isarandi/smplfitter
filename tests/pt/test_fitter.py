import os
os.environ['DATA_ROOT'] = '/work_uncached/sarandi/data'
from smplfitter.pt import BodyModel, BodyFitter
import os
import torch
import numpy as np

def test_fitter():
    body_model = BodyModel('smpl', 'neutral', num_betas=10).cuda()
    fitter = BodyFitter(body_model, enable_kid=False).cuda()
    fitter = torch.jit.script(fitter)

    pose_rotvecs = np.random.randn(2, 24 * 3).astype(np.float32) * 0.1
    shape_betas = np.random.randn(2, 10).astype(np.float32)
    trans = np.random.randn(2, 3).astype(np.float32)

    with torch.inference_mode():
        res = body_model(
            pose_rotvecs=torch.from_numpy(pose_rotvecs).cuda(),
            shape_betas=torch.from_numpy(shape_betas).cuda(),
            trans=torch.from_numpy(trans).cuda())

        fit = fitter.fit(
            target_vertices=res['vertices'], target_joints=res['joints'], num_iter=3,
            beta_regularizer=0.0, requested_keys=('pose_rotvecs', 'shape_betas'))

        res_fit = body_model(
            pose_rotvecs=fit['pose_rotvecs'], shape_betas=fit['shape_betas'], trans=fit['trans'])

    verts_err = torch.linalg.norm(res['vertices'] - res_fit['vertices'], axis=-1)
    joints_err = torch.linalg.norm(res['joints'] - res_fit['joints'], axis=-1)

    mean_verts_err = torch.mean(verts_err)
    mean_joints_err = torch.mean(joints_err)
    assert mean_verts_err < 5e-3
    assert mean_joints_err < 5e-3
