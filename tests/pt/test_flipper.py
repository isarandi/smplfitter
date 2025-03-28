from smplfitter.pt import BodyModel, BodyFlipper
import os
import torch
import numpy as np


def test_flipper():
    body_model = BodyModel('smpl', 'neutral').cuda()
    flipper = BodyFlipper(body_model).cuda()

    pose_rotvecs = np.random.randn(2, 24 * 3).astype(np.float32) * 0.1
    shape_betas = np.random.randn(2, 10).astype(np.float32)
    trans = np.random.randn(2, 3).astype(np.float32)

    with torch.inference_mode():
        res = body_model(
            pose_rotvecs=torch.from_numpy(pose_rotvecs).cuda(),
            shape_betas=torch.from_numpy(shape_betas).cuda(),
            trans=torch.from_numpy(trans).cuda())

        flip = flipper.flip(
            pose_rotvecs=torch.from_numpy(pose_rotvecs).cuda(),
            shape_betas=torch.from_numpy(shape_betas).cuda(),
            trans=torch.from_numpy(trans).cuda())

        res_new = body_model(
            pose_rotvecs=flip['pose_rotvecs'],
            shape_betas=flip['shape_betas'],
            trans=flip['trans'])

        flipped_verts = flipper.flip_vertices(res['vertices'])

    verts_err = torch.linalg.norm(flipped_verts - res_new['vertices'], axis=-1)

    mean_verts_err = torch.mean(verts_err)
    assert mean_verts_err < 1e-2
