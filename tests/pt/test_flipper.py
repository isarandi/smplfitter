import os

import numpy as np
import torch

from smplfitter.pt import BodyFlipper, BodyModel


def test_flipper_smpl():
    body_model = BodyModel('smpl', 'neutral', num_betas=10).cuda()
    flipper = BodyFlipper(body_model).cuda()

    pose_rotvecs = np.random.randn(2, 24 * 3).astype(np.float32) * 0.1
    shape_betas = np.random.randn(2, 10).astype(np.float32)
    trans = np.random.randn(2, 3).astype(np.float32)

    with torch.inference_mode():
        res = body_model(
            pose_rotvecs=torch.from_numpy(pose_rotvecs).cuda(),
            shape_betas=torch.from_numpy(shape_betas).cuda(),
            trans=torch.from_numpy(trans).cuda(),
        )

        flip = flipper.flip(
            pose_rotvecs=torch.from_numpy(pose_rotvecs).cuda(),
            shape_betas=torch.from_numpy(shape_betas).cuda(),
            trans=torch.from_numpy(trans).cuda(),
        )

        res_new = body_model(
            pose_rotvecs=flip['pose_rotvecs'], shape_betas=flip['shape_betas'], trans=flip['trans']
        )

        flipped_verts = flipper.flip_vertices(res['vertices'])

    verts_err = torch.linalg.norm(flipped_verts - res_new['vertices'], dim=-1)

    mean_verts_err = torch.mean(verts_err)
    assert mean_verts_err < 1e-2


def test_flipper_smplx_amass():
    """Test flipper with real AMASS poses."""
    import h5py

    amass_root = os.path.join(os.environ['DATA_ROOT'], 'amass')
    with h5py.File(f'{amass_root}/hdf5/smplx_neutral_val.h5', 'r') as f:
        n_total = len(f['pose'])
        indices = np.linspace(0, n_total - 1, 16, dtype=int)
        poses = f['pose'][indices].astype(np.float32)
        betas = f['shape_betas'][indices].astype(np.float32)

    body_model = BodyModel('smplx', 'neutral', num_betas=16).cuda()
    flipper = BodyFlipper(body_model).cuda()

    pose_t = torch.from_numpy(poses).cuda()
    betas_t = torch.from_numpy(betas).cuda()
    trans_t = torch.zeros(len(poses), 3).cuda()

    with torch.inference_mode():
        res = body_model(pose_rotvecs=pose_t, shape_betas=betas_t, trans=trans_t)

        flip = flipper.flip(pose_t, betas_t, trans_t, num_iter=3)

        res_new = body_model(
            pose_rotvecs=flip['pose_rotvecs'], shape_betas=flip['shape_betas'], trans=flip['trans']
        )

        flipped_verts = flipper.flip_vertices(res['vertices'])

    verts_err = torch.linalg.norm(flipped_verts - res_new['vertices'], dim=-1)

    mean_verts_err = torch.mean(verts_err)
    assert mean_verts_err < 5e-3
