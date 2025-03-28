import os

os.environ['DATA_ROOT'] = '/work_uncached/sarandi/data'
import torch
import numpy as np
from smplfitter.pt import BodyModel
from scipy.spatial.transform.rotation import Rotation
import smplx




def test_body_model_initialization():
    for g in ['m', 'f', 'n']:
        model = BodyModel('smpl', g)
        assert model.num_joints == 24
        assert model.num_vertices == 6890
        assert model.v_template.shape == (6890, 3)
        assert model.shapedirs.shape == (6890, 3, 300)
        assert model.posedirs.shape == (6890, 3, 207)

    for g in ['m', 'f']:
        model = BodyModel('smplh', g)
        assert model.num_joints == 52
        assert model.num_vertices == 6890

    for g in ['m', 'f', 'n']:
        model = BodyModel('smplx', g)
        assert model.num_joints == 55
        assert model.num_vertices == 10475


def test_body_model_call():
    model = BodyModel('smpl', 'neutral')
    batch_size = 2
    pose_rotvecs = np.random.randn(batch_size, 24, 3).astype(np.float32) * 0.1
    shape_betas = np.random.randn(batch_size, 10).astype(np.float32)
    trans = np.random.randn(batch_size, 3).astype(np.float32)

    with torch.inference_mode():
        output = model(
            pose_rotvecs=torch.from_numpy(pose_rotvecs),
            shape_betas=torch.from_numpy(shape_betas),
            trans=torch.from_numpy(trans),
        )

    assert 'vertices' in output
    assert 'joints' in output
    assert 'orientations' in output
    assert output['vertices'].shape == (batch_size, 6890, 3)
    assert output['joints'].shape == (batch_size, 24, 3)
    assert output['orientations'].shape == (batch_size, 24, 3, 3)
    assert output['vertices'].dtype == torch.float32
    assert output['vertices'].device == torch.device('cpu')


def test_body_model_call_cuda():
    model = BodyModel('smpl', 'neutral').cuda()
    batch_size = 2
    pose_rotvecs = np.random.randn(batch_size, 24, 3).astype(np.float32) * 0.1
    shape_betas = np.random.randn(batch_size, 10).astype(np.float32)
    trans = np.random.randn(batch_size, 3).astype(np.float32)

    with torch.inference_mode():
        output = model(
            pose_rotvecs=torch.from_numpy(pose_rotvecs).cuda(),
            shape_betas=torch.from_numpy(shape_betas).cuda(),
            trans=torch.from_numpy(trans).cuda(),
        )

    assert 'vertices' in output
    assert 'joints' in output
    assert 'orientations' in output
    assert output['vertices'].shape == (batch_size, 6890, 3)
    assert output['joints'].shape == (batch_size, 24, 3)
    assert output['orientations'].shape == (batch_size, 24, 3, 3)
    assert output['vertices'].dtype == torch.float32
    assert output['vertices'].device.type == 'cuda'
