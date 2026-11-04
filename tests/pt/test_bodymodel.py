import numpy as np
import pytest
import torch
from smplfitter.pt import BodyFitter, BodyModel


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
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


def test_vertex_subset_without_regressor():
    rng = np.random.default_rng(0)
    subset = np.sort(rng.choice(6890, 512, replace=False))
    with pytest.warns(UserWarning, match='joint_regressor_post_lbs'):
        sub_model = BodyModel('smpl', 'neutral', num_betas=10, vertex_subset=subset)
    assert sub_model.J_regressor_post_lbs.shape == (24, 512)

    full_model = BodyModel('smpl', 'neutral')
    torch.manual_seed(0)
    pose_rotvecs = torch.randn(8, 72) * 0.2
    shape_betas = torch.randn(8, 10)
    with torch.inference_mode():
        target = full_model(pose_rotvecs, shape_betas)

    # Joint error of the approximate regressor on posed subset vertices
    regressed = sub_model.J_regressor_post_lbs @ target['vertices'][:, subset]
    joint_err = torch.linalg.norm(regressed - target['joints'], dim=-1).mean()
    assert joint_err < 0.03

    # Vertex-only fit on the subset recovers the full mesh
    fit = BodyFitter(sub_model).fit(target['vertices'][:, subset], num_iter=3)
    with torch.inference_mode():
        refit = full_model(fit['pose_rotvecs'], fit['shape_betas'], fit['trans'])
    vertex_err = torch.linalg.norm(refit['vertices'] - target['vertices'], dim=-1).mean()
    assert vertex_err < 0.02
