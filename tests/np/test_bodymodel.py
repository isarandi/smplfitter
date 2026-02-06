import numpy as np
from scipy.spatial.transform.rotation import Rotation
from smplfitter.np import BodyModel


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

    output = model(pose_rotvecs=pose_rotvecs, shape_betas=shape_betas, trans=trans)

    assert 'vertices' in output
    assert 'joints' in output
    assert 'orientations' in output
    assert output['vertices'].shape == (batch_size, 6890, 3)
    assert output['joints'].shape == (batch_size, 24, 3)
    assert output['orientations'].shape == (batch_size, 24, 3, 3)


def test_rototranslate():
    model = BodyModel('smpl', 'neutral')
    R = Rotation.from_euler('xyz', np.random.randn(3)).as_matrix()
    t = np.array([1.0, 2.0, 3.0])
    pose_rotvecs = np.random.randn(24 * 3).astype(np.float32)
    shape_betas = np.random.randn(10).astype(np.float32)
    trans = np.random.randn(3).astype(np.float32)

    new_pose_rotvec, new_trans = model.rototranslate(R, t, pose_rotvecs, shape_betas, trans)

    assert new_pose_rotvec.shape == (24 * 3,)
    assert new_trans.shape == (3,)

    old_res = model.single(pose_rotvecs=pose_rotvecs, shape_betas=shape_betas, trans=trans)
    old_verts = old_res['vertices']
    old_joints = old_res['joints']

    new_res = model.single(pose_rotvecs=new_pose_rotvec, shape_betas=shape_betas, trans=new_trans)
    new_verts = new_res['vertices']
    new_joints = new_res['joints']

    expected_verts = old_verts @ R.T + t
    expected_joints = old_joints @ R.T + t

    assert np.allclose(new_verts, expected_verts, atol=1e-4)
    assert np.allclose(new_joints, expected_joints, atol=1e-4)


def test_single_instance():
    model = BodyModel('smpl', 'neutral')
    pose_rotvecs = np.zeros((24 * 3), dtype=np.float32)
    shape_betas = np.random.randn(10).astype(np.float32)
    trans = np.zeros((3,), dtype=np.float32)

    output = model.single(pose_rotvecs=pose_rotvecs, shape_betas=shape_betas, trans=trans)

    assert 'vertices' in output
    assert 'joints' in output
    assert 'orientations' in output
    assert output['vertices'].shape == (6890, 3)
    assert output['joints'].shape == (24, 3)
    assert output['orientations'].shape == (24, 3, 3)
