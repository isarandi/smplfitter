import numpy as np
import torch
from smplfitter.pt import BodyConverter, BodyModel


def test_converter():
    body_model_in = BodyModel('smpl', 'neutral').cuda()
    body_model_out = BodyModel('smplx', 'neutral').cuda()

    converter = BodyConverter(body_model_in, body_model_out).cuda()
    reverse_converter = BodyConverter(body_model_out, body_model_in).cuda()

    pose_rotvecs = np.random.randn(2, 24 * 3).astype(np.float32) * 0.1
    shape_betas = np.random.randn(2, 10).astype(np.float32)
    trans = np.random.randn(2, 3).astype(np.float32)

    with torch.inference_mode():
        res = body_model_in(
            pose_rotvecs=torch.from_numpy(pose_rotvecs).cuda(),
            shape_betas=torch.from_numpy(shape_betas).cuda(),
            trans=torch.from_numpy(trans).cuda(),
        )

        conv = converter.convert(
            pose_rotvecs=torch.from_numpy(pose_rotvecs).cuda(),
            shape_betas=torch.from_numpy(shape_betas).cuda(),
            trans=torch.from_numpy(trans).cuda(),
        )

        res_new = body_model_out(
            pose_rotvecs=conv['pose_rotvecs'], shape_betas=conv['shape_betas'], trans=conv['trans']
        )

        new_verts_backconverted = reverse_converter.convert_vertices(res_new['vertices'])

    verts_err = torch.linalg.norm(res['vertices'] - new_verts_backconverted, dim=-1)
    mean_verts_err = torch.mean(verts_err)
    assert mean_verts_err < 1e-2
