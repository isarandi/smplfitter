import torch


def kabsch(X, Y):
    A = X.mT @ Y
    U, _, Vh = torch.linalg.svd(A)
    T = U @ Vh
    has_reflection = (torch.det(T) < 0).unsqueeze(-1).unsqueeze(-1)
    T_mirror = T - 2 * U[..., -1:] @ Vh[..., -1:, :]
    return torch.where(has_reflection, T_mirror, T)


def rotvec2mat(rotvec):
    angle = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    axis = torch.nan_to_num(rotvec / angle)

    sin_axis = torch.sin(angle) * axis
    cos_angle = torch.cos(angle)
    cos1_axis = (1.0 - cos_angle) * axis
    _, axis_y, axis_z = torch.unbind(axis, dim=-1)
    cos1_axis_x, cos1_axis_y, _ = torch.unbind(cos1_axis, dim=-1)
    sin_axis_x, sin_axis_y, sin_axis_z = torch.unbind(sin_axis, dim=-1)
    tmp = cos1_axis_x * axis_y
    m01 = tmp - sin_axis_z
    m10 = tmp + sin_axis_z
    tmp = cos1_axis_x * axis_z
    m02 = tmp + sin_axis_y
    m20 = tmp - sin_axis_y
    tmp = cos1_axis_y * axis_z
    m12 = tmp - sin_axis_x
    m21 = tmp + sin_axis_x
    diag = cos1_axis * axis + cos_angle
    m00, m11, m22 = torch.unbind(diag, dim=-1)
    matrix = torch.stack((m00, m01, m02, m10, m11, m12, m20, m21, m22), dim=-1)
    return torch.unflatten(matrix, -1, (3, 3))


def mat2rotvec(rotmat):
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = torch.unbind(rotmat.flatten(-2, -1), dim=-1)
    p10p01 = r10 + r01
    p10m01 = r10 - r01
    p02p20 = r02 + r20
    p02m20 = r02 - r20
    p21p12 = r21 + r12
    p21m12 = r21 - r12
    p00p11 = r00 + r11
    p00m11 = r00 - r11
    _1p22 = 1.0 + r22
    _1m22 = 1.0 - r22

    trace = torch.diagonal(rotmat, dim1=-2, dim2=-1).sum(-1)
    cond0 = torch.stack((p21m12, p02m20, p10m01, 1.0 + trace), dim=-1)
    cond1 = torch.stack((_1m22 + p00m11, p10p01, p02p20, p21m12), dim=-1)
    cond2 = torch.stack((p10p01, _1m22 - p00m11, p21p12, p02m20), dim=-1)
    cond3 = torch.stack((p02p20, p21p12, _1p22 - p00p11, p10m01), dim=-1)

    trace_pos = (trace > 0.0).unsqueeze(-1)
    d00_large = torch.logical_and(r00 > r11, r00 > r22).unsqueeze(-1)
    d11_large = (r11 > r22).unsqueeze(-1)
    q = torch.where(
        trace_pos, cond0, torch.where(d00_large, cond1, torch.where(d11_large, cond2, cond3))
    )

    xyz, w = torch.split(q, (3, 1), dim=-1)
    norm = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    return (torch.nan_to_num(2.0 / norm) * torch.atan2(norm, w)) * xyz
