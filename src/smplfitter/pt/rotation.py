from __future__ import annotations

import math

import torch


def divide_no_nan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Safe division that returns zero (with zero gradient) where the denominator is zero."""
    safe_b = torch.where(b == 0, torch.ones_like(b), b)
    return torch.where(b == 0, torch.zeros_like(a / safe_b), a / safe_b)


def proj_SO3(A: torch.Tensor) -> torch.Tensor:
    """Project (..., 3, 3) matrices onto SO(3) — closest rotation in Frobenius norm.

    Delegates to the SVD-based implementation (:func:`proj_SO3_svd`): its single batched
    cusolver call is faster in eager mode at small batch sizes than the many elementwise
    kernels of the closed-form projection. :func:`proj_SO3_analytic` is preferable inside
    CUDA graphs or compiled code (it is branch-free, with no data-dependent control flow
    or host-device syncs) and at very large batch sizes.
    """
    return proj_SO3_svd(A)


def proj_SO3_analytic(A: torch.Tensor) -> torch.Tensor:
    """Project (..., 3, 3) matrices onto SO(3) via a closed-form polar decomposition.

    Instead of an SVD, this solves the symmetric 3x3 eigenproblem of ``M = An^T An``
    (``An = A / |A|_F``), whose eigenvectors are the right singular vectors ``V``:

      * The eigenvalues come from Smith's trigonometric solution of the characteristic
        cubic; the extreme eigenvector whose eigenvalue gap is larger is extracted first
        (never the ill-conditioned middle one), and the remaining two follow from
        diagonalizing the 2x2 restriction of ``M`` to its orthogonal complement
        (half-angle atan2).
      * ``U = normalize(An V)`` columnwise with Gram-Schmidt; ``u3 = u1 x u2`` and
        ``v3 = v1 x v2`` keep both bases right-handed, which bakes in the det-flip
        (reflection) correction, so ``R = U V^T`` always has det +1. ``u1`` is never
        degenerate since ``sigma1 >= sqrt(1/3)`` after normalization; ``u2`` falls back to
        an arbitrary orthogonal direction when ``sigma2 ~ 0`` (where the closest rotation
        is non-unique anyway).

    All selection is branch-free (``torch.where``). Internal math runs in float64; the
    result is cast back to the input dtype.
    """
    orig_dtype = A.dtype
    eps = 1e-12
    Ad = A.double()

    # Scale-normalize so all thresholds below are relative.
    fro = torch.linalg.norm(Ad.flatten(-2, -1), dim=-1)
    An = Ad / fro.clamp_min(1e-30).unsqueeze(-1).unsqueeze(-1)

    M = An.mT @ An  # symmetric PSD, trace = 1
    lam1, lam2, lam3 = _sym_eigvals3(M)

    eye = torch.eye(3, dtype=torch.float64, device=A.device)
    e0 = eye[0]  # unit fallback direction, broadcastable and built device-side

    # Extreme eigenvector with the larger gap (always conditioned by >= (lam1-lam3)/2).
    use_top = (lam1 - lam2) >= (lam2 - lam3)
    lam_ext = torch.where(use_top, lam1, lam3)
    v_a = _normalize_or(_eigvec_raw(M, lam_ext), e0, eps)  # e0 only when M ~ isotropic: valid

    # 2x2 eigenproblem in the orthogonal complement of v_a.
    p = _any_orthogonal(v_a)
    q = torch.linalg.cross(v_a, p, dim=-1)
    Mp = _matvec(M, p)
    Mq = _matvec(M, q)
    mpp = (p * Mp).sum(-1)
    mpq = (p * Mq).sum(-1)
    mqq = (q * Mq).sum(-1)
    th = 0.5 * torch.atan2(2.0 * mpq, mpp - mqq)
    c = torch.cos(th).unsqueeze(-1)
    s = torch.sin(th).unsqueeze(-1)
    v_big = c * p + s * q  # larger remaining eigenvalue
    v_small = -s * p + c * q  # smaller remaining eigenvalue

    use_top_u = use_top.unsqueeze(-1)
    v1 = torch.where(use_top_u, v_a, v_big)
    v2 = torch.where(use_top_u, v_big, v_small)
    v3 = torch.linalg.cross(v1, v2, dim=-1)

    u1 = _normalize_or(_matvec(An, v1), e0, eps)  # |An v1| = sigma1 >= sqrt(1/3): safe
    u2 = _matvec(An, v2)
    u2 = u2 - (u2 * u1).sum(-1, keepdim=True) * u1
    u2 = _normalize_or(u2, _any_orthogonal(u1), eps)
    u3 = torch.linalg.cross(u1, u2, dim=-1)

    U = torch.stack([u1, u2, u3], dim=-1)  # columns
    V = torch.stack([v1, v2, v3], dim=-1)  # columns
    R = U @ V.mT

    # Guard the fully degenerate A ~ 0 case -> identity.
    R = torch.where((fro > 1e-20).unsqueeze(-1).unsqueeze(-1), R, eye)
    return R.to(orig_dtype)


def proj_SO3_svd(A: torch.Tensor) -> torch.Tensor:
    """Project (..., 3, 3) matrices onto SO(3) — closest rotation in Frobenius norm.

    Computes the rotation part of the polar decomposition via SVD, with a sign flip
    on the last singular vector when the naive product is a reflection.
    """
    U, _, Vh = torch.linalg.svd(A)
    T = U @ Vh
    has_reflection = (torch.det(T) < 0).unsqueeze(-1).unsqueeze(-1)
    T_mirror = T - 2 * U[..., -1:] @ Vh[..., -1:, :]
    return torch.where(has_reflection, T_mirror, T)


def _sym_eigvals3(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eigenvalues (lam1 >= lam2 >= lam3) of a symmetric 3x3 matrix, closed form.

    Smith's trigonometric solution of the characteristic cubic.
    """
    a00 = M[..., 0, 0]
    a11 = M[..., 1, 1]
    a22 = M[..., 2, 2]
    a01 = M[..., 0, 1]
    a02 = M[..., 0, 2]
    a12 = M[..., 1, 2]

    q = (a00 + a11 + a22) / 3.0
    p1 = a01 * a01 + a02 * a02 + a12 * a12
    p2 = (a00 - q) ** 2 + (a11 - q) ** 2 + (a22 - q) ** 2 + 2.0 * p1
    p = torch.sqrt(p2 / 6.0)

    b00 = a00 - q
    b11 = a11 - q
    b22 = a22 - q
    det_shifted = (
        b00 * (b11 * b22 - a12 * a12)
        - a01 * (a01 * b22 - a12 * a02)
        + a02 * (a01 * a12 - b11 * a02)
    )
    p3 = p * p * p
    r = divide_no_nan(det_shifted, 2.0 * p3).clamp(-1.0, 1.0)
    phi = torch.acos(r) / 3.0

    two_p = 2.0 * p
    lam1 = q + two_p * torch.cos(phi)
    lam3 = q + two_p * torch.cos(phi + (2.0 * math.pi / 3.0))
    lam2 = 3.0 * q - lam1 - lam3
    return lam1, lam2, lam3


def _eigvec_raw(M: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """An (unnormalized) eigenvector of symmetric 3x3 ``M`` for eigenvalue ``lam``.

    Null vector of ``M - lam I``, taken as the largest-norm cross product of its row
    pairs. Well-conditioned only if ``lam`` is separated from the other eigenvalues, so
    callers must pass the extreme eigenvalue with the larger gap.
    """
    eye = torch.eye(3, dtype=M.dtype, device=M.device)
    N = M - lam.unsqueeze(-1).unsqueeze(-1) * eye
    r0 = N[..., 0, :]
    r1 = N[..., 1, :]
    r2 = N[..., 2, :]
    c0 = torch.linalg.cross(r0, r1, dim=-1)
    c1 = torch.linalg.cross(r1, r2, dim=-1)
    c2 = torch.linalg.cross(r2, r0, dim=-1)
    n0 = (c0 * c0).sum(-1, keepdim=True)
    n1 = (c1 * c1).sum(-1, keepdim=True)
    n2 = (c2 * c2).sum(-1, keepdim=True)
    best = torch.where(n0 >= n1, c0, c1)
    n_best = torch.where(n0 >= n1, n0, n1)
    return torch.where(n_best >= n2, best, c2)


def _any_orthogonal(u: torch.Tensor) -> torch.Tensor:
    """A unit vector orthogonal to unit vector ``u`` (..., 3), branch-free.

    Crosses ``u`` with the standard-basis axis it is least aligned with, so the cross
    norm is bounded below by sqrt(1 - 1/3) for unit ``u``.
    """
    absu = u.abs()
    a0, a1, a2 = torch.unbind(absu, dim=-1)
    is0 = torch.logical_and(a0 <= a1, a0 <= a2)
    is1 = torch.logical_and(a1 <= a0, a1 <= a2)
    e = torch.stack(
        [
            is0.to(u.dtype),
            torch.logical_and(is1, torch.logical_not(is0)).to(u.dtype),
            torch.logical_and(torch.logical_not(is0), torch.logical_not(is1)).to(u.dtype),
        ],
        dim=-1,
    )
    w = torch.linalg.cross(u, e, dim=-1)
    return w / torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-30)


def _normalize_or(x: torch.Tensor, fallback: torch.Tensor, eps: float) -> torch.Tensor:
    """Unit-normalize ``x`` along the last dim; where |x| ~ 0 return the (unit) fallback."""
    n = torch.linalg.norm(x, dim=-1, keepdim=True)
    ok = n > eps
    xn = x / torch.where(ok, n, torch.ones_like(n))
    return torch.where(ok, xn, fallback)


def _matvec(A: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.einsum('...ij,...j->...i', A, v)


def kabsch(X, Y):
    return proj_SO3(X.mT @ Y)


def align_unit_vectors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Closed-form rotation that maps unit vector ``a`` to unit vector ``b``.

    Returns (..., 3, 3). Built from Rodrigues on the axis-angle
    ``angle * (a x b) / |a x b|`` with ``angle = atan2(|a x b|, a . b)``.
    The parallel (a == b) and antiparallel (a == -b) limits stay finite —
    ``divide_no_nan`` returns a zero rotvec, which gives the identity matrix.
    The antiparallel choice is arbitrary (no canonical 180-deg rotation).
    """
    cross = torch.linalg.cross(a, b, dim=-1)
    dot = (a * b).sum(dim=-1, keepdim=True)
    sin_a = torch.linalg.norm(cross, dim=-1, keepdim=True)
    angle = torch.atan2(sin_a, dot)
    rotvec = divide_no_nan(cross * angle, sin_a)
    return rotvec2mat(rotvec)


def project_onto_plane(v: torch.Tensor, n_hat: torch.Tensor) -> torch.Tensor:
    """Component of ``v`` perpendicular to the unit vector ``n_hat``.

    Batched over leading dims; ``n_hat`` broadcasts against ``v``.
    """
    parallel = (v * n_hat).sum(dim=-1, keepdim=True) * n_hat
    return v - parallel


def rotvec2mat(rotvec):
    angle = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    axis = divide_no_nan(rotvec, angle)

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
    return (divide_no_nan(torch.full_like(norm, 2.0), norm) * torch.atan2(norm, w)) * xyz
