"""Out-of-core shared-shape fitting: one beta vector over more frames than fit in memory.

`fit_streaming` reproduces `BodyFitter.fit(share_beta=True)` without ever holding all
frames at once. The batch fitter alternates a per-frame rotation update with a shape solve
whose only cross-frame coupling is the summed normal equations of the betas; here that sum
is accumulated chunk by chunk (10x10 + 10 numbers of shared state), and the per-frame
rotations that each pass refines are kept in a compact store. The data source is a
callable that returns a fresh iterable of chunks, called once per pass.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch

from .bodyfitter import BodyFitter
from .rotation import mat2rotvec

Chunk = tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]


def fit_streaming(
    fitter: BodyFitter,
    chunks: Callable[[], Iterable[Chunk]],
    num_iter: int = 1,
    beta_regularizer: float = 1,
    beta_regularizer2: float = 0,
    kid_regularizer: Optional[float] = None,
    final_adjust_rots: bool = True,
    initial_shape_betas: Optional[torch.Tensor] = None,
    initial_kid_factor: Optional[torch.Tensor] = None,
    requested_keys: Optional[list[str]] = None,
    max_chunk_frames: Optional[int] = 4096,
) -> dict[str, torch.Tensor]:
    """Fit one shared body shape and per-frame poses over a stream of frame chunks.

    Equivalent to ``fitter.fit(..., share_beta=True)`` over the concatenation of all chunks,
    but processes one chunk at a time. Only the shape normal equations and one rotation
    matrix set per frame are held across chunks; the (large) target vertex data streams.

    Parameters:
        fitter: The BodyFitter (body model, regularization structure).
        chunks: Callable returning an iterable of ``(target_vertices, target_joints,
            vertex_weights, joint_weights)`` tuples (joints and weights may be None, as in
            ``fit``). It is called once per pass (``num_iter`` times plus one final pass) and
            must yield the same frames in the same chunking and order every time.
        num_iter: Number of alternation iterations, as in ``fit``.
        beta_regularizer: L2 regularization weight for the shared betas.
        beta_regularizer2: Regularization for the first two betas.
        kid_regularizer: Regularization for the kid factor (only if ``fitter.enable_kid``).
        final_adjust_rots: Final per-frame refinement of the part orientations, as in
            ``fit``.
        initial_shape_betas: Optional shape prior/reference (regularization target), shape
            (num_betas,) or (1, num_betas).
        initial_kid_factor: Same for the kid factor.
        requested_keys: Which per-frame outputs to return (default ``['pose_rotvecs']``).
        max_chunk_frames: Chunks larger than this are processed in consecutive slices of at
            most this many frames, which bounds peak memory independently of how the caller
            chunks the data (~0.65 MB per frame for a 1048-point subset on the CPU, i.e.
            the default keeps the working set under ~3 GB). None disables the splitting.

    Returns:
        The same dictionary as ``fit`` with ``share_beta=True``: ``shape_betas`` (repeated per
        frame), ``trans``, ``orientations`` and the requested rotation formats, concatenated
        over all frames in stream order.
    """
    if requested_keys is None:
        requested_keys = ['pose_rotvecs']
    acc = _SharedShapeAccumulator(
        fitter,
        beta_regularizer,
        beta_regularizer2,
        kid_regularizer,
        initial_shape_betas,
        initial_kid_factor,
    )
    for _ in range(num_iter):
        for chunk in _slices(chunks(), max_chunk_frames):
            acc.update(*chunk)
        acc.solve()
    results = [
        acc.finalize(*chunk, final_adjust_rots, requested_keys)
        for chunk in _slices(chunks(), max_chunk_frames)
    ]
    acc.check_pass_complete()
    return {k: torch.cat([r[k] for r in results], dim=0) for k in results[0]}


def _as_chunk(chunk) -> Chunk:
    """Accept (verts,), (verts, joints) or the full 4-tuple."""
    chunk = tuple(chunk)
    return chunk + (None,) * (4 - len(chunk))


def _slices(chunk_iter, max_frames: Optional[int]):
    """Yield the caller's chunks, splitting any longer than max_frames into consecutive
    slices (deterministic, so replays stay positionally aligned)."""
    for chunk in chunk_iter:
        chunk = _as_chunk(chunk)
        n = chunk[0].shape[0]
        if max_frames is None or n <= max_frames:
            yield chunk
            continue
        for i in range(0, n, max_frames):
            yield tuple(None if x is None else x[i : i + max_frames] for x in chunk)


class _SharedShapeAccumulator:
    """Streams frames through the shape/pose alternation of the batch fitter.

    Per pass: ``update`` for every chunk (rotations refined from the stored ones, shape
    normal equations accumulated), then ``solve``. Rotations are stored positionally, so
    chunks must arrive in the same order in every pass (frame counts are checked at pass
    boundaries). ``finalize`` runs the frozen-shape per-frame stage of ``fit`` on a chunk
    (with the rotations from the last ``update``, as the batch fitter does).
    """

    def __init__(
        self,
        fitter: BodyFitter,
        beta_regularizer: float,
        beta_regularizer2: float,
        kid_regularizer: Optional[float],
        initial_shape_betas: Optional[torch.Tensor],
        initial_kid_factor: Optional[torch.Tensor],
    ):
        self.f = fitter
        self.bm = fitter.body_model
        self.device = self.bm.v_template.device
        self.n_shared = fitter.n_betas + (1 if fitter.enable_kid else 0)
        # Regularization enters as extra rows exactly as in lstsq_partial_share.
        reg = torch.cat(
            [
                torch.full((2,), beta_regularizer2, device=self.device),
                torch.full((fitter.n_betas - 2,), beta_regularizer, device=self.device),
            ]
        )
        ref = torch.zeros(fitter.n_betas, device=self.device)
        if initial_shape_betas is not None:
            isb = torch.as_tensor(initial_shape_betas, device=self.device).reshape(-1)
            ref[: isb.shape[0]] = isb
        if fitter.enable_kid:
            kid_reg = beta_regularizer if kid_regularizer is None else kid_regularizer
            reg = torch.cat([reg, torch.tensor([kid_reg], device=self.device)])
            kid_ref = (
                torch.zeros(1, device=self.device)
                if initial_kid_factor is None
                else torch.as_tensor(initial_kid_factor, device=self.device).reshape(1)
            )
            ref = torch.cat([ref, kid_ref])
        self.reg_diag = reg
        self.reg_rhs = (reg * ref).unsqueeze(-1)
        self.shape: Optional[torch.Tensor] = None  # (n_shared,) after first solve
        self.rots: list[torch.Tensor] = []  # per-chunk (B, J, 3, 3)
        self._G = torch.zeros(self.n_shared, self.n_shared, device=self.device)
        self._g = torch.zeros(self.n_shared, 1, device=self.device)
        self._i_chunk = 0
        self._n_frames_pass = 0
        self._n_frames_prev: Optional[int] = None

    # -- one pass ---------------------------------------------------------------------

    def update(self, target_vertices, target_joints, vertex_weights, joint_weights):
        target_vertices, target_joints, _ = self._center(target_vertices, target_joints)
        glob_rotmats = self._rotations_for_chunk(
            target_vertices, target_joints, vertex_weights, joint_weights
        )
        A, b, w, _, _, _, _ = self._design(
            glob_rotmats, target_vertices, target_joints, vertex_weights, joint_weights
        )
        wA = w.unsqueeze(-1) * A
        self._G += (wA.mT @ A).sum(dim=0)
        self._g += (wA.mT @ b).sum(dim=0)
        # The batch solver appends the regularization rows once PER FRAME (they are part of
        # each frame's design and get summed with the rest); reproduce that scaling.
        n = A.shape[0]
        self._G += n * torch.diag(self.reg_diag)
        self._g += n * self.reg_rhs
        self._store_rots(glob_rotmats)

    def solve(self):
        G = self._G
        g = self._g
        chol, _ = torch.linalg.cholesky_ex(G)
        self.shape = torch.cholesky_solve(g, chol).squeeze(-1)
        self._G.zero_()
        self._g.zero_()
        self._end_pass()

    def check_pass_complete(self):
        self._end_pass()

    # -- final per-frame stage (shape frozen) -----------------------------------------

    def finalize(
        self,
        target_vertices,
        target_joints,
        vertex_weights,
        joint_weights,
        final_adjust_rots: bool,
        requested_keys: list[str],
    ) -> dict[str, torch.Tensor]:
        assert self.shape is not None, 'solve() must run before finalize()'
        target_vertices, target_joints, target_mean = self._center(target_vertices, target_joints)
        # The batch fitter's final shape solve reuses the rotations of its last loop
        # iteration; the final pass here likewise takes the stored rotations as they are.
        glob_rotmats = self._stored_rots(target_vertices.shape[0])
        self._store_rots(glob_rotmats)
        A, b, w, mean_A, mean_b, v_ext, j_ext = self._design(
            glob_rotmats, target_vertices, target_joints, vertex_weights, joint_weights
        )
        x = self.shape.expand(A.shape[0], -1)
        trans = mean_b.squeeze(1) - torch.matmul(mean_A.squeeze(1), x.unsqueeze(-1)).squeeze(-1)
        betas = x[:, : self.f.n_betas]
        kid = x[:, self.f.n_betas] if self.f.enable_kid else None
        shape_ext = x if self.f.enable_kid else betas
        ref_verts = (
            v_ext[..., 0]
            + torch.einsum('bvcs,bs->bvc', v_ext[..., 1:], shape_ext)
            + trans.unsqueeze(1)
        )
        ref_joints = (
            j_ext[..., 0]
            + torch.einsum('bvcs,bs->bvc', j_ext[..., 1:], shape_ext)
            + trans.unsqueeze(1)
        )
        if final_adjust_rots:
            glob_rotmats = self.f._fit_global_rotations_dependent(
                target_vertices,
                target_joints,
                ref_verts,
                ref_joints,
                vertex_weights,
                joint_weights,
                glob_rotmats,
                betas,
                None,
                trans,
                kid,
                True,
            )
        result: dict[str, torch.Tensor] = dict(
            shape_betas=betas, trans=trans + target_mean, orientations=glob_rotmats
        )
        if kid is not None:
            result['kid_factor'] = kid
        if 'relative_orientations' in requested_keys or 'pose_rotvecs' in requested_keys:
            parent_indices = self.bm.kintree_parents_tensor[1:].to(self.device)
            parent_glob = torch.cat(
                [
                    torch.eye(3, device=self.device).expand(glob_rotmats.shape[0], 1, 3, 3),
                    torch.index_select(glob_rotmats, 1, parent_indices),
                ],
                dim=1,
            )
            result['relative_orientations'] = parent_glob.transpose(-1, -2) @ glob_rotmats
        if 'pose_rotvecs' in requested_keys:
            rv = mat2rotvec(result['relative_orientations'])
            result['pose_rotvecs'] = rv.view(rv.shape[0], -1)
        return result

    # -- internals ---------------------------------------------------------------------

    def _center(self, target_vertices, target_joints):
        if target_joints is None:
            mean = torch.mean(target_vertices, dim=1)
            return target_vertices - mean[:, None], None, mean
        mean = torch.mean(torch.cat([target_vertices, target_joints], dim=1), dim=1)
        return target_vertices - mean[:, None], target_joints - mean[:, None], mean

    def _rotations_for_chunk(self, target_vertices, target_joints, vertex_weights, joint_weights):
        """Pass 1: rotations from the template mesh. Later passes: refine the stored
        rotations against the mesh posed with them under the current shape (this is
        exactly the batch fitter's per-iteration rotation update)."""
        f = self.f
        if self.shape is None:
            assert self._i_chunk == len(self.rots), 'chunk order changed between passes'
            return f._fit_global_rotations(
                target_vertices,
                target_joints,
                f.default_mesh_tf[None],
                f.body_model.J_template[None],
                vertex_weights,
                joint_weights,
                True,
            )
        prev = self._stored_rots(target_vertices.shape[0])
        _, _, _, mean_A, mean_b, v_ext, j_ext = self._design(
            prev, target_vertices, target_joints, vertex_weights, joint_weights
        )
        x = self.shape.expand(prev.shape[0], -1)
        trans = mean_b.squeeze(1) - torch.matmul(mean_A.squeeze(1), x.unsqueeze(-1)).squeeze(-1)
        shape_ext = x if f.enable_kid else x[:, : f.n_betas]
        ref_verts = (
            v_ext[..., 0]
            + torch.einsum('bvcs,bs->bvc', v_ext[..., 1:], shape_ext)
            + trans.unsqueeze(1)
        )
        ref_joints = (
            j_ext[..., 0]
            + torch.einsum('bvcs,bs->bvc', j_ext[..., 1:], shape_ext)
            + trans.unsqueeze(1)
        )
        return (
            f._fit_global_rotations(
                target_vertices,
                target_joints,
                ref_verts,
                ref_joints if target_joints is not None else None,
                vertex_weights,
                joint_weights,
                True,
            )
            @ prev
        )

    def _design(self, glob_rotmats, target_vertices, target_joints, vertex_weights, joint_weights):
        """Centered, weight-ready shape design of a chunk: A (B, 3N, S), b (B, 3N, 1),
        w (B, 3N), plus the row means and the posed vertex/joint extension tensors
        (position + Jacobian w.r.t. shape) needed to reconstruct the mesh for a given
        shape. Mirrors the shape stage of the batch fitter."""
        f = self.f
        bm = self.bm
        glob_rotmats = glob_rotmats.float()
        B = target_vertices.shape[0]
        parent_indices = bm.kintree_parents_tensor[1:].to(self.device)
        parent_glob = torch.cat(
            [
                torch.eye(3, device=self.device).expand(B, 1, 3, 3),
                torch.index_select(glob_rotmats, 1, parent_indices),
            ],
            dim=1,
        )
        rel_rotmats = parent_glob.transpose(-1, -2) @ glob_rotmats

        n_ext = f.J_template_ext.shape[2]
        j_ext = torch.empty(
            B, bm.num_joints, 3, n_ext, device=self.device, dtype=f.J_template_ext.dtype
        )
        j_ext.index_copy_(1, f.root_index, f.J_template_ext[:1].expand(B, 1, 3, n_ext))
        level_start = 0
        for k in range(f.num_fk_levels):
            n = f.fk_level_sizes[k]
            js = f.fk_js.narrow(0, level_start, n)
            ps = f.fk_ps.narrow(0, level_start, n)
            level_start += n
            j_ext.index_copy_(
                1,
                js,
                j_ext.index_select(1, ps)
                + torch.einsum(
                    'bnCc,ncs->bnCs',
                    glob_rotmats.index_select(1, ps),
                    f.bone_ext.index_select(0, js),
                ),
            )
        translations_ext = j_ext - torch.einsum('bjCc,jcs->bjCs', glob_rotmats, f.J_template_ext)
        rot_params = rel_rotmats[:, 1:].reshape(-1, (bm.num_joints - 1) * 9)
        v_posed = bm.v_template + torch.einsum('vcp,bp->bvc', bm.posedirs, rot_params)

        v_rotated = torch.einsum('bjCc,vj,bvc->bvC', glob_rotmats, bm.weights, v_posed)
        shapedirs = (
            torch.cat([bm.shapedirs, bm.kid_shapedir[:, :, None]], dim=2)
            if f.enable_kid
            else bm.shapedirs
        )
        v_grad = torch.einsum('bjCc,lj,lcs->blCs', glob_rotmats, bm.weights, shapedirs)
        v_ext = torch.cat([v_rotated.unsqueeze(-1), v_grad], dim=3)
        v_ext += torch.einsum('vj,bjcs->bvcs', bm.weights, translations_ext)

        if target_joints is None:
            target_both = target_vertices
            pos = v_ext[..., 0]
            jac = v_ext[..., 1:]
        else:
            target_both = torch.cat([target_vertices, target_joints], dim=1)
            pos = torch.cat([v_ext[..., 0], j_ext[..., 0]], dim=1)
            jac = torch.cat([v_ext[..., 1:], j_ext[..., 1:]], dim=1)
        A = jac
        b = target_both - pos
        if target_joints is not None and vertex_weights is not None and joint_weights is not None:
            weights = torch.cat([vertex_weights, joint_weights], dim=1)
        elif target_joints is None and vertex_weights is not None:
            weights = vertex_weights
        else:
            weights = torch.ones(A.shape[:2], dtype=torch.float32, device=self.device)
        w_sum_A = torch.sum(weights.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=True)
        w_sum_b = torch.sum(weights.unsqueeze(-1), dim=1, keepdim=True)
        mean_A = torch.where(
            w_sum_A == 0,
            torch.zeros_like(w_sum_A),
            torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * A, dim=1, keepdim=True) / w_sum_A,
        )
        mean_b = torch.where(
            w_sum_b == 0,
            torch.zeros_like(w_sum_b),
            torch.sum(weights.unsqueeze(-1) * b, dim=1, keepdim=True) / w_sum_b,
        )
        A = (A - mean_A).reshape(B, -1, self.n_shared)
        b = (b - mean_b).reshape(B, -1, 1)
        w = torch.repeat_interleave(weights.reshape(B, -1), 3, dim=1)
        return A, b, w, mean_A, mean_b, v_ext, j_ext

    def _stored_rots(self, n_frames: int) -> torch.Tensor:
        if self._i_chunk >= len(self.rots) or self.rots[self._i_chunk].shape[0] != n_frames:
            raise ValueError(
                f'chunks() must replay the same chunks in the same order every pass '
                f'(chunk {self._i_chunk}: got {n_frames} frames, expected '
                f'{self.rots[self._i_chunk].shape[0] if self._i_chunk < len(self.rots) else "none"})'
            )
        return self.rots[self._i_chunk]

    def _store_rots(self, glob_rotmats):
        if self._i_chunk < len(self.rots):
            self.rots[self._i_chunk] = glob_rotmats
        else:
            self.rots.append(glob_rotmats)
        self._i_chunk += 1
        self._n_frames_pass += glob_rotmats.shape[0]

    def _end_pass(self):
        if self._n_frames_prev is not None and self._n_frames_pass != self._n_frames_prev:
            raise ValueError(
                f'chunks() yielded {self._n_frames_pass} frames this pass but '
                f'{self._n_frames_prev} in the previous pass; it must replay the same frames'
            )
        if self._i_chunk != len(self.rots):
            raise ValueError('chunks() yielded fewer chunks than in the previous pass')
        self._n_frames_prev = self._n_frames_pass
        self._n_frames_pass = 0
        self._i_chunk = 0
