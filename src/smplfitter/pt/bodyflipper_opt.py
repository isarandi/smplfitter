from __future__ import annotations

import math
from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn

from .bodyflipper import BodyFlipper
from .rotation import rotvec2mat, mat2rotvec

if TYPE_CHECKING:
    import smplfitter.pt


def rot6d_to_rotmat(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to rotation matrix via Gram-Schmidt."""
    # rot6d: (..., 6) -> (..., 3, 3)
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]

    b1 = a1 / (torch.linalg.norm(a1, dim=-1, keepdim=True) + 1e-8)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = b2 / (torch.linalg.norm(b2, dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)


def rotmat_to_rot6d(rotmat: torch.Tensor) -> torch.Tensor:
    """Extract 6D representation (first two columns) from rotation matrix."""
    # rotmat: (..., 3, 3) -> (..., 6) as [col0, col1]
    return torch.cat([rotmat[..., :, 0], rotmat[..., :, 1]], dim=-1)


class BodyFlipperOpt(nn.Module):
    """
    Horizontally flips SMPL-like body model parameters with optional gradient descent refinement.

    Uses the closed-form BodyFlipper for initialization, then optionally refines
    with Adam optimization to minimize vertex error.

    Parameters:
        body_model: A body model whose parameters are to be transformed.
    """

    def __init__(self, body_model: 'smplfitter.pt.BodyModel'):
        super().__init__()
        self.body_model = body_model
        self.flipper = BodyFlipper(body_model)

    def flip(
        self,
        pose_rotvecs: torch.Tensor,
        shape_betas: torch.Tensor,
        trans: torch.Tensor,
        kid_factor: Optional[torch.Tensor] = None,
        num_iter: int = 1,
        refine_steps: int = 0,
        refine_lr: float = 0.03,
        warmup_ratio: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        """
        Returns the body model parameters that represent the horizontally flipped 3D human.

        Parameters:
            pose_rotvecs: Input body part orientations as rotation vectors (batch_size, num_joints*3).
            shape_betas: Input beta coefficients representing body shape.
            trans: Input translation parameters (meters).
            kid_factor: Coefficient for the kid blendshape.
            num_iter: Number of iterations for the closed-form fitter.
            refine_steps: Number of Adam optimization steps for refinement. 0 disables refinement.
            refine_lr: Learning rate for Adam optimizer.

        Returns:
            Dictionary with pose_rotvecs, shape_betas, trans, and optionally kid_factor.
        """
        # Get target flipped vertices
        with torch.no_grad():
            inp = self.body_model(pose_rotvecs, shape_betas, trans, kid_factor=kid_factor)
            target_verts = self.flipper.flip_vertices(inp['vertices'])

        # Get initial estimate from closed-form flipper
        init = self.flipper.flip(pose_rotvecs, shape_betas, trans, kid_factor, num_iter)

        if refine_steps == 0:
            return init

        # Refine with gradient descent
        return self._refine(
            target_verts=target_verts,
            init_pose=init['pose_rotvecs'],
            init_betas=init['shape_betas'],
            init_trans=init['trans'],
            init_kid_factor=init.get('kid_factor'),
            num_steps=refine_steps,
            lr=refine_lr,
            warmup_ratio=warmup_ratio,
        )

    def _refine(
        self,
        target_verts: torch.Tensor,
        init_pose: torch.Tensor,
        init_betas: torch.Tensor,
        init_trans: torch.Tensor,
        init_kid_factor: Optional[torch.Tensor],
        num_steps: int,
        lr: float,
        warmup_ratio: float,
    ) -> dict[str, torch.Tensor]:
        """Refine parameters using Adam optimization with 6D rotation representation.

        Optimizes relative rotations in 6D space (first two columns of rotmat) which is
        continuous and avoids singularities. Converts to rotmat via Gram-Schmidt.
        """
        num_joints = self.body_model.num_joints

        # Convert initial rotvecs to 6D representation
        init_rotvecs = init_pose.detach().view(-1, num_joints, 3)
        init_rotmats = rotvec2mat(init_rotvecs)  # (batch, num_joints, 3, 3)
        init_rot6d = rotmat_to_rot6d(init_rotmats)  # (batch, num_joints, 6)

        # Create optimizable parameters
        rot6d = init_rot6d.clone().requires_grad_(True)
        betas = init_betas.detach().clone().requires_grad_(True)
        trans = init_trans.detach().clone().requires_grad_(True)

        params = [rot6d, betas, trans]

        kid_factor = None
        if init_kid_factor is not None:
            kid_factor = init_kid_factor.detach().clone().requires_grad_(True)
            params.append(kid_factor)

        optimizer = torch.optim.Adam(params, lr=lr, betas=(0.97, 0.999))
        warmup_steps = int(num_steps * warmup_ratio)

        for step in range(num_steps):
            # Cosine LR schedule with linear warmup
            if step < warmup_steps:
                current_lr = lr * (step + 1) / warmup_steps
            else:
                progress = (step - warmup_steps) / max(1, num_steps - warmup_steps)
                current_lr = lr * 0.5 * (1.0 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.zero_grad()

            # Convert 6D to rotmat via Gram-Schmidt
            rotmats = rot6d_to_rotmat(rot6d)

            result = self.body_model(
                rel_rotmats=rotmats,
                shape_betas=betas,
                trans=trans,
                kid_factor=kid_factor,
            )

            # Vertex reconstruction loss (euclidean distance per vertex)
            loss = torch.mean(torch.linalg.norm(result['vertices'] - target_verts, dim=-1))
            loss.backward()

            optimizer.step()

        # Convert final 6D to rotvecs
        with torch.no_grad():
            rotmats_final = rot6d_to_rotmat(rot6d)
            pose_rotvecs = mat2rotvec(rotmats_final).view(-1, num_joints * 3)

        result = {
            'pose_rotvecs': pose_rotvecs,
            'shape_betas': betas.detach(),
            'trans': trans.detach(),
        }
        if kid_factor is not None:
            result['kid_factor'] = kid_factor.detach()

        return result
