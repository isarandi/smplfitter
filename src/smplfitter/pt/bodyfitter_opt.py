from __future__ import annotations

import math
from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn

from .bodyfitter import BodyFitter
from .rotation import rotvec2mat, mat2rotvec

if TYPE_CHECKING:
    import smplfitter.pt


def rot6d_to_rotmat(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to rotation matrix via Gram-Schmidt."""
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]

    b1 = a1 / (torch.linalg.norm(a1, dim=-1, keepdim=True) + 1e-8)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = b2 / (torch.linalg.norm(b2, dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)


def rotmat_to_rot6d(rotmat: torch.Tensor) -> torch.Tensor:
    """Extract 6D representation (first two columns) from rotation matrix."""
    return torch.cat([rotmat[..., :, 0], rotmat[..., :, 1]], dim=-1)


class BodyFitterOpt(nn.Module):
    """
    Fits body model parameters to target vertices/joints, with optional gradient descent
    refinement.

    Uses the closed-form BodyFitter for initialization, then optionally refines
    with Adam optimization to minimize vertex/joint error.

    Parameters:
        body_model: The body model instance to fit.
        enable_kid: Enables the use of a kid blendshape.
    """

    def __init__(
        self,
        body_model: 'smplfitter.pt.BodyModel',
        enable_kid: bool = False,
    ):
        super().__init__()
        self.body_model = body_model
        self.fitter = BodyFitter(body_model, enable_kid=enable_kid)
        self.enable_kid = enable_kid

    def fit(
        self,
        target_vertices: torch.Tensor,
        target_joints: Optional[torch.Tensor] = None,
        vertex_weights: Optional[torch.Tensor] = None,
        joint_weights: Optional[torch.Tensor] = None,
        num_iter: int = 1,
        beta_regularizer: float = 1,
        beta_regularizer2: float = 0,
        share_beta: bool = False,
        final_adjust_rots: bool = True,
        scale_target: bool = False,
        scale_fit: bool = False,
        refine_steps: int = 0,
        refine_lr: float = 0.03,
        warmup_ratio: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """
        Fits body model parameters to target vertices and optionally joints.

        First runs the fast closed-form BodyFitter, then optionally refines with Adam.

        Parameters:
            target_vertices: Target mesh vertices (batch_size, num_vertices, 3).
            target_joints: Target joint locations (batch_size, num_joints, 3).
            vertex_weights: Per-vertex confidence weights.
            joint_weights: Per-joint confidence weights.
            num_iter: Iterations for the closed-form fitter (1-3 typical).
            beta_regularizer: L2 penalty on shape params.
            beta_regularizer2: Secondary regularization for first two betas.
            share_beta: Share shape across batch.
            final_adjust_rots: Sequential joint adjustment after fitting.
            scale_target: Estimate scale factor for target vertices.
            scale_fit: Estimate scale factor for fitted mesh.
            refine_steps: Number of Adam steps for refinement. 0 disables refinement.
            refine_lr: Learning rate for Adam optimizer.
            warmup_ratio: Fraction of steps for linear LR warmup.

        Returns:
            Dictionary with pose_rotvecs, shape_betas, trans, and optionally kid_factor.
        """
        init = self.fitter.fit(
            target_vertices,
            target_joints=target_joints,
            vertex_weights=vertex_weights,
            joint_weights=joint_weights,
            num_iter=num_iter,
            beta_regularizer=beta_regularizer,
            beta_regularizer2=beta_regularizer2,
            share_beta=share_beta,
            final_adjust_rots=final_adjust_rots if refine_steps == 0 else False,
            scale_target=scale_target,
            scale_fit=scale_fit,
            requested_keys=['pose_rotvecs', 'shape_betas', 'trans'],
        )

        if refine_steps == 0:
            return init

        return self._refine(
            target_vertices=target_vertices,
            target_joints=target_joints,
            vertex_weights=vertex_weights,
            joint_weights=joint_weights,
            init_pose=init['pose_rotvecs'],
            init_betas=init['shape_betas'],
            init_trans=init['trans'],
            init_kid_factor=init.get('kid_factor'),
            beta_regularizer=beta_regularizer,
            num_steps=refine_steps,
            lr=refine_lr,
            warmup_ratio=warmup_ratio,
        )

    def _refine(
        self,
        target_vertices: torch.Tensor,
        target_joints: Optional[torch.Tensor],
        vertex_weights: Optional[torch.Tensor],
        joint_weights: Optional[torch.Tensor],
        init_pose: torch.Tensor,
        init_betas: torch.Tensor,
        init_trans: torch.Tensor,
        init_kid_factor: Optional[torch.Tensor],
        beta_regularizer: float,
        num_steps: int,
        lr: float,
        warmup_ratio: float,
    ) -> dict[str, torch.Tensor]:
        """Refine parameters using Adam optimization with 6D rotation representation.

        Optimizes in global rotation space for faster convergence, since gradients
        on distal joints don't need to propagate through the kinematic chain.
        """
        num_joints = self.body_model.num_joints
        device = self.body_model.v_template.device

        # Convert initial relative rotvecs to global rotation matrices
        init_rotvecs = init_pose.detach().view(-1, num_joints, 3)
        init_rel_rotmats = rotvec2mat(init_rotvecs)
        init_glob_list = [init_rel_rotmats[:, 0]]
        for i in range(1, num_joints):
            ip = self.body_model.kintree_parents[i]
            init_glob_list.append(init_glob_list[ip] @ init_rel_rotmats[:, i])
        init_glob = torch.stack(init_glob_list, dim=1)
        init_rot6d = rotmat_to_rot6d(init_glob)

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

            glob_rotmats = rot6d_to_rotmat(rot6d)
            result = self.body_model(
                glob_rotmats=glob_rotmats,
                shape_betas=betas,
                trans=trans,
                kid_factor=kid_factor,
            )

            # Vertex loss
            loss = torch.tensor(0.0, device=target_vertices.device)

            if vertex_weights is not None:
                v_diff = result['vertices'] - target_vertices
                loss = loss + torch.mean(
                    vertex_weights.unsqueeze(-1) * torch.linalg.norm(v_diff, dim=-1, keepdim=True)
                )
            else:
                loss = loss + torch.mean(
                    torch.linalg.norm(result['vertices'] - target_vertices, dim=-1)
                )

            # Joint loss
            if target_joints is not None:
                if joint_weights is not None:
                    j_diff = result['joints'] - target_joints
                    loss = loss + torch.mean(
                        joint_weights.unsqueeze(-1)
                        * torch.linalg.norm(j_diff, dim=-1, keepdim=True)
                    )
                else:
                    loss = loss + torch.mean(
                        torch.linalg.norm(result['joints'] - target_joints, dim=-1)
                    )

            # L2 regularization on shape betas (skip first two, like the analytical fitter)
            if beta_regularizer > 0 and betas.shape[1] > 2:
                loss = loss + beta_regularizer * torch.mean(betas[:, 2:] ** 2)

            loss.backward()
            optimizer.step()

        # Convert final global 6D to parent-relative rotvecs
        with torch.no_grad():
            glob_rotmats_final = rot6d_to_rotmat(rot6d)
            parent_indices = self.body_model.kintree_parents_tensor[1:].to(device)
            parent_glob = torch.cat(
                [
                    torch.eye(3, device=device).expand(glob_rotmats_final.shape[0], 1, 3, 3),
                    glob_rotmats_final.index_select(1, parent_indices),
                ],
                dim=1,
            )
            rel_rotmats = parent_glob.transpose(-1, -2) @ glob_rotmats_final
            pose_rotvecs = mat2rotvec(rel_rotmats).view(-1, num_joints * 3)

        result = {
            'pose_rotvecs': pose_rotvecs,
            'shape_betas': betas.detach(),
            'trans': trans.detach(),
        }
        if kid_factor is not None:
            result['kid_factor'] = kid_factor.detach()

        return result
