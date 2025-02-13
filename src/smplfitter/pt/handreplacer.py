import torch
import numpy as np
from smplfitter.pt.bodymodel import BodyModel
from smplfitter.pt.bodyfitter import BodyFitter
from smplfitter.pt.bodyconverter import BodyConverter, load_vertex_converter_csr, load_pickle
import os
import pickle
import torch.nn as nn


class HandReplacer(nn.Module):
    """
    Replaces the hand vertices of SMPL with the hand pose of SMPLH.
    """

    def __init__(self, hand_pose_source: torch.Tensor):
        """
        Initialize the HandReplacer.

        Parameters:
            hand_indices_path (str): Path to the hand vertex indices pickle file.
            smplx_bm_config (dict): Configuration for SMPL-X body model.
        """
        # Load hand vertex indices
        super().__init__()
        DATA_ROOT = os.getenv('DATA_ROOT', '.')
        hand_indices = load_pickle(f'{DATA_ROOT}/body_models/smplx/MANO_SMPLX_vertex_ids.pkl')
        smplx_hand_indices_all = list(hand_indices['left_hand']) + list(hand_indices['right_hand'])
        smplx2smpl_csr = load_vertex_converter_csr(
            f'{DATA_ROOT}/body_models/smplx2smpl_deftrafo_setup.pkl'
        )

        smpl_hand_indices_all = (smplx2smpl_csr[:, smplx_hand_indices_all] > 0.5).nonzero()[0]
        self.smplh_bm = BodyModel('smplh16', 'neutral')
        template = self.smplh_bm.single()['vertices']
        hand_min_x = torch.min(torch.abs(template[smpl_hand_indices_all])[:, 0])
        self.hand_mix_weight = smootherstep(
            torch.abs(template[:, 0]), hand_min_x - 0.1, hand_min_x
        )
        self.hand_indices_all = torch.tensor(smpl_hand_indices_all, dtype=torch.long)

        # Initialize converters and body model
        self.smplh_bm = BodyModel('smplh16', 'neutral')
        self.smplh_fitter = BodyFitter(self.smplh_bm, num_betas=16)
        self.hand_pose_source = hand_pose_source
        self.vertex_weights = torch.ones((1, self.smplh_bm.num_vertices), dtype=torch.float32)
        self.vertex_weights[:, self.hand_indices_all] = 1e-1

    def mirror_rotvecs(self, hand_pose: torch.Tensor) -> torch.Tensor:
        hflip_multiplier = torch.tensor(
            [1, -1, -1], dtype=hand_pose.dtype, device=hand_pose.device
        )
        return (hand_pose.reshape(-1, 3) * hflip_multiplier).reshape(-1)

    def copy_hand_params(self, smplh_pose: torch.Tensor) -> None:
        """
        Copy hand parameters from a source to SMPL-X pose.

        Parameters:
            smplx_pose (torch.Tensor): SMPL-X pose to modify.
        """
        start = 22
        left_range = slice(start * 3, (start + 15) * 3)
        right_range = slice((start + 15) * 3, (start + 30) * 3)
        smplh_pose[:, left_range] = self.mirror_rotvecs(self.hand_pose_source[right_range])
        smplh_pose[:, right_range] = self.hand_pose_source[right_range]

    @torch.jit.export
    def replace_hand(self, smpl_verts: torch.Tensor) -> torch.Tensor:
        fit = self.smplh_fitter.fit(
            target_vertices=smpl_verts,
            num_iter=3,
            beta_regularizer=0.0,
            final_adjust_rots=False,
            vertex_weights=self.vertex_weights.repeat(smpl_verts.shape[0], 1),
            requested_keys=['pose_rotvecs', 'shape_betas'],
        )
        self.copy_hand_params(fit['pose_rotvecs'])
        new_res = self.smplh_bm(fit['pose_rotvecs'], fit['shape_betas'], fit['trans'])
        new_verts = new_res['vertices']
        return smpl_verts + (new_verts - smpl_verts) * self.hand_mix_weight[:, np.newaxis]


def smootherstep(x, x0, x1):
    y = torch.clip((x - x0) / (x1 - x0), 0.0, 1.0)
    return y**3 * (y * (y * 6.0 - 15.0) + 10.0)
