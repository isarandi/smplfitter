"""PyTorch implementation of body models and fitters."""

import functools
import os
import torch
import warnings
from smplfitter.pt.bodymodel import BodyModel
from smplfitter.pt.bodyfitter import BodyFitter
from smplfitter.pt.bodyconverter import BodyConverter
from smplfitter.pt.bodyflipper import BodyFlipper
from typing import Optional


__all__ = [
    'BodyModel', 'BodyFitter', 'BodyConverter', 'BodyFlipper',
    'get_cached_body_model', 'get_cached_fit_fn',
    'fit']


@functools.lru_cache()
def get_cached_body_model(model_name='smpl', gender='neutral', model_root=None):
    return get_body_model(model_name, gender, model_root)


def get_body_model(model_name, gender, model_root=None):
    if model_root is None:
        DATA_ROOT = os.getenv('DATA_ROOT', default='.')
        model_root = f'{DATA_ROOT}/body_models/{model_name}'
    return BodyModel(model_root=model_root, gender=gender, model_name=model_name)


@functools.lru_cache()
def get_cached_fit_fn(
        body_model_name='smpl', gender='neutral', num_betas=10, enable_kid=False,
        requested_keys=('pose_rotvecs', 'shape_betas', 'trans'),
        beta_regularizer=1.0, beta_regularizer2=0.0, num_iter=3, vertex_subset=None,
        joint_regressor=None,
        share_beta=False, final_adjust_rots=True, scale_target=False,
        scale_fit=False, scale_regularizer=0.0, kid_regularizer=None, device='cuda'):
    with torch.device(device):
        body_model = BodyModel(gender=gender, model_name=body_model_name)
        fitter = BodyFitter(
            body_model, num_betas=num_betas, enable_kid=enable_kid,
            vertex_subset=torch.as_tensor(vertex_subset) if vertex_subset is not None else None,
            joint_regressor=joint_regressor)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitter = torch.jit.script(fitter)

        def fit_fn(
                verts: torch.Tensor, joints: Optional[torch.Tensor] = None,
                vertex_weights: Optional[torch.Tensor] = None,
                joint_weights: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
            res = fitter.fit(
                verts, target_joints=joints, vertex_weights=vertex_weights,
                joint_weights=joint_weights, num_iter=num_iter,
                beta_regularizer=beta_regularizer, beta_regularizer2=beta_regularizer2,
                scale_regularizer=scale_regularizer,
                kid_regularizer=kid_regularizer,
                share_beta=share_beta, final_adjust_rots=final_adjust_rots,
                scale_target=scale_target, scale_fit=scale_fit,
                requested_keys=list(requested_keys))
            return {k: v for k, v in res.items()}

        def wrapped(verts, joints, vertex_weights, joint_weights):
            verts_resh = verts.view(-1, fitter.num_vertices, 3)
            joints_resh = joints.view(-1, body_model.num_joints, 3) if joints is not None else None
            vertex_weights_resh = (
                vertex_weights.view(-1, fitter.num_vertices)
                if vertex_weights is not None else None)
            joint_weights_resh = (
                joint_weights.view(-1, body_model.num_joints)
                if joint_weights is not None else None)
            res = fit_fn(verts_resh, joints_resh, vertex_weights_resh, joint_weights_resh)
            return {k: v.view(*verts.shape[:-2], *v.shape[1:]) for k, v in res.items()}

        return wrapped


def fit(verts: torch.Tensor, joints: Optional[torch.Tensor] = None,
        vertex_weights: Optional[torch.Tensor] = None, joint_weights: Optional[torch.Tensor] = None,
        body_model_name='smpl', gender='neutral', num_betas=10, enable_kid=False,
        requested_keys=('pose_rotvecs', 'shape_betas', 'trans'), beta_regularizer=1.0,
        beta_regularizer2=0.0, num_iter=3, vertex_subset=None, joint_regressor=None,
        share_beta=False,
        final_adjust_rots=True, scale_target=False, scale_fit=False, scale_regularizer=0.0,
        kid_regularizer=None):
    fit_fn = get_cached_fit_fn(
        body_model_name, gender, num_betas, enable_kid, requested_keys, beta_regularizer,
        beta_regularizer2, num_iter, vertex_subset, joint_regressor, share_beta, final_adjust_rots,
        scale_target, scale_fit, scale_regularizer, kid_regularizer, device=str(verts.device))
    return fit_fn(verts, joints, vertex_weights, joint_weights)
