from __future__ import annotations
# This module finds joint regressors for subsets of body model vertices.
# That is, it finds a linear regressor that maps a subset of vertices to the joints.
# And this is done for the post-LBS case, i.e. the vertices and joints are posed with linear
# blend skinning.
# This is useful in body model fitting in case the user does not provide the joint positions.
# In that case we must run a regressor on the (already posed) vertices.
#
# The joint regressors need to create convex combinations, i.e. the weights must be non-negative
# and sum to 1 for each joint. They should also be sparse and spatially compact.
# That is, each joint should only depend on a few vertices that are spatially close to each other.
# This is achieved by adding a regularization term that encourages the weights to be
# sparse (L1/2 norm), and a term that computes the weighted variance of the template vertices.

import os

import numpy as np
from .. import pt as smplfitter_pt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = os.getenv('DATA_ROOT', default='.')
torch.backends.cudnn.benchmark = True


def main():
    for body_model_name in ['smpl', 'smplx']:
        canonical_verts = torch.from_numpy(
            np.load(f'{DATA_ROOT}/nlf/canonical_vertices_{body_model_name}.npy')
        ).float()
        body_model = smplfitter_pt.BodyModel(body_model_name).to(DEVICE)
        dataset = RandomBodyParamDataset(num_betas=16, num_joints=body_model.num_joints)
        dataloader = DataLoader(dataset, batch_size=None, num_workers=4, pin_memory=True)

        for n_verts_subset in reversed(
            [32, 64, 128, 256, 512, 1024, 2048, 4096, body_model.num_vertices]
        ):
            print(f'Fitting joint regressor for {n_verts_subset} vertices')
            out_path = (
                f'{DATA_ROOT}/body_models/{body_model_name}/'
                f'vertex_subset_joint_regr_post_lbs_{n_verts_subset}.npy'
            )
            # if os.path.exists(out_path):
            #     continue

            if n_verts_subset == body_model.num_vertices:
                i_verts = torch.arange(body_model.num_vertices)
            else:
                subset = np.load(
                    f'{DATA_ROOT}/body_models/{body_model_name}/vertex_subset_{n_verts_subset}.npz'
                )
                i_verts = torch.from_numpy(subset['i_verts'])

            model = ConvexCombiningRegressor(len(i_verts), body_model.num_joints).to(DEVICE)
            trainer = ConvexCombiningRegressorTrainer(
                model=model,
                body_model=body_model,
                template_verts=canonical_verts[i_verts],
                vertex_subset=i_verts,
                regul_lambda=6e-5,
            )
            optimizer = optim.Adam(model.parameters(), lr=1e0)
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda step: 1.0 if step < int(37500 * 0.9) else 1e-3
            )
            trainer.train_loop(
                dataloader, total_steps=37500, optimizer=optimizer, scheduler=scheduler
            )
            model.threshold_for_sparsity(1e-3)

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 30)
            trainer.train_loop(
                dataloader, total_steps=37500 + 12500, optimizer=optimizer, scheduler=scheduler
            )
            model.threshold_for_sparsity(1e-3)

            J_subset = model.get_w().cpu().detach().numpy().T
            print(f'Sparsity ratio: {sparsity_ratio(J_subset)}')
            np.save(out_path, J_subset)


class ConvexCombiningRegressor(nn.Module):
    def __init__(self, n_in_points, n_out_points):
        super().__init__()
        self.n_in = n_in_points
        self.n_out = n_out_points

        self.weight_mask = nn.Parameter(torch.ones(n_in_points, n_out_points), requires_grad=False)
        self.weights = nn.Parameter(torch.empty(n_in_points, n_out_points))
        nn.init.uniform_(self.weights, -1, 1)

    def forward(self, x):
        w = self.get_w()
        return torch.einsum('bjc,jJ->bJc', x, w)

    def get_w(self):
        w = torch.nn.functional.softplus(self.weights) * self.weight_mask
        return w / w.sum(dim=0, keepdim=True)

    def threshold_for_sparsity(self, threshold=1e-3):
        with torch.no_grad():
            self.weight_mask.data = (torch.abs(self.get_w()) > threshold).float()


class ConvexCombiningRegressorTrainer:
    def __init__(self, model, body_model, template_verts, vertex_subset, regul_lambda=3e-5):
        self.model = model
        self.body_model = body_model
        self.template_verts = template_verts.to(DEVICE)
        self.vertex_subset = vertex_subset
        self.regul_lambda = regul_lambda
        self.current_step = 0

    def train_loop(self, dataloader, total_steps, optimizer, scheduler=None):
        progress_bar = tqdm(
            total=total_steps,
            desc='Training',
            unit='step',
            dynamic_ncols=True,
            initial=self.current_step,
        )

        for batch in dataloader:
            batch = EasyDict({k: v.to(DEVICE) for k, v in batch.items()})
            need_metrics = self.current_step % 100 == 0
            losses, metrics = self.train_step(
                batch, optimizer, scheduler, need_metrics=need_metrics
            )
            self.current_step += 1
            progress_bar.update(1)

            if need_metrics:
                losses_and_metrics_str = ', '.join(
                    f'{k}: {v.item():.4f}' for k, v in (losses | metrics).items()
                )
                progress_bar.set_postfix_str(losses_and_metrics_str)

            if self.current_step >= total_steps:
                break

    def forward_train(self, inps):
        preds = EasyDict()
        r = self.body_model(inps.pose, inps.shape)
        preds.postjoints = self.body_model.J_regressor_post_lbs @ r['vertices']
        preds.pose3d = self.model(r['vertices'][:, self.vertex_subset])
        preds.pose3d_gt = r['joints']
        return preds

    def train_step(self, inps, optimizer, scheduler=None, need_metrics=True):
        self.model.train()
        optimizer.zero_grad()
        preds = self.forward_train(inps)
        losses = self.compute_losses(inps, preds)
        losses.loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if need_metrics:
            with torch.no_grad():
                metrics = self.compute_metrics(inps, preds)
        else:
            metrics = None

        return losses, metrics

    def compute_losses(self, inps, preds):
        losses = EasyDict()
        losses.main_loss = torch.mean(torch.abs(preds.pose3d_gt - preds.pose3d))

        w = self.model.get_w()
        losses.regul = torch.sum(soft_sqrt(torch.abs(w), 1e-5)) / w.shape[1]
        # losses.supp = mean_spatial_support(self.template_verts, w)
        losses.loss = (
            losses.main_loss + self.regul_lambda * losses.regul  # + self.supp_lambda * losses.supp
        )
        return losses

    def compute_metrics(self, inps, preds):
        m = EasyDict()
        dist = torch.norm(preds.pose3d_gt - preds.pose3d, dim=-1)
        m.pck1 = pck(dist, 0.01)
        m.pck2 = pck(dist, 0.02)
        m.pck3 = pck(dist, 0.03)
        m.pck7 = pck(dist, 0.07)
        m.euclidean = torch.mean(dist)
        m.l1 = torch.mean(torch.abs(preds.pose3d_gt - preds.pose3d))
        m.max_supp = torch.max(
            torch.sqrt(spatial_support(self.template_verts, self.model.get_w()))
        )

        dist = torch.norm(preds.pose3d_gt - preds.postjoints, dim=-1)
        m.pck1_post = pck(dist, 0.01)
        return m


class RandomBodyParamDataset(IterableDataset):
    def __init__(self, batch_size=72, num_joints=24, num_betas=16):
        self.batch_size = batch_size
        self.num_joints = num_joints
        self.num_betas = num_betas
        self.rng = torch.Generator()

    def __iter__(self):
        with torch.no_grad():
            while True:
                pose = self.random_smpl_pose() / 3
                shape = torch.empty((self.batch_size, self.num_betas))
                nn.init.trunc_normal_(shape, mean=0.0, std=1.0, a=-2.0, b=2.0, generator=self.rng)
                yield {'pose': pose, 'shape': shape * 10}

    def random_rotvec(self, batch_size):
        rand = torch.rand(3, batch_size, generator=self.rng)
        r1 = torch.sqrt(1 - rand[0])
        r2 = torch.sqrt(rand[0])
        t1 = 2 * np.pi * rand[1]
        t2 = 2 * np.pi * rand[2]
        cost2 = torch.cos(t2)
        xyz = torch.stack([torch.sin(t1) * r1, torch.cos(t1) * r1, torch.sin(t2) * r2], dim=0)
        return (xyz / torch.sqrt(1 - cost2**2 * rand[0]) * 2 * torch.acos(cost2 * r2)).T

    def random_smpl_pose(self):
        return self.random_rotvec(self.batch_size * self.num_joints).reshape(
            self.batch_size, self.num_joints * 3
        )


def sparsity_ratio(J):
    return np.count_nonzero(np.abs(J) > 1e-4) / J.shape[0]


def soft_sqrt(x, eps):
    return x / torch.sqrt(x + eps)


def spatial_support(template, weights):
    weighted_mean = torch.matmul(weights.t(), template)
    sq_diff = (template[np.newaxis, :] - weighted_mean[:, np.newaxis]) ** 2
    sq_dists = torch.sum(sq_diff, dim=-1)
    return torch.einsum('Jj,jJ->J', sq_dists, torch.abs(weights))


def pck(x, t):
    return (x <= t).float().mean()


class EasyDict(dict):
    def __init__(self, d=None, **kwargs):
        super().__init__()
        if d is None:
            d = {}
        else:
            d = dict(d)
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')) and k not in ('update', 'pop'):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        super().__setitem__(name, value)

    def __getattr__(self, name):
        try:
            return super().__getitem__(name)
        except KeyError:
            raise AttributeError(name)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k, v in d.items():
            setattr(self, k, v)

    def pop(self, k, d=None):
        delattr(self, k)
        return super().pop(k, d)


if __name__ == '__main__':
    main()
