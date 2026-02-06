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

import fleras
import fleras.callbacks
import fleras.optimizers
import numpy as np
from .. import tf as smplfitter_tf
import tensorflow as tf
from fleras import EasyDict


def main():
    DATA_ROOT = os.getenv('DATA_ROOT', default='.')
    canonical_verts = np.load(f'{DATA_ROOT}/nlf/canonical_vertices_smpl.npy')
    train_ds_random = make_random_ds(3 * 24)

    for n_verts_subset in reversed([32, 64, 128, 256, 512, 1024, 2048, 4096, 6890]):
        print(f'Fitting joint regressor for {n_verts_subset} vertices')
        out_path = (
            f'{DATA_ROOT}/body_models/smpl/vertex_subset_joint_regr_post_lbs_{n_verts_subset}_.npy'
        )
        # if os.path.exists(out_path):
        #    print(f'File {out_path} already exists, skipping')
        #    continue
        if n_verts_subset == 6890:
            i_verts = np.arange(6890)
        else:
            vertex_subset = np.load(
                f'{DATA_ROOT}/body_models/smpl/vertex_subset_{n_verts_subset}.npz'
            )
            i_verts = vertex_subset['i_verts']

        model = ConvexCombiningRegressor(n_verts_subset, 24)
        bm = smplfitter_tf.get_cached_body_model('smpl', 'neutral')
        trainer = ConvexCombiningRegressorTrainer(
            model,
            regul_lambda=3e-5,
            supp_lambda=3e-1,
            random_seed=0,
            pose3d_template=canonical_verts[i_verts],
            body_model=bm,
            vertex_subset=i_verts,
        )

        trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule()))
        trainer.fit_epochless(
            train_ds_random,
            steps=37500,
            verbose=1,
            callbacks=[fleras.callbacks.ProgbarLogger(), tf.keras.callbacks.History()],
        )
        model.layer.threshold_for_sparsity(1e-3)

        trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=30))
        trainer.fit_epochless(
            train_ds_random,
            steps=12500,
            verbose=1,
            callbacks=[fleras.callbacks.ProgbarLogger(), tf.keras.callbacks.History()],
        )
        model.layer.threshold_for_sparsity(1e-3)

        J_subset = model.layer.get_w().numpy().T
        print(f'Sparsity ratio: {sparsity_ratio(J_subset)}')

        np.save(out_path, J_subset)


def sparsity_ratio(J):
    is_nonzero = np.abs(J) > 1e-4
    return np.count_nonzero(is_nonzero) / J.shape[0]


@fleras.optimizers.schedules.wrap(jit_compile=True)
def lr_schedule(step):
    n_total_steps = 37500
    if step < int(n_total_steps * 0.9):
        return 1e0
    else:
        return 1e-3


def random_rotvec(rng, batch_size):
    rand = rng.rand(3, batch_size).astype(np.float32)
    r1 = np.sqrt(1 - rand[0])
    r2 = np.sqrt(rand[0])
    pi2 = np.pi * 2
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    cost2 = np.cos(t2)
    xyz = np.stack([np.sin(t1) * r1, np.cos(t1) * r1, np.sin(t2) * r2], axis=0)
    return (xyz / np.sqrt(1 - cost2**2 * rand[0]) * 2 * np.arccos(cost2 * r2)).transpose(1, 0)


def random_smpl_pose(rng, batch_size):
    return random_rotvec(rng, batch_size * 24).reshape(batch_size, 24 * 3)


def make_random_ds(batch_size):
    def generate_random_example():
        tf_rng = tf.random.Generator.from_non_deterministic_state()
        np_rng = np.random.RandomState()

        while True:
            yield dict(
                pose=tf.constant(random_smpl_pose(np_rng, batch_size)) / 3,
                shape=tf_rng.truncated_normal(shape=(batch_size, 16)) * 10,
            )

    return tf.data.Dataset.from_generator(
        generate_random_example,
        output_signature=dict(
            pose=tf.TensorSpec(shape=(batch_size, 24 * 3), dtype=tf.float32),
            shape=tf.TensorSpec(shape=(batch_size, 16), dtype=tf.float32),
        ),
    )


class ConvexCombiningRegressor(tf.keras.Model):
    def __init__(self, n_in_points, n_out_points):
        super().__init__()
        self.layer = ConvexCombinationLayer(n_in_points, n_out_points)

    def call(self, inp):
        return self.layer(inp)


class ConvexCombiningRegressorTrainer(fleras.ModelTrainer):
    def __init__(
        self,
        model,
        regul_lambda,
        supp_lambda,
        pose3d_template,
        body_model,
        vertex_subset,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.regul_lambda = regul_lambda
        self.supp_lambda = supp_lambda
        self.pose3d_template = tf.convert_to_tensor(pose3d_template, dtype=tf.float32)
        self.body_model = body_model
        self.vertex_subset = vertex_subset

    def forward_train(self, inps, training):
        preds = EasyDict()
        r = self.body_model(inps.pose, inps.shape)
        preds.postjoints = self.body_model.J_regressor @ r['vertices']
        preds.pose3d = self.model(tf.gather(r['vertices'], self.vertex_subset, axis=1))
        preds.pose3d_gt = r['joints']
        return preds

    def compute_losses(self, inps, preds):
        losses = EasyDict()
        losses.main_loss = tf.reduce_mean(tf.abs(preds.pose3d_gt - preds.pose3d))

        w = self.model.layer.get_w()
        losses.regul = tf.reduce_sum(soft_sqrt(tf.abs(w), 1e-5)) / w.shape[1]
        losses.supp = mean_spatial_support(self.pose3d_template, w)
        losses.loss = (
            losses.main_loss + self.regul_lambda * losses.regul + self.supp_lambda * losses.supp
        )
        return losses

    def compute_metrics(self, inps, preds, training):
        m = EasyDict()
        dist = tf.linalg.norm(preds.pose3d_gt - preds.pose3d, axis=-1)
        m.pck1 = pck(dist, 0.01)
        m.pck2 = pck(dist, 0.02)
        m.pck3 = pck(dist, 0.03)
        m.pck7 = pck(dist, 0.07)
        m.euclidean = tf.reduce_mean(dist)
        m.l1 = tf.reduce_mean(tf.abs(preds.pose3d_gt - preds.pose3d))
        m.max_supp = tf.reduce_max(
            tf.sqrt(spatial_support(self.pose3d_template, self.model.layer.get_w()))
        )

        dist = tf.linalg.norm(preds.pose3d_gt - preds.postjoints, axis=-1)
        m.pck1_post = pck(dist, 0.01)
        return m


class ConvexCombinationLayer(tf.keras.layers.Layer):
    def __init__(self, n_in_points, n_out_points):
        super().__init__()
        self.n_in_points = n_in_points
        self.n_out_points = n_out_points

        self.weight_mask = self.add_weight(
            'weight_mask',
            shape=(self.n_in_points, self.n_out_points),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Ones(),
            trainable=False,
        )

        self.w = self.add_weight(
            'weights',
            shape=(self.n_in_points, self.n_out_points),
            dtype=tf.float32,
            initializer=tf.keras.initializers.random_uniform(-1, 1),
        )

    def call(self, inputs):
        return tf.einsum('bjc,jJ->bJc', inputs, self.get_w())

    def get_w(self):
        return normalize_weights(tf.nn.softplus(self.w) * self.weight_mask)

    def threshold_for_sparsity(self, threshold):
        self.weight_mask.assign(tf.cast(tf.abs(self.get_w()) > threshold, tf.float32))


def soft_sqrt(x, eps):
    return x / tf.sqrt(x + eps)


def normalize_weights(w):
    return w / tf.reduce_sum(w, axis=0, keepdims=True)


def block_concat(inp):
    return tf.concat([tf.concat(arrs, axis=1) for arrs in inp], axis=0)


def sum_squared_difference(a, b, axis=-1):
    return tf.reduce_sum(tf.math.squared_difference(a, b), axis=axis)


def mean_spatial_support(template, weights):
    weighted_mean = convert_pose(template, weights)
    sum_squared_diff = sum_squared_difference(
        template[tf.newaxis, :], weighted_mean[:, tf.newaxis]
    )  # Jj
    return tf.einsum('Jj,jJ->', sum_squared_diff, tf.abs(weights)) / weights.shape[1]


def spatial_support(template, weights):
    weighted_mean = convert_pose(template, weights)
    sq_diff = tf.math.squared_difference(
        template[tf.newaxis, :], weighted_mean[:, tf.newaxis]
    )  # Jjc
    sq_dists = tf.reduce_sum(sq_diff, axis=-1)  # Jj
    return tf.einsum('Jj,jJ->J', sq_dists, tf.abs(weights))


def convert_pose(pose, weights):
    return tf.matmul(weights, pose, transpose_a=True)


def pck(x, t):
    return tf.reduce_mean(tf.cast(x <= t, tf.float32))


if __name__ == '__main__':
    main()
