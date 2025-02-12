"""TensorFlow implementation of body models and the model fitter."""

import functools
import os

import tensorflow as tf
from smplfitter.tf.bodymodel import BodyModel
from smplfitter.tf.bodyfitter import BodyFitter

__all__ = ['BodyModel', 'BodyFitter', 'get_cached_body_model', 'get_cached_fit_fn']

@functools.lru_cache()
def get_cached_body_model(model_name='smpl', gender='neutral', model_root=None):
    return get_body_model(model_name, gender, model_root)


def get_body_model(model_name, gender, model_root=None):
    if model_root is None:
        DATA_ROOT = os.getenv('DATA_ROOT', '.')
        model_root = f'{DATA_ROOT}/body_models/{model_name}'
    return BodyModel(model_root=model_root, gender=gender, model_name=model_name)


@functools.lru_cache()
def get_cached_fit_fn(
        body_model_name='smpl', gender='neutral', num_betas=10, enable_kid=False,
        requested_keys=('pose_rotvecs', 'trans', 'shape_betas', 'joints'),
        l2_regularizer=1, l2_regularizer2=0, num_iter=3, vertex_subset=None, share_beta=False,
        final_adjust_rots=True, weighted=False, scale_target=False, scale_fit=False,
        scale_regularizer=0, kid_regularizer=None):
    return get_fit_fn(
        body_model_name, gender, num_betas, enable_kid, requested_keys, l2_regularizer,
        l2_regularizer2, num_iter, vertex_subset, share_beta, final_adjust_rots, weighted,
        scale_target, scale_fit, scale_regularizer, kid_regularizer)


def get_fit_fn(
        body_model_name='smpl', gender='neutral', num_betas=10, enable_kid=False,
        requested_keys=('pose_rotvecs', 'trans', 'shape_betas', 'joints'),
        l2_regularizer=1, l2_regularizer2=0, num_iter=3, vertex_subset=None, share_beta=False,
        final_adjust_rots=True, weighted=False, scale_target=False, scale_fit=False,
        scale_regularizer=0, kid_regularizer=None):
    body_model = get_cached_body_model(body_model_name, gender)
    fitter = BodyFitter(
        body_model, num_betas=num_betas, enable_kid=enable_kid, vertex_subset=vertex_subset)

    if weighted:
        @tf.function(
            input_signature=[
                tf.TensorSpec([None, fitter.num_vertices, 3], tf.float32),
                tf.TensorSpec([None, body_model.num_joints, 3], tf.float32),
                tf.TensorSpec([None, fitter.num_vertices], tf.float32),
                tf.TensorSpec([None, body_model.num_joints], tf.float32),
            ])
        def fit_fn(verts, joints, vertex_weights, joint_weights):
            res = fitter.fit(verts, target_joints=joints, vertex_weights=vertex_weights,
                             joint_weights=joint_weights, num_iter=num_iter,
                             beta_regularizer=l2_regularizer, beta_regularizer2=l2_regularizer2,
                             scale_regularizer=scale_regularizer, kid_regularizer=kid_regularizer,
                             share_beta=share_beta, final_adjust_rots=final_adjust_rots,
                             scale_target=scale_target, scale_fit=scale_fit,
                             requested_keys=requested_keys)
            return {k: v for k, v in res.items() if v is not None}

        def wrapped(verts, joints, vertex_weights, joint_weights):
            if isinstance(verts, tf.RaggedTensor):
                res = wrapped(
                    verts.flat_values, joints.flat_values, vertex_weights.flat_values,
                    joint_weights.flat_values)
                return tf.nest.map_structure(
                    lambda x: tf.RaggedTensor.from_nested_row_splits(
                        x, verts.nested_row_splits), res)
            else:
                verts_resh = tf.reshape(verts, [-1, fitter.num_vertices, 3])
                joints_resh = tf.reshape(joints, [-1, body_model.num_joints, 3])
                vertex_weights_resh = tf.reshape(vertex_weights, [-1, fitter.num_vertices])
                joint_weights_resh = tf.reshape(joint_weights, [-1, body_model.num_joints])
                res = fit_fn(verts_resh, joints_resh, vertex_weights_resh, joint_weights_resh)
                return {k: tf.reshape(v, verts.shape[:-2] + v.shape[1:]) for k, v in res.items()}
    else:
        @tf.function(
            input_signature=[
                tf.TensorSpec([None, fitter.num_vertices, 3], tf.float32),
                tf.TensorSpec([None, body_model.num_joints, 3], tf.float32),
            ])
        def fit_fn(verts, joints):
            res = fitter.fit(verts, target_joints=joints, num_iter=num_iter,
                             beta_regularizer=l2_regularizer, beta_regularizer2=l2_regularizer2,
                             scale_regularizer=scale_regularizer, kid_regularizer=kid_regularizer,
                             share_beta=share_beta, final_adjust_rots=final_adjust_rots,
                             scale_target=scale_target, scale_fit=scale_fit,
                             requested_keys=requested_keys)
            return {k: v for k, v in res.items() if v is not None}

        def wrapped(verts, joints):
            if isinstance(verts, tf.RaggedTensor):
                res = wrapped(verts.flat_values, joints.flat_values)
                return tf.nest.map_structure(
                    lambda x: tf.RaggedTensor.from_nested_row_splits(
                        x, verts.nested_row_splits), res)
            else:
                verts_resh = tf.reshape(verts, [-1, fitter.num_vertices, 3])
                joints_resh = tf.reshape(joints, [-1, body_model.num_joints, 3])
                res = fit_fn(verts_resh, joints_resh)
                return {k: tf.reshape(v, verts.shape[:-2] + v.shape[1:]) for k, v in res.items()}

    return wrapped
