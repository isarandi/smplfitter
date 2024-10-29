import numpy as np
from smplfitter.tf.fitter import SMPLFitter
import tensorflow as tf

import smplfitter.tf


class Converter(tf.Module):
    def __init__(self):
        super().__init__()
        self.smpl_female = smplfitter.tf.SMPLBodyModel(model_name='smpl', gender='female')
        self.smpl_male = smplfitter.tf.SMPLBodyModel(model_name='smpl', gender='male')
        self.smpl_neutral = smplfitter.tf.SMPLBodyModel(model_name='smpl', gender='neutral')

        self.smplx_female = smplfitter.tf.SMPLBodyModel(model_name='smplx', gender='female')
        self.smplx_male = smplfitter.tf.SMPLBodyModel(model_name='smplx', gender='male')
        self.smplx_neutral = smplfitter.tf.SMPLBodyModel(model_name='smplx', gender='neutral')

        self.smplxlh_female = smplfitter.tf.SMPLBodyModel(model_name='smplxlh', gender='female')
        self.smplxlh_male = smplfitter.tf.SMPLBodyModel(model_name='smplxlh', gender='male')
        self.smplxlh_neutral = smplfitter.tf.SMPLBodyModel(model_name='smplxlh', gender='neutral')

        self.smplh16_female = smplfitter.tf.SMPLBodyModel(model_name='smplh16', gender='female')
        self.smplh16_male = smplfitter.tf.SMPLBodyModel(model_name='smplh16', gender='male')
        self.smplh16_neutral = smplfitter.tf.SMPLBodyModel(model_name='smplh16', gender='neutral')

        self.smplh_female = smplfitter.tf.SMPLBodyModel(model_name='smplh', gender='female')
        self.smplh_male = smplfitter.tf.SMPLBodyModel(model_name='smplh', gender='male')

        self.smpl2smplx_mat = (spu.load_pickle(
            f'{DATA_ROOT}/body_models/smpl2smplx_deftrafo_setup.pkl'
        )['mtx'].tocsr()[:, :6890]).astype(np.float32)
        self.smplx2smpl_mat = (spu.load_pickle(
            f'{DATA_ROOT}/body_models/smplx2smpl_deftrafo_setup.pkl'
        )['mtx'].tocsr()[:, :10475]).astype(np.float32)

    @tf.function(input_signature=[
        tf.TensorSpec([], tf.string),
        tf.TensorSpec([], tf.string),
        tf.TensorSpec([], tf.string),
        tf.TensorSpec([], tf.string),
        tf.TensorSpec([None, None], tf.float32),
        tf.TensorSpec([None, None], tf.float32),
        tf.TensorSpec([None, 3], tf.float32),
        tf.TensorSpec([], tf.int32), ])
    def convert(self, body_model_in, gender_in, body_model_out, gender_out, pose, betas, trans,
                num_betas_out=10):
        if body_model_in == 'smpl':
            if gender_in == 'female':
                res = self.smpl_female(pose, betas, trans)
            elif gender_in == 'male':
                res = self.smpl_male(pose, betas, trans)
            else:
                res = self.smpl_neutral(pose, betas, trans)
        elif body_model_in == 'smplx':
            if gender_in == 'female':
                res = self.smplx_female(pose, betas, trans)
            elif gender_in == 'male':
                res = self.smplx_male(pose, betas, trans)
            else:
                res = self.smplx_neutral(pose, betas, trans)
        elif body_model_in == 'smplxlh':
            if gender_in == 'female':
                res = self.smplxlh_female(pose, betas, trans)
            elif gender_in == 'male':
                res = self.smplxlh_male(pose, betas, trans)
            else:
                res = self.smplxlh_neutral(pose, betas, trans)
        elif body_model_in == 'smplh':
            if gender_in == 'female':
                res = self.smplh_female(pose, betas, trans)
            else:
                res = self.smplh_male(pose, betas, trans)
        else:
            if gender_in == 'female':
                res = self.smplh16_female(pose, betas, trans)
            elif gender_in == 'male':
                res = self.smplh16_male(pose, betas, trans)
            else:
                res = self.smplh16_neutral(pose, betas, trans)

        if (body_model_in == 'smpl' or body_model_in == 'smplh' or body_model_in == 'smplh16') and (
                body_model_out == 'smplx' or body_model_out == 'smplxlh'):
            verts = self.smpl_verts_to_smplx_verts(res['vertices'])
        elif (body_model_in == 'smplx' or body_model_in == 'smplxlh') and (
                body_model_out == 'smpl' or body_model_out == 'smplh' or body_model_out ==
                'smplh16'):
            verts = self.smplx_verts_to_smpl_verts(res['vertices'])
        else:
            verts = res['vertices']

        if body_model_out == 'smpl':
            if gender_out == 'female':
                fit = SMPLFitter(self.smpl_female,
                                 num_betas=num_betas_out).fit(
                    target_vertices=verts, n_iter=1, beta_regularizer=0, final_adjust_rots=True,
                    requested_keys=[
                        'pose_rotvecs',
                        'shape_betas'])
                return fit['pose_rotvecs'], fit['shape_betas'], fit['trans']
            elif gender_out == 'male':
                fit = SMPLFitter(self.smpl_male,
                                 num_betas=num_betas_out).fit(
                    target_vertices=verts, n_iter=1, beta_regularizer=0, final_adjust_rots=True,
                    requested_keys=[
                        'pose_rotvecs',
                        'shape_betas'])
                return fit['pose_rotvecs'], fit['shape_betas'], fit['trans']
            else:
                fit = SMPLFitter(self.smpl_neutral,
                                 num_betas=num_betas_out).fit(
                    target_vertices=verts, n_iter=1, beta_regularizer=0, final_adjust_rots=True,
                    requested_keys=[
                        'pose_rotvecs',
                        'shape_betas'])
                return fit['pose_rotvecs'], fit['shape_betas'], fit['trans']
        elif body_model_out == 'smplx':
            if gender_out == 'female':
                fit = SMPLFitter(self.smplx_female,
                                 num_betas=num_betas_out).fit(
                    target_vertices=verts, n_iter=1, beta_regularizer=0, final_adjust_rots=True,
                    requested_keys=[
                        'pose_rotvecs',
                        'shape_betas'])
                return fit['pose_rotvecs'], fit['shape_betas'], fit['trans']
            elif gender_out == 'male':
                fit = SMPLFitter(self.smplx_male,
                                 num_betas=num_betas_out).fit(
                    target_vertices=verts, n_iter=1, beta_regularizer=0, final_adjust_rots=True,
                    requested_keys=[
                        'pose_rotvecs',
                        'shape_betas'])
                return fit['pose_rotvecs'], fit['shape_betas'], fit['trans']
            else:
                fit = SMPLFitter(self.smplx_neutral,
                                 num_betas=num_betas_out).fit(
                    target_vertices=verts, n_iter=1, beta_regularizer=0, final_adjust_rots=True,
                    requested_keys=[
                        'pose_rotvecs',
                        'shape_betas'])
                return fit['pose_rotvecs'], fit['shape_betas'], fit['trans']
        elif body_model_out == 'smplxlh':
            if gender_out == 'female':
                fit = SMPLFitter(self.smplxlh_female,
                                 num_betas=num_betas_out).fit(
                    target_vertices=verts, n_iter=1, beta_regularizer=0, final_adjust_rots=True,
                    requested_keys=[
                        'pose_rotvecs',
                        'shape_betas'])
                return fit['pose_rotvecs'], fit['shape_betas'], fit['trans']
            elif gender_out == 'male':
                fit = SMPLFitter(self.smplxlh_male,
                                 num_betas=num_betas_out).fit(
                    target_vertices=verts, n_iter=1, beta_regularizer=0, final_adjust_rots=True,
                    requested_keys=[
                        'pose_rotvecs',
                        'shape_betas'])
                return fit['pose_rotvecs'], fit['shape_betas'], fit['trans']
            else:
                fit = SMPLFitter(self.smplxlh_neutral,
                                 num_betas=num_betas_out).fit(
                    target_vertices=verts, n_iter=1, beta_regularizer=0, final_adjust_rots=True,
                    requested_keys=[
                        'pose_rotvecs',
                        'shape_betas'])
                return fit['pose_rotvecs'], fit['shape_betas'], fit['trans']
        elif body_model_out == 'smplh':
            if gender_out == 'female':
                fit = SMPLFitter(self.smplh_female,
                                 num_betas=num_betas_out).fit(
                    target_vertices=verts, n_iter=1, beta_regularizer=0, final_adjust_rots=True,
                    requested_keys=[
                        'pose_rotvecs',
                        'shape_betas'])
                return fit['pose_rotvecs'], fit['shape_betas'], fit['trans']
            else:
                fit = SMPLFitter(self.smplh_male,
                                 num_betas=num_betas_out).fit(
                    target_vertices=verts, n_iter=1, beta_regularizer=0, final_adjust_rots=True,
                    requested_keys=[
                        'pose_rotvecs',
                        'shape_betas'])
                return fit['pose_rotvecs'], fit['shape_betas'], fit['trans']
        else:
            if gender_out == 'female':
                fit = SMPLFitter(self.smplh16_female,
                                 num_betas=num_betas_out).fit(
                    target_vertices=verts, n_iter=1, beta_regularizer=0, final_adjust_rots=True,
                    requested_keys=[
                        'pose_rotvecs',
                        'shape_betas'])
                return fit['pose_rotvecs'], fit['shape_betas'], fit['trans']
            elif gender_out == 'male':
                fit = SMPLFitter(self.smplh16_male,
                                 num_betas=num_betas_out).fit(
                    target_vertices=verts, n_iter=1, beta_regularizer=0, final_adjust_rots=True,
                    requested_keys=[
                        'pose_rotvecs',
                        'shape_betas'])
                return fit['pose_rotvecs'], fit['shape_betas'], fit['trans']
            else:
                fit = SMPLFitter(self.smplh16_neutral,
                                 num_betas=num_betas_out).fit(
                    target_vertices=verts, n_iter=1, beta_regularizer=0, final_adjust_rots=True,
                    requested_keys=[
                        'pose_rotvecs',
                        'shape_betas'])
                return fit['pose_rotvecs'], fit['shape_betas'], fit['trans']

    def smpl_verts_to_smplx_verts(self, smpl_verts):
        smpl_verts2smplx_coo = self.smpl2smplx_mat.tocoo()
        tf_csr = tf.raw_ops.SparseTensorToCSRSparseMatrix(
            indices=tf.cast(np.stack([smpl_verts2smplx_coo.row, smpl_verts2smplx_coo.col], axis=1),
                            tf.int64),
            values=tf.cast(smpl_verts2smplx_coo.data, tf.float32),
            dense_shape=tf.cast(smpl_verts2smplx_coo.shape, tf.int64))
        v = tf.reshape(tf.transpose(smpl_verts, (1, 0, 2)), (6890, -1))
        r = tf.raw_ops.SparseMatrixMatMul(a=tf_csr, b=v)
        return tf.transpose(tf.reshape(r, (10475, -1, 3)), (1, 0, 2))

    def smplx_verts_to_smpl_verts(self, smplx_verts):
        smplx_verts2smpl_coo = self.smplx2smpl_mat.tocoo()
        tf_csr = tf.raw_ops.SparseTensorToCSRSparseMatrix(
            indices=tf.cast(np.stack([smplx_verts2smpl_coo.row, smplx_verts2smpl_coo.col], axis=1),
                            tf.int64),
            values=tf.cast(smplx_verts2smpl_coo.data, tf.float32),
            dense_shape=tf.cast(smplx_verts2smpl_coo.shape, tf.int64))
        v = tf.reshape(tf.transpose(smplx_verts, (1, 0, 2)), (10475, -1))
        r = tf.raw_ops.SparseMatrixMatMul(a=tf_csr, b=v)
        return tf.transpose(tf.reshape(r, (6890, -1, 3)), (1, 0, 2))
