import tensorflow as tf

import smplfitter.common
from smplfitter.tf.rotation import mat2rotvec, rotvec2mat


class SMPLBodyModel:
    def __init__(self, model_name='smpl', gender='neutral', model_root=None, unit='m', num_betas=None):
        """
        Args:
            model_root: path to pickle files for the model (see https://smpl.is.tue.mpg.de).
            gender: 'neutral', 'f' or 'm'
        """
        self.gender = gender
        self.model_name = model_name
        tensors, nontensors = smplfitter.common.initialize(model_name, gender, model_root, num_betas)
        self.v_template = tf.constant(tensors['v_template'], tf.float32)
        self.shapedirs = tf.constant(tensors['shapedirs'], tf.float32)
        self.posedirs = tf.constant(tensors['posedirs'], tf.float32)
        self.J_regressor = tf.constant(tensors['J_regressor'], tf.float32)
        self.J_template = tf.constant(tensors['J_template'], tf.float32)
        self.J_shapedirs = tf.constant(tensors['J_shapedirs'], tf.float32)
        self.kid_shapedir = tf.constant(tensors['kid_shapedir'], tf.float32)
        self.kid_J_shapedir = tf.constant(tensors['kid_J_shapedir'], tf.float32)
        self.weights = tf.constant(tensors['weights'], tf.float32)
        self.kintree_parents = nontensors['kintree_parents']
        self.faces = nontensors['faces']
        self.num_joints = nontensors['num_joints']
        self.num_vertices = nontensors['num_vertices']
        self.unit_factor = dict(mm=1000, cm=100, m=1)[unit]

    def __call__(
            self, pose_rotvecs=None, shape_betas=None, trans=None, kid_factor=None,
            rel_rotmats=None, glob_rotmats=None, *, return_vertices=True):
        """Calculate the SMPL body model vertices, joint positions, and orientations given the input
         pose, shape parameters.

        Args:
            pose_rotvecs (tf.Tensor, optional): Tensor representing rotation vectors for each
            joint pose.
            shape_betas (tf.Tensor, optional): Tensor representing shape coefficients (betas) for
            the body shape.
            trans (tf.Tensor, optional): Tensor representing the translation vector to apply
            after posing.
            kid_factor (float, optional): Adjustment factor for child shapes. Defaults to 0.
            rel_rotmats (tf.Tensor, optional): Tensor representing the rotation matrices for each
            joint in the pose.
            glob_rotmats (tf.Tensor, optional): Tensor representing global rotation matrices for
            the pose.
            return_vertices (bool, optional): Flag indicating whether to return the body model
            vertices. Defaults to True.

        Returns:
            dict: Dictionary containing 3D body model vertices ('vertices'), joint positions (
            'joints'),
                  and orientation matrices for each joint ('orientations').
        """
        if isinstance(shape_betas, tf.RaggedTensor):
            res = self(
                pose_rotvecs=pose_rotvecs.flat_values, shape_betas=shape_betas.flat_values,
                trans=trans.flat_values, return_vertices=return_vertices)
            return tf.nest.map_structure(
                lambda x: tf.RaggedTensor.from_row_splits(x, shape_betas.row_splits), res)

        batch_size = check_batch_size(pose_rotvecs, shape_betas, trans, rel_rotmats, glob_rotmats)
        if rel_rotmats is not None:
            rel_rotmats = tf.cast(rel_rotmats, tf.float32)
        elif pose_rotvecs is not None:
            pose_rotvecs = tf.cast(pose_rotvecs, tf.float32)
            rel_rotmats = rotvec2mat(tf.reshape(pose_rotvecs, (batch_size, self.num_joints, 3)))
        elif glob_rotmats is None:
            rel_rotmats = tf.eye(3, batch_shape=[batch_size, self.num_joints])

        if glob_rotmats is None:
            glob_rotmats = [rel_rotmats[:, 0]]
            for i_joint in range(1, self.num_joints):
                i_parent = self.kintree_parents[i_joint]
                glob_rotmats.append(glob_rotmats[i_parent] @ rel_rotmats[:, i_joint])
            glob_rotmats = tf.stack(glob_rotmats, axis=1)

        parent_glob_rotmats = tf.concat([
            tf.broadcast_to(tf.eye(3), tf.shape(glob_rotmats[:, :1])),
            tf.gather(glob_rotmats, self.kintree_parents[1:], axis=1)], axis=1)

        if rel_rotmats is None:
            rel_rotmats = tf.linalg.matmul(parent_glob_rotmats, glob_rotmats, transpose_a=True)

        shape_betas = (tf.cast(shape_betas, tf.float32) if shape_betas is not None
                       else tf.zeros((batch_size, 0), tf.float32))
        num_betas = tf.minimum(tf.shape(shape_betas)[1], self.shapedirs.shape[2])

        if kid_factor is None:
            kid_factor = tf.zeros((1,), tf.float32)
        else:
            kid_factor = tf.cast(kid_factor, tf.float32)
        j = (self.J_template +
             tf.einsum(
                 'jcs,bs->bjc', self.J_shapedirs[:, :, :num_betas], shape_betas[:, :num_betas]) +
             tf.einsum('jc,b->bjc', self.kid_J_shapedir, kid_factor))

        j_parent = tf.concat([
            tf.broadcast_to(tf.zeros(3), tf.shape(j[:, :1])),
            tf.gather(j, self.kintree_parents[1:], axis=1)], axis=1)
        bones = j - j_parent
        rotated_bones = tf.einsum('bjCc,bjc->bjC', parent_glob_rotmats, bones)

        glob_positions = [j[:, 0]]
        for i_joint in range(1, self.num_joints):
            i_parent = self.kintree_parents[i_joint]
            glob_positions.append(glob_positions[i_parent] + rotated_bones[:, i_joint])
        glob_positions = tf.stack(glob_positions, axis=1)

        if trans is None:
            trans = tf.zeros((1, 3), tf.float32)
        else:
            trans = tf.cast(trans, tf.float32)

        if not return_vertices:
            return dict(joints=glob_positions + trans[:, tf.newaxis], orientations=glob_rotmats)

        pose_feature = tf.reshape(rel_rotmats[:, 1:], [-1, (self.num_joints - 1) * 3 * 3])
        v_posed = (
                self.v_template +
                tf.einsum(
                    'vcp,bp->bvc', self.shapedirs[:, :, :num_betas], shape_betas[:, :num_betas]) +
                tf.einsum('vcp,bp->bvc', self.posedirs, pose_feature) +
                tf.einsum('vc,b->bvc', self.kid_shapedir, kid_factor))

        translations = glob_positions - tf.einsum('bjCc,bjc->bjC', glob_rotmats, j)
        vertices = (
                tf.einsum('bjCc,vj,bvc->bvC', glob_rotmats, self.weights, v_posed) +
                self.weights @ translations)

        return dict(
            joints=(glob_positions + trans[:, tf.newaxis]) * self.unit_factor,
            vertices=(vertices + trans[:, tf.newaxis]) * self.unit_factor,
            orientations=glob_rotmats)

    def single(self, *args, return_vertices=True, **kwargs):
        args = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), args)
        kwargs = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), kwargs)
        if len(args) == 0 and len(kwargs) == 0:
            kwargs['shape_betas'] = tf.zeros((1, 0), tf.float32)
        result = self(*args, return_vertices=return_vertices, **kwargs)
        return tf.nest.map_structure(lambda x: tf.squeeze(x, 0), result)

    def rototranslate(
            self, R, t, pose_rotvecs, shape_betas, trans, kid_factor=0, post_translate=True):
        """Rotate and translate the SMPL body model carefully, taking into account that the
        global orientation is applied with the pelvis as anchor, not the origin of the canonical
        coordinate system!
        The translation vector needs to be changed accordingly, too, not just the pose.
        """
        current_rotmat = rotvec2mat(pose_rotvecs[..., :3])
        new_rotmat = R @ current_rotmat
        new_pose_rotvec = tf.concat([mat2rotvec(new_rotmat), pose_rotvecs[3:]], axis=0)

        pelvis = (
                self.J_template[0] +
                self.J_shapedirs[0, :, :shape_betas.shape[0]] @ shape_betas +
                self.kid_J_shapedir[0] * kid_factor) * self.unit_factor
        if post_translate:
            new_trans = (
                    tf.matmul(trans, R, transpose_b=True) + t +
                    tf.matmul(pelvis, R - tf.eye(3), transpose_b=True))
        else:
            new_trans = (
                    tf.matmul(trans - t, R, transpose_b=True) +
                    tf.matmul(pelvis, R - tf.eye(3), transpose_b=True))
        return new_pose_rotvec, new_trans


def check_batch_size(pose_rotvecs, shape_betas, trans, rel_rotmats, glob_rotmats):
    batch_sizes = [
        tf.shape(x)[0] for x in [pose_rotvecs, shape_betas, trans, rel_rotmats, glob_rotmats]
        if x is not None]

    if len(batch_sizes) == 0:
        raise RuntimeError(
            'At least one argument must be given among pose_rotvecs, shape_betas, trans, '
            'rel_rotmats.')

    return batch_sizes[0]
