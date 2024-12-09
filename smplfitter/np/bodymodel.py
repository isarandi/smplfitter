import numpy as np
import smplfitter.common
from smplfitter.np.rotation import rotvec2mat, mat2rotvec
from smplfitter.np.util import matmul_transp_a


class SMPLBodyModel:
    def __init__(self, model_name='smpl', gender='neutral', model_root=None, unit='m', num_betas=None):
        """
        Args:
            model_root: path to pickle files for the model (see https://smpl.is.tue.mpg.de).
            gender: 'neutral' (default) or 'f' or 'm'
            model_type: 'basic' or 'shapeagnostic' (the latter is designed to ignore any
                influence from the shape betas except the influence on the joint locations,
                i.e., it always yields average-BMI body shapes but allows changing the skeleton.)
        """
        self.gender = gender
        self.model_name = model_name
        tensors, nontensors = smplfitter.common.initialize(model_name, gender, model_root, num_betas)
        self.v_template = np.array(tensors['v_template'], np.float32)
        self.shapedirs = np.array(tensors['shapedirs'], np.float32)
        self.posedirs = np.array(tensors['posedirs'], np.float32)
        self.J_regressor = np.array(tensors['J_regressor'], np.float32)
        self.J_template = np.array(tensors['J_template'], np.float32)
        self.J_shapedirs = np.array(tensors['J_shapedirs'], np.float32)
        self.kid_shapedir = np.array(tensors['kid_shapedir'], np.float32)
        self.kid_J_shapedir = np.array(tensors['kid_J_shapedir'], np.float32)
        self.weights = np.array(tensors['weights'], np.float32)
        self.kintree_parents = nontensors['kintree_parents']
        self.faces = nontensors['faces']
        self.num_joints = nontensors['num_joints']
        self.num_vertices = nontensors['num_vertices']
        self.unit_factor = dict(mm=1000, cm=100, m=1)[unit]

    def __call__(
            self, pose_rotvecs=None, shape_betas=None, trans=None, kid_factor=None,
            rel_rotmats=None, glob_rotmats=None, *, return_vertices=True):
        """Calculate the SMPL body model vertices, joint positions and orientations given the input
        pose and shape parameters.

        Args:
            pose_rotvecs (np.ndarray): An array of shape (batch_size, num_joints * 3),
                representing the rotation vectors for each joint in the pose.
            shape_betas (np.ndarray): An array of shape (batch_size, num_shape_coeffs),
                representing the shape coefficients (betas) for the body shape.
            trans (np.ndarray, optional): An array of shape (batch_size, 3), representing the
                translation of the root joint. Defaults to None, in which case a zero translation is
                applied.
            return_vertices (bool, optional): A flag indicating whether to return the body model
                vertices. If False, only joint positions and orientations are returned.
                Defaults to True.

        Returns:
            A dictionary containing the following keys and values:
                - 'vertices': An array of shape (batch_size, num_vertices, 3), representing the
                    3D body model vertices in the posed state. This key is only present if
                    `return_vertices` is True.
                - 'joints': An array of shape (batch_size, num_joints, 3), representing the 3D
                    positions of the body joints.
                - 'orientations': An array of shape (batch_size, num_joints, 3, 3), representing
                    the 3D orientation matrices for each joint.
        """

        batch_size = check_batch_size(pose_rotvecs, shape_betas, trans, rel_rotmats)

        if rel_rotmats is not None:
            rel_rotmats = np.asarray(rel_rotmats, np.float32)
        elif pose_rotvecs is not None:
            pose_rotvecs = np.asarray(pose_rotvecs, np.float32)
            rel_rotmats = rotvec2mat(np.reshape(pose_rotvecs, (batch_size, self.num_joints, 3)))
        elif glob_rotmats is None:
            rel_rotmats = np.tile(
                np.eye(3, dtype=np.float32), [batch_size, self.num_joints, 1, 1])

        if glob_rotmats is None:
            glob_rotmats = [rel_rotmats[:, 0]]
            for i_joint in range(1, self.num_joints):
                i_parent = self.kintree_parents[i_joint]
                glob_rotmats.append(glob_rotmats[i_parent] @ rel_rotmats[:, i_joint])
            glob_rotmats = np.stack(glob_rotmats, axis=1)

        parent_indices = self.kintree_parents[1:]
        parent_glob_rotmats = np.concatenate([
            np.tile(np.eye(3), [glob_rotmats.shape[0], 1, 1, 1]),
            glob_rotmats[:, parent_indices]], axis=1)

        if rel_rotmats is None:
            rel_rotmats = matmul_transp_a(parent_glob_rotmats, glob_rotmats)

        if shape_betas is None:
            shape_betas = np.zeros((batch_size, 0), np.float32)
        else:
            shape_betas = np.asarray(shape_betas, np.float32)
        num_betas = np.minimum(shape_betas.shape[1], self.shapedirs.shape[2])

        if kid_factor is None:
            kid_factor = np.zeros((1,), np.float32)
        else:
            kid_factor = np.float32(kid_factor)

        j = (self.J_template +
             np.einsum(
                 'jcs,bs->bjc', self.J_shapedirs[:, :, :num_betas],
                 shape_betas[:, :num_betas]) +
             np.einsum('jc,b->bjc', self.kid_J_shapedir, kid_factor))

        glob_rotmats = [rel_rotmats[:, 0]]
        glob_positions = [j[:, 0]]

        for i_joint in range(1, self.num_joints):
            i_parent = self.kintree_parents[i_joint]
            glob_rotmats.append(glob_rotmats[i_parent] @ rel_rotmats[:, i_joint])
            glob_positions.append(
                glob_positions[i_parent] +
                np.einsum('bCc,bc->bC', glob_rotmats[i_parent], j[:, i_joint] - j[:, i_parent]))

        glob_rotmats = np.stack(glob_rotmats, axis=1)
        glob_positions = np.stack(glob_positions, axis=1)

        if trans is None:
            trans = np.zeros((1, 3), np.float32)
        else:
            trans = trans.astype(np.float32)

        if not return_vertices:
            return dict(
                joints=(glob_positions + trans[:, np.newaxis]) * self.unit_factor,
                orientations=glob_rotmats)

        pose_feature = np.reshape(rel_rotmats[:, 1:], [-1, (self.num_joints - 1) * 3 * 3])
        v_posed = (
                self.v_template +
                np.einsum(
                    'vcp,bp->bvc', self.shapedirs[:, :, :num_betas], shape_betas[:, :num_betas]) +
                np.einsum('vcp,bp->bvc', self.posedirs, pose_feature) +
                np.einsum('vc,b->bvc', self.kid_shapedir, kid_factor))

        translations = glob_positions - np.einsum('bjCc,bjc->bjC', glob_rotmats, j)
        vertices = (
                np.einsum('bjCc,vj,bvc->bvC', glob_rotmats, self.weights, v_posed) +
                self.weights @ translations)

        return dict(
            vertices=vertices * self.unit_factor + trans[:, np.newaxis],
            joints=glob_positions * self.unit_factor + trans[:, np.newaxis],
            orientations=glob_rotmats)

    def single(self, *args, return_vertices=True, **kwargs):
        args = [np.expand_dims(x, axis=0) for x in args]
        kwargs = {k: np.expand_dims(v, axis=0) for k, v in kwargs.items()}
        if len(args) == 0 and len(kwargs) == 0:
            kwargs['shape_betas'] = np.zeros((1, 0), np.float32)
        result = self(*args, return_vertices=return_vertices, **kwargs)
        return {k: np.squeeze(v, axis=0) for k, v in result.items()}

    def rototranslate(
            self, R, t, pose_rotvecs, shape_betas, trans, kid_factor=0, post_translate=True):
        """Rotate and translate the SMPL body model carefully, taking into account that the
        global orientation is applied with the pelvis as anchor, not the origin of the canonical
        coordinate system!
        The translation vector needs to be changed accordingly, too, not just the pose.
        """
        current_rotmat = rotvec2mat(pose_rotvecs[:3])
        new_rotmat = R @ current_rotmat
        new_pose_rotvec = np.concatenate(
            [mat2rotvec(new_rotmat), pose_rotvecs[3:]], axis=0)

        pelvis = (
                         self.J_template[0] +
                         self.J_shapedirs[0, :, :shape_betas.shape[0]] @ shape_betas +
                         self.kid_J_shapedir[0] * kid_factor
                 ) * self.unit_factor

        if post_translate:
            new_trans = pelvis @ (R.T - np.eye(3)) + trans @ R.T + t
        else:
            new_trans = pelvis @ (R.T - np.eye(3)) + (trans - t) @ R.T
        return new_pose_rotvec, new_trans

    def transform(
            self, extrinsic_matrix, pose_rotvecs, shape_betas, trans, kid_factor=0):
        """Rotate and translate the SMPL body model carefully, taking into account that the
        global orientation is applied with the pelvis as anchor, not the origin of the canonical
        coordinate system!
        The translation vector needs to be changed accordingly, too, not just the pose.
        """
        from scipy.spatial.transform import Rotation
        #current_rotmat = rotvec2mat(pose_rotvecs[:3])
        current_rotmat = Rotation.from_rotvec(pose_rotvecs[:3]).as_matrix()
        new_rotmat = extrinsic_matrix[:3, :3] @ current_rotmat
        new_pose_rotvec = np.concatenate(
            [Rotation.from_matrix(new_rotmat).as_rotvec(), pose_rotvecs[3:]], axis=0)
        pelvis = (
                         self.J_template[0] +
                         self.J_shapedirs[0, :, :shape_betas.shape[0]] @ shape_betas +
                         self.kid_J_shapedir[0] * kid_factor
                 ) * self.unit_factor
        new_trans = pelvis @ (R.T - np.eye(3)) + trans @ R.T + extrinsic_matrix[:3, 3]
        return new_pose_rotvec, new_trans



def check_batch_size(pose_rotvecs, shape_betas, trans, rel_rotmats):
    batch_sizes = [
        np.asarray(x).shape[0] for x in [pose_rotvecs, shape_betas, trans, rel_rotmats]
        if x is not None]

    if len(batch_sizes) == 0:
        raise RuntimeError(
            'At least one argument must be given among pose_rotvecs, shape_betas, trans, '
            'rel_rotmats.')

    if not all(b == batch_sizes[0] for b in batch_sizes[1:]):
        raise RuntimeError('The batch sizes must be equal.')

    return batch_sizes[0]
