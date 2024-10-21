import tensorflow as tf

from smplfit.tf.lstsq import lstsq, lstsq_partial_share
from smplfit.tf.rotation import kabsch, mat2rotvec
from smplfit.tf.util import safe_nan_to_zero


class Fitter:
    def __init__(
            self, body_model, num_betas, enable_kid=False, vertex_subset=None, subset_faces=None):
        self.body_model = body_model
        self.n_betas = num_betas
        self.enable_kid = enable_kid

        if vertex_subset is None:
            vertex_subset = tf.range(body_model.num_vertices)

        self.vertex_subset = vertex_subset
        self.default_mesh_tf = tf.gather(body_model.single(
            shape_betas=tf.zeros([0], tf.float32))['vertices'], self.vertex_subset, axis=0)

        self.J_template_ext = tf.concat(
            [tf.reshape(body_model.J_template, [-1, 3, 1]),
             body_model.J_shapedirs[:, :, :self.n_betas]] +
            ([tf.reshape(body_model.kid_J_shapedir, [-1, 3, 1])] if enable_kid else []),
            axis=2)

        self.children_and_self = [[i_joint] for i_joint in range(body_model.num_joints)]
        for i_joint in range(1, body_model.num_joints):
            i_parent = body_model.kintree_parents[i_joint]
            self.children_and_self[i_parent].append(i_joint)

        self.descendants_and_self = [[i_joint] for i_joint in range(body_model.num_joints)]
        for i_joint in range(body_model.num_joints - 1, 0, -1):
            i_parent = body_model.kintree_parents[i_joint]
            self.descendants_and_self[i_parent].extend(self.descendants_and_self[i_joint])

        self.shapedirs = tf.gather(body_model.shapedirs, self.vertex_subset, axis=0)
        self.kid_shapedir = tf.gather(body_model.kid_shapedir, self.vertex_subset, axis=0)
        self.v_template = tf.gather(body_model.v_template, self.vertex_subset, axis=0)
        self.weights = tf.gather(body_model.weights, self.vertex_subset, axis=0)
        self.posedirs = tf.gather(body_model.posedirs, self.vertex_subset, axis=0)
        self.num_vertices = tf.shape(self.v_template)[0]

    def fit(self, to_fit, n_iter=1, l2_regularizer=5e-6, l2_regularizer2=0,
            initial_vertices=None, joints_to_fit=None,
            initial_joints=None, requested_keys=(), allow_nan=False, vertex_weights=None,
            joint_weights=None, share_beta=False, final_adjust_rots=True, scale_target=False,
            scale_fit=False, scale_regularizer=0, kid_regularizer=None):

        # Subtract mean first for better numerical stability (and add it back later)
        if joints_to_fit is None:
            to_fit_mean = tf.reduce_mean(to_fit, axis=1)
            to_fit = to_fit - to_fit_mean[:, tf.newaxis]
        else:
            to_fit_mean = tf.reduce_mean(tf.concat([to_fit, joints_to_fit], axis=1), axis=1)
            to_fit = to_fit - to_fit_mean[:, tf.newaxis]
            joints_to_fit = joints_to_fit - to_fit_mean[:, tf.newaxis]

        if initial_vertices is None:
            initial_vertices = self.default_mesh_tf[tf.newaxis]
        if initial_joints is None:
            initial_joints = self.body_model.J_template[tf.newaxis]

        glob_rots = self.fit_global_rotations(
            to_fit, initial_vertices, joints_to_fit, initial_joints,
            vertex_weights=vertex_weights, joint_weights=joint_weights)

        for i in range(n_iter - 1):
            result = self.estimate_shape(
                to_fit, glob_rots, joints_to_fit, l2_regularizer=l2_regularizer,
                l2_regularizer2=l2_regularizer2,
                vertex_weights=vertex_weights, joint_weights=joint_weights,
                requested_keys=['vertices'] + (['joints'] if joints_to_fit is not None else []),
                share_beta=share_beta, scale_target=False, scale_fit=False,
                kid_regularizer=kid_regularizer)
            glob_rots = self.fit_global_rotations(
                to_fit, result['vertices'], joints_to_fit, result['joints'],
                vertex_weights=vertex_weights, joint_weights=joint_weights) @ glob_rots

        result = self.estimate_shape(
            to_fit, glob_rots, joints_to_fit, l2_regularizer=l2_regularizer,
            l2_regularizer2=l2_regularizer2,
            vertex_weights=vertex_weights, joint_weights=joint_weights,
            requested_keys=['vertices'] + (['joints'] if joints_to_fit is not None else []),
            share_beta=share_beta, scale_target=scale_target, scale_fit=scale_fit,
            scale_regularizer=scale_regularizer, kid_regularizer=kid_regularizer)

        if final_adjust_rots:
            if scale_target:
                factor = result['scale_corr'][:, tf.newaxis, tf.newaxis]
                glob_rots = self.fit_global_rotations_dependent(
                    glob_rots, to_fit * factor, result['vertices'], result['shape_betas'],
                    result['kid_factor'], result['trans'], joints_to_fit * factor, result['joints'],
                    vertex_weights=vertex_weights, joint_weights=joint_weights)
            if scale_fit:
                factor = result['scale_corr'][:, tf.newaxis, tf.newaxis]

                def scale_adjust(x):
                    return factor * x + (1 - factor) * tf.expand_dims(result['trans'], -2)

                glob_rots = self.fit_global_rotations_dependent(
                    glob_rots, to_fit, scale_adjust(result['vertices']),
                    result['shape_betas'], result['kid_factor'], result['trans'],
                    joints_to_fit, scale_adjust(result['joints']),
                    vertex_weights=vertex_weights, joint_weights=joint_weights)
            else:
                glob_rots = self.fit_global_rotations_dependent(
                    glob_rots, to_fit, result['vertices'], result['shape_betas'],
                    result['kid_factor'], result['trans'], joints_to_fit, result['joints'],
                    vertex_weights=vertex_weights, joint_weights=joint_weights)

        if 'joints' in requested_keys or 'vertices' in requested_keys:
            forw = self.body_model(
                glob_rotmats=glob_rots, shape_betas=result['shape_betas'], trans=result['trans'],
                kid_factor=result['kid_factor'])

        # Add the mean back
        result['trans'] = result['trans'] + to_fit_mean
        if 'joints' in requested_keys:
            result['joints'] = forw['joints'] + to_fit_mean[:, tf.newaxis]
        if 'vertices' in requested_keys:
            result['vertices'] = forw['vertices'] + to_fit_mean[:, tf.newaxis]

        result['orientations'] = glob_rots

        # Provide other requested rotation formats
        if 'relative_orientations' in requested_keys or 'pose_rotvecs' in requested_keys:
            parent_glob_rotmats = tf.concat([
                tf.broadcast_to(tf.eye(3), tf.shape(glob_rots[:, :1])),
                tf.gather(glob_rots, self.body_model.kintree_parents[1:], axis=1)], axis=1)
            result['relative_orientations'] = tf.linalg.matmul(
                parent_glob_rotmats, glob_rots, transpose_a=True)

        if 'pose_rotvecs' in requested_keys:
            rotvecs = mat2rotvec(result['relative_orientations'])
            result['pose_rotvecs'] = tf.reshape(rotvecs, [tf.shape(rotvecs)[0], -1])

        if not allow_nan:
            return {k: safe_nan_to_zero(v) if v is not None else None for k, v in result.items()}
        return result

    def estimate_shape(
            self, to_fit, glob_rotmats, joints_to_fit, l2_regularizer=5e-6, l2_regularizer2=0,
            vertex_weights=None, joint_weights=None, requested_keys=(),
            share_beta=False, scale_target=False, scale_fit=False, scale_regularizer=0,
            kid_regularizer=None):
        if scale_target and scale_fit:
            raise ValueError("Only one of estim_scale_target and estim_scale_fit can be True")

        glob_rotmats = tf.cast(glob_rotmats, tf.float32)
        batch_size = tf.shape(to_fit)[0]

        parent_glob_rot_mats = tf.concat([
            tf.broadcast_to(tf.eye(3), tf.shape(glob_rotmats[:, :1])),
            tf.gather(glob_rotmats, self.body_model.kintree_parents[1:], axis=1)], axis=1)
        rel_rotmats = tf.linalg.matmul(parent_glob_rot_mats, glob_rotmats, transpose_a=True)

        glob_positions_ext = [tf.repeat(self.J_template_ext[tf.newaxis, 0], batch_size, axis=0)]
        for i_joint, i_parent in enumerate(self.body_model.kintree_parents[1:], start=1):
            glob_positions_ext.append(
                glob_positions_ext[i_parent] +
                tf.einsum(
                    'bCc,cs->bCs', glob_rotmats[:, i_parent],
                    self.J_template_ext[i_joint] - self.J_template_ext[i_parent]))
        glob_positions_ext = tf.stack(glob_positions_ext, axis=1)
        translations_ext = glob_positions_ext - tf.einsum(
            'bjCc,jcs->bjCs', glob_rotmats, self.J_template_ext)

        rot_params = tf.reshape(rel_rotmats[:, 1:], [-1, (self.body_model.num_joints - 1) * 3 * 3])
        v_posed = self.v_template + tf.einsum(
            'vcp,bp->bvc', self.posedirs, rot_params)
        v_rotated = tf.einsum('bjCc,vj,bvc->bvC', glob_rotmats, self.weights, v_posed)

        shapedirs = (
            tf.concat([
                self.shapedirs[:, :, :self.n_betas],
                self.kid_shapedir[:, :, tf.newaxis]], axis=2) if self.enable_kid
            else self.shapedirs[:, :, :self.n_betas])
        v_grad_rotated = tf.einsum(
            'bjCc,lj,lcs->blCs', glob_rotmats, self.weights, shapedirs)

        v_rotated_ext = tf.concat([v_rotated[:, :, :, tf.newaxis], v_grad_rotated], axis=3)
        v_translations_ext = tf.einsum('vj,bjcs->bvcs', self.weights, translations_ext)
        v_posed_posed_ext = v_translations_ext + v_rotated_ext

        if joints_to_fit is None:
            to_fit_both = to_fit
            pos_both = v_posed_posed_ext[..., 0]
            jac_pos_both = v_posed_posed_ext[..., 1:]
        else:
            to_fit_both = tf.concat([to_fit, joints_to_fit], axis=1)
            pos_both = tf.concat([v_posed_posed_ext[..., 0], glob_positions_ext[..., 0]], axis=1)
            jac_pos_both = tf.concat(
                [v_posed_posed_ext[..., 1:], glob_positions_ext[..., 1:]], axis=1)

        if scale_target:
            A = tf.concat([jac_pos_both, -to_fit_both[..., tf.newaxis]], axis=3)
        elif scale_fit:
            A = tf.concat([jac_pos_both, pos_both[..., tf.newaxis]], axis=3)
        else:
            A = jac_pos_both

        b = to_fit_both - pos_both
        mean_A = tf.reduce_mean(A, axis=1, keepdims=True)
        mean_b = tf.reduce_mean(b, axis=1, keepdims=True)
        A = A - mean_A
        b = b - mean_b

        if vertex_weights is not None and joint_weights is not None:
            weights = tf.concat([vertex_weights, joint_weights], axis=1)
        else:
            weights = tf.ones(tf.shape(A)[:2], tf.float32)

        n_params = (
                self.n_betas + (1 if self.enable_kid else 0) +
                (1 if scale_target or scale_fit else 0))
        A = tf.reshape(A, [batch_size, -1, n_params])
        b = tf.reshape(b, [batch_size, -1, 1])
        w = tf.repeat(tf.reshape(weights, [batch_size, -1]), 3, axis=1)
        l2_regularizer = tf.convert_to_tensor(l2_regularizer, tf.float32)
        l2_regularizer2 = tf.convert_to_tensor(l2_regularizer2, tf.float32)  # regul of first two

        l2_regularizer_all = tf.concat([
            tf.fill([2], l2_regularizer2),
            tf.fill([self.n_betas - 2], l2_regularizer),
        ], axis=0)

        if self.enable_kid:
            if kid_regularizer is None:
                kid_regularizer = l2_regularizer
            else:
                kid_regularizer = tf.convert_to_tensor(kid_regularizer, tf.float32)
            l2_regularizer_all = tf.concat(
                [l2_regularizer_all, kid_regularizer[tf.newaxis]], axis=0)

        if scale_target or scale_fit:
            scale_regularizer = tf.convert_to_tensor(scale_regularizer, tf.float32)
            l2_regularizer_all = tf.concat(
                [l2_regularizer_all, scale_regularizer[tf.newaxis]], axis=0)

        if share_beta:
            x = lstsq_partial_share(
                A, b, w, l2_regularizer_all, n_shared=self.n_betas + (1 if self.enable_kid else 0))
        else:
            x = lstsq(A, b, w, l2_regularizer_all)

        x = tf.squeeze(x, -1)
        new_trans = tf.squeeze(mean_b, 1) - tf.linalg.matvec(tf.squeeze(mean_A, 1), x)
        new_shape = x[:, :self.n_betas]
        new_kid_factor = None
        new_scale_corr = None

        if self.enable_kid:
            new_kid_factor = x[:, self.n_betas]
        if scale_target or scale_fit:
            new_scale_corr = x[:, -1] + 1
            if scale_fit:
                new_shape /= new_scale_corr[..., tf.newaxis]

        result = dict(
            shape_betas=new_shape, kid_factor=new_kid_factor, trans=new_trans,
            relative_orientations=rel_rotmats, joints=None, vertices=None,
            scale_corr=new_scale_corr)

        if self.enable_kid:
            new_shape = tf.concat([new_shape, new_kid_factor[:, tf.newaxis]], axis=1)

        if 'joints' in requested_keys:
            result['joints'] = (
                    glob_positions_ext[..., 0] +
                    tf.einsum('bvcs,bs->bvc', glob_positions_ext[..., 1:], new_shape) +
                    new_trans[:, tf.newaxis])

        if 'vertices' in requested_keys:
            result['vertices'] = (
                    v_posed_posed_ext[..., 0] +
                    tf.einsum('bvcs,bs->bvc', v_posed_posed_ext[..., 1:], new_shape) +
                    new_trans[:, tf.newaxis])
        return result

    def fit_global_rotations(
            self, target, reference, target_joints=None, reference_joints=None, vertex_weights=None,
            joint_weights=None):
        glob_rots = []
        mesh_weight = 1e-6
        joint_weight = 1 - mesh_weight

        if target_joints is None or reference_joints is None:
            target_joints = self.body_model.J_regressor @ target
            reference_joints = self.body_model.J_regressor @ reference

        part_assignment = tf.argmax(self.weights, axis=1)
        # Disable the rotation of toes separately from the feet
        part_assignment = tf.where(part_assignment == 10, tf.cast(7, tf.int64), part_assignment)
        part_assignment = tf.where(part_assignment == 11, tf.cast(8, tf.int64), part_assignment)

        for i in range(self.body_model.num_joints):
            # Disable the rotation of toes separately from the feet
            if i == 10:
                glob_rots.append(glob_rots[7])
                continue
            elif i == 11:
                glob_rots.append(glob_rots[8])
                continue

            selector = tf.where(part_assignment == i)[:, 0]
            default_body_part = tf.gather(reference, selector, axis=1)
            estim_body_part = tf.gather(target, selector, axis=1)
            weights_body_part = (
                tf.gather(vertex_weights, selector, axis=1)[..., tf.newaxis] * mesh_weight
                if vertex_weights is not None else mesh_weight)

            default_joints = tf.gather(reference_joints, self.children_and_self[i], axis=1)
            estim_joints = tf.gather(target_joints, self.children_and_self[i], axis=1)
            weights_joints = (
                tf.gather(joint_weights, self.children_and_self[i], axis=1)[
                    ..., tf.newaxis] * joint_weight
                if joint_weights is not None else joint_weight)

            body_part_mean_reference = tf.reduce_mean(
                default_joints, axis=1, keepdims=True)
            default_points = tf.concat([
                (default_body_part - body_part_mean_reference) * weights_body_part,
                (default_joints - body_part_mean_reference) * weights_joints], axis=1)

            body_part_mean_target = tf.reduce_mean(
                estim_joints, axis=1, keepdims=True)

            estim_points = (
                tf.concat([
                    (estim_body_part - body_part_mean_target),
                    (estim_joints - body_part_mean_target)], axis=1))

            glob_rot = kabsch(estim_points, default_points)
            glob_rots.append(glob_rot)

        return tf.stack(glob_rots, axis=1)

    def fit_global_rotations_dependent(
            self, glob_rots_prev, target, reference, shape_betas, kid_factor,
            trans, target_joints=None, reference_joints=None, vertex_weights=None,
            joint_weights=None, all_descendants=False):
        glob_rots = []

        if target_joints is None or reference_joints is None:
            target_joints = self.body_model.J_regressor @ target
            reference_joints = self.body_model.J_regressor @ reference

        part_assignment = tf.argmax(self.weights, axis=1)
        part_assignment = tf.where(part_assignment == 10, tf.cast(7, tf.int64), part_assignment)
        part_assignment = tf.where(part_assignment == 11, tf.cast(8, tf.int64), part_assignment)

        j = (self.body_model.J_template +
             tf.einsum(
                 'jcs,...s->...jc', self.body_model.J_shapedirs[:, :, :self.n_betas], shape_betas))
        if kid_factor is not None:
            j += tf.einsum('jc,...->...jc', self.body_model.kid_J_shapedir, kid_factor)

        j_parent = tf.concat([
            tf.broadcast_to(tf.zeros(3), tf.shape(j[:, :1])),
            tf.gather(j, self.body_model.kintree_parents[1:], axis=1)], axis=1)
        bones = j - j_parent

        glob_positions = []

        for i in range(self.body_model.num_joints):
            if i == 0:
                glob_position = j[:, i] + trans
            else:
                i_parent = self.body_model.kintree_parents[i]
                glob_position = (
                        glob_positions[i_parent] +
                        tf.linalg.matvec(glob_rots[i_parent], bones[:, i]))
            glob_positions.append(glob_position)

            if i == 10:
                glob_rots.append(glob_rots[7])
                continue
            elif i == 11:
                glob_rots.append(glob_rots[8])
                continue

            if not all_descendants:
                vertex_selector = tf.where(part_assignment == i)[:, 0]
                joint_selector = self.children_and_self[i]
            else:
                vertex_selector = tf.concat(
                    [tf.where(part_assignment == i)[:, 0]
                     for i in self.descendants_and_self[i]], axis=0)
                joint_selector = self.descendants_and_self[i]

            default_body_part = tf.gather(reference, vertex_selector, axis=1)
            estim_body_part = tf.gather(target, vertex_selector, axis=1)
            weights_body_part = (
                tf.gather(vertex_weights, vertex_selector, axis=1)[..., tf.newaxis]
                if vertex_weights is not None else tf.constant(1, tf.float32))

            default_joints = tf.gather(reference_joints, joint_selector, axis=1)
            estim_joints = tf.gather(target_joints, joint_selector, axis=1)
            weights_joints = (
                tf.gather(joint_weights, joint_selector, axis=1)[..., tf.newaxis]
                if joint_weights is not None else tf.constant(1, tf.float32))

            reference_point = glob_position[:, tf.newaxis]
            default_reference_point = reference_joints[:, i:i + 1]
            default_points = tf.concat([
                (default_body_part - default_reference_point) * weights_body_part,
                (default_joints - default_reference_point) * weights_joints], axis=1)
            estim_points = (
                tf.concat([
                    (estim_body_part - reference_point),
                    (estim_joints - reference_point)], axis=1))
            glob_rot = kabsch(estim_points, default_points) @ glob_rots_prev[:, i]
            glob_rots.append(glob_rot)

        return tf.stack(glob_rots, axis=1)
