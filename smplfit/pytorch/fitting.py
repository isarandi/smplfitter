import torch

from smplfit.pytorch.lstsq import lstsq, lstsq_partial_share
from smplfit.pytorch.rotation import kabsch, mat2rotvec


class Fitter:
    def __init__(self, body_model, num_betas, enable_kid=False, vertex_subset=None):
        self.body_model = body_model
        self.n_betas = num_betas
        self.enable_kid = enable_kid
        device = body_model.v_template.device

        if vertex_subset is None:
            vertex_subset = torch.arange(body_model.num_vertices, dtype=torch.int64, device=device)
        else:
            vertex_subset = torch.as_tensor(vertex_subset, dtype=torch.int64, device=device)

        self.vertex_subset = vertex_subset
        self.default_mesh_tf = body_model.single(
            shape_betas=torch.zeros(
                [0], dtype=torch.float32,
                device=self.body_model.v_template.device))['vertices'][self.vertex_subset]

        self.J_template_ext = torch.cat(
            [body_model.J_template.view(-1, 3, 1),
             body_model.J_shapedirs[:, :, :self.n_betas]] +
            ([body_model.kid_J_shapedir.view(-1, 3, 1)] if enable_kid else []),
            dim=2)

        self.children_and_self = [[i_joint] for i_joint in range(body_model.num_joints)]
        for i_joint in range(1, body_model.num_joints):
            i_parent = body_model.kintree_parents[i_joint]
            self.children_and_self[i_parent].append(i_joint)

        self.descendants_and_self = [[i_joint] for i_joint in range(body_model.num_joints)]
        for i_joint in range(body_model.num_joints - 1, 0, -1):
            i_parent = body_model.kintree_parents[i_joint]
            self.descendants_and_self[i_parent].extend(self.descendants_and_self[i_joint])

        self.shapedirs = body_model.shapedirs.index_select(0, self.vertex_subset)
        self.kid_shapedir = body_model.kid_shapedir.index_select(0, self.vertex_subset)
        self.v_template = body_model.v_template.index_select(0, self.vertex_subset)
        self.weights = body_model.weights.index_select(0, self.vertex_subset)
        self.posedirs = body_model.posedirs.index_select(0, self.vertex_subset)
        self.num_vertices = self.v_template.shape[0]


    def fit(self, to_fit, n_iter=1, l2_regularizer=5e-6, l2_regularizer2=0,
            initial_vertices=None, joints_to_fit=None,
            initial_joints=None, requested_keys=(), allow_nan=True, vertex_weights=None,
            joint_weights=None, share_beta=False, final_adjust_rots=True, scale_target=False,
            scale_fit=False, scale_regularizer=0, kid_regularizer=None):

        # Subtract mean first for better numerical stability (and add it back later)
        if joints_to_fit is None:
            to_fit_mean = torch.mean(to_fit, dim=1)
            to_fit = to_fit - to_fit_mean[:, None]
        else:
            to_fit_mean = torch.mean(torch.cat([to_fit, joints_to_fit], dim=1), dim=1)
            to_fit = to_fit - to_fit_mean[:, None]
            joints_to_fit = joints_to_fit - to_fit_mean[:, None]

        if initial_vertices is None:
            initial_vertices = self.default_mesh_tf[None]
        if initial_joints is None:
            initial_joints = self.body_model.J_template[None]

        glob_rots = self.fit_global_rotations(
            to_fit, initial_vertices, joints_to_fit, initial_joints,
            vertex_weights=vertex_weights, joint_weights=joint_weights)
        device = self.body_model.v_template.device
        parent_indices = self.body_model.kintree_parents_tensor[1:].to(device)

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
                factor = result['scale_corr'][:, None, None]
                glob_rots = self.fit_global_rotations_dependent(
                    glob_rots, to_fit * factor, result['vertices'], result['shape_betas'],
                    result['kid_factor'], result['trans'], joints_to_fit * factor, result['joints'],
                    vertex_weights=vertex_weights, joint_weights=joint_weights)
            if scale_fit:
                factor = result['scale_corr'][:, None, None]
                glob_rots = self.fit_global_rotations_dependent(
                    glob_rots, to_fit, factor * result['vertices'] + (1 - factor) * result['trans'].unsqueeze(-2),
                    result['shape_betas'], result['kid_factor'], result['trans'],
                    joints_to_fit, factor * result['joints'] + (1 - factor) * result['trans'].unsqueeze(-2),
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
            result['joints'] = forw['joints'] + to_fit_mean[:, None]
        if 'vertices' in requested_keys:
            result['vertices'] = forw['vertices'] + to_fit_mean[:, None]

        result['orientations'] = glob_rots

        # Provide other requested rotation formats
        if 'relative_orientations' in requested_keys or 'pose_rotvecs' in requested_keys:
            parent_glob_rotmats = torch.cat([
                torch.eye(3, device=device).expand(glob_rots.shape[0], 1, 3, 3),
                torch.index_select(glob_rots, 1, parent_indices)], dim=1)
            result['relative_orientations'] = torch.matmul(
                parent_glob_rotmats.transpose(-1, -2), glob_rots)

        if 'pose_rotvecs' in requested_keys:
            rotvecs = mat2rotvec(result['relative_orientations'])
            result['pose_rotvecs'] = rotvecs.view(rotvecs.shape[0], -1)

        return result

    def estimate_shape(
            self, to_fit, glob_rotmats, joints_to_fit, l2_regularizer=5e-6, l2_regularizer2=0,
            vertex_weights=None, joint_weights=None, requested_keys=(),
            share_beta=False, scale_target=False, scale_fit=False, scale_regularizer=0,
            kid_regularizer=None):
        if scale_target and scale_fit:
            raise ValueError("Only one of estim_scale_target and estim_scale_fit can be True")

        glob_rotmats = glob_rotmats.float()
        batch_size = to_fit.shape[0]

        device = self.body_model.v_template.device
        parent_indices = self.body_model.kintree_parents_tensor[1:].to(device)

        parent_glob_rot_mats = torch.cat([
            torch.eye(3, device=device).expand(glob_rotmats.shape[0], 1, 3, 3),
            torch.index_select(glob_rotmats, 1, parent_indices)], dim=1)
        rel_rotmats = torch.matmul(parent_glob_rot_mats.transpose(-1, -2), glob_rotmats)

        glob_positions_ext = [self.J_template_ext[None, 0].expand(batch_size, -1, -1)]
        for i_joint, i_parent in enumerate(self.body_model.kintree_parents[1:], start=1):
            glob_positions_ext.append(
                glob_positions_ext[i_parent] +
                torch.einsum('bCc,cs->bCs', glob_rotmats[:, i_parent],
                             self.J_template_ext[i_joint] - self.J_template_ext[i_parent]))
        glob_positions_ext = torch.stack(glob_positions_ext, dim=1)
        translations_ext = glob_positions_ext - torch.einsum(
            'bjCc,jcs->bjCs', glob_rotmats, self.J_template_ext)

        rot_params = rel_rotmats[:, 1:].reshape(-1, (self.body_model.num_joints - 1) * 3 * 3)
        v_posed = self.v_template + torch.einsum('vcp,bp->bvc', self.posedirs, rot_params)
        v_rotated = torch.einsum('bjCc,vj,bvc->bvC', glob_rotmats, self.weights, v_posed)

        shapedirs = (
            torch.cat([
                self.shapedirs[:, :, :self.n_betas],
                self.kid_shapedir[:, :, None]], dim=2) if self.enable_kid
            else self.shapedirs[:, :, :self.n_betas])
        v_grad_rotated = torch.einsum('bjCc,lj,lcs->blCs', glob_rotmats, self.weights, shapedirs)

        v_rotated_ext = torch.cat([v_rotated.unsqueeze(-1), v_grad_rotated], dim=3)
        v_translations_ext = torch.einsum('vj,bjcs->bvcs', self.weights, translations_ext)
        v_posed_posed_ext = v_translations_ext + v_rotated_ext

        if joints_to_fit is None:
            to_fit_both = to_fit
            pos_both = v_posed_posed_ext[..., 0]
            jac_pos_both = v_posed_posed_ext[..., 1:]
        else:
            to_fit_both = torch.cat([to_fit, joints_to_fit], dim=1)
            pos_both = torch.cat([v_posed_posed_ext[..., 0], glob_positions_ext[..., 0]], dim=1)
            jac_pos_both = torch.cat(
                [v_posed_posed_ext[..., 1:], glob_positions_ext[..., 1:]], dim=1)

        if scale_target:
            A = torch.cat([jac_pos_both, -to_fit_both.unsqueeze(-1)], dim=3)
        elif scale_fit:
            A = torch.cat([jac_pos_both, pos_both.unsqueeze(-1)], dim=3)
        else:
            A = jac_pos_both

        b = to_fit_both - pos_both
        mean_A = torch.mean(A, dim=1, keepdim=True)
        mean_b = torch.mean(b, dim=1, keepdim=True)
        A = A - mean_A
        b = b - mean_b

        if vertex_weights is not None and joint_weights is not None:
            weights = torch.cat([vertex_weights, joint_weights], dim=1)
        else:
            weights = torch.ones(A.shape[:2], dtype=torch.float32, device=device)

        n_params = (
                self.n_betas + (1 if self.enable_kid else 0) +
                (1 if scale_target or scale_fit else 0))
        A = A.reshape(batch_size, -1, n_params)
        b = b.reshape(batch_size, -1, 1)
        w = weights.reshape(batch_size, -1).repeat(1, 3)

        l2_regularizer_all = torch.cat([
            torch.full((2,), l2_regularizer2, device=device),
            torch.full((self.n_betas - 2,), l2_regularizer, device=device)
        ])

        if self.enable_kid:
            if kid_regularizer is None:
                kid_regularizer = l2_regularizer
            l2_regularizer_all = torch.cat([l2_regularizer_all, torch.tensor([kid_regularizer])])

        if scale_target or scale_fit:
            l2_regularizer_all = torch.cat([l2_regularizer_all, torch.tensor([scale_regularizer])])

        if share_beta:
            x = lstsq_partial_share(
                A, b, w, l2_regularizer_all, n_shared=self.n_betas + (1 if self.enable_kid else 0))
        else:
            x = lstsq(A, b, w, l2_regularizer_all)

        x = x.squeeze(-1)
        new_trans = mean_b.squeeze(1) - torch.matmul(mean_A.squeeze(1), x.unsqueeze(-1)).squeeze(-1)
        new_shape = x[:, :self.n_betas]
        new_kid_factor = None
        new_scale_corr = None

        if self.enable_kid:
            new_kid_factor = x[:, self.n_betas]
        if scale_target or scale_fit:
            new_scale_corr = x[:, -1] + 1
            if scale_fit:
                new_shape /= new_scale_corr.unsqueeze(-1)

        result = dict(
            shape_betas=new_shape, kid_factor=new_kid_factor, trans=new_trans,
            relative_orientations=rel_rotmats, joints=None, vertices=None,
            scale_corr=new_scale_corr)

        if self.enable_kid:
            new_shape = torch.cat([new_shape, new_kid_factor.unsqueeze(-1)], dim=1)

        if 'joints' in requested_keys:
            result['joints'] = (
                    glob_positions_ext[..., 0] +
                    torch.einsum('bvcs,bs->bvc', glob_positions_ext[..., 1:], new_shape) +
                    new_trans.unsqueeze(1))

        if 'vertices' in requested_keys:
            result['vertices'] = (
                    v_posed_posed_ext[..., 0] +
                    torch.einsum('bvcs,bs->bvc', v_posed_posed_ext[..., 1:], new_shape) +
                    new_trans.unsqueeze(1))
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

        part_assignment = torch.argmax(self.weights, dim=1)
        # Disable the rotation of toes separately from the feet
        part_assignment = torch.where(part_assignment == 10, torch.tensor(7, dtype=torch.int64),
                                      part_assignment)
        part_assignment = torch.where(part_assignment == 11, torch.tensor(8, dtype=torch.int64),
                                      part_assignment)

        for i in range(self.body_model.num_joints):
            # Disable the rotation of toes separately from the feet
            if i == 10:
                glob_rots.append(glob_rots[7])
                continue
            elif i == 11:
                glob_rots.append(glob_rots[8])
                continue

            selector = torch.where(part_assignment == i)[0]
            default_body_part = reference[:, selector]
            estim_body_part = target[:, selector]
            weights_body_part = (
                vertex_weights[:, selector].unsqueeze(-1) * mesh_weight
                if vertex_weights is not None else mesh_weight)

            default_joints = reference_joints[:, self.children_and_self[i]]
            estim_joints = target_joints[:, self.children_and_self[i]]
            weights_joints = (
                joint_weights[:, self.children_and_self[i]].unsqueeze(-1) * joint_weight
                if joint_weights is not None else joint_weight)

            body_part_mean_reference = torch.mean(default_joints, dim=1, keepdim=True)
            default_points = torch.cat([
                (default_body_part - body_part_mean_reference) * weights_body_part,
                (default_joints - body_part_mean_reference) * weights_joints
            ], dim=1)

            body_part_mean_target = torch.mean(estim_joints, dim=1, keepdim=True)
            estim_points = torch.cat([
                (estim_body_part - body_part_mean_target),
                (estim_joints - body_part_mean_target)
            ], dim=1)

            glob_rot = kabsch(estim_points, default_points)
            glob_rots.append(glob_rot)

        return torch.stack(glob_rots, dim=1)

    def fit_global_rotations_dependent(
            self, glob_rots_prev, target, reference, shape_betas, kid_factor,
            trans, target_joints=None, reference_joints=None, vertex_weights=None,
            joint_weights=None, all_descendants=False):
        glob_rots = []

        if target_joints is None or reference_joints is None:
            target_joints = self.body_model.J_regressor @ target
            reference_joints = self.body_model.J_regressor @ reference

        device = self.body_model.v_template.device
        part_assignment = torch.argmax(self.weights, dim=1)
        part_assignment = torch.where(
            part_assignment == 10, torch.tensor(7, dtype=torch.int64, device=device),
            part_assignment)
        part_assignment = torch.where(
            part_assignment == 11, torch.tensor(8, dtype=torch.int64, device=device),
            part_assignment)

        j = (self.body_model.J_template +
             torch.einsum('jcs,...s->...jc', self.body_model.J_shapedirs[:, :, :self.n_betas],
                          shape_betas))
        if kid_factor is not None:
            j += torch.einsum('jc,...->...jc', self.body_model.kid_J_shapedir, kid_factor)

        parent_indices = self.body_model.kintree_parents_tensor[1:].to(device)
        j_parent = torch.cat([
            torch.zeros(1, 3, device=device).expand(j.shape[0], -1, -1),
            torch.index_select(j, 1, parent_indices)], dim=1)
        bones = j - j_parent

        glob_positions = []

        for i in range(self.body_model.num_joints):
            if i == 0:
                glob_position = j[:, i] + trans
            else:
                i_parent = self.body_model.kintree_parents[i]
                glob_position = (
                        glob_positions[i_parent] +
                        torch.matmul(glob_rots[i_parent], bones[:, i].unsqueeze(-1)).squeeze(-1))
            glob_positions.append(glob_position)

            if i == 10:
                glob_rots.append(glob_rots[7])
                continue
            elif i == 11:
                glob_rots.append(glob_rots[8])
                continue

            if not all_descendants:
                vertex_selector = torch.where(part_assignment == i)[0]
                joint_selector = self.children_and_self[i]
            else:
                vertex_selector = torch.cat(
                    [torch.where(part_assignment == i)[0]
                     for i in self.descendants_and_self[i]], dim=0)
                joint_selector = self.descendants_and_self[i]

            default_body_part = reference[:, vertex_selector]
            estim_body_part = target[:, vertex_selector]
            weights_body_part = (
                vertex_weights[:, vertex_selector].unsqueeze(-1)
                if vertex_weights is not None else torch.tensor(
                    1.0, dtype=torch.float32, device=device))

            default_joints = reference_joints[:, joint_selector]
            estim_joints = target_joints[:, joint_selector]
            weights_joints = (
                joint_weights[:, joint_selector].unsqueeze(-1)
                if joint_weights is not None else torch.tensor(
                    1.0, dtype=torch.float32, device=device))

            reference_point = glob_position.unsqueeze(1)
            default_reference_point = reference_joints[:, i:i + 1]
            default_points = torch.cat([
                (default_body_part - default_reference_point) * weights_body_part,
                (default_joints - default_reference_point) * weights_joints
            ], dim=1)
            estim_points = torch.cat([
                (estim_body_part - reference_point),
                (estim_joints - reference_point)
            ], dim=1)
            glob_rot = kabsch(estim_points, default_points) @ glob_rots_prev[:, i]
            glob_rots.append(glob_rot)

        return torch.stack(glob_rots, dim=1)
