"""JAX implementation of body model forward pass."""

from __future__ import annotations

from typing import Optional
import jax.numpy as jnp
from .. import common as smplfitter_common
from .rotation import mat2rotvec, rotvec2mat


class BodyModel:
    """SMPL family body model implemented in JAX."""

    def __init__(
        self,
        model_name: str = 'smpl',
        gender: str = 'neutral',
        model_root: Optional[str] = None,
        num_betas: Optional[int] = None,
        device=None,
    ):
        import jax

        self.gender = gender
        self.model_name = model_name

        if device is None:
            device = jax.devices()[0]

        data = smplfitter_common.initialize(
            model_name, gender, model_root, num_betas, None, None, None, None
        )

        with jax.default_device(device):
            self.v_template = jnp.array(data.v_template, dtype=jnp.float32)
            self.shapedirs = jnp.array(data.shapedirs, dtype=jnp.float32)
            self.posedirs = jnp.array(data.posedirs, dtype=jnp.float32)
            self.J_template = jnp.array(data.J_template, dtype=jnp.float32)
            self.J_shapedirs = jnp.array(data.J_shapedirs, dtype=jnp.float32)
            self.kid_shapedir = jnp.array(data.kid_shapedir, dtype=jnp.float32)
            self.kid_J_shapedir = jnp.array(data.kid_J_shapedir, dtype=jnp.float32)
            self.weights = jnp.array(data.weights, dtype=jnp.float32)
            self.J_regressor_post_lbs = jnp.array(
                data.J_regressor_post_lbs, dtype=jnp.float32
            )

        self.kintree_parents = data.kintree_parents
        self.num_joints = data.num_joints
        self.num_vertices = data.num_vertices
        self.num_betas = self.shapedirs.shape[2]

    def __call__(
        self,
        pose_rotvecs: Optional[jnp.ndarray] = None,
        shape_betas: Optional[jnp.ndarray] = None,
        trans: Optional[jnp.ndarray] = None,
        kid_factor: Optional[jnp.ndarray] = None,
        rel_rotmats: Optional[jnp.ndarray] = None,
        glob_rotmats: Optional[jnp.ndarray] = None,
        return_vertices: bool = True,
    ) -> dict[str, jnp.ndarray]:
        batch_size = 0
        for arg in [pose_rotvecs, shape_betas, trans, rel_rotmats, glob_rotmats]:
            if arg is not None:
                batch_size = arg.shape[0]
                break

        if batch_size == 0:
            result = dict(
                joints=jnp.empty((0, self.num_joints, 3)),
                orientations=jnp.empty((0, self.num_joints, 3, 3)),
            )
            if return_vertices:
                result['vertices'] = jnp.empty((0, self.num_vertices, 3))
            return result

        # Get rotation matrices
        if rel_rotmats is not None:
            pass
        elif pose_rotvecs is not None:
            rel_rotmats = rotvec2mat(pose_rotvecs.reshape(batch_size, self.num_joints, 3))
        elif glob_rotmats is None:
            rel_rotmats = jnp.tile(jnp.eye(3), (batch_size, self.num_joints, 1, 1))

        # Compute global rotations via kinematic chain
        if glob_rotmats is None:
            assert rel_rotmats is not None
            glob_rotmats_list = [rel_rotmats[:, 0]]
            for i_joint in range(1, self.num_joints):
                i_parent = self.kintree_parents[i_joint]
                glob_rotmats_list.append(glob_rotmats_list[i_parent] @ rel_rotmats[:, i_joint])
            glob_rotmats = jnp.stack(glob_rotmats_list, axis=1)

        parent_indices1 = jnp.array(self.kintree_parents[1:])
        parent_glob_rotmats1 = glob_rotmats[:, parent_indices1]

        if rel_rotmats is None:
            rel_rotmats1 = jnp.swapaxes(parent_glob_rotmats1, -1, -2) @ glob_rotmats[:, 1:]
        else:
            rel_rotmats1 = rel_rotmats[:, 1:]

        # Shape
        if shape_betas is None:
            shape_betas = jnp.zeros((batch_size, 0))
        num_betas = min(shape_betas.shape[1], self.shapedirs.shape[2])

        if kid_factor is None:
            kid_factor = jnp.zeros((1,))

        # Joint positions in T-pose
        j = (
            self.J_template
            + jnp.einsum(
                'jcs,bs->bjc', self.J_shapedirs[:, :, :num_betas], shape_betas[:, :num_betas]
            )
            + jnp.einsum('jc,b->bjc', self.kid_J_shapedir, kid_factor)
        )

        # Forward kinematics
        bones1 = j[:, 1:] - j[:, parent_indices1]
        rotated_bones1 = jnp.einsum('bjCc,bjc->bjC', parent_glob_rotmats1, bones1)

        glob_positions_list = [j[:, 0]]
        for i_joint in range(1, self.num_joints):
            i_parent = self.kintree_parents[i_joint]
            glob_positions_list.append(
                glob_positions_list[i_parent] + rotated_bones1[:, i_joint - 1]
            )
        glob_positions = jnp.stack(glob_positions_list, axis=1)

        if trans is None:
            trans = jnp.zeros((1, 3))

        if not return_vertices:
            return dict(joints=glob_positions + trans[:, None], orientations=glob_rotmats)

        # Pose blend shapes
        pose_feature = rel_rotmats1.reshape(-1, (self.num_joints - 1) * 9)

        v_posed = (
            self.v_template
            + jnp.einsum(
                'vcp,bp->bvc', self.shapedirs[:, :, :num_betas], shape_betas[:, :num_betas]
            )
            + jnp.einsum('vcp,bp->bvc', self.posedirs, pose_feature)
            + jnp.einsum('vc,b->bvc', self.kid_shapedir, kid_factor)
        )

        # Linear blend skinning
        translations = glob_positions - jnp.einsum('bjCc,bjc->bjC', glob_rotmats, j)
        vertices = (
            jnp.einsum('bjCc,vj,bvc->bvC', glob_rotmats, self.weights, v_posed)
            + self.weights @ translations
        )

        return dict(
            joints=glob_positions + trans[:, None],
            vertices=vertices + trans[:, None],
            orientations=glob_rotmats,
        )

    def rototranslate(
        self, R, t, pose_rotvecs, shape_betas, trans, kid_factor=0, post_translate=True
    ):
        """Rotate and translate the body in parametric form.

        See np.BodyModel.rototranslate for full documentation.
        """
        current_rotmat = rotvec2mat(pose_rotvecs[:3])
        new_rotmat = R @ current_rotmat
        new_pose_rotvec = jnp.concatenate([mat2rotvec(new_rotmat), pose_rotvecs[3:]], axis=0)

        pelvis = (
            self.J_template[0]
            + self.J_shapedirs[0, :, :shape_betas.shape[0]] @ shape_betas
            + self.kid_J_shapedir[0] * kid_factor
        )

        eye = jnp.eye(3, dtype=jnp.float32)
        if post_translate:
            new_trans = pelvis @ (R.T - eye) + trans @ R.T + t
        else:
            new_trans = pelvis @ (R.T - eye) + (trans - t) @ R.T
        return new_pose_rotvec, new_trans

    def single(self, *args, return_vertices=True, **kwargs):
        """Single instance forward pass (no batch dimension in inputs/outputs)."""
        args = [jnp.expand_dims(x, axis=0) for x in args]
        kwargs = {k: jnp.expand_dims(v, axis=0) for k, v in kwargs.items()}
        if len(args) == 0 and len(kwargs) == 0:
            kwargs['shape_betas'] = jnp.zeros((1, 0), jnp.float32)
        result = self(*args, return_vertices=return_vertices, **kwargs)
        return {k: jnp.squeeze(v, axis=0) for k, v in result.items()}
