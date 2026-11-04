"""Online temporal smoothing of per-frame body estimates, for one person at a time.

Per-frame estimators such as Neural Localizer Fields (NLF) predict vertices and joints with an
uncertainty per point. Their results jitter from frame to frame, and their absolute scale along
the viewing ray is ambiguous. :class:`CausalPoseSmoother` turns such a stream into a smooth
stream of body model parameters. It is causal: the result for a frame depends only on that frame
and the earlier ones, so it can run on live input.

Each :meth:`CausalPoseSmoother.update` runs these steps on a sliding window of recent frames:

1. **Weights.** Each point gets the weight ``uncertainty ** -1.5``, normalized over the window so
   that the relative reliability between frames is preserved.
2. **Scale.** A first fit with a shared shape estimates a scale correction per frame. It is
   normalized by its median over the window and applied about the camera center, which removes
   the scale ambiguity along the viewing ray.
3. **Root.** The root trajectory is filtered with a wide kernel. Each frame is then rescaled along
   its own viewing ray, so that the root follows the filtered trajectory.
4. **Points.** All points are filtered with a robust moving median (the IRLS iterations of a
   weighted geometric median), evaluated at the newest frame.
5. **Fit.** A fit with a shared shape runs over the window of smoothed frames. The filtered root
   of this fit then gives a final translation along the viewing ray of the newest frame.

All temporal filtering runs on world coordinates, which do not depend on the camera. The camera
center only sets the center of the scale correction and the direction of the viewing rays. So a
camera that moves, even abruptly, does not enter the filtered trajectory, provided that
:meth:`CausalPoseSmoother.update` receives its center for each frame.

The kernels suit frame rates below about 40 fps. They are past-only versions of two-sided kernels
designed for offline smoothing of whole videos: the mass of the future half is folded into the
past, so that the weight of the current frame relative to the others stays the same.
"""

from __future__ import annotations

import collections
import importlib
from typing import Any, Optional

import numpy as np

__all__ = ['CausalPoseSmoother']


class CausalPoseSmoother:
    """Smooths a stream of one person's vertices and joints online, into body model parameters.

    Use one instance per person, and call :meth:`reset` at a shot change or when a track restarts.
    The filters run in NumPy, and the fits run in the backend of ``body_model``.

    Parameters:
        body_model: Full-mesh body model from :mod:`smplfitter.np` or :mod:`smplfitter.nb`. Its
            model type, gender and number of betas also apply to the internal fits. Numba is
            faster, and NumPy needs no extra dependency.
        camera_center: Camera optical center (3,) in the coordinates of the points, for a static
            camera. The default suits points in camera coordinates. For a moving camera, pass
            the center of each frame to :meth:`update` instead.
        window: Number of frames in the sliding window, including the newest one.
        eps: Robustness constant of the moving median, in the unit of the points. Point
            deviations well above it are treated as outliers.
        n_iter: Number of iterations of the moving median.
        num_fit_iter: Number of iterations of the final fit.
        fit_vertex_subset_size: The fits run on a precomputed vertex subset of this size, for
            speed, while the temporal filters use all points. If None, they use the full mesh.
        do_scale: Apply the scale correction (step 2 in :mod:`smplfitter.smoothing`).
        do_root: Apply the root correction (step 3).
        do_offset: Apply the final translation of the fitted root (step 5).
        model_root: Path to the body model files, as in the ``BodyModel`` of the backend.

    Example:

    .. code-block:: python

        from smplfitter.nb import BodyModel
        from smplfitter.smoothing import CausalPoseSmoother

        smoother = CausalPoseSmoother(BodyModel('smpl', 'neutral', num_betas=10))
        for frame in stream:
            result = smoother.update(
                frame.vertices, frame.joints, frame.vertex_uncertainties, frame.joint_uncertainties
            )
            result['pose_rotvecs'], result['shape_betas'], result['trans']
    """

    def __init__(
        self,
        body_model: Any,
        camera_center: Any = (0.0, 0.0, 0.0),
        window: int = 5,
        eps: float = 5e-2,
        n_iter: int = 10,
        num_fit_iter: int = 3,
        fit_vertex_subset_size: Optional[int] = 1024,
        do_scale: bool = True,
        do_root: bool = True,
        do_offset: bool = True,
        model_root: Optional[str] = None,
    ):
        backend = _array_backend(body_model)
        self.body_model = body_model
        if fit_vertex_subset_size is None:
            fit_model = body_model
        else:
            fit_model = backend.BodyModel(
                model_name=body_model.model_name,
                gender=body_model.gender,
                model_root=model_root,
                num_betas=body_model.num_betas,
                vertex_subset_size=fit_vertex_subset_size,
            )
        self.fit_vertices = np.asarray(fit_model.vertex_subset)
        self.fitter = backend.BodyFitter(fit_model)
        # Fewer vertices have less total weight in the fit, so the regularizers scale along.
        self.regularizer_factor = (fit_model.num_vertices + body_model.num_joints) / (
            body_model.num_vertices + body_model.num_joints
        )
        self.camera_center = np.asarray(camera_center, np.float32)
        self.eps = eps
        self.n_iter = n_iter
        self.num_fit_iter = num_fit_iter
        self.do_scale = do_scale
        self.do_root = do_root
        self.do_offset = do_offset

        # Frame weights, oldest to newest.
        self.point_kernel = np.array([0.2, 0.6, 2.0, 6.0, 9.0], np.float32)
        self.root_kernel = np.array([0.1, 0.2, 2.0, 4.0, 6.0], np.float32)
        self.offset_kernel = np.array([2.0, 4.0, 3.0], np.float32)

        self._vertices: collections.deque = collections.deque(maxlen=window)
        self._joints: collections.deque = collections.deque(maxlen=window)
        self._vertex_uncertainties: collections.deque = collections.deque(maxlen=window)
        self._joint_uncertainties: collections.deque = collections.deque(maxlen=window)
        self._camera_centers: collections.deque = collections.deque(maxlen=window)
        self._smooth_points: collections.deque = collections.deque(maxlen=window)
        self._smooth_weights: collections.deque = collections.deque(maxlen=window)

    def reset(self) -> None:
        """Forget all frames, for example at a shot change or when a track restarts."""
        for history in (
            self._vertices,
            self._joints,
            self._vertex_uncertainties,
            self._joint_uncertainties,
            self._camera_centers,
            self._smooth_points,
            self._smooth_weights,
        ):
            history.clear()

    def update(
        self,
        vertices: np.ndarray,
        joints: np.ndarray,
        vertex_uncertainties: np.ndarray,
        joint_uncertainties: np.ndarray,
        camera_center: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """Adds the newest frame and returns its smoothed fit.

        Non-finite values in the points that a fit uses give NaN results, without an error, until
        the frame has left the windows of the raw and of the smoothed frames: for up to twice the
        window length.

        Parameters:
            vertices: Vertices of the frame (V, 3), in meters.
            joints: Joints of the frame (J, 3), in meters.
            vertex_uncertainties: Positive uncertainty per vertex (V,).
            joint_uncertainties: Positive uncertainty per joint (J,).
            camera_center: Camera optical center of this frame (3,), in the coordinates of the
                points. If None, the ``camera_center`` given at construction applies.

        Returns:
            Dictionary with the fit for this frame, for use as ``BM(pose, betas, trans)``
                - **pose_rotvecs** -- Pose in concatenated rotation vector format.
                - **shape_betas** -- Shape parameters, shared over the current window.
                - **trans** -- Translation.
                - **vertices** -- Vertices (V, 3) of the fitted body.
                - **joints** -- Joints (J, 3) of the fitted body.
        """
        self._vertices.append(np.asarray(vertices, np.float32))
        self._joints.append(np.asarray(joints, np.float32))
        self._vertex_uncertainties.append(np.asarray(vertex_uncertainties, np.float32))
        self._joint_uncertainties.append(np.asarray(joint_uncertainties, np.float32))
        self._camera_centers.append(
            self.camera_center if camera_center is None else np.asarray(camera_center, np.float32)
        )

        window_vertices = np.stack(self._vertices)
        window_joints = np.stack(self._joints)
        centers = np.stack(self._camera_centers)
        vertex_weights = np.stack(self._vertex_uncertainties) ** -1.5
        joint_weights = np.stack(self._joint_uncertainties) ** -1.5
        vertex_weights /= max(np.mean(vertex_weights), 1e-12)
        joint_weights /= max(np.mean(joint_weights), 1e-12)

        if self.do_scale:
            window_vertices, window_joints = self._correct_scale(
                window_vertices, window_joints, vertex_weights, joint_weights, centers
            )
        if self.do_root:
            self._correct_root(window_vertices, window_joints, joint_weights, centers)

        smooth_points, smooth_weights = _robust_moving_median(
            np.concatenate([window_vertices, window_joints], axis=1),
            np.concatenate([vertex_weights, joint_weights], axis=1),
            _kernel_tail(self.point_kernel, len(window_vertices)),
            self.eps,
            self.n_iter,
        )
        self._smooth_points.append(smooth_points.astype(np.float32))
        self._smooth_weights.append(smooth_weights.astype(np.float32))
        return self._fit_newest(window_vertices.shape[1], centers[-1])

    def _correct_scale(self, vertices, joints, vertex_weights, joint_weights, centers):
        """Rescales each frame about its camera center by its median-normalized scale."""
        factor = self.regularizer_factor
        fit = self.fitter.fit(
            vertices[:, self.fit_vertices],
            joints,
            vertex_weights[:, self.fit_vertices],
            joint_weights,
            num_iter=1,
            beta_regularizer=10.0 * factor,
            beta_regularizer2=0.2 * factor,
            scale_regularizer=1.0 * factor,
            share_beta=True,
            final_adjust_rots=False,
            scale_target=True,
            requested_keys=('pose_rotvecs',),
        )
        scale = np.asarray(fit['scale_corr'], np.float32).reshape(len(vertices))
        scale = (scale / max(np.median(scale), 1e-6))[:, np.newaxis, np.newaxis]
        centers = centers[:, np.newaxis]
        return (vertices - centers) * scale + centers, (joints - centers) * scale + centers

    def _correct_root(self, vertices, joints, joint_weights, centers) -> None:
        """Moves each frame in place along its viewing ray, towards its filtered root.

        ``roots`` is a view of ``joints`` on purpose: the filter for a frame sees the earlier
        frames with their corrections already applied, which makes the correction recursive.
        """
        roots = joints[:, 0]
        for i in range(len(joints)):
            filtered_root, _ = _robust_moving_median(
                roots[: i + 1, np.newaxis],
                joint_weights[: i + 1, 0, np.newaxis],
                _kernel_tail(self.root_kernel, i + 1),
                self.eps,
                self.n_iter,
            )
            factor = _ray_factor(filtered_root[0], roots[i], centers[i])
            vertices[i] = (vertices[i] - centers[i]) * factor + centers[i]
            joints[i] = (joints[i] - centers[i]) * factor + centers[i]

    def _fit_newest(self, num_vertices: int, center: np.ndarray) -> dict[str, np.ndarray]:
        """Fits the window of smoothed frames, and returns the result for the newest frame."""
        points = np.stack(self._smooth_points)
        weights = np.stack(self._smooth_weights)
        fit = self.fitter.fit(
            points[:, :num_vertices][:, self.fit_vertices],
            points[:, num_vertices:],
            weights[:, :num_vertices][:, self.fit_vertices],
            weights[:, num_vertices:],
            num_iter=self.num_fit_iter,
            beta_regularizer=10.0 * self.regularizer_factor,
            beta_regularizer2=0.0,
            share_beta=True,
            final_adjust_rots=True,
            requested_keys=('pose_rotvecs', 'shape_betas', 'trans'),
        )
        posed = self.body_model(
            pose_rotvecs=fit['pose_rotvecs'], shape_betas=fit['shape_betas'], trans=fit['trans']
        )
        fit_joints = np.asarray(posed['joints'], np.float32)

        offset = np.zeros(3, np.float32)
        if self.do_offset:
            kernel = _kernel_tail(
                self.offset_kernel, min(len(fit_joints), len(self.offset_kernel))
            )
            filtered_root, _ = _robust_moving_median(
                fit_joints[-len(kernel) :, 0, np.newaxis],
                weights[-len(kernel) :, num_vertices, np.newaxis],
                kernel,
                self.eps,
                self.n_iter,
            )
            root = fit_joints[-1, 0]
            factor = _ray_factor(filtered_root[0], root, center)
            offset = ((factor - 1.0) * (root - center)).astype(np.float32)

        return dict(
            vertices=np.asarray(posed['vertices'][-1], np.float32) + offset,
            joints=fit_joints[-1] + offset,
            pose_rotvecs=np.asarray(fit['pose_rotvecs'][-1], np.float32),
            shape_betas=np.asarray(fit['shape_betas'][-1], np.float32),
            trans=np.asarray(fit['trans'][-1], np.float32) + offset,
        )


def _array_backend(body_model):
    """The smplfitter backend module of the body model, which must work on NumPy arrays."""
    name = '.'.join(type(body_model).__module__.split('.')[:2])
    if name not in ('smplfitter.np', 'smplfitter.nb'):
        raise ValueError(
            'CausalPoseSmoother needs a BodyModel from smplfitter.np or smplfitter.nb, '
            f'not from {name}'
        )
    return importlib.import_module(name)


def _robust_moving_median(points, weights, kernel, eps, n_iter):
    """Weighted geometric median over a window, by IRLS, evaluated at the newest frame.

    Parameters:
        points: Window of point sets (W, P, 3), oldest first.
        weights: Base weights per point (W, P).
        kernel: Frame weights (W,), oldest first.
        eps: Robustness constant of the reweighting, in the unit of the points.
        n_iter: Number of iterations.

    Returns:
        The filtered points (P, 3) and their filtered weights (P,).
    """
    frame_weights = kernel[:, np.newaxis]
    combined = frame_weights * weights
    filtered = _weighted_mean(points, combined)
    for _ in range(n_iter):
        distances = np.linalg.norm(points - filtered, axis=-1)
        combined = frame_weights * (weights / (distances + eps))
        filtered = _weighted_mean(points, combined)
    distances = np.linalg.norm(points - filtered, axis=-1)
    combined = frame_weights * (weights / (distances + eps))
    filtered_weights = np.sum(combined * weights, axis=0) / np.maximum(
        np.sum(combined, axis=0), 1e-12
    )
    return filtered, filtered_weights


def _weighted_mean(points, weights):
    denominator = np.maximum(np.sum(weights, axis=0), 1e-12)[..., np.newaxis]
    return np.sum(weights[..., np.newaxis] * points, axis=0) / denominator


def _ray_factor(filtered_root, root, center) -> float:
    """Scale along the viewing ray of ``root`` that brings it closest to ``filtered_root``."""
    ray = root - center
    return float(np.dot(filtered_root - center, ray)) / max(float(np.dot(ray, ray)), 1e-9)


def _kernel_tail(kernel, count):
    """The newest ``count`` kernel weights, padded on the old side with the oldest weight."""
    if count <= len(kernel):
        return kernel[-count:]
    return np.concatenate([np.full(count - len(kernel), kernel[0], np.float32), kernel])
