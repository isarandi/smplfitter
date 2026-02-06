How-To Guides
=============

.. contents:: On this page
   :local:
   :depth: 1

Download Body Model Files
-------------------------

Body model files must be downloaded from the official websites (they cannot be
redistributed due to licensing). The easiest way is to use the built-in downloader:

.. code-block:: bash

   python -m smplfitter.download

This will prompt for your email and password (register at
https://smpl.is.tue.mpg.de/ first), then download SMPL, SMPL-X, SMPL+H, and
MANO model files.

By default, files are saved to the location given by the ``SMPLFITTER_BODY_MODELS``
environment variable, or ``$DATA_ROOT/body_models``, or
``~/.local/share/smplfitter/body_models``. You can also pass a path directly:

.. code-block:: bash

   python -m smplfitter.download /path/to/body_models

To tell SMPLFitter where to find the models at runtime:

.. code-block:: bash

   # Option 1: package-specific variable (recommended)
   export SMPLFITTER_BODY_MODELS=/path/to/body_models

   # Option 2: generic data root
   export DATA_ROOT=/path/to/data   # looks for $DATA_ROOT/body_models/

   # Option 3: pass directly in code
   body_model = BodyModel('smpl', 'neutral', model_root='/path/to/body_models/smpl')


Fit Pose and Shape to Vertices
------------------------------

Given a batch of vertex locations in correspondence with the body model template:

.. code-block:: python

   import torch
   from smplfitter.pt import BodyModel, BodyFitter

   body_model = BodyModel('smpl', 'neutral', num_betas=10).cuda()
   fitter = BodyFitter(body_model).cuda()
   fitter = torch.jit.script(fitter)  # optional, faster after first call

   result = fitter.fit(target_vertices=vertices, num_iter=3, beta_regularizer=1)
   result['pose_rotvecs']  # (batch, 72)
   result['shape_betas']   # (batch, 10)
   result['trans']          # (batch, 3)

If you also have joint locations (e.g., from a joint regressor or a separate prediction),
pass them as ``target_joints``. Joints are weighted heavily during orientation fitting
to ensure neighboring body parts link up correctly:

.. code-block:: python

   result = fitter.fit(
       target_vertices=vertices,  # (batch, 6890, 3)
       target_joints=joints,      # (batch, 24, 3)
       num_iter=3,
       beta_regularizer=1,
   )

If ``target_joints`` is omitted, they are computed from the vertices via the model's
joint regressor.


Fit with Known Shape
--------------------

If you already know the shape parameters (e.g., from a previous fit on the same person),
use :meth:`~smplfitter.pt.BodyFitter.fit_with_known_shape` to only estimate pose and
translation:

.. code-block:: python

   result = fitter.fit_with_known_shape(
       shape_betas=known_betas,     # (batch, 10)
       target_vertices=vertices,
       target_joints=joints,
       num_iter=1,
       final_adjust_rots=True,
   )

This is faster and more robust than joint pose+shape fitting, since the shape
is no longer a free variable.


Fit with Known Pose
-------------------

To estimate only the body shape given known pose parameters (e.g., from a motion capture
system), use :meth:`~smplfitter.pt.BodyFitter.fit_with_known_pose`:

.. code-block:: python

   result = fitter.fit_with_known_pose(
       pose_rotvecs=known_pose,     # (batch, 72)
       target_vertices=vertices,
       target_joints=joints,
       beta_regularizer=1,
       share_beta=True,             # estimate one shape for the whole batch
   )

Since SMPL is linear in shape, this is a single linear least-squares solve
(no iteration needed).


Use Vertex and Joint Weights
----------------------------

Assign per-vertex and per-joint confidence weights to downweight noisy or occluded
parts of the input:

.. code-block:: python

   # Example: set low weight for uncertain vertices
   vertex_weights = confidence_scores          # (batch, 6890), higher = more trusted
   joint_weights = joint_confidence_scores     # (batch, 24)

   result = fitter.fit(
       target_vertices=vertices,
       target_joints=joints,
       vertex_weights=vertex_weights,
       joint_weights=joint_weights,
       num_iter=3,
       beta_regularizer=1,
   )

Weights are used in both the Kabsch orientation fitting and the least-squares shape
fitting. For example, if your vertex estimates come with per-vertex uncertainty from
a model like NLF, use the inverse uncertainty as the weight.


Share Shape Across a Video Sequence
-----------------------------------

For multiple frames of the same person, estimate a single shared shape vector
while allowing pose to vary per frame:

.. code-block:: python

   # Stack all frames into one batch
   all_vertices = torch.stack(frame_vertices)   # (n_frames, 6890, 3)

   result = fitter.fit(
       target_vertices=all_vertices,
       num_iter=3,
       beta_regularizer=1,
       share_beta=True,
   )

   result['shape_betas']    # (1, 10) - single shared shape
   result['pose_rotvecs']   # (n_frames, 72) - per-frame pose

Sharing shape produces more consistent and accurate shape estimates, especially
when individual frames have noisy or partial observations.


Fit a Vertex Subset
-------------------

For faster fitting or when only a subset of vertices is available, create the body
model with a vertex subset:

.. code-block:: python

   import numpy as np

   subset_indices = np.array([0, 100, 200, ...])  # indices into the full mesh
   body_model = BodyModel('smpl', 'neutral', vertex_subset=subset_indices).cuda()
   fitter = BodyFitter(body_model).cuda()

   result = fitter.fit(
       target_vertices=subset_vertices,  # (batch, len(subset_indices), 3)
       num_iter=3,
   )

The subset should cover the entire body, with several vertices per body part,
otherwise orientation fitting will be underdetermined.

Using 1024 vertices still allows high-quality fits while significantly improving
throughput.

If you want to use joints with a vertex subset and don't have externally computed
joints, provide a compatible joint regressor of size
``(num_joints, len(subset_indices))``:

.. code-block:: python

   body_model = BodyModel(
       'smpl', 'neutral',
       vertex_subset=subset_indices,
       joint_regressor_post_lbs=my_regressor,
   ).cuda()


Estimate Scale
--------------

When the metric scale of your input is unknown, the fitter can estimate a scale
correction factor. There are two modes:

**Scale the target** (input is rescaled to match the model):

.. code-block:: python

   result = fitter.fit(
       target_vertices=vertices,
       num_iter=3,
       scale_target=True,
   )
   result['scale_corr']  # (batch,) estimated scale factor

This tends to bias toward smaller bodies, since vertex error is lower for
smaller meshes. For video, divide by the mean estimated scale across frames to
correct for this.

**Scale the fit** (model output is rescaled to match the input):

.. code-block:: python

   result = fitter.fit(
       target_vertices=vertices,
       num_iter=3,
       scale_fit=True,
       scale_regularizer=1,  # penalize scale deviation from 1
   )

This mode is incompatible with ``share_beta=True`` because it creates a
nonlinear coupling between scale and shape.


Flip Body Parameters Left-Right
--------------------------------

Mirror body parameters (e.g., for data augmentation):

.. code-block:: python

   from smplfitter.pt import BodyModel, BodyFlipper

   body_model = BodyModel('smpl', 'neutral').cuda()
   flipper = BodyFlipper(body_model).cuda()

   result = flipper.flip(
       pose_rotvecs=pose,     # (batch, 72)
       shape_betas=betas,     # (batch, 10)
       trans=trans,            # (batch, 3)
       num_iter=1,
   )

   result['pose_rotvecs']  # flipped pose
   result['shape_betas']   # flipped shape (accounts for model asymmetry)
   result['trans']          # flipped translation

The flipper accounts for left-right asymmetry in the body model template by
internally re-fitting after vertex flipping, rather than just swapping joint
indices.


Convert Between Body Model Types
---------------------------------

Use :class:`~smplfitter.pt.BodyConverter` to convert parameters between body model
types (e.g., SMPL to SMPL-X). The converter uses a closed-form fitting algorithm
that is thousands of times faster than the official SMPL-X transfer tool.

.. code-block:: python

   import torch
   from smplfitter.pt import BodyModel, BodyConverter

   # Create body models
   smpl = BodyModel('smpl', 'neutral').cuda()
   smplx = BodyModel('smplx', 'neutral').cuda()

   # Create converter
   converter = BodyConverter(smpl, smplx).cuda()
   converter = torch.jit.script(converter)

   # Convert parameters
   result = converter.convert(
       pose_rotvecs=smpl_pose,      # (batch, 72)
       shape_betas=smpl_betas,      # (batch, 10)
       trans=smpl_trans,            # (batch, 3)
       num_iter=3,
   )

   smplx_pose = result['pose_rotvecs']   # (batch, 55*3)
   smplx_betas = result['shape_betas']   # (batch, 10)
   smplx_trans = result['trans']         # (batch, 3)

The converter also supports:

* ``known_output_pose_rotvecs``: If pose is already known, only fit shape
* ``known_output_shape_betas``: If shape is already known, only fit pose


Use the Convenience Functions
-----------------------------

For one-off fitting without manually creating model and fitter objects:

.. code-block:: python

   from smplfitter.pt import fit

   result = fit(
       verts=vertices,
       joints=joints,
       body_model_name='smpl',
       gender='neutral',
       num_betas=10,
       num_iter=3,
   )

For repeated fitting (e.g., in a training loop), use a cached and JIT-compiled
fitter:

.. code-block:: python

   from smplfitter.pt import get_cached_fit_fn

   fit_fn = get_cached_fit_fn(
       body_model_name='smpl',
       gender='neutral',
       num_betas=10,
       num_iter=3,
       device='cuda',
   )

   # Call repeatedly - model and JIT compilation are cached
   result = fit_fn(vertices, joints, vertex_weights, joint_weights)

The cached function returns the same compiled fitter on subsequent calls with
the same arguments.


Use a Different Backend
-----------------------

The same API is available in NumPy and TensorFlow. Replace the import:

**NumPy:**

.. code-block:: python

   from smplfitter.np import BodyModel, BodyFitter

   body_model = BodyModel('smpl', 'neutral', num_betas=10)
   fitter = BodyFitter(body_model)
   result = fitter.fit(target_vertices=vertices_np, num_iter=3)

**TensorFlow:**

.. code-block:: python

   from smplfitter.tf import BodyModel, BodyFitter

   body_model = BodyModel('smpl', 'neutral', num_betas=10)
   fitter = BodyFitter(body_model)
   result = fitter.fit(target_vertices=vertices_tf, num_iter=3)

The PyTorch backend is the most complete and generally recommended. The NumPy
backend is useful when no GPU framework is available. The TensorFlow backend
integrates with TF computation graphs.


Tune the Beta Regularizer
-------------------------

The ``beta_regularizer`` parameter controls the L2 penalty on shape parameters
(excluding the first two, which encode overall size and BMI):

- ``beta_regularizer=0``: No regularization. Best if inputs are clean and complete.
- ``beta_regularizer=1``: Moderate regularization. Good default for noisy inputs.
- ``beta_regularizer=10``: Strong regularization. Use for very noisy or partial inputs.

The separate ``beta_regularizer2`` parameter controls regularization on the
first two shape components (size/BMI). By default it is 0, meaning size is
always freely estimated.

.. code-block:: python

   result = fitter.fit(
       target_vertices=vertices,
       num_iter=3,
       beta_regularizer=1,    # regularize shape components 3+
       beta_regularizer2=0,   # don't regularize size/BMI
   )


Request Specific Output Keys
-----------------------------

By default, fitting returns all available outputs. To reduce computation,
request only what you need:

.. code-block:: python

   result = fitter.fit(
       target_vertices=vertices,
       num_iter=3,
       requested_keys=['pose_rotvecs', 'shape_betas', 'trans'],
   )

Available keys: ``pose_rotvecs``, ``shape_betas``, ``trans``, ``vertices``,
``joints``, ``orientations``, ``relative_orientations``, ``kid_factor``,
``scale_corr``.
