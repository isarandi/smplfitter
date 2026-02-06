SMPLFitter: The Fast Way From Vertices to Parametric 3D Humans
================================================================

.. image:: _static/images/example.gif
   :alt: on_image
   :width: 500px

This repository contains code for efficiently fitting parametric SMPL/SMPL+H/SMPL-X human body models to nonparametric 3D vertex and joint locations. The input needs to be in correspondence with the body template - this code does not handle unordered input point clouds.

Example use cases:

* You extracted nonparametric vertex and joint estimates from an RGB image, e.g. using `Neural Localizer Fields (NLF) <https://virtualhumans.mpi-inf.mpg.de/nlf>`_, and want to express this estimate in parametric form, for example to feed it to another model that expects body model parameters.
* You want to convert between body models. For example, you have SMPL parameters from some dataset but need SMPL-X parameters as input to some pretrained model (or vice versa).

We provide the implementation in **PyTorch, TensorFlow and NumPy**.

The algorithm is **fast**, optimized for **batch** processing, can run on the **GPU** and is **differentiable**. There are no learnable parameters here, nor sensitivity to initialization. Just solving equation systems.

It can fit a batch of 4096 instances in 423 ms on a single RTX 3090 GPU giving a throughput of **9481 fits per second**. At the small batch size regime (batch size 32), the throughput is still 1839 fits/second. When using a subset of 1024 vertices (which still allows high-quality fits), one can fit a batch of 16384 instances in 440 ms. For 25 fps videos, this means you can fit SMPL params to every frame of 10 minutes of nonparametric motion data in less than half a second.

Installation
------------

.. code-block:: bash

   pip install smplfitter


Download Body Model Files
--------------------------

You need to download the body model data files from the corresponding websites for this code to work. You only need the ones that you plan to use. There should be a ``DATA_ROOT`` environment variable under which a ``body_models`` directory should look like this:

.. code-block:: text

   $DATA_ROOT/body_models
   ├── smpl
   │   ├── basicmodel_f_lbs_10_207_0_v1.1.0.pkl
   │   ├── basicmodel_m_lbs_10_207_0_v1.1.0.pkl
   │   ├── basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
   │   └── kid_template.npy
   ├── smplx
   │   ├── kid_template.npy
   │   ├── SMPLX_FEMALE.npz
   │   ├── SMPLX_MALE.npz
   │   └── SMPLX_NEUTRAL.npz
   ├── smplh
   │   ├── kid_template.npy
   │   ├── SMPLH_FEMALE.pkl
   │   └── SMPLH_MALE.pkl
   ├── smplh16
   │   ├── kid_template.npy
   │   ├── female/model.npz
   │   ├── male/model.npz
   │   └── neutral/model.npz
   ├── smpl2smplx_deftrafo_setup.pkl
   └── smplx2smpl_deftrafo_setup.pkl

You can refer to the relevant `script <https://github.com/isarandi/PosePile/tree/main/posepile/get_body_models.sh>`_ in the PosePile repo about how to download these files.

Usage Examples
--------------

Basic Fitting
^^^^^^^^^^^^^

.. code-block:: python

   import torch
   from smplfitter.pt import BodyModel, BodyFitter

   body_model = BodyModel('smpl', 'neutral').cuda()
   fitter = BodyFitter(body_model, num_betas=10).cuda()
   fitter = torch.jit.script(fitter)

   batch_size = 30
   vertices = torch.rand((batch_size, 6890, 3)).cuda()
   joints = torch.rand((batch_size, 24, 3)).cuda()

   fit_res = fitter.fit(vertices, joints, n_iter=3, beta_regularizer=1)
   fit_res['pose_rotvecs'], fit_res['shape_betas'], fit_res['trans']

Body Model Conversion (Transfer)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import torch
   from smplfitter.pt import BodyConverter

   bm_in = BodyModel('smpl', 'neutral')
   bm_out = BodyModel('smplx', 'neutral')
   smpl2smplx = BodyConverter(bm_in, bm_out).cuda()
   smpl2smplx = torch.jit.script(smpl2smplx)

   batch_size = 30
   pose_rotvecs_in = torch.rand((batch_size, 72)).cuda()
   shape_betas_in = torch.rand((batch_size, 10)).cuda()
   trans_in = torch.rand((batch_size, 3)).cuda()

   out = smpl2smplx.convert(pose_rotvecs_in, shape_betas_in, trans_in)
   out['pose_rotvecs'], out['shape_betas'], out['trans']


Citation
--------

If you find this code useful, please consider citing our work.
This algorithm was developed for and described in the following paper:

.. code-block:: bibtex

   @article{sarandi24nlf,
       title = {Neural Localizer Fields for Continuous 3D Human Pose and Shape Estimation},
       author = {Sárándi, István and Pons-Moll, Gerard},
       journal = {Advances in Neural Information Processing Systems (NeurIPS)},
       year = {2024},
   }


.. toctree::
   :maxdepth: 3
   :caption: Contents

   howto
   API Reference <api/smplfitter/index>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
