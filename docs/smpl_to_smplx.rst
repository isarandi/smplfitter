SMPL to SMPL-X Conversion
=========================

SMPLFitter can convert SMPL body parameters to SMPL-X parameters orders of magnitude faster than the official SMPL-X transfer tool, while achieving comparable accuracy.

Converting between body model formats (e.g., SMPL to SMPL-X) is a common need when:

* A dataset provides annotations in one format but your model expects another
* You want to leverage pretrained models that use a different body representation
* You need to combine data from different sources with different body models

The official `SMPL-X transfer tool <https://github.com/vchoutas/smplx/tree/main/transfer_model>`_ is accurate but uses a slow iterative optimization algorithm, taking several minutes per mesh.

SMPLFitter's :class:`~smplfitter.pt.BodyConverter` uses a simpler, closed-form fitting algorithm that works with 1-3 iterations.


Benchmark Results
-----------------

We compared SMPLFitter against the official SMPL-X converter on 33 sample meshes from the official SMPLX repository  (the same evaluation setup as Table 10 in the `NLF paper <https://arxiv.org/abs/2407.07532>`_).

**Official SMPL-X Converter:**

.. list-table::
   :header-rows: 1
   :widths: 20 30 20

   * - Iterations
     - Time
     - Error (mm)
   * - 25
     - 3 min 18 s
     - 14.0
   * - 50
     - 16 min 35 s
     - 6.2
   * - 100
     - 33 min
     - 5.0

**SMPLFitter (PyTorch, GPU):**

.. list-table::
   :header-rows: 1
   :widths: 20 30 20

   * - Iterations
     - Time (33 meshes)
     - Error (mm)
   * - 1
     - 43 ms
     - 8.3
   * - 2
     - 75 ms
     - 8.0
   * - 3
     - 110 ms
     - 8.0
   * - 5
     - 190 ms
     - 8.1

**SMPLFitter (TensorFlow, GPU):**

.. list-table::
   :header-rows: 1
   :widths: 20 30 20

   * - Iterations
     - Time (33 meshes)
     - Error (mm)
   * - 1
     - 25 ms
     - 8.3
   * - 2
     - 35 ms
     - 8.0
   * - 3
     - 50 ms
     - 8.0
   * - 5
     - 78 ms
     - 8.1

Key Takeaways
-------------

* **Thousands of times faster**: SMPLFitter converts 33 meshes in 50 ms vs 33 minutes for the official tool
* **Good accuracy**: 8 mm error vs 5-14 mm depending on official tool iterations
* **Batch processing**: Process thousands of meshes in seconds
* **No initialization sensitivity**: Closed-form solution, deterministic results

Usage
-----

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

Reproducing the Benchmark
-------------------------

The benchmark script is available at ``benchmark/bench_converter.py``:

.. code-block:: bash

   python benchmark/bench_converter.py --backends pt tf
