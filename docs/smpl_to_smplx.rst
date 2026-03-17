SMPL-to-SMPL-X Conversion Benchmark
====================================

:class:`~smplfitter.pt.BodyConverter` was benchmarked against the official
`SMPL-X transfer tool <https://github.com/vchoutas/smplx/tree/main/transfer_model>`_
on 33 sample meshes from the official SMPL-X repository (the same evaluation setup as
Table 10 in the `NLF paper <https://arxiv.org/abs/2407.07532>`_).

Error is measured as per-vertex Euclidean distance (mm) between the ground-truth SMPL-X
mesh and the converted mesh.

Official SMPL-X Converter
-------------------------

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

SMPLFitter (PyTorch, GPU)
-------------------------

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

SMPLFitter (TensorFlow, GPU)
-----------------------------

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

Reproducing
-----------

.. code-block:: bash

   python benchmark/bench_converter.py --backends pt tf
