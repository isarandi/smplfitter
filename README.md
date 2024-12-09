# SMPLFitter: The Fast Way From Vertices to Parametric 3D Humans

<img src="docs/source/_static/figures/example.gif" alt="on_image" width="500"/>

This repository contains code for efficiently fitting parametric SMPL/SMPL+H/SMPL-X human body models to nonparametric 3D vertex and joint locations. The input needs to be in correspondence with the body template - this code does not handle unordered input point clouds.

Example use cases:
* You extracted nonparametric vertex and joint estimates from an RGB image, e.g. using [Neural Localizer Fields (NLF)](https://virtualhumans.mpi-inf.mpg.de/nlf), and want to express this estimate in parametric form, for example to feed it to another model that expects body model parameters.
* You want to convert between body models. For example, you have SMPL parameters from some dataset but need SMPL-X parameters as input to some pretrained model (or vice versa).

We provide the implementation in **PyTorch, TensorFlow and NumPy**. 

The algorithm is **fast**, optimized for **batch** processing, can run on the **GPU** and is **differentiable**. There are no learnable parameters here, nor sensitivity to initialization. Just solving equation systems.

It can fit a batch of 4096 instances in 423 ms on a single RTX 3090 GPU giving a throughput of **9481 fits per second**. At the small batch size regime (batch size 32), the throughput is still 1839 fits/second. When using a subset of 1024 vertices (which still allows high-quality fits), one can fit a batch of 16384 instances in 440 ms. For 25 fps videos, this means you can fit SMPL params to every frame of 10 minutes of nonparametric motion data in less than half a second.


## Installation

```bash
pip install git+https://github.com/isarandi/smplfitter.git
```

(Packaging for PyPI is planned for later.)

### Download Body Model Files

You need to download the body model data files from the corresponding websites for this code to work. You only need the ones that you plan to use. There should be a `DATA_ROOT` environment variable under which a `body_models` directory should look like this:

```
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
```

You can refer to the relevant [script](https://github.com/isarandi/PosePile/tree/main/posepile/get_body_models.sh) in the PosePile repo about how to download these files.

## Usage Examples

### Basic Fitting

```python
import torch
from smplfitter import SMPLBodyModel, SMPLFitter

body_model = SMPLBodyModel('smpl', 'neutral').cuda()  # create the body model to be fitted
fitter = SMPLFitter(body_model, num_betas=10).cuda()  # create the fitter
fitter = torch.jit.script(fitter)  # optional: compile the fitter for faster execution

# Obtain a batch of nonparametric vertex and joint locations (here we use dummy random data)
batch_size = 30
vertices = torch.rand((batch_size, 6890, 3)).cuda()
joints = torch.rand((batch_size, 24, 3)).cuda()

# Do the fitting!
fit_res = fitter.fit(vertices, joints, n_iter=3, beta_regularizer=1)
fit_res['pose_rotvecs'], fit_res['shape_betas'], fit_res['trans']
```

### Body Model Conversion (Transfer)

```python
import torch
from smplfitter import SMPLConverter

smpl2smplx = SMPLConverter('smpl', 'neutral', 'smplx', 'neutral').cuda()
smpl2smplx = torch.jit.script(smpl2smplx)  # optional: compile the converter for faster execution

batch_size = 30
pose_rotvecs_in = torch.rand((batch_size, 72)).cuda()
shape_betas_in = torch.rand((batch_size, 10)).cuda()
trans_in = torch.rand((batch_size, 3)).cuda()

out = smpl2smplx.convert(pose_rotvecs_in, shape_betas_in, trans_in)
out['pose_rotvecs'], out['shape_betas'], out['trans']
```


## The Algorithm

SMPL(-X/+H) is a parametric body model that takes body part orientations $\theta$ and body shape vector $\beta$ as inputs and yields vertex and joint locations as outputs. SMPLfit approximates the **inverse operation**: it takes vertex and joint locations as inputs and yields orientations $\theta$ and shape $\beta$ as outputs.

Our algorithm alternates between fitting orientations and fitting shape. A good result can be obtained already with 1-3 iterations.

We illustrate the steps with the following example. Given the depicted RGB image, we used [Neural Localizer Fields (NLF)](https://virtualhumans.mpi-inf.mpg.de/nlf) to obtain nonparametric vertex and joint locations as follows:

<img src="docs/source/_static/figures/image.png" alt="on_image" width="300"/>
<img src="docs/source/_static/figures/on_image.png" alt="on_image" width="300"/>
<img src="docs/source/_static/figures/pose3d.png" alt="on_image" width="300"/>

### Fitting Body Part Orientations

We first **partition the body** based on the linear blend skinning (LBS) weights provided with the SMPL model, and then **independently fit the global orientation of each body part**.

We illustrate the partitioning on both the target and the template reference:

<img src="docs/source/_static/figures/pose3d_parts_color.png" alt="tmean_parts" width="450"/>
<img src="docs/source/_static/figures/tmean_parts.png" alt="tmean_parts" width="450"/>

We can separate the body parts for better visualization:

<img src="docs/source/_static/figures/pose3d_parts_color_blown.png" alt="tmean_parts" width="450"/>
<img src="docs/source/_static/figures/tmean_parts_blown.png" alt="tmean_parts" width="450"/>


For each body part, we have a set of target vertices from the input and a set of reference vertices, initially from the default SMPL template mesh. We apply the **Kabsch algorithm** to get the least-squares optimal rotation between the target and the reference vertices of each body part.

The first figure below shows the body parts of the target (blue) and reference (red) pairwise aligned at the centroid. The second figure shows the result of the Kabsch algorithm, with each body part independently and optimally rotated. In the third figure we show the SMPL mesh with the estimated orientations applied (red), compared with the target mesh (blue). (Note how the body shape and size do not align well yet.)

<img src="docs/source/_static/figures/parts_overlay1.png" alt="tmean_parts" width="300"/>
<img src="docs/source/_static/figures/parts_overlay1_rot.png" alt="tmean_parts" width="300"/>

<img src="docs/source/_static/figures/overlay_after_rot1.png" alt="tmean_parts" width="300"/>


In subsequent iterations, the reference is no longer the default template, but the SMPL mesh posed and shaped according to the current parametric estimate.

We found it important to **weight the joints much higher than the vertices** when solving for the orientations. This ensures that the independently estimated orientations correctly "link up" without error accumulation, since neighboring body parts share a joint.

### Fitting Body Shape

In SMPL, the L stands for linear, and so SMPL has a nice property: as long as the body part orientations $\theta$ are fixed, the **mapping from shape vector to vertex locations is linear** (the same applies to the joints), so solving for the shape boils down to a linear least squares problem.

SMPL is linear because going from shape vector to vertex location only involves matrix multiplications and adding constants: multiplying by the blend shapes, adding the template, adding pose-dependent blendshapes (which are constant for a fixed pose), multiplying by rotation matrices and multiplying by the linear blend skinning weights. (Technically, it's affine, not linear, but it boils down to the same thing.)

This means that for each vertex index $i$ there is a matrix $A_i(\theta)\in \mathbb{R}^{3\times 10}$ and vector $b_i(\theta)\in \mathbb{R}^3$ such that the vertex location can be written as

$$v_i(\theta, \beta) = A_i(\theta) \beta + b_i(\theta).$$

Substituting $\beta=0$ shows that $b_i(\theta)=v_i(\theta, 0)$, in other words, calculating $b_i$ is the same as applying the forward function of SMPL with the zero shape vector. Calculating $A_i$ boils down to the same SMPL forward computation, but extended with forward-mode automatic differentiation. That is, in each step of the forward computation, we track the Jacobian alongside the values.

Once we have the matrices $A_i$ and vectors $b_i$, we can **concatenate them** into a single large matrix $A$ and vector $b$, and solve the combined linear system $A\beta = v - b$ for the shape vector $\beta$ by linear least squares. Specifically, we do this via Cholesky decomposition.

The shape vector can be regularized with an L2 penalty, which is useful when the input is noisy. We found that it is best not to regularize the first two shape parameters, which correspond to the global scaling (body size and BMI) of the body.

Below we show the overlay of target and reference before  and after shape fitting (observe how the silhouette is much better matched after shape fitting):

<img src="docs/source/_static/figures/overlay_after_rot1.png" alt="tmean_parts" width="450"/>
<img src="docs/source/_static/figures/overlay_after_rot1_shape.png" alt="tmean_parts" width="450"/>

### Final Orientation Adjustment

In the orientation estimation step described above, we fit the orientations of the body parts independently, using the centroid as the rotation anchor point (following the Kabsch algorithm). However, even if we perfectly estimate these independent orientations, we may have translational errors due to bone length mismatches.

Therefore, we optionally perform a final adjustment of the orientations, where we sequentially (i.e. non-independently) adjust the orientation of each body part along the kinematic tree, starting from the root joint. Different from the previously described orientation fitting, we now use the position of the proximal joint of the body part as the anchor (instead of the centroid of the body part). This way a modified orientation can compensate for the translational error and bring the distal body parts closer to the target, preventing one joint error from impacting all subsequent parts in the chain.

Below we illustrate the effect of this step. The first image shows the situation after two iterations of orientation and shape fitting, but before the final adjustment. The second one is after the adjustment.

<img src="docs/source/_static/figures/overlay_after_rot2_shape.png" alt="tmean_parts" width="450"/>

<img src="docs/source/_static/figures/overlay_after_rot3.png" alt="tmean_parts" width="450"/>

The difference is visually subtle, but can be seen around the upper right arm and armpit area, as well as the heel of the right foot.

### Handling of Incomplete Input

The algorithm can also fit to a **subset of the vertices**. However, the subset should properly cover the whole body including several vertices on each body part, otherwise the orientation fitting will be underdetermined and fail.

The input target joint locations are also optional. If they are not given, they are calculated from the input vertices using the SMPL joint regressor. (If you want to use a subset of vertices as the target and don't want to provide joints, then you need to provide a joint regressor $\mathcal{J}$ of size $n_\text{joints}\times n_\text{subset}$ that works for that subset of vertices - a simple way to get such a regressor could be natural neighbor interpolation weights based on the template mesh, while a more complicated way would be to learn it from a dataset like AMASS).

### Weights

Our algorithm supports **vertex and joint weights**. The weights are used in both the least-squares orientation and shape fitting. The idea is to downweight noisy or unreliable parts of the input. For example, if your nonparametric vertices and joints come with uncertainty estimates, such as in the case of NLF, you can use the inverse of the uncertainty (or some power of thereof) as the weight.

### Sharing Shape Parameters Across Instances

If you have several nonparametric body estimates for the same person in different poses (such as from a video), you can fit all of them together, estimating one shared shape vector for all instances of the batch, while not sharing the orientation parameters. This way the body shape will be more accurately estimated, as there are multiple observations to estimate it from and the output will be more consistent over time.

### Scaling (experimental)
If you don't trust your input to have the correct metric scale, the shape fitting step can be extended by a scale fitting as well. There are two ways to do this: 1) we estimate a scale factor for the target body (your input), or 2) we estimate a scale factor for the reference body (the resulting output). Neither is really satisfactory, hence experimental. The former will bias the estimation towards smaller bodies as the vertex-to-vertex error that is being minimized scales with body size, and so the solver will favor scaling down the target and producing a beta vector for a smaller person. Estimating the scale for the reference body causes an issue if we want to share the betas among many instances, since the problem becomes quadratic (the beta variables get multiplied by the scale factor variable). For just one instance, this is doable, though messes with the L2 regularization strength. Tentatively, we recommend using the first option, but accounting for the scale bias by e.g. dividing by the mean estimated scale for a longer video sequence.


## Citation

If you find this code useful, please consider citing our work.
This algorithm was developed for and described in the following paper:

```
@article{sarandi24nlf,
    title = {Neural Localizer Fields for Continuous 3D Human Pose and Shape Estimation},
    author = {Sárándi, István and Pons-Moll, Gerard},
    journal = {Advances in Neural Information Processing Systems (NeurIPS)},
    year = {2024},
}
```
