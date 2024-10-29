# Installation

## Installing the library
```bash
pip install git+https://github.com/isarandi/smplfitter.git
```

(Packaging for PyPI is planned for later.)

## Downloading the body model files

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