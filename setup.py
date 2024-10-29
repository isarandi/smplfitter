from setuptools import setup

setup(
    name='smplfitter',
    version='0.1.2',
    author='István Sárándi',
    author_email='istvan.sarandi@uni-tuebingen.de',
    packages=['smplfitter'],
    scripts=[],
    description='Inverse kinematics solver and body shape fitter for SMPL-family body models, including a reimplementation of these body models for NumPy, PyTorch and TensorFlow',
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy',
#        'chumpy', # only for SMPL+H
    ],
    extras_require={
        'tensorflow': ['tensorflow'],
        'pytorch': ['torch'],
    },
)
