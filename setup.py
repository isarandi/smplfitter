from setuptools import setup

setup(
    name='smplfit',
    version='0.1.2',
    author='István Sárándi',
    author_email='istvan.sarandi@uni-tuebingen.de',
    packages=['smplfit'],
    scripts=[],
    description='Inverse kinematics solver and body shape fitter for SMPL-family body models, including a reimplementation of these body models for NumPy, PyTorch and TensorFlow',
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'addict',
        'scipy',
#        'chumpy',
    ],
    extras_require={
        'tensorflow': ['tensorflow'],
        'pytorch': ['torch'],
    },
)
