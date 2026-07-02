"""Benchmark smplfitter backends.

Run with: python benchmark/run_benchmark.py
"""

import numpy as np
import sqlite3
import time
from abc import ABC, abstractmethod
from pathlib import Path

ALL_BACKENDS = ['numpy', 'numba', 'pytorch', 'tensorflow', 'jax', 'smplx']
ALL_MODELS = ['smpl', 'smplx']
FIT_METHODS = ['fit', 'fit_known_shape', 'fit_known_pose']


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--backends', nargs='+', choices=ALL_BACKENDS)
    parser.add_argument('--models', nargs='+', choices=ALL_MODELS, default=['smpl'])
    parser.add_argument('--bench', choices=['forward', 'fit', 'both'], default='both')
    parser.add_argument('--db', default='results.db')
    args = parser.parse_args()

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    fit_batch_sizes = [1, 32, 1024]
    fit_vertex_variants = [('full', None), ('v1024', 1024)]
    fit_num_iter = 3
    out_dir = Path(__file__).parent
    conn = init_db(out_dir / args.db)

    for model_name in args.models:
        if args.bench in ('forward', 'both'):
            for mode, return_vertices in [('with_vertices', True), ('joints_only', False)]:
                print(f'\nBenchmarking forward: {model_name} {mode}')
                backends = make_backends(
                    model_name=model_name, return_vertices=return_vertices, which=args.backends
                )
                results = run_benchmarks(backends, batch_sizes)

                names = [b.name for b in backends]
                print(f"{'Batch':>8} |" + ''.join(f' {n:>12} |' for n in names))
                for r in results:
                    row = f"{r['batch_size']:>8} |"
                    for name in names:
                        row += f' {r[name]:>12.2f} |'
                    print(row)

                for r in results:
                    for name in names:
                        conn.execute(
                            'INSERT OR REPLACE INTO results (backend, model, mode, batch_size, time_ms) VALUES (?, ?, ?, ?, ?)',
                            (name, model_name, mode, r['batch_size'], r[name]),
                        )
                conn.commit()

        if args.bench in ('fit', 'both'):
            for vlabel, vsize in fit_vertex_variants:
                print(f'\nBenchmarking fit: {model_name} verts={vlabel} num_iter={fit_num_iter}')
                backends = make_fit_backends(
                    model_name=model_name,
                    vertex_subset_size=vsize,
                    num_iter=fit_num_iter,
                    which=args.backends,
                )
                if not backends:
                    print('  (no backends available for this configuration)')
                    continue
                for method in FIT_METHODS:
                    print(f'\n  method={method}')
                    results = run_fit_benchmarks(backends, fit_batch_sizes, method)

                    names = [b.name for b in backends]
                    print(f"{'Batch':>8} |" + ''.join(f' {n:>14} |' for n in names))
                    for r in results:
                        row = f"{r['batch_size']:>8} |"
                        for name in names:
                            val = r.get(name)
                            row += f' {val:>14.2f} |' if val is not None else f' {"--":>14} |'
                        print(row)

                    for r in results:
                        for name in names:
                            if r.get(name) is None:
                                continue
                            conn.execute(
                                'INSERT OR REPLACE INTO fit_results (backend, model, method, vertex_count, batch_size, num_iter, time_ms) VALUES (?, ?, ?, ?, ?, ?, ?)',
                                (
                                    name,
                                    model_name,
                                    method,
                                    vlabel,
                                    r['batch_size'],
                                    fit_num_iter,
                                    r[name],
                                ),
                            )
                    conn.commit()

    conn.close()
    print(f'\nSaved results to {out_dir / args.db}')


def run_benchmarks(backends, batch_sizes):
    num_joints = backends[0].num_joints
    num_betas = backends[0].num_betas
    results = []

    for batch_size in batch_sizes:
        pose = np.random.randn(batch_size, num_joints * 3).astype(np.float32) * 0.1
        shape = np.random.randn(batch_size, num_betas).astype(np.float32) * 0.5
        trans = np.random.randn(batch_size, 3).astype(np.float32)
        n_iter = max(10, 50 // batch_size)
        timings = {'batch_size': batch_size}

        for backend in backends:
            inputs = backend.prepare(pose, shape, trans)
            backend.run(inputs)
            backend.sync()

            start = time.perf_counter()
            for _ in range(n_iter):
                backend.run(inputs)
            backend.sync()
            timings[backend.name] = (time.perf_counter() - start) / n_iter * 1000

        results.append(timings)
    return results


def run_fit_benchmarks(backends, batch_sizes, method):
    """Time `method` for each backend across `batch_sizes`. Targets generated with NumPy ref."""
    ref_model = backends[0]._ref_model
    num_joints = ref_model.num_joints
    num_betas = ref_model.num_betas
    results = []

    for batch_size in batch_sizes:
        np.random.seed(42)
        pose = np.random.randn(batch_size, num_joints * 3).astype(np.float32) * 0.1
        shape = np.random.randn(batch_size, num_betas).astype(np.float32) * 0.5
        trans = np.random.randn(batch_size, 3).astype(np.float32)
        forw = ref_model(pose_rotvecs=pose, shape_betas=shape, trans=trans)
        target_verts = np.ascontiguousarray(forw['vertices'], dtype=np.float32)
        target_joints = np.ascontiguousarray(forw['joints'], dtype=np.float32)

        n_iter = max(3, 30 // batch_size)
        timings = {'batch_size': batch_size}

        for backend in backends:
            try:
                prepared = backend.prepare_fit(pose, shape, target_verts, target_joints)
                backend.run_fit(method, prepared)
                backend.sync()

                start = time.perf_counter()
                for _ in range(n_iter):
                    backend.run_fit(method, prepared)
                backend.sync()
                timings[backend.name] = (time.perf_counter() - start) / n_iter * 1000
            except Exception as e:
                print(f'    {backend.name}: failed ({type(e).__name__}: {e})')
                timings[backend.name] = None

        results.append(timings)
    return results


def make_backends(model_name='smpl', return_vertices=True, which=None):
    if which is None:
        which = ALL_BACKENDS
    backends = []

    if 'numpy' in which:
        backends.append(NumPyBackend(model_name=model_name, return_vertices=return_vertices))

    if 'numba' in which:
        backends.append(NumbaBackend(model_name=model_name, return_vertices=return_vertices))

    if 'pytorch' in which:
        try:
            import torch

            for compile_mode in [None, 'script', 'compile']:
                backends.append(
                    PyTorchBackend(
                        model_name=model_name,
                        gpu=False,
                        compile_mode=compile_mode,
                        return_vertices=return_vertices,
                    )
                )
                if torch.cuda.is_available():
                    backends.append(
                        PyTorchBackend(
                            model_name=model_name,
                            gpu=True,
                            compile_mode=compile_mode,
                            return_vertices=return_vertices,
                        )
                    )
        except ImportError:
            pass

    if 'tensorflow' in which:
        try:
            import tensorflow as tf

            backends.append(
                TensorFlowBackend(
                    model_name=model_name,
                    gpu=False,
                    function=True,
                    return_vertices=return_vertices,
                )
            )
            if tf.config.list_physical_devices('GPU'):
                backends.append(
                    TensorFlowBackend(
                        model_name=model_name,
                        gpu=True,
                        function=True,
                        return_vertices=return_vertices,
                    )
                )
        except ImportError:
            pass

    if 'jax' in which:
        try:
            import jax

            backends.append(
                JAXBackend(
                    model_name=model_name, gpu=False, jit=True, return_vertices=return_vertices
                )
            )
            if jax.devices()[0].platform == 'gpu':
                backends.append(
                    JAXBackend(
                        model_name=model_name, gpu=True, jit=True, return_vertices=return_vertices
                    )
                )
        except ImportError:
            pass

    if 'smplx' in which:
        try:
            import smplx as smplx_lib  # noqa: F401
            import torch

            backends.append(
                SMPLXLibBackend(
                    model_type=model_name,
                    gpu=False,
                    compile_mode='compile',
                    return_vertices=return_vertices,
                )
            )
            if torch.cuda.is_available():
                backends.append(
                    SMPLXLibBackend(
                        model_type=model_name,
                        gpu=True,
                        compile_mode='compile',
                        return_vertices=return_vertices,
                    )
                )
        except ImportError:
            pass

    return backends


def make_fit_backends(model_name='smpl', vertex_subset_size=None, num_iter=3, which=None):
    """Build backends for the fit benchmark. Skips backends that don't support the configuration."""
    if which is None:
        which = ALL_BACKENDS

    import smplfitter.np as smpl_np

    ref_model = smpl_np.BodyModel(
        model_name=model_name, num_betas=10, vertex_subset_size=vertex_subset_size
    )
    backends = []
    fit_kwargs = dict(
        model_name=model_name,
        vertex_subset_size=vertex_subset_size,
        num_iter=num_iter,
        ref_model=ref_model,
    )

    if 'numpy' in which:
        backends.append(NumPyFitBackend(**fit_kwargs))

    if 'numba' in which:
        backends.append(NumbaFitBackend(**fit_kwargs))

    if 'pytorch' in which:
        try:
            import torch

            for compile_mode in [None, 'script', 'compile']:
                backends.append(
                    PyTorchFitBackend(gpu=False, compile_mode=compile_mode, **fit_kwargs)
                )
                if torch.cuda.is_available():
                    backends.append(
                        PyTorchFitBackend(gpu=True, compile_mode=compile_mode, **fit_kwargs)
                    )
        except ImportError:
            pass

    if 'tensorflow' in which:
        try:
            import tensorflow as tf

            backends.append(TensorFlowFitBackend(gpu=False, **fit_kwargs))
            if tf.config.list_physical_devices('GPU'):
                backends.append(TensorFlowFitBackend(gpu=True, **fit_kwargs))
        except ImportError:
            pass

    if 'jax' in which and vertex_subset_size is None:
        try:
            import jax

            backends.append(JAXFitBackend(gpu=False, **fit_kwargs))
            if jax.devices()[0].platform == 'gpu':
                backends.append(JAXFitBackend(gpu=True, **fit_kwargs))
        except ImportError:
            pass

    return backends


def init_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            backend TEXT,
            model TEXT,
            mode TEXT,
            batch_size INTEGER,
            time_ms REAL,
            PRIMARY KEY (backend, model, mode, batch_size)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fit_results (
            backend TEXT,
            model TEXT,
            method TEXT,
            vertex_count TEXT,
            batch_size INTEGER,
            num_iter INTEGER,
            time_ms REAL,
            PRIMARY KEY (backend, model, method, vertex_count, batch_size, num_iter)
        )
    """)
    conn.commit()
    return conn


# Backend implementations


class Backend(ABC):
    name: str

    @abstractmethod
    def prepare(self, pose, shape, trans):
        pass

    @abstractmethod
    def run(self, inputs):
        pass

    def sync(self):
        pass

    @property
    @abstractmethod
    def num_joints(self):
        pass

    @property
    @abstractmethod
    def num_betas(self):
        pass

    def to_numpy(self, value):
        return value


class NumPyBackend(Backend):
    name = 'NumPy'

    def __init__(self, model_name='smpl', return_vertices=True):
        import smplfitter.np as smpl_np

        self.model = smpl_np.BodyModel(model_name=model_name, num_betas=10)
        self.return_vertices = return_vertices

    def prepare(self, pose, shape, trans):
        return dict(pose_rotvecs=pose, shape_betas=shape, trans=trans)

    def run(self, inputs):
        return self.model(**inputs, return_vertices=self.return_vertices)

    @property
    def num_joints(self):
        return self.model.num_joints

    @property
    def num_betas(self):
        return self.model.num_betas


class NumbaBackend(Backend):
    name = 'Numba'

    def __init__(self, model_name='smpl', return_vertices=True):
        import smplfitter.nb as smpl_nb

        self.model = smpl_nb.BodyModel(model_name=model_name, num_betas=10)
        self.return_vertices = return_vertices
        # Warmup JIT
        pose = np.zeros((1, self.num_joints * 3), dtype=np.float32)
        shape = np.zeros((1, self.num_betas), dtype=np.float32)
        trans = np.zeros((1, 3), dtype=np.float32)
        self.model(
            pose_rotvecs=pose, shape_betas=shape, trans=trans, return_vertices=return_vertices
        )

    def prepare(self, pose, shape, trans):
        return dict(pose_rotvecs=pose, shape_betas=shape, trans=trans)

    def run(self, inputs):
        return self.model(**inputs, return_vertices=self.return_vertices)

    @property
    def num_joints(self):
        return self.model.num_joints

    @property
    def num_betas(self):
        return self.model.num_betas


class PyTorchBackend(Backend):
    def __init__(self, model_name='smpl', gpu=False, compile_mode=None, return_vertices=True):
        import torch
        import smplfitter.pt as smpl_pt

        self.torch = torch
        self.gpu = gpu
        self.return_vertices = return_vertices

        model = smpl_pt.BodyModel(model_name=model_name, num_betas=10)
        if gpu:
            model = model.cuda()
        if compile_mode == 'compile':
            model = torch.compile(model)
        elif compile_mode == 'script':
            model = torch.jit.script(model)
        self.model = model

        suffix = ' GPU' if gpu else ' CPU'
        mode_name = {'compile': 'PT compile', 'script': 'PT script', None: 'PT eager'}[
            compile_mode
        ]
        self.name = mode_name + suffix

    def prepare(self, pose, shape, trans):
        inputs = dict(
            pose_rotvecs=self.torch.from_numpy(pose),
            shape_betas=self.torch.from_numpy(shape),
            trans=self.torch.from_numpy(trans),
        )
        if self.gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        return inputs

    def run(self, inputs):
        return self.model(**inputs, return_vertices=self.return_vertices)

    def sync(self):
        if self.gpu:
            self.torch.cuda.synchronize()

    @property
    def num_joints(self):
        return self.model.num_joints

    @property
    def num_betas(self):
        return self.model.num_betas

    def to_numpy(self, value):
        return value.cpu().numpy() if self.gpu else value.numpy()


class TensorFlowBackend(Backend):
    def __init__(self, model_name='smpl', gpu=False, function=False, return_vertices=True):
        import tensorflow as tf
        import smplfitter.tf as smpl_tf

        self.tf = tf
        self.gpu = gpu
        self.return_vertices = return_vertices
        self.device = '/GPU:0' if gpu else '/CPU:0'
        self.model = smpl_tf.BodyModel(model_name=model_name, num_betas=10)
        self._num_betas = self.model.shapedirs.shape[2]

        if function:

            @tf.function(jit_compile=True)
            def run_model(pose_rotvecs, shape_betas, trans):
                return self.model(
                    pose_rotvecs=pose_rotvecs,
                    shape_betas=shape_betas,
                    trans=trans,
                    return_vertices=return_vertices,
                )

            self._run = run_model
        else:
            self._run = lambda **kw: self.model(**kw, return_vertices=self.return_vertices)

        suffix = ' GPU' if gpu else ' CPU'
        self.name = ('TF function' if function else 'TensorFlow') + suffix

    def prepare(self, pose, shape, trans):
        with self.tf.device(self.device):
            return dict(
                pose_rotvecs=self.tf.constant(pose),
                shape_betas=self.tf.constant(shape),
                trans=self.tf.constant(trans),
            )

    def run(self, inputs):
        with self.tf.device(self.device):
            return self._run(**inputs)

    def sync(self):
        if self.gpu:
            with self.tf.device(self.device):
                self.tf.constant([0.0]).numpy()

    @property
    def num_joints(self):
        return self.model.num_joints

    @property
    def num_betas(self):
        return self._num_betas

    def to_numpy(self, value):
        return value.numpy()


class JAXBackend(Backend):
    def __init__(self, model_name='smpl', gpu=False, jit=False, return_vertices=True):
        import jax
        import jax.numpy as jnp
        import smplfitter.jax as smpl_jax

        self.jax = jax
        self.jnp = jnp
        self.gpu = gpu
        self.return_vertices = return_vertices
        self.backend = 'gpu' if gpu else 'cpu'
        self.device = jax.devices(self.backend)[0]
        self.model = smpl_jax.BodyModel(model_name=model_name, num_betas=10, device=self.device)

        def run_model(pose_rotvecs, shape_betas, trans):
            return self.model(
                pose_rotvecs=pose_rotvecs,
                shape_betas=shape_betas,
                trans=trans,
                return_vertices=return_vertices,
            )

        if jit:
            run_model = jax.jit(run_model, backend=self.backend)
        self._run = run_model

        suffix = ' GPU' if gpu else ' CPU'
        self.name = ('JAX jit' if jit else 'JAX') + suffix

        # Warmup
        with jax.default_device(self.device):
            pose = jnp.zeros((1, self.num_joints * 3), dtype=jnp.float32)
            shape = jnp.zeros((1, self.num_betas), dtype=jnp.float32)
            trans = jnp.zeros((1, 3), dtype=jnp.float32)
            self._run(pose, shape, trans)

    def prepare(self, pose, shape, trans):
        with self.jax.default_device(self.device):
            return dict(
                pose_rotvecs=self.jnp.array(pose),
                shape_betas=self.jnp.array(shape),
                trans=self.jnp.array(trans),
            )

    def run(self, inputs):
        return self._run(**inputs)

    def sync(self):
        result = self._run(
            **self.prepare(
                np.zeros((1, self.num_joints * 3), dtype=np.float32),
                np.zeros((1, self.num_betas), dtype=np.float32),
                np.zeros((1, 3), dtype=np.float32),
            )
        )
        self.jax.block_until_ready(result['joints'])

    @property
    def num_joints(self):
        return self.model.num_joints

    @property
    def num_betas(self):
        return self.model.num_betas

    def to_numpy(self, value):
        return np.array(value)


class SMPLXLibBackend(Backend):
    def __init__(self, model_type='smpl', gpu=False, compile_mode=None, return_vertices=True):
        import os
        import torch
        import smplx as smplx_lib
        from smplfitter.common import _chumpy_stub_modules

        self.torch = torch
        self.gpu = gpu
        self.return_vertices = return_vertices
        self.model_type = model_type

        DATA_ROOT = os.environ.get('DATA_ROOT', '.')
        with _chumpy_stub_modules():
            # For SMPLX, use use_pca=False to work with full 45-dim hand poses
            kwargs = {'gender': 'neutral', 'num_betas': 10, 'ext': 'npz'}
            if model_type == 'smplx':
                kwargs['use_pca'] = False
            self.model = smplx_lib.create(
                f'{DATA_ROOT}/body_models', model_type=model_type, **kwargs
            )
        if gpu:
            self.model = self.model.cuda()

        suffix = ' GPU' if gpu else ' CPU'
        self.name = 'smplx lib' + suffix

    def prepare(self, pose, shape, trans):
        # Split pose tensor based on model type
        # SMPL: global_orient (0:3) + body_pose (3:72)
        # SMPLX: global_orient (0:3) + body_pose (3:66) + jaw (66:69) + leye (69:72) + reye (72:75) + lhand (75:120) + rhand (120:165)
        batch_size = pose.shape[0]

        if self.model_type == 'smpl':
            global_orient = self.torch.from_numpy(pose[:, 0:3])
            body_pose = self.torch.from_numpy(pose[:, 3:72])
            inputs = dict(global_orient=global_orient, body_pose=body_pose)
        else:  # smplx
            global_orient = self.torch.from_numpy(pose[:, 0:3])
            body_pose = self.torch.from_numpy(pose[:, 3:66])
            jaw_pose = self.torch.from_numpy(pose[:, 66:69])
            leye_pose = self.torch.from_numpy(pose[:, 69:72])
            reye_pose = self.torch.from_numpy(pose[:, 72:75])
            left_hand_pose = self.torch.from_numpy(pose[:, 75:120])
            right_hand_pose = self.torch.from_numpy(pose[:, 120:165])
            expression = self.torch.zeros(batch_size, 10)
            inputs = dict(
                global_orient=global_orient,
                body_pose=body_pose,
                jaw_pose=jaw_pose,
                leye_pose=leye_pose,
                reye_pose=reye_pose,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                expression=expression,
            )

        betas = self.torch.from_numpy(shape)
        transl = self.torch.from_numpy(trans)

        if self.gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
            betas = betas.cuda()
            transl = transl.cuda()

        inputs['betas'] = betas
        inputs['transl'] = transl
        return inputs

    def run(self, inputs):
        return self.model(**inputs, return_verts=self.return_vertices)

    def sync(self):
        if self.gpu:
            self.torch.cuda.synchronize()

    @property
    def num_joints(self):
        return self.model.NUM_BODY_JOINTS + 1

    @property
    def num_betas(self):
        return 10

    def to_numpy(self, value):
        return value.cpu().numpy() if self.gpu else value.numpy()


# --- Fit backends ---


class FitBackend(ABC):
    """Backend wrapping a (model, fitter) pair for the fit benchmark.

    Each subclass populates ``self._fit_fns`` with three closures (one per method) that
    take only the per-batch tensor inputs; constants like num_iter and requested_keys
    are baked in so jit/compile boundaries don't retrace on them.
    """

    name: str

    def __init__(self, model_name, vertex_subset_size, num_iter, ref_model):
        self.model_name = model_name
        self.vertex_subset_size = vertex_subset_size
        self.num_iter = num_iter
        self._ref_model = ref_model

    @abstractmethod
    def prepare_fit(self, pose, shape, target_verts, target_joints): ...

    def run_fit(self, method, prepared):
        fn = self._fit_fns[method]
        if method == 'fit':
            return fn(prepared['target_vertices'], prepared['target_joints'])
        if method == 'fit_known_shape':
            return fn(
                prepared['shape_betas'], prepared['target_vertices'], prepared['target_joints']
            )
        if method == 'fit_known_pose':
            return fn(
                prepared['pose_rotvecs'], prepared['target_vertices'], prepared['target_joints']
            )
        raise ValueError(method)

    def sync(self):
        pass


class NumPyFitBackend(FitBackend):
    name = 'NumPy'

    def __init__(self, model_name, vertex_subset_size, num_iter, ref_model):
        super().__init__(model_name, vertex_subset_size, num_iter, ref_model)
        import smplfitter.np as smpl_np

        self.model = smpl_np.BodyModel(
            model_name=model_name, num_betas=10, vertex_subset_size=vertex_subset_size
        )
        self.fitter = smpl_np.BodyFitter(self.model)
        ni = num_iter
        self._fit_fns = {
            'fit': lambda tv, tj: self.fitter.fit(
                target_vertices=tv,
                target_joints=tj,
                num_iter=ni,
                beta_regularizer=1.0,
                requested_keys=['pose_rotvecs'],
            ),
            'fit_known_shape': lambda sb, tv, tj: self.fitter.fit_with_known_shape(
                shape_betas=sb,
                target_vertices=tv,
                target_joints=tj,
                num_iter=ni,
                requested_keys=['pose_rotvecs'],
            ),
            'fit_known_pose': lambda pr, tv, tj: self.fitter.fit_with_known_pose(
                pose_rotvecs=pr,
                target_vertices=tv,
                target_joints=tj,
                beta_regularizer=0.0,
                requested_keys=['shape_betas'],
            ),
        }

    def prepare_fit(self, pose, shape, target_verts, target_joints):
        return dict(
            pose_rotvecs=pose,
            shape_betas=shape,
            target_vertices=target_verts,
            target_joints=target_joints,
        )


class NumbaFitBackend(FitBackend):
    name = 'Numba'

    def __init__(self, model_name, vertex_subset_size, num_iter, ref_model):
        super().__init__(model_name, vertex_subset_size, num_iter, ref_model)
        import smplfitter.nb as smpl_nb

        self.model = smpl_nb.BodyModel(
            model_name=model_name, num_betas=10, vertex_subset_size=vertex_subset_size
        )
        self.fitter = smpl_nb.BodyFitter(self.model)
        ni = num_iter
        self._fit_fns = {
            'fit': lambda tv, tj: self.fitter.fit(
                target_vertices=tv,
                target_joints=tj,
                num_iter=ni,
                beta_regularizer=1.0,
                requested_keys=['pose_rotvecs'],
            ),
            'fit_known_shape': lambda sb, tv, tj: self.fitter.fit_with_known_shape(
                shape_betas=sb,
                target_vertices=tv,
                target_joints=tj,
                num_iter=ni,
                requested_keys=['pose_rotvecs'],
            ),
            'fit_known_pose': lambda pr, tv, tj: self.fitter.fit_with_known_pose(
                pose_rotvecs=pr,
                target_vertices=tv,
                target_joints=tj,
                beta_regularizer=0.0,
                requested_keys=['shape_betas'],
            ),
        }

    def prepare_fit(self, pose, shape, target_verts, target_joints):
        return dict(
            pose_rotvecs=pose,
            shape_betas=shape,
            target_vertices=target_verts,
            target_joints=target_joints,
        )


class PyTorchFitBackend(FitBackend):
    def __init__(
        self,
        model_name,
        vertex_subset_size,
        num_iter,
        ref_model,
        gpu,
        compile_mode,
    ):
        super().__init__(model_name, vertex_subset_size, num_iter, ref_model)
        import torch
        import smplfitter.pt as smpl_pt

        self.torch = torch
        self.gpu = gpu
        device = 'cuda' if gpu else 'cpu'
        self.model = smpl_pt.BodyModel(
            model_name=model_name, num_betas=10, vertex_subset_size=vertex_subset_size
        ).to(device)
        self.fitter = smpl_pt.BodyFitter(self.model).to(device)

        if compile_mode == 'script':
            self.model = torch.jit.script(self.model)
            self.fitter = torch.jit.script(self.fitter)

        ni = num_iter

        def fit_fn(tv, tj):
            return self.fitter.fit(
                target_vertices=tv,
                target_joints=tj,
                num_iter=ni,
                beta_regularizer=1.0,
                requested_keys=['pose_rotvecs'],
            )

        def fit_known_shape_fn(sb, tv, tj):
            return self.fitter.fit_with_known_shape(
                shape_betas=sb,
                target_vertices=tv,
                target_joints=tj,
                num_iter=ni,
                requested_keys=['pose_rotvecs'],
            )

        def fit_known_pose_fn(pr, tv, tj):
            return self.fitter.fit_with_known_pose(
                pose_rotvecs=pr,
                target_vertices=tv,
                target_joints=tj,
                beta_regularizer=0.0,
                requested_keys=['shape_betas'],
            )

        if compile_mode == 'compile':
            fit_fn = torch.compile(fit_fn)
            fit_known_shape_fn = torch.compile(fit_known_shape_fn)
            fit_known_pose_fn = torch.compile(fit_known_pose_fn)

        self._fit_fns = {
            'fit': fit_fn,
            'fit_known_shape': fit_known_shape_fn,
            'fit_known_pose': fit_known_pose_fn,
        }

        suffix = ' GPU' if gpu else ' CPU'
        mode_name = {'compile': 'PT compile', 'script': 'PT script', None: 'PT eager'}[
            compile_mode
        ]
        self.name = mode_name + suffix

    def prepare_fit(self, pose, shape, target_verts, target_joints):
        device = 'cuda' if self.gpu else 'cpu'
        return dict(
            pose_rotvecs=self.torch.from_numpy(pose).to(device),
            shape_betas=self.torch.from_numpy(shape).to(device),
            target_vertices=self.torch.from_numpy(target_verts).to(device),
            target_joints=self.torch.from_numpy(target_joints).to(device),
        )

    def sync(self):
        if self.gpu:
            self.torch.cuda.synchronize()


class TensorFlowFitBackend(FitBackend):
    def __init__(self, model_name, vertex_subset_size, num_iter, ref_model, gpu):
        super().__init__(model_name, vertex_subset_size, num_iter, ref_model)
        import tensorflow as tf
        import smplfitter.tf as smpl_tf

        self.tf = tf
        self.gpu = gpu
        self.device = '/GPU:0' if gpu else '/CPU:0'
        with tf.device(self.device):
            self.model = smpl_tf.BodyModel(
                model_name=model_name, num_betas=10, vertex_subset_size=vertex_subset_size
            )
            self.fitter = smpl_tf.BodyFitter(self.model)
        ni = num_iter

        @tf.function(jit_compile=True)
        def fit_fn(tv, tj):
            return self.fitter.fit(
                target_vertices=tv,
                target_joints=tj,
                num_iter=ni,
                beta_regularizer=1.0,
                requested_keys=['pose_rotvecs'],
            )

        @tf.function(jit_compile=True)
        def fit_known_shape_fn(sb, tv, tj):
            return self.fitter.fit_with_known_shape(
                shape_betas=sb,
                target_vertices=tv,
                target_joints=tj,
                num_iter=ni,
                requested_keys=['pose_rotvecs'],
            )

        @tf.function(jit_compile=True)
        def fit_known_pose_fn(pr, tv, tj):
            return self.fitter.fit_with_known_pose(
                pose_rotvecs=pr,
                target_vertices=tv,
                target_joints=tj,
                beta_regularizer=0.0,
                requested_keys=['shape_betas'],
            )

        self._fit_fns = {
            'fit': fit_fn,
            'fit_known_shape': fit_known_shape_fn,
            'fit_known_pose': fit_known_pose_fn,
        }

        suffix = ' GPU' if gpu else ' CPU'
        self.name = 'TF function' + suffix

    def prepare_fit(self, pose, shape, target_verts, target_joints):
        with self.tf.device(self.device):
            return dict(
                pose_rotvecs=self.tf.constant(pose),
                shape_betas=self.tf.constant(shape),
                target_vertices=self.tf.constant(target_verts),
                target_joints=self.tf.constant(target_joints),
            )

    def run_fit(self, method, prepared):
        with self.tf.device(self.device):
            return super().run_fit(method, prepared)

    def sync(self):
        if self.gpu:
            with self.tf.device(self.device):
                self.tf.constant([0.0]).numpy()


class JAXFitBackend(FitBackend):
    def __init__(self, model_name, vertex_subset_size, num_iter, ref_model, gpu):
        super().__init__(model_name, vertex_subset_size, num_iter, ref_model)
        import jax
        import jax.numpy as jnp
        import smplfitter.jax as smpl_jax

        self.jax = jax
        self.jnp = jnp
        self.gpu = gpu
        self.backend = 'gpu' if gpu else 'cpu'
        self.device = jax.devices(self.backend)[0]
        self.model = smpl_jax.BodyModel(model_name=model_name, num_betas=10, device=self.device)
        self.fitter = smpl_jax.BodyFitter(self.model)
        ni = num_iter

        def fit_fn(tv, tj):
            return self.fitter.fit(
                target_vertices=tv,
                target_joints=tj,
                num_iter=ni,
                beta_regularizer=1.0,
                requested_keys=['pose_rotvecs'],
            )

        def fit_known_shape_fn(sb, tv, tj):
            return self.fitter.fit_with_known_shape(
                shape_betas=sb,
                target_vertices=tv,
                target_joints=tj,
                num_iter=ni,
                requested_keys=['pose_rotvecs'],
            )

        def fit_known_pose_fn(pr, tv, tj):
            return self.fitter.fit_with_known_pose(
                pose_rotvecs=pr,
                target_vertices=tv,
                target_joints=tj,
                beta_regularizer=0.0,
                requested_keys=['shape_betas'],
            )

        self._fit_fns = {
            'fit': jax.jit(fit_fn, backend=self.backend),
            'fit_known_shape': jax.jit(fit_known_shape_fn, backend=self.backend),
            'fit_known_pose': jax.jit(fit_known_pose_fn, backend=self.backend),
        }

        suffix = ' GPU' if gpu else ' CPU'
        self.name = 'JAX jit' + suffix
        self._last_result = None

    def prepare_fit(self, pose, shape, target_verts, target_joints):
        with self.jax.default_device(self.device):
            return dict(
                pose_rotvecs=self.jnp.array(pose),
                shape_betas=self.jnp.array(shape),
                target_vertices=self.jnp.array(target_verts),
                target_joints=self.jnp.array(target_joints),
            )

    def run_fit(self, method, prepared):
        self._last_result = super().run_fit(method, prepared)
        return self._last_result

    def sync(self):
        if self._last_result is not None:
            self.jax.block_until_ready(self._last_result)


if __name__ == '__main__':
    main()
