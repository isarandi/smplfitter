"""Generate benchmark plots from results.db."""

import sqlite3
import matplotlib.pyplot as plt
from pathlib import Path


def query_throughput(conn, backend, model, mode):
    """Query throughput (items/sec) for a backend, model, and mode."""
    cursor = conn.execute(
        'SELECT batch_size, batch_size / (time_ms / 1000.0) as throughput '
        'FROM results WHERE backend = ? AND model = ? AND mode = ? ORDER BY batch_size',
        (backend, model, mode),
    )
    rows = cursor.fetchall()
    if not rows:
        return [], []
    batch_sizes, throughputs = zip(*rows)
    return list(batch_sizes), list(throughputs)


# Color = framework, linestyle = device (solid=GPU, dashed=CPU)
STYLE = {
    'NumPy': dict(color='C0', linestyle='--', marker='o'),
    'Numba': dict(color='C1', linestyle='--', marker='s'),
    'PT eager CPU': dict(color='C2', linestyle='--', marker='o'),
    'PT eager GPU': dict(color='C2', linestyle='-', marker='o'),
    'PT script CPU': dict(color='C6', linestyle='--', marker='s'),
    'PT script GPU': dict(color='C6', linestyle='-', marker='s'),
    'PT compile CPU': dict(color='C2', linestyle='--', marker='^'),
    'PT compile GPU': dict(color='C2', linestyle='-', marker='^'),
    'PT eager dec CPU': dict(color='C8', linestyle='--', marker='o'),
    'PT eager dec GPU': dict(color='C8', linestyle='-', marker='o'),
    'PT script dec CPU': dict(color='C9', linestyle='--', marker='s'),
    'PT script dec GPU': dict(color='C9', linestyle='-', marker='s'),
    'PT compile dec CPU': dict(color='C7', linestyle='--', marker='^'),
    'PT compile dec GPU': dict(color='C7', linestyle='-', marker='^'),
    'TF function CPU': dict(color='C3', linestyle='--', marker='D'),
    'TF function GPU': dict(color='C3', linestyle='-', marker='D'),
    'JAX jit CPU': dict(color='C4', linestyle='--', marker='v'),
    'JAX jit GPU': dict(color='C4', linestyle='-', marker='v'),
    'smplx lib CPU': dict(color='C5', linestyle='--', marker='P'),
    'smplx lib GPU': dict(color='C5', linestyle='-', marker='P'),
    'smplx compile CPU': dict(color='C5', linestyle='--', marker='X'),
    'smplx compile GPU': dict(color='C5', linestyle='-', marker='X'),
}


def plot_results():
    out_dir = Path(__file__).parent
    conn = sqlite3.connect(out_dir / 'results.db')

    # Get all models and backends in DB
    models = [r[0] for r in conn.execute('SELECT DISTINCT model FROM results')]
    backends = [r[0] for r in conn.execute('SELECT DISTINCT backend FROM results')]
    backends = [b for b in STYLE if b in backends]
    # Exclude all smplx backends
    backends = [b for b in backends if 'smplx' not in b]

    for model in models:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, (mode, title) in zip(
            axes, [('with_vertices', 'With Vertices'), ('joints_only', 'Joints Only')]
        ):
            for backend in backends:
                batch_sizes, throughputs = query_throughput(conn, backend, model, mode)
                if batch_sizes:
                    ax.plot(
                        batch_sizes,
                        throughputs,
                        **STYLE[backend],
                        label=backend,
                        linewidth=2,
                        markersize=8,
                    )

            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Throughput (items/sec)')
            ax.set_title(f'{title} ({model.upper()})')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f'benchmark_{model}.png' if len(models) > 1 else 'benchmark.png'
        plt.savefig(out_dir / filename, dpi=150)
        print(f'Saved {out_dir / filename}')

    conn.close()


def plot_vs_smplx():
    """Plot only smplfitter PT vs official smplx library."""
    out_dir = Path(__file__).parent
    conn = sqlite3.connect(out_dir / 'results.db')

    # Get all models in DB
    models = [r[0] for r in conn.execute('SELECT DISTINCT model FROM results')]
    backends_to_plot = [
        'PT compile CPU',
        'PT compile GPU',
        'smplx compile CPU',
        'smplx compile GPU',
    ]

    for model in models:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        for ax, (mode, title) in zip(
            axes, [('with_vertices', 'With Vertices'), ('joints_only', 'Joints Only')]
        ):
            for backend in backends_to_plot:
                batch_sizes, throughputs = query_throughput(conn, backend, model, mode)
                if batch_sizes:
                    ax.plot(
                        batch_sizes,
                        throughputs,
                        **STYLE[backend],
                        label=backend,
                        linewidth=2,
                        markersize=8,
                    )

            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Throughput (items/sec)')
            ax.set_title(f'{title} ({model.upper()})')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = (
            f'benchmark_vs_smplx_{model}.png' if len(models) > 1 else 'benchmark_vs_smplx.png'
        )
        plt.savefig(out_dir / filename, dpi=150)
        print(f'Saved {out_dir / filename}')

    conn.close()


def query_fit_metric(conn, backend, model, method, vertex_count, metric):
    """Query a fit-benchmark column. metric is 'throughput' or 'time'."""
    if metric == 'throughput':
        select = 'batch_size, batch_size / (time_ms / 1000.0)'
    elif metric == 'time':
        select = 'batch_size, time_ms'
    else:
        raise ValueError(metric)
    cursor = conn.execute(
        f'SELECT {select} FROM fit_results WHERE backend = ? AND model = ? AND method = ? '
        f'AND vertex_count = ? ORDER BY batch_size',
        (backend, model, method, vertex_count),
    )
    rows = cursor.fetchall()
    if not rows:
        return [], []
    batch_sizes, values = zip(*rows)
    return list(batch_sizes), list(values)


def _plot_fit(metric, ylabel, filename_stem, only_backends=None):
    out_dir = Path(__file__).parent
    conn = sqlite3.connect(out_dir / 'results.db')

    models = [r[0] for r in conn.execute('SELECT DISTINCT model FROM fit_results')]
    vertex_counts = [r[0] for r in conn.execute('SELECT DISTINCT vertex_count FROM fit_results')]
    methods = ['fit', 'fit_known_shape', 'fit_known_pose']
    backends = [r[0] for r in conn.execute('SELECT DISTINCT backend FROM fit_results')]
    backends = [b for b in STYLE if b in backends]
    if only_backends is not None:
        backends = [b for b in backends if b in only_backends]

    for model in models:
        nrows = len(vertex_counts)
        ncols = len(methods)
        _, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

        for i, vcount in enumerate(vertex_counts):
            for j, method in enumerate(methods):
                ax = axes[i][j]
                for backend in backends:
                    batch_sizes, values = query_fit_metric(
                        conn, backend, model, method, vcount, metric
                    )
                    if batch_sizes:
                        ax.plot(
                            batch_sizes,
                            values,
                            **STYLE[backend],
                            label=backend,
                            linewidth=2,
                            markersize=8,
                        )
                ax.set_xlabel('Batch Size')
                ax.set_ylabel(ylabel)
                ax.set_title(f'{method} — verts={vcount} ({model.upper()})')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f'{filename_stem}_{model}.png' if len(models) > 1 else f'{filename_stem}.png'
        plt.savefig(out_dir / filename, dpi=150)
        print(f'Saved {out_dir / filename}')

    conn.close()


def plot_fit_results():
    _plot_fit('throughput', 'Throughput (items/sec)', 'benchmark_fit')


def plot_fit_time():
    _plot_fit('time', 'Batch time (ms)', 'benchmark_fit_time')


def plot_fit_compile_vs_numba():
    only = ['PT compile CPU', 'Numba']
    _plot_fit('throughput', 'Throughput (items/sec)', 'benchmark_fit_compile_vs_numba', only)
    _plot_fit('time', 'Batch time (ms)', 'benchmark_fit_compile_vs_numba_time', only)


def plot_fit_decoupled_vs_regular():
    only = [
        'PT eager CPU',
        'PT eager dec CPU',
        'PT eager GPU',
        'PT eager dec GPU',
        'PT script CPU',
        'PT script dec CPU',
        'PT script GPU',
        'PT script dec GPU',
        'PT compile CPU',
        'PT compile dec CPU',
        'PT compile GPU',
        'PT compile dec GPU',
    ]
    _plot_fit('throughput', 'Throughput (items/sec)', 'benchmark_fit_decoupled_vs_regular', only)
    _plot_fit('time', 'Batch time (ms)', 'benchmark_fit_decoupled_vs_regular_time', only)


if __name__ == '__main__':
    plot_results()
    plot_vs_smplx()
    plot_fit_results()
    plot_fit_time()
    plot_fit_compile_vs_numba()
    plot_fit_decoupled_vs_regular()
