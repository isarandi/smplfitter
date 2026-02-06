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
    'PT compile CPU': dict(color='C2', linestyle='--', marker='^'),
    'PT compile GPU': dict(color='C2', linestyle='-', marker='^'),
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


if __name__ == '__main__':
    plot_results()
    plot_vs_smplx()
