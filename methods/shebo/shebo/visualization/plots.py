"""Visualization tools for SHEBO results."""

from typing import Dict, List, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure


def plot_convergence_history(
    convergence_history: List[bool],
    window_size: int = 10,
    figsize: tuple = (12, 5)
) -> Figure:
    """Plot convergence rate over optimization.

    Args:
        convergence_history: List of convergence booleans
        window_size: Window for moving average
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    iterations = np.arange(len(convergence_history))
    convergence_array = np.array(convergence_history, dtype=float)

    # Plot raw convergence
    ax1.scatter(
        iterations,
        convergence_array,
        alpha=0.5,
        c=['red' if not c else 'green' for c in convergence_history],
        s=20
    )
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Converged (1) / Failed (0)')
    ax1.set_title('Convergence Status per Iteration')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)

    # Plot moving average
    if len(convergence_history) >= window_size:
        moving_avg = np.convolve(
            convergence_array,
            np.ones(window_size) / window_size,
            mode='valid'
        )
        ax2.plot(
            np.arange(window_size - 1, len(convergence_history)),
            moving_avg,
            linewidth=2,
            color='blue'
        )
        ax2.fill_between(
            np.arange(window_size - 1, len(convergence_history)),
            moving_avg,
            alpha=0.3
        )

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel(f'Convergence Rate (window={window_size})')
    ax2.set_title('Moving Average Convergence Rate')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    return fig


def plot_performance_history(
    performance_history: List[float],
    convergence_history: List[bool],
    figsize: tuple = (12, 5)
) -> Figure:
    """Plot performance metrics over time.

    Args:
        performance_history: List of performance values
        convergence_history: List of convergence booleans
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    iterations = np.arange(len(performance_history))
    perf_array = np.array(performance_history)
    conv_array = np.array(convergence_history)

    # Separate converged and failed
    converged_mask = conv_array == True
    failed_mask = ~converged_mask

    # Plot all performance values
    if converged_mask.any():
        ax1.scatter(
            iterations[converged_mask],
            perf_array[converged_mask],
            alpha=0.6,
            c='green',
            label='Converged',
            s=30
        )

    if failed_mask.any():
        ax1.scatter(
            iterations[failed_mask],
            perf_array[failed_mask],
            alpha=0.6,
            c='red',
            label='Failed',
            s=30,
            marker='x'
        )

    # Plot best performance line
    best_perf = []
    current_best = float('inf')
    for i, (perf, conv) in enumerate(zip(performance_history, convergence_history)):
        if conv and perf < current_best:
            current_best = perf
        best_perf.append(current_best if current_best != float('inf') else None)

    valid_best = [(i, p) for i, p in enumerate(best_perf) if p is not None]
    if valid_best:
        best_iters, best_vals = zip(*valid_best)
        ax1.plot(best_iters, best_vals, 'b--', linewidth=2, label='Best', alpha=0.8)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Performance (iterations)')
    ax1.set_title('Performance Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Histogram of converged performances
    if converged_mask.any():
        ax2.hist(
            perf_array[converged_mask],
            bins=20,
            alpha=0.7,
            color='green',
            edgecolor='black'
        )
        ax2.axvline(
            perf_array[converged_mask].min(),
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Best: {perf_array[converged_mask].min():.1f}'
        )
        ax2.set_xlabel('Performance (iterations)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Converged Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_constraint_timeline(
    discovered_constraints: Dict[str, Dict[str, Any]],
    total_iterations: int,
    figsize: tuple = (12, 6)
) -> Figure:
    """Plot constraint discovery timeline.

    Args:
        discovered_constraints: Dictionary of discovered constraints
        total_iterations: Total number of iterations
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if not discovered_constraints:
        ax.text(
            0.5, 0.5,
            'No constraints discovered',
            ha='center',
            va='center',
            fontsize=14
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig

    # Sort by first seen
    sorted_constraints = sorted(
        discovered_constraints.items(),
        key=lambda x: x[1]['first_seen']
    )

    # Color map by severity
    severity_colors = {
        'low': 'yellow',
        'medium': 'orange',
        'high': 'red'
    }

    y_pos = 0
    for con_name, info in sorted_constraints:
        first_seen = info['first_seen']
        frequency = info['frequency']
        severity = info['severity']
        color = severity_colors.get(severity, 'gray')

        # Plot horizontal bar
        ax.barh(
            y_pos,
            total_iterations - first_seen,
            left=first_seen,
            height=0.8,
            color=color,
            alpha=0.7,
            edgecolor='black',
            linewidth=1
        )

        # Annotate
        label = f"{con_name}\n(freq: {frequency}, sev: {severity})"
        ax.text(
            first_seen + 5,
            y_pos,
            label,
            va='center',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        y_pos += 1

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_yticks([])
    ax.set_title('Constraint Discovery Timeline', fontsize=14, fontweight='bold')
    ax.set_xlim(0, total_iterations)
    ax.grid(True, axis='x', alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='yellow', alpha=0.7, label='Low severity'),
        Patch(facecolor='orange', alpha=0.7, label='Medium severity'),
        Patch(facecolor='red', alpha=0.7, label='High severity')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    return fig


def plot_parameter_space_2d(
    all_params: List[np.ndarray],
    convergence_history: List[bool],
    param_indices: tuple = (0, 1),
    param_names: Optional[List[str]] = None,
    figsize: tuple = (10, 8)
) -> Figure:
    """Plot 2D parameter space with convergence outcomes.

    Args:
        all_params: List of parameter vectors
        convergence_history: List of convergence booleans
        param_indices: Tuple of (x_param_idx, y_param_idx)
        param_names: Optional parameter names
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    params_array = np.array(all_params)
    conv_array = np.array(convergence_history)

    x_idx, y_idx = param_indices
    x_vals = params_array[:, x_idx]
    y_vals = params_array[:, y_idx]

    # Plot converged and failed with different markers
    converged_mask = conv_array == True
    failed_mask = ~converged_mask

    if converged_mask.any():
        ax.scatter(
            x_vals[converged_mask],
            y_vals[converged_mask],
            c='green',
            marker='o',
            s=100,
            alpha=0.6,
            label='Converged',
            edgecolors='black',
            linewidth=0.5
        )

    if failed_mask.any():
        ax.scatter(
            x_vals[failed_mask],
            y_vals[failed_mask],
            c='red',
            marker='x',
            s=100,
            alpha=0.6,
            label='Failed',
            linewidth=2
        )

    # Labels
    if param_names and len(param_names) > max(x_idx, y_idx):
        ax.set_xlabel(param_names[x_idx], fontsize=12)
        ax.set_ylabel(param_names[y_idx], fontsize=12)
    else:
        ax.set_xlabel(f'Parameter {x_idx}', fontsize=12)
        ax.set_ylabel(f'Parameter {y_idx}', fontsize=12)

    ax.set_title('Parameter Space Exploration', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_ensemble_uncertainty(
    X_test: np.ndarray,
    predictions: Dict[str, np.ndarray],
    param_idx: int = 0,
    figsize: tuple = (12, 5)
) -> Figure:
    """Plot ensemble predictions with uncertainty.

    Args:
        X_test: Test parameter array
        predictions: Dictionary with 'mean', 'epistemic_uncertainty', etc.
        param_idx: Parameter index to plot against
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    x_vals = X_test[:, param_idx]
    mean = predictions['mean']
    epistemic = np.sqrt(predictions['epistemic_uncertainty'])
    aleatoric = np.sqrt(predictions['aleatoric_uncertainty'])

    # Sort by x for plotting
    sort_idx = np.argsort(x_vals)
    x_sorted = x_vals[sort_idx]
    mean_sorted = mean[sort_idx]
    epistemic_sorted = epistemic[sort_idx]
    aleatoric_sorted = aleatoric[sort_idx]

    # Plot mean prediction with uncertainty bands
    ax1.plot(x_sorted, mean_sorted, 'b-', linewidth=2, label='Mean prediction')
    ax1.fill_between(
        x_sorted,
        mean_sorted - epistemic_sorted,
        mean_sorted + epistemic_sorted,
        alpha=0.3,
        label='Epistemic uncertainty'
    )
    ax1.axhline(y=0.5, color='r', linestyle='--', label='Decision boundary')
    ax1.set_xlabel(f'Parameter {param_idx}')
    ax1.set_ylabel('Convergence Probability')
    ax1.set_title('Ensemble Prediction')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)

    # Plot uncertainty breakdown
    ax2.plot(x_sorted, epistemic_sorted, 'b-', linewidth=2, label='Epistemic')
    ax2.plot(x_sorted, aleatoric_sorted, 'r-', linewidth=2, label='Aleatoric')
    ax2.plot(
        x_sorted,
        epistemic_sorted + aleatoric_sorted,
        'k--',
        linewidth=2,
        label='Total'
    )
    ax2.set_xlabel(f'Parameter {param_idx}')
    ax2.set_ylabel('Uncertainty (std)')
    ax2.set_title('Uncertainty Decomposition')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def save_all_plots(
    result: Any,
    output_dir: str = 'plots'
) -> None:
    """Save all plots for SHEBO result.

    Args:
        result: SHEBOResult object
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Convergence history
    fig = plot_convergence_history(result.convergence_history)
    fig.savefig(os.path.join(output_dir, 'convergence_history.png'), dpi=150)
    plt.close(fig)

    # Performance history
    fig = plot_performance_history(
        result.performance_history,
        result.convergence_history
    )
    fig.savefig(os.path.join(output_dir, 'performance_history.png'), dpi=150)
    plt.close(fig)

    # Constraint timeline
    if result.discovered_constraints:
        fig = plot_constraint_timeline(
            result.discovered_constraints.get('constraints', {}),
            result.iterations
        )
        fig.savefig(os.path.join(output_dir, 'constraint_timeline.png'), dpi=150)
        plt.close(fig)

    # Parameter space (first two params)
    if len(result.all_params) > 0 and len(result.all_params[0]) >= 2:
        fig = plot_parameter_space_2d(
            result.all_params,
            result.convergence_history
        )
        fig.savefig(os.path.join(output_dir, 'parameter_space.png'), dpi=150)
        plt.close(fig)

    print(f"Plots saved to {output_dir}/")
