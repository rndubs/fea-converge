"""
Visualization utilities for CONFIG optimizer results.

Provides functions for plotting optimization progress, violation trajectories,
and parameter space exploration.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Any
from pathlib import Path


def plot_optimization_progress(
    results: Dict[str, Any],
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot optimization progress over iterations.

    Args:
        results: Results dictionary from CONFIGController.optimize()
        save_path: Optional path to save figure
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    y_observed = results['y_observed']
    n_feasible = results['n_feasible']
    iterations = np.arange(1, len(y_observed) + 1)

    # Determine feasible points
    feasible_mask = np.zeros(len(y_observed), dtype=bool)
    for i in range(len(y_observed)):
        all_feasible = all(
            results['constraint_values'][name][i] <= 0
            for name in results['constraint_values'].keys()
        )
        feasible_mask[i] = all_feasible

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Objective values
    ax1.scatter(
        iterations[~feasible_mask],
        y_observed[~feasible_mask],
        c='red',
        alpha=0.5,
        label='Infeasible',
        marker='x'
    )
    ax1.scatter(
        iterations[feasible_mask],
        y_observed[feasible_mask],
        c='green',
        alpha=0.7,
        label='Feasible',
        marker='o'
    )

    # Plot best feasible trajectory
    if results['best_x'] is not None:
        best_trajectory = []
        current_best = float('inf')
        for i in range(len(y_observed)):
            if feasible_mask[i] and y_observed[i] < current_best:
                current_best = y_observed[i]
            best_trajectory.append(current_best if current_best != float('inf') else np.nan)

        ax1.plot(
            iterations,
            best_trajectory,
            'b--',
            linewidth=2,
            label='Best Feasible'
        )

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Objective Value')
    ax1.set_title(f'CONFIG Optimization Progress ({n_feasible} feasible points)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Constraint violations
    colors = plt.cm.tab10(np.linspace(0, 1, len(results['constraint_values'])))
    for idx, (name, values) in enumerate(results['constraint_values'].items()):
        ax2.plot(
            iterations,
            values,
            label=name,
            color=colors[idx],
            alpha=0.7
        )

    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, label='Feasibility boundary')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Constraint Value')
    ax2.set_title('Constraint Trajectories (negative = satisfied)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_parameter_evolution(
    results: Dict[str, Any],
    param_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot evolution of parameters over iterations.

    Args:
        results: Results dictionary from CONFIGController.optimize()
        param_names: Optional list of parameter names
        save_path: Optional path to save figure
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    X_observed = results['X_observed']
    n_params = X_observed.shape[1]
    iterations = np.arange(1, len(X_observed) + 1)

    if param_names is None:
        param_names = [f'x_{i}' for i in range(n_params)]

    # Determine feasible points
    feasible_mask = np.zeros(len(X_observed), dtype=bool)
    for i in range(len(X_observed)):
        all_feasible = all(
            results['constraint_values'][name][i] <= 0
            for name in results['constraint_values'].keys()
        )
        feasible_mask[i] = all_feasible

    fig, axes = plt.subplots(n_params, 1, figsize=(12, 3 * n_params))
    if n_params == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        # Plot infeasible points
        ax.scatter(
            iterations[~feasible_mask],
            X_observed[~feasible_mask, i],
            c='red',
            alpha=0.5,
            label='Infeasible',
            marker='x'
        )
        # Plot feasible points
        ax.scatter(
            iterations[feasible_mask],
            X_observed[feasible_mask, i],
            c='green',
            alpha=0.7,
            label='Feasible',
            marker='o'
        )

        # Mark best point
        if results['best_x'] is not None:
            ax.axhline(
                y=results['best_x'][i],
                color='blue',
                linestyle='--',
                linewidth=2,
                label='Best Feasible'
            )

        ax.set_xlabel('Iteration')
        ax.set_ylabel(param_names[i])
        ax.set_title(f'Parameter Evolution: {param_names[i]}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_2d_landscape(
    results: Dict[str, Any],
    param_idx: tuple = (0, 1),
    param_names: Optional[tuple] = None,
    resolution: int = 50,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot 2D landscape of observed points (for 2D problems or slices).

    Args:
        results: Results dictionary from CONFIGController.optimize()
        param_idx: Indices of parameters to plot (default: first two)
        param_names: Optional tuple of parameter names
        resolution: Grid resolution for background (not used for scatter)
        save_path: Optional path to save figure
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    X_observed = results['X_observed']
    y_observed = results['y_observed']

    if param_names is None:
        param_names = (f'x_{param_idx[0]}', f'x_{param_idx[1]}')

    # Determine feasible points
    feasible_mask = np.zeros(len(X_observed), dtype=bool)
    for i in range(len(X_observed)):
        all_feasible = all(
            results['constraint_values'][name][i] <= 0
            for name in results['constraint_values'].keys()
        )
        feasible_mask[i] = all_feasible

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot infeasible points
    infeasible = ax.scatter(
        X_observed[~feasible_mask, param_idx[0]],
        X_observed[~feasible_mask, param_idx[1]],
        c=y_observed[~feasible_mask],
        cmap='Reds',
        alpha=0.6,
        s=100,
        marker='x',
        label='Infeasible'
    )

    # Plot feasible points
    feasible = ax.scatter(
        X_observed[feasible_mask, param_idx[0]],
        X_observed[feasible_mask, param_idx[1]],
        c=y_observed[feasible_mask],
        cmap='Greens',
        alpha=0.7,
        s=100,
        marker='o',
        edgecolors='black',
        linewidths=1,
        label='Feasible'
    )

    # Mark best point
    if results['best_x'] is not None:
        ax.scatter(
            results['best_x'][param_idx[0]],
            results['best_x'][param_idx[1]],
            c='blue',
            s=300,
            marker='*',
            edgecolors='black',
            linewidths=2,
            label='Best Feasible',
            zorder=5
        )

    ax.set_xlabel(param_names[0])
    ax.set_ylabel(param_names[1])
    ax.set_title('CONFIG Optimization Landscape')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(feasible, ax=ax)
    cbar.set_label('Objective Value')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def create_optimization_report(
    results: Dict[str, Any],
    param_names: Optional[List[str]] = None,
    output_dir: str = "./config_report",
    show: bool = False
) -> Path:
    """
    Create a complete optimization report with all visualizations.

    Args:
        results: Results dictionary from CONFIGController.optimize()
        param_names: Optional list of parameter names
        output_dir: Directory to save report figures
        show: Whether to display plots

    Returns:
        Path to output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Plot 1: Optimization progress
    plot_optimization_progress(
        results,
        save_path=str(output_path / "optimization_progress.png"),
        show=show
    )

    # Plot 2: Parameter evolution
    plot_parameter_evolution(
        results,
        param_names=param_names,
        save_path=str(output_path / "parameter_evolution.png"),
        show=show
    )

    # Plot 3: 2D landscape (if applicable)
    if results['X_observed'].shape[1] >= 2:
        plot_2d_landscape(
            results,
            param_names=tuple(param_names[:2]) if param_names else None,
            save_path=str(output_path / "2d_landscape.png"),
            show=show
        )

    # Plot 4: Violation trajectory (from monitor)
    if 'violation_statistics' in results:
        # This would use the violation monitor's plot method
        pass

    print(f"Report saved to: {output_path}")
    return output_path
