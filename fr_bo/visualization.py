"""
Visualization and reporting tools for FR-BO optimization.

Provides:
- Convergence plots
- Parameter space visualization (2D projections)
- Risk heatmaps
- Real-time dashboards
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class OptimizationVisualizer:
    """Visualization tools for FR-BO optimization results."""

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        plt.style.use("seaborn-v0_8-darkgrid")

    def plot_convergence_history(
        self,
        trials: List[any],
        save_path: Optional[str] = None,
    ):
        """
        Plot optimization convergence history.

        Args:
            trials: List of TrialRecord objects
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)

        # Extract data
        trial_numbers = [t.trial_number for t in trials]
        objectives = [t.objective_value for t in trials]
        converged = [t.result.converged for t in trials]

        # Running best
        running_best = []
        best_so_far = float("inf")
        for obj, conv in zip(objectives, converged):
            if conv and obj < best_so_far:
                best_so_far = obj
            running_best.append(best_so_far)

        # 1. Objective values over time
        ax = axes[0, 0]
        colors = ["green" if c else "red" for c in converged]
        ax.scatter(trial_numbers, objectives, c=colors, alpha=0.6, s=30)
        ax.plot(trial_numbers, running_best, "b-", linewidth=2, label="Best so far")
        ax.set_xlabel("Trial Number")
        ax.set_ylabel("Objective Value")
        ax.set_title("Optimization Progress")
        ax.legend()
        ax.set_yscale("log")

        # 2. Success rate over time (rolling window)
        ax = axes[0, 1]
        window_size = 20
        success_rates = []
        for i in range(len(trials)):
            start = max(0, i - window_size + 1)
            window = converged[start:i+1]
            success_rates.append(sum(window) / len(window))

        ax.plot(trial_numbers, success_rates, "g-", linewidth=2)
        ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="50% baseline")
        ax.set_xlabel("Trial Number")
        ax.set_ylabel("Success Rate")
        ax.set_title(f"Success Rate (rolling window={window_size})")
        ax.set_ylim([0, 1])
        ax.legend()

        # 3. Phase comparison
        ax = axes[1, 0]
        phases = {}
        for trial in trials:
            if trial.phase not in phases:
                phases[trial.phase] = {"converged": 0, "total": 0}
            phases[trial.phase]["total"] += 1
            if trial.result.converged:
                phases[trial.phase]["converged"] += 1

        phase_names = list(phases.keys())
        success_counts = [phases[p]["converged"] for p in phase_names]
        total_counts = [phases[p]["total"] for p in phase_names]
        success_rates_phase = [s/t if t > 0 else 0 for s, t in zip(success_counts, total_counts)]

        ax.bar(phase_names, success_rates_phase, color=["blue", "green", "orange"][:len(phase_names)])
        ax.set_ylabel("Success Rate")
        ax.set_title("Success Rate by Phase")
        ax.set_ylim([0, 1])

        # 4. Iteration histogram
        ax = axes[1, 1]
        converged_iters = [t.result.iterations for t in trials if t.result.converged]
        failed_iters = [t.result.iterations for t in trials if not t.result.converged]

        if converged_iters:
            ax.hist(converged_iters, bins=20, alpha=0.7, color="green", label="Converged")
        if failed_iters:
            ax.hist(failed_iters, bins=20, alpha=0.7, color="red", label="Failed")

        ax.set_xlabel("Iterations")
        ax.set_ylabel("Count")
        ax.set_title("Iteration Distribution")
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_parameter_space_2d(
        self,
        trials: List[any],
        param_encoder: callable,
        save_path: Optional[str] = None,
        method: str = "pca",
    ):
        """
        Plot 2D projection of parameter space with success/failure regions.

        Args:
            trials: List of TrialRecord objects
            param_encoder: Function to encode parameters
            save_path: Optional path to save figure
            method: Dimensionality reduction method ("pca" or "tsne")
        """
        # Encode all parameters
        X = np.array([param_encoder(t.parameters) for t in trials])
        converged = np.array([t.result.converged for t in trials])
        objectives = np.array([t.objective_value for t in trials])

        # Dimensionality reduction
        if method == "pca":
            reducer = PCA(n_components=2)
            X_2d = reducer.fit_transform(X)
            title_suffix = f"(PCA, explained var: {reducer.explained_variance_ratio_.sum():.2%})"
        elif method == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
            X_2d = reducer.fit_transform(X)
            title_suffix = "(t-SNE)"
        else:
            raise ValueError(f"Unknown method: {method}")

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Success/Failure classification
        ax = axes[0]
        success_mask = converged
        ax.scatter(
            X_2d[success_mask, 0],
            X_2d[success_mask, 1],
            c="green",
            alpha=0.6,
            s=50,
            label="Converged",
            edgecolors="black",
            linewidths=0.5,
        )
        ax.scatter(
            X_2d[~success_mask, 0],
            X_2d[~success_mask, 1],
            c="red",
            alpha=0.6,
            s=50,
            marker="x",
            label="Failed",
            linewidths=2,
        )

        # Mark best trial
        best_idx = np.argmin(objectives[converged]) if np.any(converged) else 0
        converged_indices = np.where(converged)[0]
        if len(converged_indices) > 0:
            best_global_idx = converged_indices[best_idx]
            ax.scatter(
                X_2d[best_global_idx, 0],
                X_2d[best_global_idx, 1],
                c="gold",
                s=300,
                marker="*",
                label="Best",
                edgecolors="black",
                linewidths=2,
                zorder=10,
            )

        ax.set_xlabel(f"Component 1")
        ax.set_ylabel(f"Component 2")
        ax.set_title(f"Parameter Space - Success/Failure {title_suffix}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Objective value heatmap
        ax = axes[1]
        scatter = ax.scatter(
            X_2d[converged, 0],
            X_2d[converged, 1],
            c=objectives[converged],
            cmap="viridis_r",
            alpha=0.7,
            s=80,
            edgecolors="black",
            linewidths=0.5,
        )
        # Failed trials in red
        ax.scatter(
            X_2d[~converged, 0],
            X_2d[~converged, 1],
            c="lightgray",
            alpha=0.3,
            s=50,
            marker="x",
            linewidths=1,
        )

        plt.colorbar(scatter, ax=ax, label="Objective Value")
        ax.set_xlabel(f"Component 1")
        ax.set_ylabel(f"Component 2")
        ax.set_title(f"Parameter Space - Objective Values {title_suffix}")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_failure_probability_heatmap(
        self,
        failure_model: any,
        param_bounds: Tuple[np.ndarray, np.ndarray],
        param_names: Tuple[str, str],
        resolution: int = 50,
        save_path: Optional[str] = None,
    ):
        """
        Plot 2D heatmap of failure probability.

        Args:
            failure_model: Trained failure classifier
            param_bounds: Tuple of (lower_bounds, upper_bounds) for 2D slice
            param_names: Names of the two parameters
            resolution: Grid resolution
            save_path: Optional path to save figure
        """
        import torch

        # Create grid
        x1 = np.linspace(param_bounds[0][0], param_bounds[1][0], resolution)
        x2 = np.linspace(param_bounds[0][1], param_bounds[1][1], resolution)
        X1, X2 = np.meshgrid(x1, x2)

        # Evaluate failure probability at grid points
        grid_points = np.column_stack([X1.ravel(), X2.ravel()])
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

        with torch.no_grad():
            latent_dist = failure_model(grid_tensor)
            if hasattr(failure_model, "likelihood"):
                pred_dist = failure_model.likelihood(latent_dist)
                failure_probs = pred_dist.mean.numpy()
            else:
                failure_probs = torch.sigmoid(latent_dist.mean).numpy()

        failure_probs = failure_probs.reshape(X1.shape)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot heatmap
        im = ax.contourf(X1, X2, failure_probs, levels=20, cmap="RdYlGn_r", alpha=0.8)
        contours = ax.contour(X1, X2, failure_probs, levels=[0.2, 0.5, 0.8], colors="black", linewidths=2)
        ax.clabel(contours, inline=True, fontsize=10)

        plt.colorbar(im, ax=ax, label="Failure Probability")
        ax.set_xlabel(param_names[0])
        ax.set_ylabel(param_names[1])
        ax.set_title("Failure Probability Landscape")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def create_interactive_dashboard(
        self,
        trials: List[any],
        save_path: Optional[str] = "dashboard.html",
    ):
        """
        Create interactive Plotly dashboard.

        Args:
            trials: List of TrialRecord objects
            save_path: Path to save HTML dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Optimization Progress",
                "Success Rate Over Time",
                "Objective Distribution",
                "Iteration vs Time",
            ),
        )

        # Extract data
        trial_numbers = [t.trial_number for t in trials]
        objectives = [t.objective_value for t in trials]
        converged = [t.result.converged for t in trials]
        iterations = [t.result.iterations for t in trials]
        times = [t.result.time_elapsed for t in trials]

        # Running best
        running_best = []
        best_so_far = float("inf")
        for obj, conv in zip(objectives, converged):
            if conv and obj < best_so_far:
                best_so_far = obj
            running_best.append(best_so_far if best_so_far != float("inf") else None)

        # 1. Optimization progress
        colors = ["green" if c else "red" for c in converged]
        fig.add_trace(
            go.Scatter(
                x=trial_numbers,
                y=objectives,
                mode="markers",
                marker=dict(color=colors, size=6),
                name="Trials",
                text=[f"Trial {t}<br>Obj: {o:.4f}" for t, o in zip(trial_numbers, objectives)],
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=trial_numbers,
                y=running_best,
                mode="lines",
                line=dict(color="blue", width=2),
                name="Best",
            ),
            row=1,
            col=1,
        )

        # 2. Success rate
        window_size = 20
        success_rates = []
        for i in range(len(trials)):
            start = max(0, i - window_size + 1)
            window = converged[start:i+1]
            success_rates.append(sum(window) / len(window))

        fig.add_trace(
            go.Scatter(
                x=trial_numbers,
                y=success_rates,
                mode="lines",
                line=dict(color="green", width=2),
                name="Success Rate",
            ),
            row=1,
            col=2,
        )

        # 3. Objective distribution
        converged_objs = [o for o, c in zip(objectives, converged) if c]
        failed_objs = [o for o, c in zip(objectives, converged) if not c]

        if converged_objs:
            fig.add_trace(
                go.Histogram(x=converged_objs, name="Converged", marker_color="green", opacity=0.7),
                row=2,
                col=1,
            )
        if failed_objs:
            fig.add_trace(
                go.Histogram(x=failed_objs, name="Failed", marker_color="red", opacity=0.7),
                row=2,
                col=1,
            )

        # 4. Iterations vs Time
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=times,
                mode="markers",
                marker=dict(color=colors, size=6),
                name="Trials",
                text=[f"Trial {t}" for t in trial_numbers],
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="FR-BO Optimization Dashboard",
        )

        fig.update_xaxes(title_text="Trial Number", row=1, col=1)
        fig.update_yaxes(title_text="Objective (log)", type="log", row=1, col=1)
        fig.update_xaxes(title_text="Trial Number", row=1, col=2)
        fig.update_yaxes(title_text="Success Rate", row=1, col=2)
        fig.update_xaxes(title_text="Objective Value", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_xaxes(title_text="Iterations", row=2, col=2)
        fig.update_yaxes(title_text="Time (s)", row=2, col=2)

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_parameter_importance(
        self,
        importance_scores: np.ndarray,
        param_names: List[str],
        save_path: Optional[str] = None,
    ):
        """
        Plot parameter importance from ARD lengthscales.

        Args:
            importance_scores: Array of importance scores
            param_names: List of parameter names
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Sort by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        sorted_scores = importance_scores[sorted_indices]
        sorted_names = [param_names[i] for i in sorted_indices]

        # Plot
        bars = ax.barh(sorted_names, sorted_scores, color="steelblue")

        # Color top 3 differently
        for i in range(min(3, len(bars))):
            bars[i].set_color("darkgreen")

        ax.set_xlabel("Importance Score")
        ax.set_title("Parameter Importance (from ARD)")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
