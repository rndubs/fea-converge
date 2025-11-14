"""
Visualization tools for GP Classification optimization.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap

from .data import TrialDatabase
from .models import DualModel


class OptimizationVisualizer:
    """
    Visualization suite for GP Classification optimization results.
    """

    def __init__(
        self,
        database: TrialDatabase,
        dual_model: DualModel,
        parameter_names: List[str],
    ):
        """
        Initialize visualizer.

        Args:
            database: Trial database
            dual_model: Trained dual model
            parameter_names: Ordered list of parameter names
        """
        self.database = database
        self.dual_model = dual_model
        self.parameter_names = parameter_names

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.dpi"] = 100

    def plot_convergence_landscape_2d(
        self,
        param1_name: str,
        param2_name: str,
        resolution: int = 50,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot 2D convergence probability heatmap.

        Args:
            param1_name: First parameter name (x-axis)
            param2_name: Second parameter name (y-axis)
            resolution: Grid resolution
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        # Get parameter bounds
        bounds1 = self.database.parameter_bounds[param1_name]
        bounds2 = self.database.parameter_bounds[param2_name]

        # Create grid
        x = np.linspace(bounds1[0], bounds1[1], resolution)
        y = np.linspace(bounds2[0], bounds2[1], resolution)
        X, Y = np.meshgrid(x, y)

        # Get predictions
        idx1 = self.parameter_names.index(param1_name)
        idx2 = self.parameter_names.index(param2_name)

        grid_points = []
        for i in range(resolution):
            for j in range(resolution):
                params = np.zeros(len(self.parameter_names))
                for k, name in enumerate(self.parameter_names):
                    lower, upper = self.database.parameter_bounds[name]
                    params[k] = (lower + upper) / 2.0
                params[idx1] = X[i, j]
                params[idx2] = Y[i, j]
                grid_points.append(params)

        grid_tensor = torch.tensor(grid_points, dtype=torch.float64)
        probs, _ = self.dual_model.predict_convergence(grid_tensor)
        P = probs.reshape(resolution, resolution).numpy()

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Custom colormap: red -> yellow -> green
        colors = ["#d73027", "#fee08b", "#1a9850"]
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list("convergence", colors, N=n_bins)

        # Plot heatmap
        im = ax.contourf(X, Y, P, levels=20, cmap=cmap, alpha=0.8)

        # Add contour lines
        contours = ax.contour(X, Y, P, levels=[0.5, 0.7, 0.9], colors="black", linewidths=1.5)
        ax.clabel(contours, inline=True, fontsize=10)

        # Plot trial points
        df = self.database.get_all_trials()
        if not df.empty:
            converged = df[df["converged"] == True]
            failed = df[df["converged"] == False]

            if not converged.empty:
                ax.scatter(
                    converged[param1_name],
                    converged[param2_name],
                    c="blue",
                    marker="o",
                    s=50,
                    alpha=0.6,
                    edgecolors="black",
                    label="Converged",
                )

            if not failed.empty:
                ax.scatter(
                    failed[param1_name],
                    failed[param2_name],
                    c="red",
                    marker="x",
                    s=50,
                    alpha=0.6,
                    label="Failed",
                )

        # Mark best trial
        best_trial = self.database.get_best_trial()
        if best_trial:
            ax.scatter(
                best_trial.parameters[param1_name],
                best_trial.parameters[param2_name],
                c="gold",
                marker="*",
                s=300,
                edgecolors="black",
                linewidths=2,
                label="Best",
                zorder=10,
            )

        # Labels and title
        ax.set_xlabel(param1_name.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel(param2_name.replace("_", " ").title(), fontsize=12)
        ax.set_title("Convergence Probability Landscape", fontsize=14, fontweight="bold")

        # Colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("P(Converged)", fontsize=12)

        # Legend
        ax.legend(loc="best", fontsize=10)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_uncertainty_map(
        self,
        param1_name: str,
        param2_name: str,
        resolution: int = 50,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot classification uncertainty map.

        Args:
            param1_name: First parameter name (x-axis)
            param2_name: Second parameter name (y-axis)
            resolution: Grid resolution
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        # Similar grid generation as convergence landscape
        bounds1 = self.database.parameter_bounds[param1_name]
        bounds2 = self.database.parameter_bounds[param2_name]

        x = np.linspace(bounds1[0], bounds1[1], resolution)
        y = np.linspace(bounds2[0], bounds2[1], resolution)
        X, Y = np.meshgrid(x, y)

        idx1 = self.parameter_names.index(param1_name)
        idx2 = self.parameter_names.index(param2_name)

        grid_points = []
        for i in range(resolution):
            for j in range(resolution):
                params = np.zeros(len(self.parameter_names))
                for k, name in enumerate(self.parameter_names):
                    lower, upper = self.database.parameter_bounds[name]
                    params[k] = (lower + upper) / 2.0
                params[idx1] = X[i, j]
                params[idx2] = Y[i, j]
                grid_points.append(params)

        grid_tensor = torch.tensor(grid_points, dtype=torch.float64)
        _, stds = self.dual_model.predict_convergence(grid_tensor)
        U = stds.reshape(resolution, resolution).numpy()

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot uncertainty
        im = ax.contourf(X, Y, U, levels=20, cmap="YlOrRd", alpha=0.8)

        # Add contour lines
        contours = ax.contour(X, Y, U, levels=5, colors="black", linewidths=1, alpha=0.5)
        ax.clabel(contours, inline=True, fontsize=9)

        # Plot trial points
        df = self.database.get_all_trials()
        if not df.empty:
            ax.scatter(
                df[param1_name],
                df[param2_name],
                c="blue",
                marker="o",
                s=30,
                alpha=0.4,
                edgecolors="black",
                label="Trials",
            )

        # Labels
        ax.set_xlabel(param1_name.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel(param2_name.replace("_", " ").title(), fontsize=12)
        ax.set_title("Prediction Uncertainty Map", fontsize=14, fontweight="bold")

        # Colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Std. Deviation (σ)", fontsize=12)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_optimization_history(self, save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot optimization history: convergence rate and best objective over time.

        Args:
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        df = self.database.get_all_trials()

        if df.empty:
            raise ValueError("No trials to plot")

        # Calculate cumulative statistics
        iterations = range(1, len(df) + 1)
        cumulative_converged = df["converged"].cumsum()
        convergence_rate = cumulative_converged / np.arange(1, len(df) + 1)

        # Best objective over time
        best_objectives = []
        current_best = float("inf")

        for _, row in df.iterrows():
            if row["converged"] and row["objective_value"] is not None:
                current_best = min(current_best, row["objective_value"])
            best_objectives.append(current_best if current_best != float("inf") else np.nan)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot 1: Convergence rate
        ax1.plot(iterations, convergence_rate, linewidth=2, color="#1f77b4")
        ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% baseline")
        ax1.fill_between(iterations, 0, convergence_rate, alpha=0.3, color="#1f77b4")
        ax1.set_xlabel("Iteration", fontsize=12)
        ax1.set_ylabel("Convergence Rate", fontsize=12)
        ax1.set_title("Cumulative Convergence Rate", fontsize=13, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])

        # Plot 2: Best objective
        ax2.plot(iterations, best_objectives, linewidth=2, color="#2ca02c", marker="o", markersize=3)
        ax2.set_xlabel("Iteration", fontsize=12)
        ax2.set_ylabel("Best Objective Value", fontsize=12)
        ax2.set_title("Best Objective Over Time", fontsize=13, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_parameter_importance(self, save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot parameter importance based on GP lengthscales (ARD).

        Args:
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        # Extract lengthscales from convergence model
        lengthscales = (
            self.dual_model.convergence_model.covar_module.base_kernel.lengthscale.detach()
            .squeeze()
            .numpy()
        )

        # Inverse lengthscale = importance (shorter lengthscale = more sensitive parameter)
        importance = 1.0 / lengthscales
        importance = importance / importance.sum()  # Normalize

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Bar plot
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(self.parameter_names)))
        bars = ax.barh(self.parameter_names, importance, color=colors, edgecolor="black")

        # Labels
        ax.set_xlabel("Relative Importance", fontsize=12)
        ax.set_title("Parameter Importance (ARD Lengthscales)", fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}",
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_calibration(self, n_bins: int = 10, save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot calibration curve: predicted probability vs actual convergence rate.

        Args:
            n_bins: Number of probability bins
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        # Get all predictions
        X_all, y_converged, _ = self.database.get_training_data(converged_only=False)
        probs, _ = self.dual_model.predict_convergence(X_all)

        probs = probs.numpy()
        y_true = y_converged.squeeze().numpy()

        # Bin predictions
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        predicted_probs = []
        actual_rates = []
        counts = []

        for i in range(n_bins):
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
            if mask.sum() > 0:
                predicted_probs.append(bin_centers[i])
                actual_rates.append(y_true[mask].mean())
                counts.append(mask.sum())

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", linewidth=2, alpha=0.5, label="Perfect Calibration")

        # Calibration curve
        if predicted_probs:
            ax.plot(
                predicted_probs,
                actual_rates,
                marker="o",
                markersize=8,
                linewidth=2,
                color="#1f77b4",
                label="Model Calibration",
            )

            # Add count annotations
            for x, y, count in zip(predicted_probs, actual_rates, counts):
                ax.annotate(
                    f"n={count}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=9,
                    alpha=0.7,
                )

        # Labels
        ax.set_xlabel("Predicted Probability", fontsize=12)
        ax.set_ylabel("Actual Convergence Rate", fontsize=12)
        ax.set_title("Model Calibration", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_summary_dashboard(self, save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create comprehensive summary dashboard.

        Args:
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Statistics text
        ax_stats = fig.add_subplot(gs[0, :])
        ax_stats.axis("off")

        stats = self.database.get_statistics()
        stats_text = f"""
        Optimization Summary
        {'=' * 80}
        Total Trials: {stats['total_trials']}
        Converged: {stats['converged_trials']} ({stats['convergence_rate']:.1%})
        Failed: {stats['failed_trials']}
        Best Objective: {stats.get('best_objective', 'N/A'):.6f if 'best_objective' in stats else 'N/A'}
        Mean Objective: {stats.get('mean_objective', 'N/A'):.6f if 'mean_objective' in stats else 'N/A'}
        """

        ax_stats.text(
            0.5, 0.5, stats_text, ha="center", va="center", fontsize=12, family="monospace"
        )

        # Convergence history
        ax_conv = fig.add_subplot(gs[1, :2])
        df = self.database.get_all_trials()
        iterations = range(1, len(df) + 1)
        convergence_rate = df["converged"].cumsum() / np.arange(1, len(df) + 1)
        ax_conv.plot(iterations, convergence_rate, linewidth=2)
        ax_conv.set_xlabel("Iteration")
        ax_conv.set_ylabel("Convergence Rate")
        ax_conv.set_title("Convergence Rate Over Time")
        ax_conv.grid(True, alpha=0.3)

        # Parameter importance
        ax_imp = fig.add_subplot(gs[1, 2])
        lengthscales = (
            self.dual_model.convergence_model.covar_module.base_kernel.lengthscale.detach()
            .squeeze()
            .numpy()
        )
        importance = 1.0 / lengthscales
        importance = importance / importance.sum()
        ax_imp.barh(self.parameter_names, importance)
        ax_imp.set_xlabel("Importance")
        ax_imp.set_title("Parameter Importance")

        # Trial outcomes pie chart
        ax_pie = fig.add_subplot(gs[2, 0])
        converged_count = stats["converged_trials"]
        failed_count = stats["failed_trials"]
        ax_pie.pie(
            [converged_count, failed_count],
            labels=["Converged", "Failed"],
            autopct="%1.1f%%",
            colors=["#2ca02c", "#d62728"],
        )
        ax_pie.set_title("Trial Outcomes")

        # Objective distribution (if available)
        ax_obj = fig.add_subplot(gs[2, 1:])
        if "best_objective" in stats:
            objectives = [
                t.objective_value
                for t in self.database.get_successful_trials()
                if t.objective_value is not None
            ]
            if objectives:
                ax_obj.hist(objectives, bins=20, edgecolor="black", alpha=0.7)
                ax_obj.axvline(
                    stats["best_objective"], color="red", linestyle="--", label="Best", linewidth=2
                )
                ax_obj.set_xlabel("Objective Value")
                ax_obj.set_ylabel("Frequency")
                ax_obj.set_title("Objective Distribution")
                ax_obj.legend()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig
