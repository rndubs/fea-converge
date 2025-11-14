# Contributing to FR-BO Development

This guide provides instructions for setting up the development environment, running tests, and contributing to the Failure-Robust Bayesian Optimization (FR-BO) implementation.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)
- [Using the FR-BO Optimizer](#using-the-fr-bo-optimizer)
- [Creating Synthetic Datasets](#creating-synthetic-datasets)
- [Extending the Implementation](#extending-the-implementation)
- [Code Quality](#code-quality)

## Development Setup

### Prerequisites

- Python 3.11 or higher
- `uv` package manager (recommended) or `pip`
- Git for version control

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd fea-converge
   ```

2. **Create virtual environment with uv**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**:
   ```bash
   uv pip install -e ".[dev]"
   ```

   This installs:
   - **Core dependencies**: PyTorch, BoTorch, Ax Platform, GPyTorch, NumPy, SciPy, Pandas
   - **Visualization**: Matplotlib, Plotly
   - **ML tools**: scikit-learn
   - **Development tools**: pytest, black, ruff, mypy

4. **Verify installation**:
   ```bash
   python -c "import fr_bo; print('FR-BO installed successfully')"
   ```
