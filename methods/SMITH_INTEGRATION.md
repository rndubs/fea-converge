# Smith Models Integration Guide

This guide explains how each optimization method in `./methods` integrates with the Smith contact models in `./smith-models` to understand and optimize contact convergence.

## Overview

The `fea-converge` repository provides four distinct Bayesian optimization methods for resolving contact convergence failures. Each method can build and run the Smith contact models to:

1. **Collect convergence data** from validated contact test cases
2. **Analyze convergence patterns** specific to each method's approach
3. **Train optimization models** using real FEA simulation results
4. **Validate optimization strategies** on physical contact problems

## Common Infrastructure

### SmithModelRunner Utility

All methods share a common Python utility for interacting with Smith models:

**Location:** `./methods/smith_runner.py`

**Features:**
- Build and run Smith models via the `./run_model` script
- Parse output to extract convergence metrics
- Support for Docker and local build modes
- Automatic detection of repository structure
- Comprehensive result parsing (iterations, residuals, timesteps, etc.)

**Basic Usage:**

```python
from smith_runner import SmithModelRunner

# Initialize runner
runner = SmithModelRunner(verbose=True)

# Run a single model
result = runner.run_model("die-on-slab")
print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")

# Run all models
results = runner.run_all_models()
```

### Available Smith Models

Eight validated contact test cases are available:

1. **die-on-slab** - Cylindrical die pressed and slid on flexible slab
2. **block-on-slab** - Stiff square block pressed and slid on flexible slab
3. **sphere-in-sphere** - Solid sphere pressed into hollow sphere
4. **stacked-blocks** - Four stacked blocks with stick-slip behavior
5. **hemisphere-twisting** - Hollow hemisphere indenting and twisting
6. **concentric-spheres** - Two concentric spheres with friction-dependent buckling
7. **deep-indentation** - Small stiff block fully indenting large soft block
8. **hollow-sphere-pinching** - Hollow sphere compressed between fingers

## Method-Specific Integration

### 1. CONFIG (Constrained Efficient Global Optimization)

**Focus:** Theoretical guarantees for constrained optimization

**Smith Integration Script:**
```bash
cd methods/config/examples
python run_smith_models.py
```

**What It Does:**
- Runs all Smith models to collect baseline convergence data
- Analyzes convergence statistics (iterations, timesteps, residuals)
- Identifies challenging contact scenarios for constraint definition
- Provides parameter ranges from successful runs
- Saves results to `smith_results/` for CONFIG optimizer setup

**Output Files:**
- `all_models_results.json` - Complete results for all models
- `convergence_analysis.json` - Statistical analysis of convergence

**Next Steps:**
Use the convergence data to:
1. Define constraint tolerances for CONFIG optimizer
2. Set parameter bounds based on successful configurations
3. Identify which models have strictest convergence requirements
4. Configure CONFIG with appropriate constraint configs

---

### 2. FR-BO (Failure-Robust Bayesian Optimization)

**Focus:** Dual GP modeling of convergence + failure probability

**Smith Integration Script:**
```bash
cd methods/fr_bo/examples
python run_smith_models.py
```

**What It Does:**
- Runs all Smith models with focus on failure detection
- Analyzes failure patterns and failure modes
- Computes failure rates to assess FR-BO applicability
- Categorizes models into successful/failed/non-converged
- Saves failure analysis for dual GP training

**Output Files:**
- `all_models_results.json` - Complete results for all models
- `frbo_failure_analysis.json` - Failure-specific analysis

**FR-BO Relevance:**
- **High relevance** if failure rate > 10%
- **Moderate relevance** if failure rate 5-10%
- **Low relevance** if failure rate < 5%

**Next Steps:**
Use the failure data to:
1. Train dual GP models (convergence GP + failure GP)
2. Set up failure-aware acquisition function
3. Configure early termination based on typical failure patterns
4. Validate FR-BO's advantage over standard BO

---

### 3. GP-Classification

**Focus:** Binary convergence prediction with interpretable boundaries

**Smith Integration Script:**
```bash
cd methods/gp-classification/examples
python run_smith_models.py
```

**What It Does:**
- Runs all Smith models to collect binary convergence labels
- Analyzes class balance (converged vs failed)
- Identifies decision boundary characteristics
- Assesses suitability for GP classification approach
- Saves binary classification data

**Output Files:**
- `all_models_results.json` - Complete results for all models
- `gp_classification_analysis.json` - Binary classification analysis

**GP Classification Suitability:**
- **Excellent** if class balance 20-80%
- **Usable** if class balance outside this range (may need resampling)

**Next Steps:**
Use the binary labels to:
1. Train Variational GP Classifier
2. Visualize convergence decision boundary
3. Run three-phase optimization (boundary discovery → refinement → exploitation)
4. Interpret learned boundary for physical insights

---

### 4. SHEBO (Surrogate Optimization with Hidden Constraints)

**Focus:** Ensemble neural networks with constraint discovery

**Smith Integration Script:**
```bash
cd methods/shebo/examples
python run_smith_models.py
```

**What It Does:**
- Runs all Smith models to discover hidden constraints
- Analyzes successful runs to establish constraint thresholds
- Identifies safe vs unsafe parameter regions
- Discovers implicit constraints (iteration limits, residual tolerances, etc.)
- Saves constraint discovery results

**Output Files:**
- `all_models_results.json` - Complete results for all models
- `shebo_constraint_discovery.json` - Discovered constraints

**Discovered Constraints:**
The script automatically discovers:
1. **Iteration limits** - Maximum iterations for successful runs
2. **Residual tolerances** - Required final residual thresholds
3. **Timestep completion** - Minimum timesteps to complete

**SHEBO Suitability:**
- **Excellent** if multiple hidden constraints discovered
- **Moderate** if 1 constraint discovered
- **Limited** if no constraints discovered (may need more data)

**Next Steps:**
Use the discovered constraints to:
1. Define constraint functions for SHEBO optimizer
2. Train ensemble of neural network surrogates
3. Configure constraint-aware acquisition
4. Run optimization with checkpointing

---

## Running Smith Models

### Prerequisites

**Option 1: Docker (Recommended)**
```bash
# Build Smith with Docker (one-time setup, 15-30 min)
./build-smith-docker.sh
```

**Option 2: LC HPC**
```bash
# Build Smith on LLNL HPC systems
./build-smith-lc.sh
```

**Option 3: Local Build**
Follow instructions in `smith-models/README.md`

### Running Workflows

Each method has its own workflow script:

```bash
# CONFIG method
cd methods/config/examples
python run_smith_models.py

# FR-BO method
cd methods/fr_bo/examples
python run_smith_models.py

# GP-Classification method
cd methods/gp-classification/examples
python run_smith_models.py

# SHEBO method
cd methods/shebo/examples
python run_smith_models.py
```

All scripts:
- Auto-detect repository structure
- Use shared `smith_runner` utility
- Run all 8 Smith models
- Parse and analyze results
- Save JSON output files
- Provide method-specific insights

### Expected Runtime

- **Smith build (one-time):** 15-30 minutes (Docker) or 15-60 minutes (LC HPC)
- **Single model run:** 5-30 seconds
- **All 8 models:** 1-5 minutes total

### Build Modes

The `smith_runner` automatically detects the appropriate build mode:

1. **Auto** (default) - Detects Docker vs local
2. **Docker** - Force Docker build
3. **Local** - Force local build

Override with:
```python
runner = SmithModelRunner(build_mode=BuildMode.DOCKER)
```

## Output Structure

Each method creates a `smith_results/` directory in its examples folder:

```
methods/
├── config/examples/smith_results/
│   ├── all_models_results.json
│   └── convergence_analysis.json
├── fr_bo/examples/smith_results/
│   ├── all_models_results.json
│   └── frbo_failure_analysis.json
├── gp-classification/examples/smith_results/
│   ├── all_models_results.json
│   └── gp_classification_analysis.json
└── shebo/examples/smith_results/
    ├── all_models_results.json
    └── shebo_constraint_discovery.json
```

## Result Format

### SmithModelResult

Each model run returns a `SmithModelResult` object with:

```python
@dataclass
class SmithModelResult:
    model_name: str              # Name of model
    converged: bool              # Whether simulation converged
    success: bool                # Whether run completed without errors
    iterations: int              # Total nonlinear solver iterations
    timesteps_completed: int     # Number of timesteps completed
    final_residual: float        # Final residual norm (if available)
    solve_time: float            # Total solve time in seconds (if available)
    output_files: List[Path]     # Generated ParaView files
    stdout: str                  # Standard output
    stderr: str                  # Standard error
    returncode: int              # Process return code
    error_message: str           # Error message if failed
```

### JSON Output

Results are saved in JSON format for easy loading:

```python
import json

with open('smith_results/all_models_results.json', 'r') as f:
    results = json.load(f)

# Access individual model results
die_result = results['die-on-slab']
print(f"Converged: {die_result['converged']}")
print(f"Iterations: {die_result['iterations']}")
```

## Troubleshooting

### Smith Not Built

**Error:** `Smith executable not found` or `Smith not built or installed`

**Solution:**
```bash
# Build Smith first
./build-smith-docker.sh
# or for LC HPC
./build-smith-lc.sh
```

### Model Build Failures

**Error:** `Build or executable not found`

**Solution:**
```bash
# Clean build and retry
./run_model --clean die-on-slab
```

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'smith_runner'`

**Solution:**
The scripts automatically add the parent directory to the path. If issues persist:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from smith_runner import SmithModelRunner
```

### Docker Not Running

**Error:** `Docker not running, will use local build`

**Solution:**
- Start Docker daemon, or
- Build Smith locally (see smith-models/README.md), or
- Use LC HPC build script

## Advanced Usage

### Run Specific Models

```python
from smith_runner import SmithModelRunner

runner = SmithModelRunner(verbose=True)

# Run specific models
models_to_run = ["die-on-slab", "sphere-in-sphere", "stacked-blocks"]

results = {}
for model in models_to_run:
    results[model] = runner.run_model(model)
```

### Custom Timeout

```python
# Increase timeout for complex models
result = runner.run_model("hemisphere-twisting", timeout=1200)  # 20 minutes
```

### Clean Build

```python
# Force clean build before running
result = runner.run_model("concentric-spheres", clean=True)
```

### Check Model Existence

```python
if runner.model_exists("custom-model"):
    result = runner.run_model("custom-model")
else:
    print("Model not found")
```

## Integration with Optimization

The Smith models serve as **simulation backends** for optimization:

```
Parameter vector (θ) from optimizer
         ↓
    Smith model execution
         ↓
   Convergence metrics (success, iterations, residual)
         ↓
    Feedback to optimizer
         ↓
    Next parameter suggestion
```

Each method uses the convergence data differently:

- **CONFIG:** Constraint violation monitoring
- **FR-BO:** Failure probability modeling
- **GP-Classification:** Binary convergence labels
- **SHEBO:** Hidden constraint discovery

## References

- **Smith SDK Documentation:** https://serac.readthedocs.io/
- **Tribol Contact Library:** LLNL internal documentation
- **Build Instructions:** `./smith-models/README.md`
- **Run Model Script:** `./run_model` (bash script)
- **LC HPC Guide:** https://hpc.llnl.gov/documentation/

## Support

For issues with:
- **Smith integration:** See this document and `smith_runner.py`
- **Smith build:** See `smith-models/README.md` and build scripts
- **Method-specific:** See each method's README and documentation
- **General questions:** Open an issue in the repository
