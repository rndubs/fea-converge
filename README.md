# Smith FEA Build System Setup

This repository contains the Smith FEA package configured for machine learning applications with Ax/Botorch.

## Overview

Smith is a 3D implicit nonlinear thermal-structural simulation code from Lawrence Livermore National Laboratory (LLNL). This setup is designed to enable parameter tuning of solver controls and contact settings for machine learning workflows.

## Repository Structure

```
fea-converge/
├── smith/              # Smith FEA package (git submodule)
│   ├── src/           # Smith source code
│   ├── examples/      # Example input files
│   ├── mfem/          # MFEM finite element library (submodule)
│   ├── axom/          # Axom data structures library (submodule)
│   ├── tribol/        # Tribol contact mechanics library (submodule)
│   └── scripts/       # Build and configuration scripts
├── README.md          # This file
└── build_smith.sh     # Build script
```

## Smith FEA Package

- **Repository**: https://github.com/LLNL/smith
- **License**: BSD-3-Clause
- **Documentation**: https://serac.readthedocs.io
- **Purpose**: Thermal-structural FEA with focus on emerging architectures

## Build System

Smith uses:
- **CMake** (3.14+) for build configuration
- **BLT** (Build Level Tools) as the build system foundation
- **Spack/uberenv** for dependency management
- **MPI** (mandatory requirement - MPICH or OpenMPI)

### Key Features

- **Codevelop Mode**: Build with bundled MFEM, Axom, and Tribol (enabled via `-DSMITH_ENABLE_CODEVELOP=ON`)
- **MPI Support**: Required for all builds (can run with single process)
- **Lua Input Decks**: Supports parameter configuration via Lua scripts
- **C++ API**: Direct programmatic control available

## System Requirements

### Ubuntu 24.04 Dependencies

Already installed in this environment:

```bash
# Core build tools
cmake (3.28.3)
gcc-13, g++-13, gfortran-13
python3 (3.11)

# Libraries
libopenblas-dev      # Linear algebra
lua5.2, lua5.2-dev   # Lua scripting support
mpich, libmpich-dev  # MPI (v4.2.0)

# Utilities
gettext, lsb-release, ssh
```

### Additional Dependencies (Required but not installed)

Smith requires numerous third-party libraries that are typically managed by Spack:

- **CAMP**: Compiler abstraction and metaprogramming
- **Umpire**: Portable memory management
- **RAJA**: Portable kernel execution
- **MFEM**: Finite element library (bundled as submodule)
- **Axom**: Data structures and utilities (bundled as submodule)
- **Tribol**: Contact mechanics (bundled as submodule)
- **Conduit**: Data description library
- **HDF5**: Hierarchical data format
- **Hypre**: Parallel linear solvers
- **METIS/ParMETIS**: Graph partitioning
- **SuperLU-DIST**: Sparse linear solver
- **PETSc**: Scientific computing toolkit (optional)
- **SLEPc**: Eigenvalue solvers (optional)
- **SUNDIALS**: ODE/DAE solvers (optional)
- **Strumpack**: Sparse direct solver (optional)

## Build Approaches

### Recommended: Spack/Uberenv (Requires Network Access)

This is the official build method:

```bash
cd smith

# Build all dependencies and generate host-config
python3 ./scripts/uberenv/uberenv.py \
    --spack-env-file=./scripts/spack/configs/docker/ubuntu24/spack.yaml \
    --project-json=.uberenv_config.json \
    --spec="~devtools~enzyme %gcc_13" \
    --prefix=../smith_tpls

# Configure Smith
python3 config-build.py -hc *.cmake -bp build

# Build
cd build
make -j$(nproc)

# Test
make test
```

**Build Variants**:
- `~devtools`: Disable development tools (Sphinx, CppCheck, etc.)
- `~enzyme`: Disable automatic differentiation
- `~petsc`: Disable PETSc support
- `~slepc`: Disable SLEPc support
- `+openmp`: Enable OpenMP (default ON)

### Alternative: Manual Dependency Installation

If network access is limited, dependencies must be installed manually. This is complex and not recommended for initial setup.

## MPI Configuration

Smith requires MPI but can be run in serial mode:

```bash
# Run with single MPI process (effectively serial)
mpirun -np 1 ./smith_executable input_deck.lua

# Run with multiple processes
mpirun -np 4 ./smith_executable input_deck.lua
```

MPI compilers are available at:
- C: `/usr/bin/mpicc`
- C++: `/usr/bin/mpicxx`
- Fortran: `/usr/bin/mpifort`

## Using Smith for Machine Learning

### Parameter Tuning with Lua

Smith supports Lua input decks that allow runtime configuration of:
- Solver parameters (tolerances, iterations, preconditioners)
- Contact settings (penalty parameters, friction coefficients)
- Material properties
- Time stepping controls

Example Lua input deck structure:

```lua
-- Solver configuration
solver = {
    max_iterations = 1000,
    relative_tolerance = 1.0e-6,
    absolute_tolerance = 1.0e-10,
    preconditioner = "AMG"
}

-- Contact parameters
contact = {
    penalty_parameter = 1.0e6,
    friction_coefficient = 0.3,
    contact_tolerance = 1.0e-8
}
```

### Integration with Ax/Botorch

Smith can be integrated with Bayesian optimization frameworks:

1. **Parameter Space Definition**: Define ranges for solver/contact parameters
2. **Objective Function**: Wrap Smith execution to return performance metrics
3. **Optimization Loop**: Use Ax/Botorch to explore parameter space

Example integration sketch:

```python
from ax import optimize
import subprocess

def evaluate_smith(parameters):
    """Run Smith with given parameters and return objective."""
    # Generate Lua input with parameters
    write_lua_input(parameters)
    
    # Run Smith
    result = subprocess.run(['mpirun', '-np', '1', './smith', 'input.lua'],
                          capture_output=True)
    
    # Extract performance metric
    return parse_smith_output(result.stdout)

# Optimize
best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "tolerance", "type": "range", "bounds": [1e-8, 1e-4]},
        {"name": "penalty", "type": "range", "bounds": [1e4, 1e8]},
    ],
    evaluation_function=evaluate_smith,
    objective_name="convergence_rate",
)
```

## Network Restrictions

**Important**: The current environment has network access restrictions (HTTP 403 Forbidden) that prevent:
- Spack from downloading source packages
- Uberenv from bootstrapping dependencies
- Direct source downloads from mirrors

To complete the build, you will need either:
1. **Unrestricted network access** for the Spack-based build
2. **Pre-built dependencies** from a Spack mirror
3. **Docker container** with pre-installed dependencies (see `smith/scripts/docker/`)

## Pre-built Docker Images

LLNL provides Docker images with all dependencies:

```bash
# Pull pre-built image
docker pull seracllnl/tpls:gcc-14_latest

# Or build from Dockerfile
docker build -f smith/scripts/docker/dockerfile_gcc-14 -t smith-build .
```

## Next Steps

1. **Resolve Network Access**: Enable external downloads for Spack
2. **Run Uberenv Build**: Execute the build script to install dependencies
3. **Configure Smith**: Use generated host-config to configure CMake
4. **Build and Test**: Compile Smith and run test suite
5. **Develop ML Workflow**: Create Lua templates and Python integration scripts

## Troubleshooting

### MPI Not Found
```bash
# Verify MPI installation
which mpicc
mpicc --version

# Set CMake variables
cmake -DMPI_C_COMPILER=/usr/bin/mpicc \
      -DMPI_CXX_COMPILER=/usr/bin/mpicxx \
      ..
```

### Missing Dependencies
```bash
# Check for specific library
ldconfig -p | grep <library_name>

# Install HDF5 (example)
apt-get install libhdf5-mpich-dev
```

### Spack Bootstrap Failures
- Check internet connectivity: `curl -I https://mirror.spack.io`
- Use Spack mirror: `spack mirror add <mirror_url>`
- Disable clingo bootstrap: `spack config add config:concretizer:original`

## References

- Smith Repository: https://github.com/LLNL/smith
- Smith Documentation: https://serac.readthedocs.io
- MFEM: https://mfem.org
- Spack: https://spack.io
- BLT: https://github.com/LLNL/blt
- Ax Platform: https://ax.dev
- BoTorch: https://botorch.org

## License

Smith is licensed under BSD-3-Clause.
Copyright (c) Lawrence Livermore National Security, LLC.
LLNL-CODE-805541
