# Smith Build System Documentation

## Overview

This directory contains consolidated build scripts for Smith and smith-models across different platforms:

- **macOS**: Container-based builds using Docker
- **LLNL HPC**: Container-based builds using Singularity/Apptainer
- **Native**: Direct builds on Linux systems

## Directory Structure

```
build/
├── BUILD.md                      # This file
├── docker/                       # macOS container builds
│   └── build-smith-macos.sh
├── hpc/                          # LLNL HPC builds
│   └── build-smith-llnl.sh
└── scripts/                      # Model compilation and execution
    ├── build-model.sh
    └── run-model.sh
```

---

## Quick Start

### macOS

```bash
# 1. Build Smith
./build/docker/build-smith-macos.sh

# 2. Build a model
./build/scripts/build-model.sh die-on-slab

# 3. Run the model
./build/scripts/run-model.sh die-on-slab
```

### LLNL HPC

```bash
# 1. Build Smith (interactive)
./build/hpc/build-smith-llnl.sh --system quartz

# 2. Build a model
./build/scripts/build-model.sh die-on-slab

# 3. Run the model
./build/scripts/run-model.sh die-on-slab --np 4
```

---

## Building Smith

Smith is the finite element solver framework that provides contact mechanics capabilities. You must build Smith before building any models.

### macOS (Docker)

**Prerequisites:**
- Docker Desktop installed and running
- At least 10GB free disk space
- Git submodules initialized (`git submodule update --init --recursive`)

**Basic Build:**
```bash
./build/docker/build-smith-macos.sh
```

**Advanced Options:**
```bash
# Use GCC instead of Clang
./build/docker/build-smith-macos.sh --image gcc-14

# Build with 8 jobs and run tests
./build/docker/build-smith-macos.sh -j 8 --test

# Use CUDA image (requires NVIDIA GPU)
./build/docker/build-smith-macos.sh --image cuda-12
```

**Available Docker Images:**
- `clang-19`: Clang 19.1.1 (recommended, default)
- `gcc-14`: GCC 14.2.0
- `cuda-12`: CUDA 12 with GCC 12.3.0 (requires NVIDIA GPU)

**Output:**
- Build artifacts: `./smith-build/`
- Installed files: `./smith-install/`

**Apple Silicon Notes:**
- Docker images are x86_64 and run via Rosetta 2 emulation
- This is slower than native ARM builds but works correctly
- No additional configuration needed

### LLNL HPC (Singularity)

**Prerequisites:**
- Access to LLNL LC clusters (quartz, ruby, lassen, etc.)
- Singularity/Apptainer installed
- Container image converted from Docker (see below)
- Git submodules initialized

**Converting Docker Images:**

First-time setup requires converting Docker images to Singularity format:

```bash
# Create container directory
mkdir -p ~/containers

# Convert Clang 19 image (recommended)
singularity build ~/containers/smith-clang19.sif \
    docker://seracllnl/tpls:clang-19_10-09-25_23h-54m

# Or convert GCC 14 image
singularity build ~/containers/smith-gcc14.sif \
    docker://seracllnl/tpls:gcc-14_10-09-25_23h-54m

# Or convert CUDA 12 image (for GPU nodes)
singularity build ~/containers/smith-cuda12.sif \
    docker://seracllnl/tpls:cuda-12_04-16-25_20h-55m
```

**Interactive Build:**
```bash
# Auto-detect system
./build/hpc/build-smith-llnl.sh

# Specify system explicitly
./build/hpc/build-smith-llnl.sh --system quartz --compiler clang
```

**Batch Build:**
```bash
# Submit as batch job
./build/hpc/build-smith-llnl.sh --batch --compiler gcc -j 16
```

**Supported Systems:**
- `quartz`, `ruby`, `jade`: x86_64 clusters
- `lassen`: IBM POWER9 + NVIDIA V100
- `rzadams`, `rzvernal`: Intel Sapphire Rapids

**Output:**
- Build artifacts: `./smith-build/`
- Installed files: `./smith-install/`
- Batch logs: `./smith-build/build-<jobid>.out`

---

## Building Models

After Smith is built, you can build contact models from the `smith-models/` directory.

### Available Models

**From Puso & Laursen (2003):**
- `die-on-slab`: Cylindrical die pressed and slid on flexible slab
- `block-on-slab`: Stiff square block pressed and slid on flexible slab
- `sphere-in-sphere`: Solid sphere pressed into hollow sphere

**From Zimmerman & Ateshian (2018):**
- `stacked-blocks`: Four stacked blocks with stick-slip transition
- `hemisphere-twisting`: Hollow hemisphere indenting and twisting
- `concentric-spheres`: Two concentric spheres with friction-dependent buckling
- `deep-indentation`: Small stiff block fully indenting large soft block
- `hollow-sphere-pinching`: Hollow sphere compressed between deformable fingers

### Build Commands

**Basic Build:**
```bash
./build/scripts/build-model.sh <model-name>
```

**Examples:**
```bash
# Build die-on-slab
./build/scripts/build-model.sh die-on-slab

# Clean build with 8 jobs
./build/scripts/build-model.sh sphere-in-sphere -j 8 --clean

# Force specific environment
./build/scripts/build-model.sh block-on-slab --env docker

# Verbose output
./build/scripts/build-model.sh stacked-blocks --verbose
```

**Options:**
- `-e, --env ENV`: Build environment (docker, singularity, native, auto)
- `-j, --jobs N`: Number of parallel build jobs
- `-c, --clean`: Clean build directory before building
- `-v, --verbose`: Verbose build output

**Output:**
- Build directory: `./smith-models/build/<model-name>/`
- Executable: `./smith-models/build/<model-name>/<model_name>`

---

## Running Models

After a model is built, you can run simulations.

### Run Commands

**Basic Run:**
```bash
./build/scripts/run-model.sh <model-name>
```

**Examples:**
```bash
# Run die-on-slab
./build/scripts/run-model.sh die-on-slab

# Run with 4 MPI ranks
./build/scripts/run-model.sh sphere-in-sphere --np 4

# Custom output directory
./build/scripts/run-model.sh block-on-slab --output ./results

# Batch job on HPC
./build/scripts/run-model.sh stacked-blocks --batch --np 8
```

**Options:**
- `-e, --env ENV`: Run environment (docker, singularity, native, auto)
- `-n, --np N`: Number of MPI ranks (default: 1)
- `-o, --output DIR`: Output directory (default: model build directory)
- `-p, --params FILE`: Parameter file for model (optional)
- `-b, --batch`: Submit as batch job on HPC (LLNL only)

**Output Files:**
- ParaView visualization: `<model_name>_paraview.*`
- Log files: `<model_name>.log`
- Batch job logs: `<model_name>-<jobid>.out` (HPC only)

---

## Visualization

### ParaView

All models output results in ParaView format for visualization:

1. **Open ParaView**
2. **Load file:** `smith-models/build/<model-name>/<model_name>_paraview.pvd`
3. **Click "Apply"** to load the data
4. **Use time controls** to step through the simulation

**Example:**
```bash
# After running die-on-slab
paraview smith-models/build/die-on-slab/die_on_slab_paraview.pvd
```

---

## Troubleshooting

### Docker Issues (macOS)

**Error: Docker is not running**
```bash
# Solution: Launch Docker Desktop from Applications
open -a Docker
```

**Error: No space left on device**
```bash
# Clean up Docker images and containers
docker system prune -a

# Check disk usage
docker system df
```

**Error: Image pull failed**
```bash
# Check internet connection
# Try pulling manually
docker pull --platform linux/amd64 seracllnl/tpls:clang-19_10-09-25_23h-54m
```

### HPC Issues (LLNL)

**Error: Container image not found**
```bash
# Convert Docker image to Singularity
singularity build ~/containers/smith-clang19.sif \
    docker://seracllnl/tpls:clang-19_10-09-25_23h-54m
```

**Error: Permission denied**
```bash
# Ensure container directory is in your home
ls -la ~/containers/

# Check container permissions
chmod 644 ~/containers/*.sif
```

**Batch job not starting**
```bash
# Check job status
squeue -u $USER

# View job details
scontrol show job <jobid>

# Cancel job if needed
scancel <jobid>
```

### Build Issues

**Error: Smith not found**
```bash
# Build Smith first
./build/docker/build-smith-macos.sh    # macOS
./build/hpc/build-smith-llnl.sh        # LLNL HPC

# Verify installation
ls -la smith-install/lib/cmake/
```

**Error: Model executable not found**
```bash
# Rebuild the model
./build/scripts/build-model.sh <model-name> --clean

# Check build directory
ls -la smith-models/build/<model-name>/
```

**Error: CMake configuration failed**
```bash
# Ensure Smith is properly installed
ls smith-install/lib/cmake/

# Try with verbose output
./build/scripts/build-model.sh <model-name> --verbose --clean
```

### Run Issues

**Error: Mesh file not found**

The models currently use placeholder mesh files from Smith examples. To run actual contact simulations, you need to create geometry-specific mesh files. See `smith-models/README.md` for details.

**Error: MPI not available**
```bash
# Docker/Singularity environments include MPI
# For native builds, install MPI
sudo apt-get install mpich libmpich-dev  # Ubuntu/Debian
brew install mpich                        # macOS (not recommended, use Docker)
```

---

## Advanced Topics

### Custom CMake Options

You can pass custom CMake options to model builds by modifying the build script or using environment variables:

```bash
# Example: Enable debugging symbols
cd smith-models/build/<model-name>
cmake <model-source-dir> \
    -DSmith_DIR=../../smith-install/lib/cmake \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-g -O0"
make -j8
```

### Parallel Builds

For faster builds, use multiple jobs:

```bash
# Smith build (16 cores)
./build/docker/build-smith-macos.sh -j 16

# Model build (8 cores)
./build/scripts/build-model.sh die-on-slab -j 8
```

### Multiple Compiler Builds

You can maintain builds with different compilers:

```bash
# Build with Clang
./build/docker/build-smith-macos.sh --image clang-19

# Build with GCC (creates separate build artifacts)
./build/docker/build-smith-macos.sh --image gcc-14
```

Note: Different compiler builds will overwrite `smith-install/`. To maintain multiple installations, modify the install directory in the build scripts.

### Cross-Platform Development

**Develop on macOS, Deploy on HPC:**

1. Edit code on macOS
2. Test builds locally with Docker
3. Push to Git repository
4. Pull on HPC cluster
5. Build with Singularity
6. Run production simulations

```bash
# macOS (development)
git add .
git commit -m "Update contact model parameters"
git push

# LLNL HPC (production)
git pull
./build/hpc/build-smith-llnl.sh
./build/scripts/build-model.sh die-on-slab
./build/scripts/run-model.sh die-on-slab --batch --np 32
```

---

## Performance Tips

### Build Performance

1. **Use parallel builds:** `-j $(nproc)` or `-j 16`
2. **Don't clean unless necessary:** Incremental builds are much faster
3. **Use local Docker images:** Only pull once, reuse for all builds
4. **On HPC:** Convert container images once, store in `~/containers/`

### Runtime Performance

1. **Use multiple MPI ranks:** `--np 4` or `--np 8` for parallel execution
2. **GPU acceleration:** Use CUDA image on GPU-enabled systems
3. **Batch mode on HPC:** Submit jobs to compute nodes instead of login nodes

---

## Environment Variables

The build scripts support several environment variables for customization:

```bash
# Docker image selection (macOS)
export DOCKER_IMAGE=seracllnl/tpls:gcc-14_10-09-25_23h-54m
./build/docker/build-smith-macos.sh

# Build jobs
export BUILD_JOBS=8
./build/scripts/build-model.sh die-on-slab

# Custom container location (HPC)
export CONTAINER_DIR=/usr/workspace/$USER/containers
./build/hpc/build-smith-llnl.sh
```

---

## Integration with Bayesian Optimization Methods

After building and running models, you can use the Bayesian optimization methods in this repository to tune contact parameters:

```bash
# Example: Optimize contact parameters for die-on-slab
cd config/examples
python smith_integration_example.py --model die-on-slab
```

See method-specific documentation:
- `config/README.md` - CONFIG method
- `gp-classification/README.md` - GP-Classification method
- `shebo/README.md` - SHEBO method
- `fr_bo/README.md` - FR-BO method

---

## Maintenance

### Updating Docker Images

Docker images are periodically updated by the Serac team. To update:

```bash
# Pull latest image
docker pull --platform linux/amd64 seracllnl/tpls:clang-19_10-09-25_23h-54m

# Rebuild Smith
./build/docker/build-smith-macos.sh --clean
```

### Updating Singularity Containers

```bash
# Remove old container
rm ~/containers/smith-clang19.sif

# Rebuild from latest Docker image
singularity build ~/containers/smith-clang19.sif \
    docker://seracllnl/tpls:clang-19_10-09-25_23h-54m

# Rebuild Smith
./build/hpc/build-smith-llnl.sh
```

### Cleaning Build Artifacts

```bash
# Clean Smith build
rm -rf smith-build smith-install

# Clean all model builds
rm -rf smith-models/build/

# Clean specific model
rm -rf smith-models/build/die-on-slab/

# Clean Docker images
docker system prune -a
```

---

## References

- **Smith Documentation:** https://github.com/LLNL/smith
- **Serac Docker Guide:** https://serac.readthedocs.io/en/latest/sphinx/dev_guide/docker_env.html
- **Docker Hub Images:** https://hub.docker.com/r/seracllnl/tpls/tags
- **LLNL LC Documentation:** https://hpc.llnl.gov/
- **ParaView:** https://www.paraview.org/

---

## Support

For issues:
- **Smith build issues:** https://github.com/LLNL/smith/issues
- **Model-specific issues:** See individual model README files in `smith-models/`
- **Build script issues:** Create an issue in this repository
