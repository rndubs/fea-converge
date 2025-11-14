# Contact Models for Smith

This directory contains contact test cases from two influential papers on frictional contact algorithms:

## References

**Puso, M. A., & Laursen, T. A. (2003).** *A Mortar Segment-to-Segment Frictional Contact Method for Large Deformations*. Computer Methods in Applied Mechanics and Engineering.
- Reference: https://www.osti.gov/servlets/purl/15013715

**Zimmerman, B. K., & Ateshian, G. A. (2018).** *A Surface-to-Surface Finite Element Algorithm for Large Deformation Frictional Contact in FEBIO*. Journal of Biomechanical Engineering, 140(8), 081013.
- DOI: [10.1115/1.4040497](https://doi.org/10.1115/1.4040497)
- Reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC6056201/

## Models from Puso & Laursen (2003)

1. **[die-on-slab](die-on-slab/)** - Cylindrical die pressed and slid on flexible slab (Example 2: Ironing)
2. **[block-on-slab](block-on-slab/)** - Stiff square block pressed and slid on flexible slab (Example 2 variant)
3. **[sphere-in-sphere](sphere-in-sphere/)** - Solid sphere pressed into hollow sphere (Example 3)

## Models from Zimmerman & Ateshian (2018)

4. **[stacked-blocks](stacked-blocks/)** - Four stacked blocks with dissimilar materials demonstrating stick-slip transition (Figure 5, Section 3.5)
5. **[hemisphere-twisting](hemisphere-twisting/)** - Hollow hemisphere indenting and twisting on deformable box (Figure 7, Section 3.6)
6. **[concentric-spheres](concentric-spheres/)** - Two concentric spheres compressed between rigid planes with friction-dependent buckling (Figure 9, Section 3.7)
7. **[deep-indentation](deep-indentation/)** - Small stiff block with sharp corners fully indenting large soft block (Figure 11, Section 3.8)
8. **[hollow-sphere-pinching](hollow-sphere-pinching/)** - Hollow sphere compressed between two deformable fingers demonstrating biomechanical grasping (Figure 19, Section 3.10)

## Quick Start

### Step 1: Build Smith

First, you need to build Smith using Docker. This only needs to be done once:

```bash
# From the project root directory
./build-smith-docker.sh
```

This script will:
- Pull a ~4GB Docker image with all pre-built dependencies (one-time download)
- Build Smith inside the Docker container
- Install Smith to `./smith-install` on your host machine
- The build process takes approximately 15-30 minutes

**Note for Apple Silicon (M1/M2/M3) Macs:** The Docker images are x86_64 and will run via Rosetta 2 emulation. This works correctly but may be slower.

### Step 2: Run a Model

Once Smith is built, you can run any of the contact models:

```bash
# Puso & Laursen (2003) models
./run_model die-on-slab
./run_model block-on-slab
./run_model sphere-in-sphere

# Zimmerman & Ateshian (2018) models
./run_model stacked-blocks
./run_model hemisphere-twisting
./run_model concentric-spheres
./run_model deep-indentation
./run_model hollow-sphere-pinching
```

The `run_model` script will:
1. Check that Smith has been built
2. Configure the model with CMake
3. Build the model executable
4. Run the simulation
5. Output ParaView-compatible visualization files

### Step 3: Visualize Results

After running a model, you can visualize the results in ParaView:

1. Open ParaView
2. Load the file: `build_<model-name>/<model_name>_paraview.pvd`
3. Click "Apply" to load the data
4. Use the time controls to step through the simulation

Example:
```
# For die-on-slab model:
build_die-on-slab/die_on_slab_paraview.pvd
```

## Troubleshooting

### Docker not running

**Error:** `Error: Docker is not running`

**Solution:** Launch Docker Desktop from your Applications folder and wait for it to start.

### No space left on device

**Error:** Docker image fails to pull due to disk space

**Solution:**
```bash
# Clean up unused Docker images
docker system prune -a

# Check available space
docker system df
```

### Smith build failed

**Error:** Build fails during `./build-smith-docker.sh`

**Solutions:**
1. Check Docker logs for specific errors
2. Try cleaning the build directories:
   ```bash
   rm -rf smith-build smith-install
   ./build-smith-docker.sh
   ```
3. Ensure you have at least 10GB of free disk space

### Model fails to find mesh file

**Error:** `Cannot open mesh file`

**Issue:** The models currently reference placeholder mesh files

**Solution:** The mesh files need to be created for each geometry. Currently, the models use existing Smith example meshes as placeholders. To fully implement the models, you would need to:

1. Generate appropriate mesh files for each geometry
2. Update the model source code to reference the correct mesh files
3. Assign correct element and boundary attributes

## Model Status

### ⚠️ Important Note on Mesh Files

The current implementations use **placeholder mesh files** from the Smith examples directory. To run the actual Puso & Laursen (2003) test cases, you need to:

1. **Create geometry-specific mesh files** for:
   - Cylindrical die and slab (die-on-slab)
   - Square block and slab (block-on-slab)
   - Solid sphere and hollow sphere (sphere-in-sphere)

2. **Assign proper element attributes** for:
   - Material property assignment (different attributes for different bodies)
   - Boundary conditions (fixed surfaces, driven surfaces)
   - Contact surfaces (surfaces that will interact)

3. **Update the source code** to reference the new mesh files

### Creating Mesh Files

You can create mesh files using tools like:
- **Cubit/Trelis** - LLNL's preferred meshing tool
- **Gmsh** - Open-source mesh generator
- **MFEM's mesh generation utilities** - Built-in mesh creation

The mesh files should be in MFEM format (.mesh) with appropriate:
- Element attributes for material assignment
- Boundary attributes for BC and contact surface definition

## Advanced Usage

### Building Smith Manually

If you prefer more control over the build process:

```bash
# Start the Docker container interactively
./docker-build-smith.sh

# Inside the container:
cd /home/serac/serac
python3 ./config-build.py -hc host-configs/docker/llvm@19.1.1.cmake -bp /home/serac/smith-build -ip /home/serac/smith-install
cd /home/serac/smith-build
make -j$(nproc)
make test
make install
```

### Using Different Docker Images

You can use different compiler toolchains:

```bash
# Use GCC 14
export DOCKER_IMAGE=seracllnl/tpls:gcc-14_10-09-25_23h-54m
export HOST_CONFIG=gcc@14.2.0.cmake
./build-smith-docker.sh

# Use CUDA 12 (requires NVIDIA GPU)
export DOCKER_IMAGE=seracllnl/tpls:cuda-12_04-16-25_20h-55m
export HOST_CONFIG=gcc@12.3.0_cuda.cmake
./build-smith-docker.sh
```

### Debugging Model Runs

To see more detailed output:

```bash
# Run with verbose CMake output
cd build_<model-name>
cmake ../models/<model-name> -DSmith_DIR=../smith-install/lib/cmake/smith -DCMAKE_VERBOSE_MAKEFILE=ON
make VERBOSE=1

# Run with MPI for parallel execution
mpirun -np 4 ./<model_name>
```

## Directory Structure

```
models/
├── README.md                          # This file
│
├── die-on-slab/                       # Puso & Laursen (2003)
│   ├── README.md
│   ├── CMakeLists.txt
│   └── die_on_slab.cpp
├── block-on-slab/                     # Puso & Laursen (2003)
│   ├── README.md
│   ├── CMakeLists.txt
│   └── block_on_slab.cpp
├── sphere-in-sphere/                  # Puso & Laursen (2003)
│   ├── README.md
│   ├── CMakeLists.txt
│   └── sphere_in_sphere.cpp
│
├── stacked-blocks/                    # Zimmerman & Ateshian (2018)
│   ├── README.md
│   ├── CMakeLists.txt
│   └── stacked_blocks.cpp
├── hemisphere-twisting/               # Zimmerman & Ateshian (2018)
│   ├── README.md
│   ├── CMakeLists.txt
│   └── hemisphere_twisting.cpp
├── concentric-spheres/                # Zimmerman & Ateshian (2018)
│   ├── README.md
│   ├── CMakeLists.txt
│   └── concentric_spheres.cpp
├── deep-indentation/                  # Zimmerman & Ateshian (2018)
│   ├── README.md
│   ├── CMakeLists.txt
│   └── deep_indentation.cpp
└── hollow-sphere-pinching/            # Zimmerman & Ateshian (2018)
    ├── README.md
    ├── CMakeLists.txt
    └── hollow_sphere_pinching.cpp

build_<model-name>/                    # Created during build (gitignored)
├── <model_name>                       # Executable
├── CMakeFiles/                        # Build artifacts
└── <model_name>_paraview.*            # Output files after run
```

## Additional References

- **Puso & Laursen (2003) Paper**: https://www.osti.gov/servlets/purl/15013715
- **Zimmerman & Ateshian (2018) Paper**: https://pmc.ncbi.nlm.nih.gov/articles/PMC6056201/
- **Smith Documentation**: https://github.com/LLNL/smith
- **Serac Docker Guide**: https://serac.readthedocs.io/en/latest/sphinx/dev_guide/docker_env.html
- **MFEM**: https://mfem.org/

## Contact

For issues or questions:
- Smith issues: https://github.com/LLNL/smith/issues
- Project-specific issues: Create an issue in this repository
