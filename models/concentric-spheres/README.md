# Concentric Spheres Between Rigid Planes Contact Model

## Description

This model implements the concentric spheres compression example from Figure 9 and Section 3.7 of Zimmerman & Ateshian (2018). Two thick-walled concentric spheres are compressed between rigid planes with frictional sliding contact between the spheres. This problem has been used to motivate mortar formulations for contact and represents a challenging benchmark.

## Model Setup

### Geometry

Modeled in octant symmetry (1/8 of full geometry):

- **Inner Sphere** (solid or thick-walled):
  - Details vary by test case
  - Positioned at center

- **Outer Sphere** (thick-walled hollow):
  - Outer radius: Ro = 2.0 units
  - Inner radius: Ri = 0.7 units
  - Wall thickness: 1.3 units
  - Contains inner sphere

- **Rigid Planes** (two):
  - Top plane: moves downward
  - Bottom plane: fixed
  - Compress spheres between them

### Material Properties

- **Both Spheres**:
  - Young's modulus: E = 1 MPa
  - Poisson's ratio: ν = 0.3 (inner), ν = 0.0 (outer) [varies by case]
  - Material model: Compressible Neo-Hookean

### Loading Conditions

- **Prescribed Displacement**: Applied to top rigid plane
  - uz = -10 mm (units) total compression
  - Applied over 10 seconds
  - 100 evenly spaced time steps
  - Linear ramp from 0 to -10 mm

### Contact Configuration

Three friction coefficient cases are examined:

**Case 1: Frictionless (μ = 0)**
- Contact between spheres: Frictionless
- Penalty parameter: α = 1
- Augmentation tolerance: Ptol = 0.2

**Case 2: Moderate Friction (μ = 0.5)**
- Contact between spheres: Frictional
- Friction coefficient: μ = 0.5
- Penalty parameter: α = 3
- Augmentation tolerance: Ptol = 0.2

**Case 3: High Friction (μ = 2)**
- Contact between spheres: Frictional
- Friction coefficient: μ = 2.0
- Penalty parameter: α = 8
- Augmentation tolerance: Ptol = 0.05

**Contact Pairs:**
- Sphere-to-sphere: Varies by case (see above)
- Top plane to outer sphere: Frictionless, single-pass (sphere as primary)
- Bottom plane to outer sphere: Frictionless, single-pass (sphere as primary)

### Boundary Conditions

- **Symmetry Planes**: Three planes of symmetry (x=0, y=0, z=0) with appropriate constraints
- **Bottom Plane**: Fixed rigid surface
- **Top Plane**: Prescribed displacement uz = -10 mm
- **Contact Surfaces**:
  - Outer surface of inner sphere
  - Inner surface of outer sphere
  - Outer surface of outer sphere (contacts top/bottom planes)

### Mesh Configuration

- Octant symmetry reduces computational cost
- Incompatible meshes between the two spheres (tests robustness)
- Adequate refinement to capture buckling behavior
- Linear hexahedral elements

## Expected Behavior

This benchmark demonstrates several challenging contact scenarios:

1. **Flexible-to-Flexible Contact**: Both spheres undergo significant deformation - single-pass NTS methods not applicable

2. **Friction-Dependent Buckling**:
   - **μ = 0, 0.5**: Inner sphere buckles during compression
   - **μ = 2**: No buckling - high friction prevents relative motion between spheres

3. **Gap Discontinuities**: Incompatible meshes force gap discontinuities during both contact and sliding

4. **Vertical Reaction Force**:
   - Much larger for μ = 2 due to absence of buckling
   - Buckling reduces compression force by allowing energy dissipation through deformation

5. **Node-on-Node Contact**: Initial point contact creates potential force oscillations

The paper notes that:
- Puso & Laursen (2003) noted premature failure of NTS algorithms even for frictionless case
- Areias et al. (2015) solved with friction but to smaller displacement (uz = -9 mm)
- Results show excellent agreement with ABAQUS for all friction coefficients
- Two-pass NTS approaches can fail at first timestep without special techniques
- Smooth two-pass NTS methods may fail due to locking
- The mortar method successfully handles all these challenges

## Running the Model

```bash
# From the project root directory
./run_model concentric-spheres
```

You can modify the friction coefficient in the source code to test different cases (μ = 0, 0.5, or 2.0).

## Output

The simulation produces ParaView-compatible output files showing:
- Displacement fields for both spheres
- Contact pressure distribution at sphere-sphere interface
- Stress distribution in both spheres
- Deformed geometry at each timestep
- Buckling behavior of inner sphere (for μ = 0, 0.5)
- Vertical reaction force vs. displacement curve

## Reference

Zimmerman, B. K., & Ateshian, G. A. (2018). **A Surface-to-Surface Finite Element Algorithm for Large Deformation Frictional Contact in FEBIO**. _Journal of Biomechanical Engineering_, 140(8), 081013.

DOI: [10.1115/1.4040497](https://doi.org/10.1115/1.4040497)

Available at: https://pmc.ncbi.nlm.nih.gov/articles/PMC6056201/

Also compared with:
- Puso, M. A., & Laursen, T. A. (2003). **A Mortar Segment-to-Segment Contact Method for Large Deformation Solid Mechanics**. _Computer Methods in Applied Mechanics and Engineering_.
- Areias, P., Rabczuk, T., de Melo, F. Q., & De Sa, J. C. (2015). **Coulomb Frictional Contact by Explicit Projection in the Cone for Finite Displacement Quasi-Static Problems**. _Computational Mechanics_, 55(1), 57-72.

## Notes

- This model requires octant symmetry boundary conditions
- Hollow sphere mesh needed with Ro=2.0, Ri=0.7
- Incompatible meshes between spheres are essential to the benchmark
- Rigid plane implementation requires special rigid body or constraint capabilities
- Higher penalty (α = 8) needed for μ = 2 case due to stronger coupling
- Tighter augmentation tolerance (Ptol = 0.05) for μ = 2 case
- The buckling of the inner sphere is a key physical phenomenon to capture
- Comparison with ABAQUS shows this is a reliable validation benchmark
- Node-on-node initial contact point requires careful treatment
- Single-pass analysis for sphere-plane contact (deformable sphere as primary)
- Two-pass analysis between the two deformable spheres
- This benchmark has historically motivated development of mortar methods
