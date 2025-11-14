# Stacked Blocks Contact Model

## Description

This model implements the stacked blocks contact example from Figure 5 and Section 3.5 of Zimmerman & Ateshian (2018). Four blocks with dissimilar material properties and meshes are stacked vertically and compressed, demonstrating the algorithm's ability to handle loss of contact, transverse gaps, and mixed stick-slip behavior.

## Model Setup

### Geometry

Four identical cubes stacked vertically:
- **Block Dimensions**: 2 mm × 2 mm × 2 mm each
- **Total Height**: 8 mm (4 blocks)
- **Numbering**: From top to bottom (Block 1, 2, 3, 4)

### Material Properties

- **Blocks 1 and 3** (top and third from top):
  - Young's modulus: E = 10 MPa
  - Poisson's ratio: ν = 0.1
  - Material model: Compressible Neo-Hookean

- **Blocks 2 and 4** (second and bottom):
  - Young's modulus: E = 0.3 MPa
  - Poisson's ratio: ν = 0.4
  - Material model: Compressible Neo-Hookean

### Loading Conditions

- **Prescribed Displacement**: Applied to top surface of Block 1
  - uz = -2 mm (50% compression)
  - Applied over 10 time steps
  - Ramped linearly from 0 to -2 mm

### Contact Configuration

- **Contact Method**: Surface-to-surface with ray-tracing projection
- **Enforcement**: Augmented Lagrangian
- **Contact Type**: Frictional (Coulomb)

**Interface 1** (Block 1 / Block 2):
- Friction coefficient: μ = 0.05
- Penalty parameter: α = 1
- Augmentation tolerance: Ptol = 0.2

**Interface 2** (Block 2 / Block 3):
- Friction coefficient: μ = 0.05
- Penalty parameter: α = 1
- Augmentation tolerance: Ptol = 0.2

**Interface 3** (Block 3 / Block 4):
- Friction coefficient: μ = 1.0
- Penalty parameter: α = 10
- Augmentation tolerance: Ptol = 0.2

### Boundary Conditions

- **Fixed**: Bottom surface of Block 4 (constrained in all directions)
- **Prescribed Displacement**: Top surface of Block 1 (uz = -2 mm)
- **Contact Surfaces**:
  - Three contact interfaces between adjacent blocks
  - Each interface uses incompatible meshes

### Mesh Configuration

- Dissimilar meshes at each interface to test robustness
- Sufficient refinement to capture contact behavior
- Linear hexahedral elements

## Expected Behavior

This problem demonstrates several challenging contact scenarios:

1. **Sliding Development**: Due to different friction coefficients, the top two blocks slide while the bottom two blocks stick
2. **Transverse Gap Formation**: The second block (soft) forms a "shelf" or overhang as it deforms under compression
3. **Loss of Contact**: As the shelf develops, some regions lose contact while others maintain it
4. **Mixed Stick-Slip**: The bottom interface (μ = 1.0) maintains sticking behavior throughout, while upper interfaces (μ = 0.05) develop sliding

The paper notes that:
- Dissimilar meshes and transverse gaps typically cause failure of node-to-segment algorithms
- The ray-tracing contact detection strategy handles overhanging blocks without difficulty
- The augmented Lagrangian scheme successfully differentiates between stick and slip regions

## Running the Model

```bash
# From the project root directory
./run_model stacked-blocks
```

## Output

The simulation produces ParaView-compatible output files showing:
- Displacement fields for all four blocks
- Contact pressure distribution at each interface
- Stick-slip status at each contact point
- Deformed geometry showing the "shelf" formation
- Evolution of contact patches as blocks slide

## Reference

Zimmerman, B. K., & Ateshian, G. A. (2018). **A Surface-to-Surface Finite Element Algorithm for Large Deformation Frictional Contact in FEBIO**. _Journal of Biomechanical Engineering_, 140(8), 081013.

DOI: [10.1115/1.4040497](https://doi.org/10.1115/1.4040497)

Available at: https://pmc.ncbi.nlm.nih.gov/articles/PMC6056201/

## Notes

- This model requires four separate cube meshes with different element densities
- The dissimilar meshes are essential to the benchmark - do not use identical meshes
- Augmented Lagrangian regularization is recommended over pure penalty for this problem
- The low friction coefficient (μ = 0.05) at the top interfaces allows sliding under compression
- The high friction coefficient (μ = 1.0) at the bottom interface ensures sticking
- Contact surface attributes must be carefully assigned to match the interface specifications
