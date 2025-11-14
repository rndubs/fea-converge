# Deep Indentation of Square Blocks Contact Model

## Description

This model implements the deep indentation contact example from Figure 11 and Section 3.8 of Zimmerman & Ateshian (2018). A small stiff block with sharp corners is fully compressed into a larger, softer body, developing contact between the sides of the small block and the surface of the larger block.

## Model Setup

### Geometry

- **Large Block** (bottom, soft):
  - Dimensions: 1.5 mm × 1.5 mm × 1.125 mm (width × depth × height)
  - Position: Base at z = 0

- **Small Block** (top, stiff indenter):
  - Dimensions: 0.5 mm × 0.5 mm × 0.5 mm (cube)
  - Initial position: Centered on top of large block, separated by 0.0375 mm gap
  - Initial z-position: z = 1.125 + 0.0375 = 1.1625 mm (bottom surface)

### Material Properties

- **Large Block** (soft, deformable):
  - Young's modulus: E = 1 MPa
  - Poisson's ratio: ν = 0.3
  - Material model: Compressible Neo-Hookean

- **Small Block** (stiff, indenter):
  - Young's modulus: E = 100 MPa
  - Poisson's ratio: ν = 0.3
  - Material model: Compressible Neo-Hookean

### Loading Conditions

- **Prescribed Displacement**: Applied to top surface of small block
  - uz = -0.6 mm (total indentation including gap = 0.6375 mm)
  - Applied linearly over 1 second
  - 100 evenly spaced time steps

### Contact Configuration

- **Contact Method**: Surface-to-surface (single-pass)
- **Enforcement**: Augmented Lagrangian with gap tolerance
- **Contact Type**: Frictional (Coulomb)
- **Friction Coefficient**: μ = 0.2
- **Penalty Parameter**: α = 150
- **Gap Tolerance**: Gtol = 0.0009 mm
- **Primary Surface**: Large block surface (for single-pass analysis)

### Boundary Conditions

- **Fixed**: Bottom surface of large block (all DOFs constrained)
- **Prescribed Displacement**: Top surface of small block (uz = -0.6 mm)
- **Contact Surfaces**:
  - All exposed surfaces of small block (including bottom and sides)
  - Top surface of large block
  - Side surfaces of large block (as soft material rises)

### Mesh Configuration

- **Large Block**: Well-refined to capture stress gradients near indenter
- **Small Block**: Coarser mesh in vertical direction (noted in paper)
- Linear hexahedral elements
- Adequate refinement to resolve contact at sharp corners

## Expected Behavior

This benchmark is particularly challenging due to:

1. **Sharp Corners**: The small block has sharp 90° corners that introduce stress singularities in the contact pressure

2. **Side Contact Development**: As indentation progresses, the soft material rises up against the sides of the indenting block, creating multi-surface contact

3. **Large Deformation**: The soft block undergoes significant deformation (>50% compression in contact zone)

4. **Stress Concentrations**: Peak contact pressures occur at the corners and along edges of the indenter

The paper notes that:
- Examples with sharp corners have historically motivated mortar formulations
- The proposed algorithm handles this without difficulty despite sharp corners
- Contact pressure shows large peaks at corners as expected
- Elevated pressures along edges of the indenter
- Reasonable agreement with Temizer (2012) mortar results, though slopes differ slightly
- Differences may reflect: different contact formulations, constitutive models, corner effects, or friction influence
- Frictionless case (μ = 0) shows virtually identical reaction force to frictional case

## Running the Model

```bash
# From the project root directory
./run_model deep-indentation
```

## Output

The simulation produces ParaView-compatible output files showing:
- Displacement fields for both blocks
- Contact pressure distribution with corner singularities
- Stress distribution in soft block
- Deformed geometry showing material rise
- Vertical reaction force vs. displacement curve
- Evolution of contact patch (bottom → bottom+sides)

## Reference

Zimmerman, B. K., & Ateshian, G. A. (2018). **A Surface-to-Surface Finite Element Algorithm for Large Deformation Frictional Contact in FEBIO**. _Journal of Biomechanical Engineering_, 140(8), 081013.

DOI: [10.1115/1.4040497](https://doi.org/10.1115/1.4040497)

Available at: https://pmc.ncbi.nlm.nih.gov/articles/PMC6056201/

Also compared with:
Temizer, I. (2012). **A Mixed Formulation of Mortar-Based Frictionless Contact**. _Computer Methods in Applied Mechanics and Engineering_, 223-224, 173-185.

## Notes

- The sharp corners are essential to this benchmark - do not round them
- Single-pass analysis recommended (large block as primary surface)
- Coarse mesh on indenter in vertical direction is intentional from the paper
- Very high penalty parameter (α = 150) needed due to stiffness contrast
- Gap tolerance convergence (Gtol) used instead of penalty tolerance (Ptol)
- Contact pressure is singular at corners - this is physical behavior
- Results may differ from mortar methods due to fundamental formulation differences
- The model successfully demonstrates robustness despite numerical challenges
- Can be run with μ = 0 to verify that friction has minimal effect on reaction force
