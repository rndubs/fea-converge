# Stiff Block on Flexible Slab Contact Model

## Description

This model implements a variant of Example 2 from the Puso & Laursen (2003) paper on mortar segment-to-segment frictional contact methods. A 1×1 stiff block is pressed into a flexible slab and then slid horizontally across the surface.

## Model Setup

### Geometry

- **Flexible Slab**: 9 × 4 × 3 units
- **Stiff Block**: 1 × 1 × 0.2 units
  - Initial position: centered on slab surface

### Material Properties

- **Slab** (flexible):
  - Young's modulus: E = 1.0
  - Poisson's ratio: ν = 0.3
  - Material model: Neo-Hookean

- **Block** (stiff):
  - Young's modulus: E = 1000.0
  - Poisson's ratio: ν = 0.499
  - Material model: Neo-Hookean

### Loading Conditions

1. **Pressing Phase** (t = 0.0 → 0.2):
   - Block moves downward (-z direction) by 1.4 units

2. **Sliding Phase** (t = 0.2 → 1.5):
   - Block slides horizontally (+x direction) by 4.0 units
   - Maintains vertical displacement of -1.4 units

### Contact Configuration

- **Contact Method**: Mortar segment-to-segment
- **Enforcement**: Penalty method
- **Contact Type**: Frictional
- **Friction Coefficient**: μ = 0.3 (Coulomb friction)
- **Penalty Parameter**: 500.0

### Boundary Conditions

- **Fixed**: Bottom surface of slab (attribute 5)
- **Prescribed Displacement**: Top surface of block (attribute 12)
- **Contact Surfaces**:
  - Slab top surface (attribute 6)
  - Block bottom surface (attribute 11)

## Expected Behavior

This model demonstrates the robustness of the mortar segment-to-segment method when:
- Sharp corners of the block create nearly singular stress concentrations
- Large sliding with friction occurs
- Flexible-to-flexible contact (both bodies deform)

The paper notes that this problem is particularly challenging due to:
- Stress singularities at block corners
- Need for refined mesh near corners
- Difficulty in achieving convergence with node-on-segment methods

## Running the Model

```bash
# From the project root directory
./run_model block-on-slab
```

## Output

The simulation produces ParaView-compatible output files showing:
- Displacement fields
- Contact pressure distribution (with stress concentrations at corners)
- Stress distribution
- Deformed geometry at each timestep

## Reference

Puso, M. A., & Laursen, T. A. (2003). **A Mortar Segment-to-Segment Frictional Contact Method for Large Deformations**. _Computer Methods in Applied Mechanics and Engineering_.

Available at: https://www.osti.gov/servlets/purl/15013715

## Notes

- This model requires a mesh file for the 1×1 block and slab geometry
- Mesh refinement near block corners is recommended for accuracy
- The mesh file should be generated with appropriate element attributes for material assignment
- Contact surface attributes must match those specified in the code
- MFEM must be built with STRUMPACK support for the linear solver
- Some oscillations in contact forces are expected due to stress singularities at corners
