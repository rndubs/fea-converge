# Die on Slab Contact Model

## Description

This model implements the "Ironing" test case (Example 2) from the Puso & Laursen (2003) paper on mortar segment-to-segment frictional contact methods. A cylindrical die is pressed into a flexible slab and then slid horizontally across the surface.

## Model Setup

### Geometry

- **Flexible Slab**: 9 × 4 × 3 units
- **Cylindrical Die**:
  - Thickness: 0.2 units
  - Width: 5.2 units
  - Radius: 3.0 units
  - Initial position: centered 2.5 units from left end of slab

### Material Properties

- **Slab** (flexible):
  - Young's modulus: E = 1.0
  - Poisson's ratio: ν = 0.3
  - Material model: Neo-Hookean

- **Die** (stiff):
  - Young's modulus: E = 1000.0
  - Poisson's ratio: ν = 0.499
  - Material model: Neo-Hookean

### Loading Conditions

1. **Pressing Phase** (t = 0.0 → 0.2):
   - Die moves downward (-z direction) by 1.4 units

2. **Sliding Phase** (t = 0.2 → 1.5):
   - Die slides horizontally (+x direction) by 4.0 units
   - Maintains vertical displacement of -1.4 units

### Contact Configuration

- **Contact Method**: Mortar segment-to-segment
- **Enforcement**: Penalty method
- **Contact Type**: Frictional
- **Friction Coefficient**: μ = 0.3 (Coulomb friction)
- **Penalty Parameter**: 500.0

### Boundary Conditions

- **Fixed**: Bottom surface of slab (attribute 5)
- **Prescribed Displacement**: Top surface of die (attribute 12)
- **Contact Surfaces**:
  - Slab top surface (attribute 6)
  - Die bottom surface (attribute 11)

## Expected Behavior

As the die slides across the slab surface:
- High Poisson's ratio (ν = 0.499) of slab causes transverse expansion
- Material "squirts" out laterally along edges perpendicular to sliding direction
- Demonstrates the robustness of mortar method in handling:
  - Large sliding
  - Non-smooth contact surfaces
  - Nodes sliding off contact boundaries

## Running the Model

```bash
# From the project root directory
./run_model die-on-slab
```

## Output

The simulation produces ParaView-compatible output files showing:
- Displacement fields
- Contact pressure distribution
- Stress distribution
- Deformed geometry at each timestep

## Reference

Puso, M. A., & Laursen, T. A. (2003). **A Mortar Segment-to-Segment Frictional Contact Method for Large Deformations**. _Computer Methods in Applied Mechanics and Engineering_.

Available at: https://www.osti.gov/servlets/purl/15013715

## Notes

- This model requires a mesh file for the cylindrical die and slab geometry
- The mesh file should be generated with appropriate element attributes for material assignment
- Contact surface attributes must match those specified in the code
- MFEM must be built with STRUMPACK support for the linear solver
