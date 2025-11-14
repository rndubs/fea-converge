# Solid Sphere in Hollow Sphere Contact Model

## Description

This model implements Example 3 ("Spheres in Sphere") from the Puso & Laursen (2003) paper on mortar segment-to-segment frictional contact methods. A solid sphere is placed within a hollow sphere and a uniform body force is applied to press the solid sphere into the outer hollow sphere.

## Model Setup

### Geometry

- **Solid Sphere** (inner):
  - Radius: R = 0.6 units
  - Positioned at center

- **Hollow Sphere** (outer):
  - Outer radius: Ro = 2.0 units
  - Inner radius: Ri = 0.7 units
  - Wall thickness: 1.3 units

### Material Properties

- **Solid Sphere**:
  - Young's modulus: E = 1.0
  - Poisson's ratio: ν = 0.3
  - Material model: Neo-Hookean

- **Hollow Sphere**:
  - Young's modulus: E = 1.0
  - Poisson's ratio: ν = 0.0
  - Material model: Neo-Hookean

### Loading Conditions

- **Body Force**: Uniform body force applied to solid sphere
  - Direction: Downward (-z direction)
  - Magnitude: Constant throughout simulation
  - Applied for 35 time steps

### Contact Configuration

- **Contact Method**: Mortar segment-to-segment
- **Enforcement**: Penalty method
- **Contact Type**: Frictionless
- **Penalty Parameter**: 10,000.0

### Boundary Conditions

- **Fixed**: Outer surface of hollow sphere (attribute 3)
- **Body Force**: Applied to entire solid sphere
- **Contact Surfaces**:
  - Outer surface of solid sphere (attribute 5)
  - Inner surface of hollow sphere (attribute 7)

## Expected Behavior

This is a truly flexible-to-flexible contact problem where:
- Both spheres undergo significant deformation
- The solid sphere is pressed downward into the hollow sphere
- Contact area evolves as deformation progresses
- Single-pass node-on-segment methods are not applicable (both bodies deform)

The paper notes that this problem is particularly challenging because:
- Node-on-node contact at initial point creates force oscillations
- Two-pass node-on-segment approaches can fail at first timestep without special techniques
- Smooth two-pass node-on-segment methods may fail due to locking
- The mortar method successfully handles all these challenges

## Running the Model

```bash
# From the project root directory
./run_model sphere-in-sphere
```

## Output

The simulation produces ParaView-compatible output files showing:
- Displacement fields for both spheres
- Contact pressure distribution
- Stress distribution
- Deformed geometry at each timestep
- Evolution of contact patch

## Reference

Puso, M. A., & Laursen, T. A. (2003). **A Mortar Segment-to-Segment Frictional Contact Method for Large Deformations**. _Computer Methods in Applied Mechanics and Engineering_.

Available at: https://www.osti.gov/servlets/purl/15013715

## Notes

- This model requires mesh files for both the solid sphere and hollow sphere
- The solid sphere mesh can be generated from the ball-nurbs.mesh with appropriate scaling
- The hollow sphere mesh needs to be generated with inner radius Ri=0.7 and outer radius Ro=2.0
- Mesh refinement should be adequate to capture contact patch evolution
- MFEM must be built with STRUMPACK support for the linear solver
- The body force implementation may need to be added if not already available in Smith
- Contact surface attributes must match those specified in the code
