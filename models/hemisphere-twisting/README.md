# Hemisphere Twisting on Box Contact Model

## Description

This model implements the hemisphere twisting contact example from Figure 7 and Section 3.6 of Zimmerman & Ateshian (2018). A thick-walled hollow hemisphere indents a deformable box and then rotates, developing large shear deformations and twisting contact between the hemisphere and box.

## Model Setup

### Geometry

- **Box**: L × L × L cube where L = 2 mm
  - Bottom body in the contact pair

- **Thick-walled Hollow Hemisphere**:
  - Outer radius: 1 mm
  - Inner radius: 0.5 mm
  - Wall thickness: 0.5 mm
  - Initially positioned above the box with bottom surface at z = L

### Material Properties

- **Box** (deformable):
  - Young's modulus: E = 10 MPa
  - Poisson's ratio: ν = 0.3
  - Material model: Compressible Neo-Hookean

- **Hemisphere** (stiff):
  - Young's modulus: E = 50 MPa
  - Poisson's ratio: ν = 0.3
  - Material model: Compressible Neo-Hookean

### Loading Conditions

1. **Indentation Phase** (t = 0.0 → 1.0):
   - Hemisphere moves downward by uz = -1 mm
   - Applied to top surface of hemisphere
   - Ramped linearly over time

2. **Twisting Phase** (t = 1.0 → 10.0):
   - Hemisphere rotates Rz = π radians (180°)
   - Vertical displacement held constant at uz = -1 mm
   - Applied as prescribed rotation about z-axis

### Contact Configuration

- **Contact Method**: Surface-to-surface with ray-tracing projection
- **Enforcement**: Augmented Lagrangian
- **Contact Type**: Frictional (Coulomb)
- **Friction Coefficient**: μ = 0.5
- **Penalty Parameter**: α = 10
- **Augmentation Tolerance**: Ptol = 0.05

### Boundary Conditions

- **Fixed**: Bottom surface of box (all DOFs constrained)
- **Prescribed Displacement/Rotation**: Top surface of hemisphere
  - Phase 1: uz = -1 mm (indentation)
  - Phase 2: Rz = π rad (twisting), uz = -1 mm (held)
- **Contact Surfaces**:
  - Outer surface of hemisphere (curved)
  - Top surface of box (planar)

### Mesh Configuration

- Adequate refinement to capture curved hemisphere geometry
- Sufficient elements to resolve contact patch evolution
- Linear hexahedral elements for box
- Curved elements for hemisphere

## Expected Behavior

This problem demonstrates several challenging contact scenarios:

1. **Transition from Stick to Slip**: Contact initially sticks during indentation, then transitions to predominantly slipping behavior during twisting

2. **Large Shear Deformations**: The box experiences significant shear deformation as the hemisphere rotates

3. **Evolving Contact Patch**: The contact area changes shape and location as the hemisphere twists

4. **Torque Evolution**: The applied torque initially increases during stick, then plateaus during slip at approximately Mtorque = μ × Fnormal × rcontact

The paper notes that:
- During twisting, contact transitions from predominantly sticking to predominantly slipping
- The torque curve shows initial increase then levels off during pure sliding
- Slight oscillations in torque during sliding are due to stick-slip instabilities
- Results show strong agreement with Sauer & De Lorenzis (2015)

## Running the Model

```bash
# From the project root directory
./run_model hemisphere-twisting
```

## Output

The simulation produces ParaView-compatible output files showing:
- Displacement fields for both hemisphere and box
- Rotation of the hemisphere
- Contact pressure distribution (evolving during twist)
- Stick-slip status at contact points
- Shear deformation in the box
- Torque vs. rotation angle curve

## Reference

Zimmerman, B. K., & Ateshian, G. A. (2018). **A Surface-to-Surface Finite Element Algorithm for Large Deformation Frictional Contact in FEBIO**. _Journal of Biomechanical Engineering_, 140(8), 081013.

DOI: [10.1115/1.4040497](https://doi.org/10.1115/1.4040497)

Available at: https://pmc.ncbi.nlm.nih.gov/articles/PMC6056201/

Also compared with:
Sauer, R. A., & De Lorenzis, L. (2015). **An Unbiased Computational Contact Formulation for 3D Friction**. _International Journal for Numerical Methods in Engineering_, 101(4), 251-280.

## Notes

- This model requires a hollow hemisphere mesh with specified inner/outer radii
- The hemisphere mesh should have adequate refinement to capture curvature
- Prescribing both displacement and rotation requires careful constraint management
- The twisting motion creates large sliding displacements (> hemisphere diameter)
- Stick-slip oscillations are physical and expected during the transition phase
- The torque should plateau at approximately T ≈ 0.5 × Fnormal × rcontact during pure slip
- 100 time steps recommended (10 for indentation, 90 for twisting)
