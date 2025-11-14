# Hollow Sphere Pinching Contact Model

## Description

This model implements the hollow sphere pinching example from Figure 19 and Section 3.10 of Zimmerman & Ateshian (2018). A hollow sphere is compressed between two deformable "finger" surfaces in a pinching motion, demonstrating the biomechanical relevance of frictional interactions in contact problems.

## Model Setup

### Geometry

- **Hollow Sphere** (center):
  - Outer radius: Ro = 2.5 cm = 25 mm
  - Inner radius: Ri = 2.25 cm = 22.5 mm
  - Wall thickness: 2.5 mm
  - Initially in grazing contact with both fingers

- **Deformable Fingers** (two):
  - Hollow tubes with hemispherical caps
  - Tube outer radius: 10 mm
  - Tube inner radius: 5 mm
  - Tube length: 100 mm
  - Hemisphere cap: outer radius 10 mm, inner radius 5 mm
  - Connected by revolute joint at intersection of centerlines

### Material Properties

- **Hollow Sphere** (nearly incompressible):
  - Mooney-Rivlin material
  - c1 = 1.25 MPa
  - c2 = 0 MPa
  - Bulk modulus: κ = 1250 MPa
  - Nearly incompressible (κ/c1 = 1000)

- **Fingers** (soft, deformable):
  - Young's modulus: E = 1 MPa
  - Poisson's ratio: ν = 0.3
  - Material model: Compressible Neo-Hookean
  - Represents soft tissue pads of fingers

- **Rigid Core** (inside each finger):
  - Rigid body constraint
  - Attached to inner surface of each finger tube
  - Prevents inner surface deformation (simulates bone)

### Loading Conditions

- **Revolute Joint Rotation**: 15° pinching motion
  - Applied over 1 second
  - 20 uniformly spaced time steps
  - Causes fingers to close together, compressing sphere

### Contact Configuration

- **Number of Contact Pairs**: 2 (one per finger-sphere interface)
- **Contact Method**: Single-pass surface-to-surface
- **Primary Surface**: Finger surfaces (for each pair)
- **Enforcement**: Augmented Lagrangian
- **Contact Type**: Frictional (Coulomb)
- **Friction Coefficient**: μ = 0.9 (high friction)
- **Penalty Parameter**: α = 4
- **Gap Tolerance**: Gtol = 0.01 cm = 0.1 mm

### Boundary Conditions

- **Revolute Joint**: Connects two fingers at intersection of centerlines
  - Allows rotation in pinching plane
  - Constrains other motions

- **Rigid Core**: Attached to inner surface of each finger
  - Prevents deformation of inner tube surface
  - Simulates underlying rigid bone structure

- **Contact Surfaces**:
  - Outer surfaces of both finger tubes/hemispheres
  - Outer surface of hollow sphere

### Mesh Configuration

- Adequate refinement to capture contact evolution
- Sphere mesh should be symmetric
- Finger meshes need refinement at contact zones
- Linear hexahedral elements

## Expected Behavior

This model demonstrates biomechanically relevant frictional contact behavior:

1. **Stick Behavior**: During compression, the sphere initially sticks to the finger surfaces and deforms

2. **Sphere Buckling**: Around t = 0.7 s (14° rotation), the sphere buckles inward and large sections lose contact with each finger

3. **Finger Deformation**: The soft finger material deforms as it is sandwiched between the rigid core (bone) and the sphere surface

4. **Increased Contact Area**: Finger deformation increases the contact surface area, which increases the friction force holding the ball

5. **Plowing Component**: The deformed finger geometry adds a "plowing" component to the friction force, further trapping the ball

6. **Failure Without Friction**: A frictionless simulation fails at t = 0.06 s when the ball loses contact and jumps away

The paper notes that:
- Friction is essential for this problem - frictionless case fails after 2 time steps
- The inclusion of friction allows finger deformation to develop
- Finger deformation creates a mechanical "trap" for the sphere
- High friction (μ = 0.9) prevents sliding and maintains stable grip
- This demonstrates necessity of frictional contact for certain biomechanical behaviors

## Running the Model

```bash
# From the project root directory
./run_model hollow-sphere-pinching
```

## Output

The simulation produces ParaView-compatible output files showing:
- Displacement fields for sphere and fingers
- Rotation of the revolute joint
- Contact pressure distribution (evolving)
- Deformation of finger tissues
- Buckling of the hollow sphere
- Evolution of contact patches (growing then shrinking)
- Loss and re-establishment of contact regions

## Reference

Zimmerman, B. K., & Ateshian, G. A. (2018). **A Surface-to-Surface Finite Element Algorithm for Large Deformation Frictional Contact in FEBIO**. _Journal of Biomechanical Engineering_, 140(8), 081013.

DOI: [10.1115/1.4040497](https://doi.org/10.1115/1.4040497)

Available at: https://pmc.ncbi.nlm.nih.gov/articles/PMC6056201/

## Notes

- This model demonstrates the biomechanical utility of the frictional contact algorithm
- The revolute joint implementation requires special joint/constraint capabilities in Smith
- Rigid body constraints for the inner "bone" cores are essential to the model
- The Mooney-Rivlin material for the sphere requires nonlinear material capability
- High friction coefficient (μ = 0.9) is typical for soft biological tissues
- Frictionless analysis can be run for comparison (expect failure at ~t = 0.06 s)
- The model exhibits complex contact evolution with loss and regain of contact
- Finger deformation is critical - without it, the ball would slip out
- This example shows how friction enables stable grasping in biomechanics
- Gap tolerance of 0.1 mm provides adequate constraint enforcement
- 20 time steps provide good temporal resolution of the pinching motion
