// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

// Hollow Sphere Pinching Contact Model
// Based on Figure 19 and Section 3.10 from Zimmerman & Ateshian (2018)
// "A Surface-to-Surface Finite Element Algorithm for Large Deformation
// Frictional Contact in FEBIO"
// Reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC6056201/

#include <cmath>
#include <set>
#include <string>
#include <vector>
#include <memory>

#include "axom/slic.hpp"
#include "mfem.hpp"
#include "smith/smith.hpp"

int main(int argc, char* argv[])
{
  // Initialize and automatically finalize MPI and other libraries
  smith::ApplicationManager applicationManager(argc, argv);

  // NOTE: p must be equal to 1 to work with Tribol's mortar method
  constexpr int p = 1;
  // NOTE: dim must be equal to 3
  constexpr int dim = 3;

  // Create DataStore
  std::string name = "hollow_sphere_pinching";
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, name + "_data");

  // Geometry parameters (all dimensions in mm)
  const double sphere_outer_radius = 25.0;  // 2.5 cm
  const double sphere_inner_radius = 22.5;  // 2.25 cm
  const double finger_tube_outer_radius = 10.0;  // 1 cm
  const double finger_tube_inner_radius = 5.0;   // 0.5 cm
  const double finger_tube_length = 100.0;       // 10 cm

  // Construct hollow sphere mesh
  // TODO: This requires a hollow sphere mesh with Ro=25, Ri=22.5
  // For now, using scaled ball-nurbs mesh as placeholder
  int num_refinements_sphere{3};
  mfem::Mesh sphere_mesh{SMITH_REPO_DIR "/data/meshes/ball-nurbs.mesh"};
  for (int i{0}; i < num_refinements_sphere; ++i) {
    sphere_mesh.UniformRefinement();
  }
  sphere_mesh.SetCurvature(p);
  sphere_mesh.Transform([sphere_outer_radius](const mfem::Vector& x, mfem::Vector& p) {
    p.SetSize(3);
    p(0) = sphere_outer_radius * x(0);
    p(1) = sphere_outer_radius * x(1);
    p(2) = sphere_outer_radius * x(2);
  });

  // Construct finger 1 mesh (left finger)
  // TODO: This requires a hollow tube with hemispherical cap
  // For now, using a simplified cylinder mesh
  int num_refinements_finger{3};
  mfem::Mesh finger1_mesh{SMITH_REPO_DIR "/data/meshes/onehex.mesh"};
  for (int i{0}; i < num_refinements_finger; ++i) {
    finger1_mesh.UniformRefinement();
  }
  finger1_mesh.SetCurvature(p);
  // Position finger 1 to the left of sphere
  finger1_mesh.Transform([finger_tube_length, finger_tube_outer_radius](const mfem::Vector& x, mfem::Vector& p) {
    p.SetSize(3);
    p(0) = -finger_tube_length * 0.5 - 30.0;  // Position to left of sphere
    p(1) = finger_tube_outer_radius * (x(1) - 0.5);
    p(2) = finger_tube_outer_radius * (x(2) - 0.5);
  });

  // Construct finger 2 mesh (right finger)
  mfem::Mesh finger2_mesh{SMITH_REPO_DIR "/data/meshes/onehex.mesh"};
  for (int i{0}; i < num_refinements_finger; ++i) {
    finger2_mesh.UniformRefinement();
  }
  finger2_mesh.SetCurvature(p);
  // Position finger 2 to the right of sphere
  finger2_mesh.Transform([finger_tube_length, finger_tube_outer_radius](const mfem::Vector& x, mfem::Vector& p) {
    p.SetSize(3);
    p(0) = finger_tube_length * 0.5 + 30.0;  // Position to right of sphere
    p(1) = finger_tube_outer_radius * (x(1) - 0.5);
    p(2) = finger_tube_outer_radius * (x(2) - 0.5);
  });

  std::vector<mfem::Mesh*> mesh_ptrs{&sphere_mesh, &finger1_mesh, &finger2_mesh};
  auto mesh = std::make_shared<smith::Mesh>(
      mfem::Mesh(mesh_ptrs.data(), static_cast<int>(mesh_ptrs.size())),
      "hollow_sphere_pinching_mesh", 0, 0);

  // Define boundary domains
  // TODO: Define appropriate boundaries for revolute joint and contact surfaces

  smith::LinearSolverOptions linear_options{.linear_solver = smith::LinearSolver::Strumpack, .print_level = 0};
#ifndef MFEM_USE_STRUMPACK
  SLIC_INFO_ROOT("Contact requires MFEM built with strumpack.");
  return 1;
#endif

  smith::NonlinearSolverOptions nonlinear_options{.nonlin_solver = smith::NonlinearSolver::Newton,
                                                  .relative_tol = 1.0e-10,
                                                  .absolute_tol = 1.0e-10,
                                                  .max_iterations = 200,
                                                  .print_level = 1};

  // Contact options - high friction for grasping
  smith::ContactOptions contact_options{.method = smith::ContactMethod::SingleMortar,
                                        .enforcement = smith::ContactEnforcement::AugmentedLagrangian,
                                        .type = smith::ContactType::Frictional,
                                        .penalty = 4.0,
                                        .friction_coeff = 0.9,
                                        .jacobian = smith::ContactJacobian::Exact};

  // TODO: Need to support Mooney-Rivlin material for sphere
  // For now, using Neo-Hookean as approximation
  smith::SolidMechanicsContact<p, dim, smith::Parameters<smith::L2<0>, smith::L2<0>>> solid_solver(
      nonlinear_options, linear_options, smith::solid_mechanics::default_quasistatic_options, name, mesh,
      {"bulk_mod", "shear_mod"});

  // Material properties from Zimmerman 2018 paper
  // Sphere: Mooney-Rivlin with c1 = 1.25 MPa, c2 = 0, kappa = 1250 MPa
  //         For Neo-Hookean approximation: G = 2*c1 = 2.5 MPa, K = 1250 MPa
  // Fingers: E = 1 MPa, nu = 0.3
  // Convert to bulk and shear moduli: K = E/(3(1-2*nu)), G = E/(2(1+nu))

  smith::FiniteElementState K_field(smith::StateManager::newState(smith::L2<0>{}, "bulk_mod", mesh->tag()));
  // Sphere: K = 1250 MPa (nearly incompressible)
  // Finger: K = 1/(3*(1-0.6)) = 0.833 MPa
  mfem::Vector K_values({1250.0, 0.833, 0.833});
  mfem::PWConstCoefficient K_coeff(K_values);
  K_field.project(K_coeff);
  solid_solver.setParameter(0, K_field);

  smith::FiniteElementState G_field(smith::StateManager::newState(smith::L2<0>{}, "shear_mod", mesh->tag()));
  // Sphere: G = 2*c1 = 2.5 MPa
  // Finger: G = 1/(2*(1+0.3)) = 0.385 MPa
  mfem::Vector G_values({2.5, 0.385, 0.385});
  mfem::PWConstCoefficient G_coeff(G_values);
  G_field.project(G_coeff);
  solid_solver.setParameter(1, G_field);

  smith::solid_mechanics::ParameterizedNeoHookeanSolid mat{1.0, 0.0, 0.0};
  solid_solver.setMaterial(smith::DependsOn<0, 1>{}, mat, mesh->entireBody());

  // TODO: Apply revolute joint constraint connecting the two fingers
  // This requires special joint/constraint capabilities in Smith

  // TODO: Apply rigid body constraints to inner surfaces of fingers
  // This simulates the rigid bone underlying the soft tissue

  // Apply pinching motion via prescribed displacement/rotation
  // 15 degree rotation over 1 second (20 time steps)
  // TODO: This requires revolute joint or rotational boundary conditions

  // Add the contact interactions
  // Contact pair 1: Finger 1 / Sphere
  auto contact_interaction_1 = 0;
  std::set<int> finger1_outer_surface({/* Outer surface attributes */});
  std::set<int> sphere_surface({/* Sphere surface attributes */});
  // NOTE: Actual surface attributes depend on mesh generation
  solid_solver.addContactInteraction(contact_interaction_1, finger1_outer_surface, sphere_surface, contact_options);

  // Contact pair 2: Finger 2 / Sphere
  auto contact_interaction_2 = 1;
  std::set<int> finger2_outer_surface({/* Outer surface attributes */});
  solid_solver.addContactInteraction(contact_interaction_2, finger2_outer_surface, sphere_surface, contact_options);

  // Finalize the data structures
  solid_solver.completeSetup();

  std::string paraview_name = name + "_paraview";
  solid_solver.outputStateToDisk(paraview_name);

  // Perform the quasi-static solve
  // 20 uniformly spaced time steps over 1 second
  double dt = 0.05;

  for (int i{0}; i < 20; ++i) {
    solid_solver.advanceTimestep(dt);

    // Output the sidre-based plot files
    solid_solver.outputStateToDisk(paraview_name);
  }

  return 0;
}
