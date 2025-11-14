// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

// Solid Sphere in Hollow Sphere Contact Model
// Based on Example 3 from Puso & Laursen (2003)
// "A Mortar Segment-to-Segment Frictional Contact Method for Large Deformations"
// Reference: https://www.osti.gov/servlets/purl/15013715

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
  std::string name = "sphere_in_sphere";
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, name + "_data");

  // Construct sphere meshes
  // Solid sphere: R = 0.6, E = 1.0, nu = 0.3
  // Hollow sphere: Ro = 2.0, Ri = 0.7, E = 1.0, nu = 0.0
  int num_refinements{3};

  // Create solid sphere mesh (inner sphere)
  mfem::Mesh solid_sphere_mesh{SMITH_REPO_DIR "/data/meshes/ball-nurbs.mesh"};
  for (int i{0}; i < num_refinements; ++i) {
    solid_sphere_mesh.UniformRefinement();
  }
  solid_sphere_mesh.SetCurvature(p);
  // TODO: Scale to R=0.6

  // Create hollow sphere mesh (outer sphere)
  // TODO: This would require a hollow sphere mesh with Ro=2.0, Ri=0.7
  // For now, using a placeholder cube mesh
  mfem::Mesh hollow_sphere_mesh{SMITH_REPO_DIR "/data/meshes/onehex.mesh"};
  for (int i{0}; i < num_refinements; ++i) {
    hollow_sphere_mesh.UniformRefinement();
  }
  hollow_sphere_mesh.SetCurvature(p);

  std::vector<mfem::Mesh*> mesh_ptrs{&solid_sphere_mesh, &hollow_sphere_mesh};
  auto mesh = std::make_shared<smith::Mesh>(
      mfem::Mesh(mesh_ptrs.data(), static_cast<int>(mesh_ptrs.size())),
      "sphere_in_sphere_mesh", 0, 0);

  // Define boundary domains
  mesh->addDomainOfBoundaryElements("outer_boundary", smith::by_attr<dim>(3));

  smith::LinearSolverOptions linear_options{.linear_solver = smith::LinearSolver::Strumpack, .print_level = 0};
#ifndef MFEM_USE_STRUMPACK
  SLIC_INFO_ROOT("Contact requires MFEM built with strumpack.");
  return 1;
#endif

  smith::NonlinearSolverOptions nonlinear_options{.nonlin_solver = smith::NonlinearSolver::Newton,
                                                  .relative_tol = 1.0e-13,
                                                  .absolute_tol = 1.0e-13,
                                                  .max_iterations = 200,
                                                  .print_level = 1};

  // Contact options - frictionless contact for this example
  smith::ContactOptions contact_options{.method = smith::ContactMethod::SingleMortar,
                                        .enforcement = smith::ContactEnforcement::Penalty,
                                        .type = smith::ContactType::Frictionless,
                                        .penalty = 1.0e4,
                                        .jacobian = smith::ContactJacobian::Exact};

  smith::SolidMechanicsContact<p, dim, smith::Parameters<smith::L2<0>, smith::L2<0>>> solid_solver(
      nonlinear_options, linear_options, smith::solid_mechanics::default_quasistatic_options, name, mesh,
      {"bulk_mod", "shear_mod"});

  // Material properties from Puso 2003 paper
  // Solid sphere: E = 1.0, nu = 0.3
  // Hollow sphere: E = 1.0, nu = 0.0
  // Convert to bulk and shear moduli: K = E/(3(1-2*nu)), G = E/(2(1+nu))

  smith::FiniteElementState K_field(smith::StateManager::newState(smith::L2<0>{}, "bulk_mod", mesh->tag()));
  // Solid sphere: K = 1.0/(3*(1-0.6)) = 0.833
  // Hollow sphere: K = 1.0/(3*(1-0)) = 0.333
  mfem::Vector K_values({0.833, 0.333});
  mfem::PWConstCoefficient K_coeff(K_values);
  K_field.project(K_coeff);
  solid_solver.setParameter(0, K_field);

  smith::FiniteElementState G_field(smith::StateManager::newState(smith::L2<0>{}, "shear_mod", mesh->tag()));
  // Solid sphere: G = 1.0/(2*(1+0.3)) = 0.385
  // Hollow sphere: G = 1.0/(2*(1+0)) = 0.5
  mfem::Vector G_values({0.385, 0.5});
  mfem::PWConstCoefficient G_coeff(G_values);
  G_field.project(G_coeff);
  solid_solver.setParameter(1, G_field);

  smith::solid_mechanics::ParameterizedNeoHookeanSolid mat{1.0, 0.0, 0.0};
  solid_solver.setMaterial(smith::DependsOn<0, 1>{}, mat, mesh->entireBody());

  // Fix outer boundary of hollow sphere
  solid_solver.setFixedBCs(mesh->domain("outer_boundary"));

  // Apply body force to solid sphere
  // The paper mentions uniform body force applied to the solid sphere
  // This would need to be implemented as a body force field
  // TODO: Add body force capability

  // Add the contact interaction between inner and outer sphere surfaces
  auto contact_interaction_id = 0;
  std::set<int> solid_sphere_surface_attributes({5});   // Surface of solid sphere
  std::set<int> hollow_sphere_inner_surface_attributes({7});  // Inner surface of hollow sphere
  solid_solver.addContactInteraction(contact_interaction_id, solid_sphere_surface_attributes,
                                     hollow_sphere_inner_surface_attributes, contact_options);

  // Finalize the data structures
  solid_solver.completeSetup();

  std::string paraview_name = name + "_paraview";
  solid_solver.outputStateToDisk(paraview_name);

  // Perform the quasi-static solve
  double dt = 1.0;

  for (int i{0}; i < 35; ++i) {
    solid_solver.advanceTimestep(dt);

    // Output the sidre-based plot files
    solid_solver.outputStateToDisk(paraview_name);
  }

  return 0;
}
